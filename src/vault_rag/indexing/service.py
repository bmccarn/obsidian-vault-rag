"""Deterministic incremental reconciliation of vault sources into SQLite."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np  # pyright: ignore[reportMissingImports]

from vault_rag.config import EgressPolicy, EmbeddingConfig, ResolvedProfile, ResolvedVault
from vault_rag.domain import JsonValue, SourceKind
from vault_rag.embedding import (
    EmbeddingBatch,
    configuration_fingerprint,
    observed_fingerprint,
)
from vault_rag.errors import ChunkingError, ParseError, RebuildRequiredError, StorageError
from vault_rag.ingest import (
    DiscoveryFailure,
    ParsedSource,
    discover_sources,
    parse_markdown,
    parse_text,
)
from vault_rag.ingest.chunker import (
    CHUNKER_SCHEMA_VERSION,
    ChunkRecord,
    TiktokenCounter,
    TokenCounter,
    chunk_source,
)
from vault_rag.ingest.models import DiscoveredSource, SourceSection
from vault_rag.security.paths import folded_path_key
from vault_rag.storage import (
    PendingEmbedding,
    PendingEmbeddingUpdate,
    SourceRevision,
    StoredChunk,
    VaultFingerprint,
    VectorRecord,
)
from vault_rag.storage.ports import AtomicReplaceIndexStore, IndexStore

PARSER_SCHEMA_VERSION = 4
_MAX_DIAGNOSTICS = 100
_MAX_MESSAGE_LENGTH = 500


class Embedder(Protocol):
    """The indexing portion of the embedding client contract."""

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...


@dataclass(frozen=True, slots=True, order=True)
class IndexDiagnostic:
    vault_id: str
    path: str
    category: str
    message: str


@dataclass(frozen=True, slots=True)
class IndexReport:
    total_sources: int
    added_sources: int
    changed_sources: int
    unchanged_sources: int
    deleted_sources: int
    ready_chunks: int
    pending_chunks: int
    parse_failures: int
    embedding_failures: int
    embedding_requests: int
    diagnostics: tuple[IndexDiagnostic, ...]
    elapsed_ms: int
    blocking_failures: int = 0


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    source: DiscoveredSource
    chunks: tuple[ChunkRecord, ...]
    fingerprint: VaultFingerprint
    replacing_path: str | None


@dataclass(frozen=True, slots=True)
class _EmbeddedSource:
    prepared: _PreparedSource
    vectors: tuple[VectorRecord, ...]
    pending: tuple[PendingEmbedding, ...]


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _manifest_fingerprint(vault: ResolvedVault) -> str:
    return _digest(vault.manifest.model_dump(mode="json"))


def _parser_fingerprint() -> str:
    return _digest({"schema_version": PARSER_SCHEMA_VERSION})


def _chunker_fingerprint(config: EmbeddingConfig) -> str:
    return _digest(
        {
            "schema_version": CHUNKER_SCHEMA_VERSION,
            "tokenizer": config.tokenizer,
            "target_min_tokens": config.target_min_tokens,
            "target_max_tokens": config.target_max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "max_input_tokens": config.max_input_tokens,
        }
    )


def expected_vault_fingerprints(
    profile: ResolvedProfile, embedding_config: EmbeddingConfig
) -> dict[str, VaultFingerprint]:
    """Compute the exact persisted compatibility identity for every profile vault."""
    embedding_fingerprint = configuration_fingerprint(embedding_config, profile.embedding_model)
    parser_fingerprint = _parser_fingerprint()
    chunker_fingerprint = _chunker_fingerprint(embedding_config)
    return {
        vault.manifest.id: VaultFingerprint(
            vault_id=vault.manifest.id,
            manifest_fingerprint=_manifest_fingerprint(vault),
            parser_fingerprint=parser_fingerprint,
            chunker_fingerprint=chunker_fingerprint,
            embedding_config_fingerprint=embedding_fingerprint,
        )
        for vault in profile.vaults
    }


def _bounded_message(message: str) -> str:
    return " ".join(message.split())[:_MAX_MESSAGE_LENGTH] or "operation failed"


def _metadata_items(value: JsonValue) -> list[JsonValue]:
    return list(value) if isinstance(value, list) else [value]


def _merge_metadata_values(existing: JsonValue | None, additions: Sequence[str]) -> list[JsonValue]:
    values = [] if existing is None else _metadata_items(existing)
    seen = {
        json.dumps(item, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
        for item in values
    }
    for addition in additions:
        encoded = json.dumps(addition, ensure_ascii=False, separators=(",", ":"))
        if encoded not in seen:
            values.append(addition)
            seen.add(encoded)
    return values


def _section_for_chunk(
    sections: tuple[SourceSection, ...], chunk: ChunkRecord
) -> SourceSection | None:
    return next(
        (
            section
            for section in sections
            if section.heading == chunk.heading
            and section.lines.start <= chunk.lines.start
            and chunk.lines.end <= section.lines.end
        ),
        None,
    )


def _preserve_section_metadata(
    parsed: ParsedSource, chunks: tuple[ChunkRecord, ...]
) -> tuple[ChunkRecord, ...]:
    records: list[ChunkRecord] = []
    for chunk in chunks:
        section = _section_for_chunk(parsed.sections, chunk)
        metadata = dict(chunk.metadata)
        if section is not None:
            for key, values in (
                ("wikilinks", section.wikilinks),
                ("aliases", section.aliases),
                ("tags", section.tags),
            ):
                if values or key in metadata:
                    metadata[key] = _merge_metadata_values(metadata.get(key), values)
        records.append(replace(chunk, metadata=metadata))
    return tuple(records)


def _filtered_frontmatter(parsed: ParsedSource, vault: ResolvedVault) -> ParsedSource:
    selected: dict[str, JsonValue] = {}
    for field in vault.manifest.metadata.frontmatter_fields:
        if field in parsed.frontmatter:
            selected[field] = parsed.frontmatter[field]
    return replace(parsed, frontmatter=selected)


def _parse(source: DiscoveredSource) -> ParsedSource:
    if source.kind is SourceKind.MARKDOWN:
        return parse_markdown(source)
    return parse_text(source)


def _embedding_records(
    chunks: Sequence[ChunkRecord],
    result: EmbeddingBatch,
    config_fingerprint: str,
    attempted_at: datetime,
) -> tuple[
    tuple[VectorRecord, ...],
    tuple[PendingEmbedding, ...],
    tuple[tuple[int, str, str], ...],
]:
    failure_by_index = {failure.index: failure for failure in result.failures}
    valid_indexes = all(
        not isinstance(index, bool) and 0 <= index < len(chunks) for index in failure_by_index
    )
    valid_result = (
        valid_indexes
        and len(failure_by_index) == len(result.failures)
        and len(result.vectors) + len(failure_by_index) == len(chunks)
        and (not result.vectors or result.dimensions is not None)
    )
    if not valid_result:
        failures = tuple(
            PendingEmbedding(
                chunk.id,
                "invalid_response",
                "embedding result was inconsistent",
                attempted_at,
            )
            for chunk in chunks
        )
        details = tuple(
            (index, "invalid_response", "embedding result was inconsistent")
            for index in range(len(chunks))
        )
        return (), failures, details

    vectors: list[VectorRecord] = []
    pending: list[PendingEmbedding] = []
    failure_details: list[tuple[int, str, str]] = []
    vector_index = 0
    dimensions = result.dimensions
    for index, chunk in enumerate(chunks):
        failure = failure_by_index.get(index)
        if failure is not None:
            category = _bounded_message(failure.category)[:64]
            message = _bounded_message(failure.message)
            pending.append(PendingEmbedding(chunk.id, category, message, attempted_at))
            failure_details.append((index, category, message))
            continue
        vector = result.vectors[vector_index]
        vector_index += 1
        if dimensions is None or np.asarray(vector).size != dimensions:
            message = "embedding vector dimensions were inconsistent"
            pending.append(PendingEmbedding(chunk.id, "invalid_response", message, attempted_at))
            failure_details.append((index, "invalid_response", message))
            continue
        vectors.append(
            VectorRecord(
                chunk_id=chunk.id,
                dimensions=dimensions,
                config_fingerprint=config_fingerprint,
                observed_fingerprint=observed_fingerprint(config_fingerprint, dimensions),
                vector=vector,
            )
        )
    return tuple(vectors), tuple(pending), tuple(failure_details)


def _clone_reusable_vector(chunk: ChunkRecord, reusable: VectorRecord) -> VectorRecord:
    """Give each persisted chunk an independent, normalized vector record."""
    vector = np.asarray(reusable.vector, dtype=np.float32)
    if vector.size != reusable.dimensions:
        raise ValueError("reusable vector dimensions were inconsistent")
    return VectorRecord(
        chunk_id=chunk.id,
        dimensions=reusable.dimensions,
        config_fingerprint=reusable.config_fingerprint,
        observed_fingerprint=reusable.observed_fingerprint,
        vector=vector.copy(),
    )


def _is_compatible_reusable_vector(vector: VectorRecord, embedding_config_fingerprint: str) -> bool:
    return (
        vector.config_fingerprint == embedding_config_fingerprint
        and not isinstance(vector.dimensions, bool)
        and vector.dimensions > 0
        and vector.observed_fingerprint
        == observed_fingerprint(embedding_config_fingerprint, vector.dimensions)
        and np.asarray(vector.vector).size == vector.dimensions
    )


def _embed_with_reuse(
    chunks: Sequence[ChunkRecord],
    embedding_config_fingerprint: str,
    embedding: Embedder,
    *,
    store: IndexStore | None,
    reusable: dict[tuple[str, str], VectorRecord],
    failed: dict[tuple[str, str], PendingEmbedding],
) -> tuple[
    tuple[VectorRecord, ...],
    tuple[PendingEmbedding, ...],
    tuple[tuple[int, str, str], ...],
    int,
]:
    """Reuse compatible vectors and embed one representative per content hash."""
    unresolved = tuple(
        chunk
        for chunk in chunks
        if (embedding_config_fingerprint, chunk.content_hash) not in reusable
        and (embedding_config_fingerprint, chunk.content_hash) not in failed
    )
    if store is not None and unresolved:
        for content_hash, vector in store.reusable_vectors(
            unresolved, embedding_config_fingerprint
        ).items():
            if _is_compatible_reusable_vector(vector, embedding_config_fingerprint):
                reusable[(embedding_config_fingerprint, content_hash)] = vector

    to_embed: list[ChunkRecord] = []
    seen_hashes: set[str] = set()
    for chunk in chunks:
        key = (embedding_config_fingerprint, chunk.content_hash)
        if key not in reusable and key not in failed and chunk.content_hash not in seen_hashes:
            to_embed.append(chunk)
            seen_hashes.add(chunk.content_hash)

    if to_embed:
        attempted_at = datetime.now(UTC)
        created, failed_records, _failure_details = _embedding_records(
            to_embed,
            embedding.embed([chunk.embedding_text for chunk in to_embed]),
            embedding_config_fingerprint,
            attempted_at,
        )
        chunks_by_id = {chunk.id: chunk for chunk in to_embed}
        for vector in created:
            chunk = chunks_by_id[vector.chunk_id]
            reusable[(embedding_config_fingerprint, chunk.content_hash)] = vector
        for pending_record in failed_records:
            chunk = chunks_by_id[pending_record.chunk_id]
            failed[(embedding_config_fingerprint, chunk.content_hash)] = pending_record

    vectors: list[VectorRecord] = []
    pending: list[PendingEmbedding] = []
    failure_details: list[tuple[int, str, str]] = []
    for index, chunk in enumerate(chunks):
        key = (embedding_config_fingerprint, chunk.content_hash)
        if reusable_vector := reusable.get(key):
            vectors.append(_clone_reusable_vector(chunk, reusable_vector))
            continue
        failure = failed[key]
        pending.append(
            PendingEmbedding(chunk.id, failure.category, failure.message, failure.attempted_at)
        )
        failure_details.append((index, failure.category, failure.message))
    return tuple(vectors), tuple(pending), tuple(failure_details), len(to_embed)


class Indexer:
    """Incrementally reconcile one resolved profile with its persistent index."""

    def __init__(
        self,
        profile: ResolvedProfile,
        config: EmbeddingConfig,
        embedding: Embedder,
        store: IndexStore,
        *,
        counter: TokenCounter | None = None,
    ) -> None:
        self._profile = profile
        self._config = config
        self._embedding = embedding
        self._store = store
        self._counter = counter or TiktokenCounter(config.tokenizer)

    def run(self, rebuild: bool = False) -> IndexReport:
        """Reconcile discovered sources, atomically replacing the database on rebuild."""
        started = time.monotonic()
        if rebuild and isinstance(self._store, AtomicReplaceIndexStore):
            report = self._atomic_rebuild()
        elif rebuild:
            self._store.reset_vaults(self._validate_profile())
            report = self._reconcile(self._store, fail_on_storage_error=True)
        else:
            report = self._reconcile(self._store, fail_on_storage_error=False)
        return replace(report, elapsed_ms=max(0, int((time.monotonic() - started) * 1_000)))

    def _validate_profile(self) -> tuple[str, ...]:
        vault_ids = tuple(vault.manifest.id for vault in self._profile.vaults)
        if not vault_ids or len(set(vault_ids)) != len(vault_ids):
            raise ValueError("profile must contain unique vault IDs")
        if not self._profile.embedding_model.strip():
            raise ValueError("resolved embedding model must be non-empty")
        expected_policy = (
            EgressPolicy.LOCAL_ONLY
            if any(
                vault.manifest.egress_policy is EgressPolicy.LOCAL_ONLY
                for vault in self._profile.vaults
            )
            else EgressPolicy.REMOTE_ALLOWED
        )
        if self._profile.effective_policy is not expected_policy:
            raise ValueError("resolved profile egress policy is inconsistent")
        return vault_ids

    def _fingerprints(self) -> dict[str, VaultFingerprint]:
        return expected_vault_fingerprints(self._profile, self._config)

    def _check_compatibility(
        self,
        store: IndexStore,
        fingerprints: Mapping[str, VaultFingerprint],
    ) -> tuple[frozenset[str], dict[str, VaultFingerprint]]:
        manifest_changes: set[str] = set()
        previous_fingerprints: dict[str, VaultFingerprint] = {}
        for vault_id in sorted(fingerprints):
            expected = fingerprints[vault_id]
            snapshot = store.snapshot((vault_id,))
            if (
                snapshot.manifest_fingerprint is not None
                and snapshot.manifest_fingerprint != expected.manifest_fingerprint
            ):
                manifest_changes.add(vault_id)
            stored_values = (
                snapshot.parser_fingerprint,
                snapshot.chunker_fingerprint,
                snapshot.embedding_config_fingerprint,
            )
            if snapshot.manifest_fingerprint is not None and all(
                value is not None for value in stored_values
            ):
                previous_fingerprints[vault_id] = VaultFingerprint(
                    vault_id=vault_id,
                    manifest_fingerprint=snapshot.manifest_fingerprint,
                    parser_fingerprint=stored_values[0] or "",
                    chunker_fingerprint=stored_values[1] or "",
                    embedding_config_fingerprint=stored_values[2] or "",
                )
            if all(value is None for value in stored_values):
                continue
            expected_values = (
                expected.parser_fingerprint,
                expected.chunker_fingerprint,
                expected.embedding_config_fingerprint,
            )
            if stored_values != expected_values:
                raise RebuildRequiredError(
                    f"index configuration for vault {vault_id} requires a rebuild",
                    details={"vault_id": vault_id},
                )
        return frozenset(manifest_changes), previous_fingerprints

    def _semantic_enabled(self) -> bool:
        remote_prohibited = (
            self._profile.effective_policy is EgressPolicy.LOCAL_ONLY
            and self._config.endpoint_class == "remote"
        )
        return self._profile.semantic_enabled and not remote_prohibited

    def _reconcile(self, store: IndexStore, *, fail_on_storage_error: bool) -> IndexReport:
        vault_ids = self._validate_profile()
        fingerprints = self._fingerprints()
        store.initialize()
        manifest_changes, previous_fingerprints = self._check_compatibility(store, fingerprints)

        discovery = {vault.manifest.id: discover_sources(vault) for vault in self._profile.vaults}
        discovered = tuple(source for result in discovery.values() for source in result.sources)
        discovery_failures: tuple[tuple[str, DiscoveryFailure], ...] = tuple(
            (vault_id, failure)
            for vault_id, result in discovery.items()
            for failure in result.failures
        )
        discovery_parse_failures = sum(
            1 for _vault_id, failure in discovery_failures if failure.category == ParseError.code
        )
        discovery_blocking_failures = sum(
            1 for _vault_id, failure in discovery_failures if failure.category != ParseError.code
        )
        discovered_by_key = {
            (source.vault_id, source.relative_path): source for source in discovered
        }
        old_hashes = store.source_hashes(vault_ids)
        old_keys = set(old_hashes)
        new_keys = set(discovered_by_key)
        additions = sorted(new_keys - old_keys)
        byte_changes = {
            key
            for key in new_keys & old_keys
            if discovered_by_key[key].content_hash != old_hashes[key]
        }
        unchanged_bytes = {
            key
            for key in new_keys & old_keys
            if discovered_by_key[key].content_hash == old_hashes[key]
        }
        manifest_promotions = {key for key in unchanged_bytes if key[0] in manifest_changes}
        changes = sorted(byte_changes | manifest_promotions)
        unchanged = sorted(unchanged_bytes - manifest_promotions)
        deletions = sorted(old_keys - new_keys)

        deleted_by_folded = {
            (vault_id, folded_path_key(path)): path for vault_id, path in deletions
        }
        vault_by_id = {vault.manifest.id: vault for vault in self._profile.vaults}
        diagnostics: list[IndexDiagnostic] = []
        # A file that failed discovery is absent from ``discovered_by_key``, so
        # any prior revision of it is already in ``deletions`` and stops being
        # searchable in the single closing transaction.
        diagnostics.extend(
            IndexDiagnostic(
                vault_id,
                failure.relative_path,
                failure.category,
                _bounded_message(failure.message),
            )
            for vault_id, failure in discovery_failures
        )
        parse_failures: list[tuple[DiscoveredSource, str]] = []
        prepared: list[_PreparedSource] = []
        for key in additions + changes:
            source = discovered_by_key[key]
            try:
                parsed = _filtered_frontmatter(_parse(source), vault_by_id[source.vault_id])
                chunks = _preserve_section_metadata(
                    parsed, chunk_source(parsed, self._config, self._counter)
                )
            except (ParseError, ChunkingError) as exc:
                message = _bounded_message(exc.message)
                category = exc.code
                parse_failures.append((source, message))
                diagnostics.append(
                    IndexDiagnostic(source.vault_id, source.relative_path, category, message)
                )
                continue
            prepared.append(
                _PreparedSource(
                    source=source,
                    chunks=chunks,
                    fingerprint=fingerprints[source.vault_id],
                    replacing_path=deleted_by_folded.get((source.vault_id, source.folded_path)),
                )
            )

        semantic_enabled = self._semantic_enabled()
        embedding_requests = 0
        embedding_failures = 0
        storage_failures = 0
        embedded_sources: list[_EmbeddedSource] = []
        reusable_vectors: dict[tuple[str, str], VectorRecord] = {}
        reusable_failures: dict[tuple[str, str], PendingEmbedding] = {}
        for item in prepared:
            vectors: tuple[VectorRecord, ...] = ()
            pending: tuple[PendingEmbedding, ...] = ()
            if semantic_enabled and item.chunks:
                vectors, pending, failure_details, request_count = _embed_with_reuse(
                    item.chunks,
                    item.fingerprint.embedding_config_fingerprint,
                    self._embedding,
                    store=store,
                    reusable=reusable_vectors,
                    failed=reusable_failures,
                )
                embedding_requests += request_count
                embedding_failures += len(failure_details)
                for index, category, message in failure_details:
                    diagnostics.append(
                        IndexDiagnostic(
                            item.source.vault_id,
                            item.source.relative_path,
                            category,
                            f"chunk {index}: {message}",
                        )
                    )
            embedded_sources.append(_EmbeddedSource(item, vectors, pending))

        unchanged_set = set(unchanged)
        pending_by_vault: dict[str, list[StoredChunk]] = {vault_id: [] for vault_id in vault_ids}
        if semantic_enabled:
            requeued = store.requeue_disabled_chunks(tuple(unchanged), datetime.now(UTC))
            diagnostics.extend(
                IndexDiagnostic(
                    result.vault_id,
                    "<policy>",
                    "policy_transition",
                    f"semantic embedding re-enabled for {result.chunk_count} chunks",
                )
                for result in requeued
            )
            for chunk in store.pending_chunks(vault_ids):
                if (chunk.vault_id, chunk.relative_path) in unchanged_set:
                    pending_by_vault[chunk.vault_id].append(chunk)

        pending_updates: dict[str, tuple[PendingEmbeddingUpdate, ...]] = {}
        for vault_id in vault_ids:
            stored_chunks = pending_by_vault[vault_id]
            if not stored_chunks:
                continue
            retry_chunks = tuple(
                ChunkRecord(
                    id=chunk.id,
                    vault_id=chunk.vault_id,
                    relative_path=chunk.relative_path,
                    ordinal=chunk.ordinal,
                    title=chunk.title,
                    heading=chunk.heading,
                    lines=chunk.lines,
                    text=chunk.text,
                    embedding_text=chunk.embedding_text,
                    token_count=chunk.token_count,
                    metadata=chunk.metadata,
                    content_hash=chunk.content_hash,
                )
                for chunk in stored_chunks
            )
            vectors, failures, failure_details, request_count = _embed_with_reuse(
                retry_chunks,
                fingerprints[vault_id].embedding_config_fingerprint,
                self._embedding,
                store=None,
                reusable=reusable_vectors,
                failed=reusable_failures,
            )
            embedding_requests += request_count
            vector_by_id = {vector.chunk_id: vector for vector in vectors}
            failure_by_id = {failure.chunk_id: failure for failure in failures}
            pending_updates[vault_id] = tuple(
                PendingEmbeddingUpdate(
                    chunk_id=chunk.id,
                    content_hash=chunk.content_hash,
                    vector=vector_by_id.get(chunk.id),
                    failure=failure_by_id.get(chunk.id),
                )
                for chunk in stored_chunks
            )
            embedding_failures += len(failure_details)
            for index, category, message in failure_details:
                chunk = stored_chunks[index]
                diagnostics.append(
                    IndexDiagnostic(
                        chunk.vault_id,
                        chunk.relative_path,
                        category,
                        f"pending chunk {chunk.ordinal}: {message}",
                    )
                )

        failed_manifest_vaults: set[str] = set()
        failed_reconciliation_vaults: set[str] = set()
        for source, message in parse_failures:
            try:
                store.suppress_source(
                    source.vault_id,
                    source.relative_path,
                    source.content_hash,
                    message,
                )
            except StorageError as exc:
                failed_reconciliation_vaults.add(source.vault_id)
                if source.vault_id in manifest_changes:
                    failed_manifest_vaults.add(source.vault_id)
                if fail_on_storage_error:
                    raise
                diagnostics.append(
                    IndexDiagnostic(
                        source.vault_id,
                        source.relative_path,
                        exc.code,
                        _bounded_message(exc.message),
                    )
                )
                storage_failures += 1

        for embedded in embedded_sources:
            item = embedded.prepared
            fingerprint = item.fingerprint
            revision = SourceRevision(
                source=item.source,
                chunks=item.chunks,
                vectors=embedded.vectors,
                pending=embedded.pending,
                manifest_fingerprint=fingerprint.manifest_fingerprint,
                parser_fingerprint=fingerprint.parser_fingerprint,
                chunker_fingerprint=fingerprint.chunker_fingerprint,
                embedding_config_fingerprint=fingerprint.embedding_config_fingerprint,
                update_vault_fingerprint=item.source.vault_id not in manifest_changes,
            )
            try:
                store.replace_source(revision, replacing_path=item.replacing_path)
            except StorageError as exc:
                failed_reconciliation_vaults.add(item.source.vault_id)
                if item.source.vault_id in manifest_changes:
                    failed_manifest_vaults.add(item.source.vault_id)
                if fail_on_storage_error:
                    raise
                diagnostics.append(
                    IndexDiagnostic(
                        item.source.vault_id,
                        item.source.relative_path,
                        exc.code,
                        _bounded_message(exc.message),
                    )
                )
                storage_failures += 1

        for vault_id in vault_ids:
            updates = pending_updates.get(vault_id)
            if updates is None:
                continue
            try:
                store.update_pending_embeddings(
                    vault_id,
                    fingerprints[vault_id].embedding_config_fingerprint,
                    updates,
                )
            except StorageError as exc:
                failed_reconciliation_vaults.add(vault_id)
                if fail_on_storage_error:
                    raise
                diagnostics.append(
                    IndexDiagnostic(vault_id, "<pending>", exc.code, _bounded_message(exc.message))
                )
                storage_failures += 1

        final_fingerprints = tuple(
            previous_fingerprints[vault_id]
            if vault_id in failed_manifest_vaults
            else fingerprints[vault_id]
            for vault_id in vault_ids
        )
        store.delete_sources(
            tuple(deletions),
            fingerprints=final_fingerprints,
            disable_semantics_for=() if semantic_enabled else vault_ids,
            reconciled_vault_ids=tuple(
                vault_id for vault_id in vault_ids if vault_id not in failed_reconciliation_vaults
            ),
        )

        snapshot = store.snapshot(vault_ids)
        return IndexReport(
            total_sources=len(discovered),
            added_sources=len(additions),
            changed_sources=len(changes),
            unchanged_sources=len(unchanged),
            deleted_sources=len(deletions),
            ready_chunks=snapshot.ready_count,
            pending_chunks=snapshot.pending_count,
            parse_failures=len(parse_failures) + discovery_parse_failures,
            embedding_failures=embedding_failures,
            embedding_requests=embedding_requests,
            diagnostics=tuple(sorted(diagnostics)[:_MAX_DIAGNOSTICS]),
            elapsed_ms=0,
            blocking_failures=discovery_blocking_failures + storage_failures,
        )

    def _atomic_rebuild(self) -> IndexReport:
        store = self._store
        if not isinstance(store, AtomicReplaceIndexStore):
            raise TypeError("atomic rebuild requires an atomic-replace index store")
        vault_ids = self._validate_profile()
        active_path = store.path
        active_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=active_path.parent,
            prefix=f".{active_path.name}.",
            suffix=".rebuild",
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        temporary_store = store.replacement_store(temporary_path)
        try:
            if active_path.exists():
                try:
                    store.initialize()
                except (RebuildRequiredError, StorageError):
                    pass
                else:
                    store.prepare_for_atomic_replace()
                    shutil.copy2(active_path, temporary_path)
                    temporary_store.initialize()
                    temporary_store.reset_vaults(vault_ids)
            report = self._reconcile(temporary_store, fail_on_storage_error=True)
            temporary_store.prepare_for_atomic_replace()
            # SQLite re-created this file under the process umask after mkstemp
            # unlinked it, so restore the user-only mode the design requires
            # before the rename publishes it as the active index.
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, active_path)
            return report
        finally:
            for path in (
                temporary_path,
                Path(f"{temporary_path}-wal"),
                Path(f"{temporary_path}-shm"),
            ):
                with suppress(OSError):
                    path.unlink(missing_ok=True)
