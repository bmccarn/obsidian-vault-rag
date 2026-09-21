"""Invisible PostgreSQL revision builds and atomic promotion."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import numpy as np  # pyright: ignore[reportMissingImports]
from pgvector import Vector  # type: ignore[import-untyped]
from psycopg import Error, sql  # pyright: ignore[reportMissingImports]
from psycopg.types.json import Jsonb  # pyright: ignore[reportMissingImports]

from vault_rag.domain import EmbeddingState, JsonValue
from vault_rag.embedding.fingerprint import observed_fingerprint
from vault_rag.errors import StorageError
from vault_rag.ingest.chunker import ChunkRecord
from vault_rag.storage.codecs import VECTOR_DTYPE, encode_vector
from vault_rag.storage.ports import RevisionBuildStore
from vault_rag.storage.records import (
    IndexSnapshot,
    PendingEmbedding,
    PendingEmbeddingUpdate,
    RequeueResult,
    SourceRevision,
    StoredChunk,
    VaultFingerprint,
    VectorRecord,
    canonical_json,
    validate_vault_ids,
)
from vault_rag.storage.schema import SCHEMA_VERSION

from .migrations import PostgresMigrator
from .pool import DatabaseConnection, PostgresPool
from .query_store import PostgresStore

_MAX_DIAGNOSTIC_LENGTH = 1_000
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


def _parse_pgvector_text(value: str, *, dimensions: int) -> np.ndarray | None:
    """Parse the textual pgvector representation returned by some psycopg setups."""
    try:
        values = json.loads(value)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(values, list)
        or len(values) != dimensions
        or any(type(item) not in {int, float} for item in values)
    ):
        return None
    vector = np.asarray(values, dtype=np.float32)
    return vector if vector.ndim == 1 and np.isfinite(vector).all() else None


def _normalized_vectors(revision: SourceRevision) -> dict[str, np.ndarray]:
    source = revision.source
    fingerprints = (
        revision.manifest_fingerprint,
        revision.parser_fingerprint,
        revision.chunker_fingerprint,
        revision.embedding_config_fingerprint,
    )
    if any(not fingerprint for fingerprint in fingerprints):
        raise ValueError("revision fingerprints must be non-empty")

    chunk_ids: set[str] = set()
    ordinals: set[int] = set()
    for chunk_record in revision.chunks:
        if not chunk_record.id or chunk_record.id in chunk_ids:
            raise ValueError("chunk IDs must be unique and non-empty")
        if chunk_record.ordinal in ordinals:
            raise ValueError("chunk ordinals must be unique")
        if (
            chunk_record.vault_id != source.vault_id
            or chunk_record.relative_path != source.relative_path
        ):
            raise ValueError("chunk source identity does not match revision")
        if chunk_record.ordinal < 0 or chunk_record.token_count < 0:
            raise ValueError("chunk numeric fields must be non-negative")
        canonical_json(dict(chunk_record.metadata))
        chunk_ids.add(chunk_record.id)
        ordinals.add(chunk_record.ordinal)

    vectors: dict[str, np.ndarray] = {}
    dimensions_by_fingerprint: dict[str, int] = {}
    for vector_record in revision.vectors:
        if vector_record.chunk_id not in chunk_ids or vector_record.chunk_id in vectors:
            raise ValueError("vectors must map uniquely to revision chunks")
        if vector_record.config_fingerprint != revision.embedding_config_fingerprint:
            raise ValueError("vector configuration fingerprint does not match revision")
        if not vector_record.observed_fingerprint or vector_record.dimensions < 1:
            raise ValueError("vector fingerprints and dimensions must be valid")
        prior_dimension = dimensions_by_fingerprint.setdefault(
            vector_record.observed_fingerprint, vector_record.dimensions
        )
        if prior_dimension != vector_record.dimensions:
            raise ValueError("observed vector fingerprint has inconsistent dimensions")
        raw = encode_vector(vector_record.vector, dimensions=vector_record.dimensions)
        vectors[vector_record.chunk_id] = np.frombuffer(raw, dtype=VECTOR_DTYPE).copy()

    pending_ids: set[str] = set()
    for pending_record in revision.pending:
        if pending_record.chunk_id not in chunk_ids or pending_record.chunk_id in pending_ids:
            raise ValueError("pending rows must map uniquely to revision chunks")
        if pending_record.chunk_id in vectors:
            raise ValueError("a chunk cannot be both ready and pending")
        if not pending_record.category or not pending_record.message:
            raise ValueError("pending errors must include category and message")
        if (
            pending_record.attempted_at.tzinfo is None
            or pending_record.attempted_at.utcoffset() is None
        ):
            raise ValueError("pending attempt timestamps must be timezone-aware")
        pending_ids.add(pending_record.chunk_id)
    return vectors


def _metadata_text(metadata: Mapping[str, JsonValue], key: str) -> str:
    value = metadata.get(key)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(item if isinstance(item, str) else canonical_json(item) for item in value)
    return canonical_json(value)


def _identifiers(metadata: Mapping[str, JsonValue]) -> str:
    return " ".join(_metadata_text(metadata, key) for key in ("jira", "repo", "pr"))


def _search_document_sql() -> sql.Composable:
    return sql.SQL(
        """
        setweight(to_tsvector('simple', %s), 'A') ||
        setweight(to_tsvector('simple', %s), 'A') ||
        setweight(to_tsvector('simple', %s), 'A') ||
        setweight(to_tsvector('simple', %s), 'A') ||
        setweight(to_tsvector('simple', %s), 'B') ||
        setweight(to_tsvector('simple', %s), 'B') ||
        setweight(to_tsvector('simple', %s), 'D')
        """
    )


class PostgresIndexStore(RevisionBuildStore):
    """IndexStore implementation scoped to one invisible revision UUID."""

    def __init__(
        self,
        pool: PostgresPool,
        *,
        revision_id: UUID,
        vault_id: str,
        commit_sha: str,
        expected_active_revision_id: UUID | None,
    ) -> None:
        self._pool = pool
        self._revision_uuid = revision_id
        self._vault_id = vault_id
        self._commit_sha = commit_sha
        self._expected_active_revision_id = expected_active_revision_id

    @classmethod
    def begin(
        cls,
        pool: PostgresPool,
        *,
        vault_id: str,
        configured_ref: str,
        commit_sha: str,
        manifest: Mapping[str, JsonValue] | None = None,
    ) -> PostgresIndexStore:
        if not vault_id or len(vault_id) > 200:
            raise ValueError("vault id is invalid")
        if not configured_ref or len(configured_ref) > 500:
            raise ValueError("configured ref is invalid")
        if _COMMIT_SHA.fullmatch(commit_sha) is None:
            raise ValueError("commit SHA is invalid")
        revision_id = uuid4()
        try:
            with pool.connection() as connection, connection.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 1))", (vault_id,)
                )
                connection.execute(
                    """
                    INSERT INTO vault_rag.vaults(vault_id, configured_ref, updated_at)
                    VALUES (%s, %s, now())
                    ON CONFLICT (vault_id) DO UPDATE SET
                        configured_ref = excluded.configured_ref,
                        updated_at = excluded.updated_at
                    """,
                    (vault_id, configured_ref),
                )
                vault = connection.execute(
                    "SELECT active_revision_id FROM vault_rag.vaults "
                    "WHERE vault_id = %s FOR UPDATE",
                    (vault_id,),
                ).fetchone()
                if vault is None:
                    raise StorageError("PostgreSQL vault state is missing")
                expected_active = cast(UUID | None, vault["active_revision_id"])
                active = None
                if expected_active is not None:
                    active = connection.execute(
                        "SELECT * FROM vault_rag.vault_revisions WHERE revision_id = %s",
                        (expected_active,),
                    ).fetchone()
                connection.execute(
                    """
                    INSERT INTO vault_rag.vault_revisions(
                        revision_id, vault_id, commit_sha, state, manifest_json,
                        manifest_fingerprint, parser_fingerprint,
                        chunker_fingerprint, embedding_config_fingerprint,
                        observed_fingerprint
                    ) VALUES (%s, %s, %s, 'building', %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        revision_id,
                        vault_id,
                        commit_sha,
                        None if manifest is None else Jsonb(dict(manifest)),
                        None if active is None else active["manifest_fingerprint"],
                        None if active is None else active["parser_fingerprint"],
                        None if active is None else active["chunker_fingerprint"],
                        None if active is None else active["embedding_config_fingerprint"],
                        None if active is None else active["observed_fingerprint"],
                    ),
                )
                if expected_active is not None:
                    cls._copy_active_revision(connection, expected_active, revision_id)
        except StorageError:
            raise
        except Error as exc:
            raise StorageError("could not start PostgreSQL revision") from exc
        return cls(
            pool,
            revision_id=revision_id,
            vault_id=vault_id,
            commit_sha=commit_sha,
            expected_active_revision_id=expected_active,
        )

    @property
    def revision_id(self) -> str:
        return str(self._revision_uuid)

    @property
    def database_location(self) -> str:
        return self._pool.database_location

    def initialize(self) -> None:
        PostgresMigrator(self._pool).check()
        try:
            with self._pool.connection() as connection:
                self._assert_building(connection)
        except Error as exc:
            raise StorageError("could not check PostgreSQL revision") from exc

    def source_hashes(self, vault_ids: tuple[str, ...]) -> dict[tuple[str, str], str]:
        self._validate_vault_scope(vault_ids)
        try:
            with self._pool.connection() as connection:
                rows = connection.execute(
                    """
                    SELECT relative_path, content_hash
                    FROM vault_rag.revision_sources
                    WHERE revision_id = %s ORDER BY relative_path
                    """,
                    (self._revision_uuid,),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not load source hashes") from exc
        return {
            (self._vault_id, cast(str, row["relative_path"])): cast(str, row["content_hash"])
            for row in rows
        }

    def reusable_vectors(
        self, chunks: tuple[ChunkRecord, ...], embedding_config_fingerprint: str
    ) -> dict[str, VectorRecord]:
        hashes = list(dict.fromkeys(chunk.content_hash for chunk in chunks))
        if not hashes:
            return {}
        try:
            with self._pool.connection() as connection:
                rows = connection.execute(
                    """
                    SELECT content_hash, dimensions, configuration_fingerprint,
                        observed_fingerprint, embedding
                    FROM vault_rag.embeddings
                    WHERE content_hash = ANY(%s) AND configuration_fingerprint = %s
                    """,
                    (hashes, embedding_config_fingerprint),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not load reusable vectors") from exc
        grouped: dict[str, list[VectorRecord]] = {}
        for row in rows:
            content_hash = cast(str, row["content_hash"])
            dimensions = cast(int, row["dimensions"])
            raw_vector = row["embedding"]
            vector = (
                _parse_pgvector_text(raw_vector, dimensions=dimensions)
                if isinstance(raw_vector, str)
                else np.asarray(raw_vector, dtype=np.float32)
            )
            if vector is None:
                continue
            grouped.setdefault(content_hash, []).append(
                VectorRecord(
                    chunk_id="",
                    dimensions=dimensions,
                    config_fingerprint=cast(str, row["configuration_fingerprint"]),
                    observed_fingerprint=cast(str, row["observed_fingerprint"]),
                    vector=vector,
                )
            )
        return {
            content_hash: vectors[0]
            for content_hash, vectors in grouped.items()
            if len({(item.dimensions, item.observed_fingerprint) for item in vectors}) == 1
            and vectors[0].config_fingerprint == embedding_config_fingerprint
            and vectors[0].observed_fingerprint
            == observed_fingerprint(embedding_config_fingerprint, vectors[0].dimensions)
            and np.asarray(vectors[0].vector).size == vectors[0].dimensions
        }

    def replace_source(
        self,
        revision: SourceRevision,
        *,
        replacing_path: str | None = None,
    ) -> None:
        try:
            vectors = _normalized_vectors(revision)
            payload = revision.source.text.encode("utf-8")
            expected_hash = f"sha256:{hashlib.sha256(payload).hexdigest()}"
            if (
                revision.source.vault_id != self._vault_id
                or revision.source.content_hash != expected_hash
                or revision.source.size_bytes != len(payload)
            ):
                raise ValueError("source content identity is invalid")
        except (TypeError, ValueError) as exc:
            raise StorageError(
                f"invalid source revision for {revision.source.vault_id}/"
                f"{revision.source.relative_path}",
                details={
                    "vault_id": revision.source.vault_id,
                    "path": revision.source.relative_path,
                },
            ) from exc

        source = revision.source
        vector_by_id = {vector.chunk_id: vector for vector in revision.vectors}
        pending_by_id = {pending.chunk_id: pending for pending in revision.pending}
        chunk_by_id = {chunk.id: chunk for chunk in revision.chunks}
        try:
            with self._pool.connection() as connection, connection.transaction():
                self._assert_building(connection, for_update=True)
                self._pool.register_vector(connection)
                connection.execute(
                    """
                    INSERT INTO vault_rag.source_blobs(
                        content_hash, byte_length, encoding, content
                    ) VALUES (%s, %s, 'utf-8', %s)
                    ON CONFLICT (content_hash) DO NOTHING
                    """,
                    (source.content_hash, len(payload), payload),
                )
                blob = connection.execute(
                    "SELECT byte_length, content FROM vault_rag.source_blobs "
                    "WHERE content_hash = %s",
                    (source.content_hash,),
                ).fetchone()
                if (
                    blob is None
                    or cast(int, blob["byte_length"]) != len(payload)
                    or bytes(blob["content"]) != payload
                ):
                    raise StorageError("source blob hash collision detected")
                if replacing_path is not None:
                    replaced = connection.execute(
                        "SELECT folded_path FROM vault_rag.revision_sources "
                        "WHERE revision_id = %s AND relative_path = %s",
                        (self._revision_uuid, replacing_path),
                    ).fetchone()
                    if replaced is not None:
                        if (
                            replacing_path == source.relative_path
                            or replaced["folded_path"] != source.folded_path
                        ):
                            raise StorageError("replacement path is not a case-only source rename")
                        connection.execute(
                            "DELETE FROM vault_rag.revision_sources "
                            "WHERE revision_id = %s AND relative_path = %s",
                            (self._revision_uuid, replacing_path),
                        )
                connection.execute(
                    "DELETE FROM vault_rag.revision_sources "
                    "WHERE revision_id = %s AND relative_path = %s",
                    (self._revision_uuid, source.relative_path),
                )
                connection.execute(
                    """
                    INSERT INTO vault_rag.revision_sources(
                        revision_id, relative_path, folded_path, source_kind,
                        content_hash, source_blob_hash, size_bytes, mtime_ns, parse_state,
                        diagnostic, indexed_at, metadata_json
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, 'active', NULL, now(), %s
                    )
                    """,
                    (
                        self._revision_uuid,
                        source.relative_path,
                        source.folded_path,
                        source.kind.value,
                        source.content_hash,
                        source.content_hash,
                        source.size_bytes,
                        source.mtime_ns,
                        Jsonb({}),
                    ),
                )
                for chunk_id, normalized in vectors.items():
                    record = vector_by_id[chunk_id]
                    chunk_record = chunk_by_id[chunk_id]
                    connection.execute(
                        """
                        INSERT INTO vault_rag.embeddings(
                            content_hash, configuration_fingerprint,
                            observed_fingerprint, dimensions, embedding
                        ) VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (
                            content_hash, configuration_fingerprint, observed_fingerprint
                        ) DO NOTHING
                        """,
                        (
                            chunk_record.content_hash,
                            record.config_fingerprint,
                            record.observed_fingerprint,
                            record.dimensions,
                            Vector(normalized),
                        ),
                    )
                    stored = connection.execute(
                        """
                        SELECT dimensions FROM vault_rag.embeddings
                        WHERE content_hash = %s AND configuration_fingerprint = %s
                            AND observed_fingerprint = %s
                        """,
                        (
                            chunk_record.content_hash,
                            record.config_fingerprint,
                            record.observed_fingerprint,
                        ),
                    ).fetchone()
                    if stored is None or stored["dimensions"] != record.dimensions:
                        raise StorageError("embedding identity has inconsistent dimensions")

                for chunk_record in revision.chunks:
                    vector = vector_by_id.get(chunk_record.id)
                    pending = pending_by_id.get(chunk_record.id)
                    state = (
                        EmbeddingState.READY
                        if vector is not None
                        else EmbeddingState.PENDING
                        if pending is not None
                        else EmbeddingState.DISABLED
                    )
                    metadata = dict(chunk_record.metadata)
                    heading = " ".join(chunk_record.heading)
                    connection.execute(
                        sql.SQL(
                            """
                            INSERT INTO vault_rag.revision_chunks(
                                revision_id, chunk_id, source_path, source_blob_hash,
                                ordinal, title, heading_json, start_line, end_line,
                                body, embedding_text, token_count, metadata_json,
                                content_hash, embedding_state,
                                embedding_config_fingerprint, observed_fingerprint,
                                dimensions, search_document
                            ) VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                {}
                            )
                            """
                        ).format(_search_document_sql()),
                        (
                            self._revision_uuid,
                            chunk_record.id,
                            source.relative_path,
                            source.content_hash,
                            chunk_record.ordinal,
                            chunk_record.title,
                            Jsonb(list(chunk_record.heading)),
                            chunk_record.lines.start,
                            chunk_record.lines.end,
                            chunk_record.text,
                            chunk_record.embedding_text,
                            chunk_record.token_count,
                            Jsonb(metadata),
                            chunk_record.content_hash,
                            state.value,
                            None if vector is None else vector.config_fingerprint,
                            None if vector is None else vector.observed_fingerprint,
                            None if vector is None else vector.dimensions,
                            source.relative_path,
                            chunk_record.title,
                            heading,
                            _identifiers(metadata),
                            _metadata_text(metadata, "aliases"),
                            _metadata_text(metadata, "tags"),
                            chunk_record.text,
                        ),
                    )
                    if pending is not None:
                        connection.execute(
                            """
                            INSERT INTO vault_rag.embedding_queue(
                                revision_id, chunk_id, category, message,
                                last_attempted_at, next_eligible_at
                            ) VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (
                                self._revision_uuid,
                                pending.chunk_id,
                                pending.category[:100],
                                pending.message[:_MAX_DIAGNOSTIC_LENGTH],
                                pending.attempted_at.astimezone(UTC),
                                pending.attempted_at.astimezone(UTC),
                            ),
                        )
                observed = sorted({vector.observed_fingerprint for vector in revision.vectors})
                connection.execute(
                    """
                    UPDATE vault_rag.vault_revisions SET
                        manifest_fingerprint = %s, parser_fingerprint = %s,
                        chunker_fingerprint = %s,
                        embedding_config_fingerprint = %s,
                        observed_fingerprint = %s
                    WHERE revision_id = %s AND state = 'building'
                    """,
                    (
                        revision.manifest_fingerprint,
                        revision.parser_fingerprint,
                        revision.chunker_fingerprint,
                        revision.embedding_config_fingerprint,
                        observed[0] if len(observed) == 1 else None,
                        self._revision_uuid,
                    ),
                )
        except StorageError:
            raise
        except Error as exc:
            raise StorageError(
                f"could not replace source {source.vault_id}/{source.relative_path}",
                details={"vault_id": source.vault_id, "path": source.relative_path},
            ) from exc

    def record_vault_fingerprint(self, fingerprint: VaultFingerprint) -> None:
        if fingerprint.vault_id != self._vault_id:
            raise ValueError("vault fingerprint does not match revision")
        values = (
            fingerprint.manifest_fingerprint,
            fingerprint.parser_fingerprint,
            fingerprint.chunker_fingerprint,
            fingerprint.embedding_config_fingerprint,
        )
        if any(not value for value in values):
            raise ValueError("vault fingerprints must be non-empty")
        try:
            with self._pool.connection() as connection, connection.transaction():
                self._assert_building(connection, for_update=True)
                connection.execute(
                    """
                    UPDATE vault_rag.vault_revisions SET
                        manifest_fingerprint = %s, parser_fingerprint = %s,
                        chunker_fingerprint = %s,
                        embedding_config_fingerprint = %s
                    WHERE revision_id = %s
                    """,
                    (*values, self._revision_uuid),
                )
        except Error as exc:
            raise StorageError("could not record vault fingerprint") from exc

    def update_pending_embeddings(
        self,
        vault_id: str,
        embedding_config_fingerprint: str,
        updates: tuple[PendingEmbeddingUpdate, ...],
    ) -> None:
        if vault_id != self._vault_id or not embedding_config_fingerprint:
            raise ValueError("vault id and embedding fingerprint are invalid")
        if not updates:
            return
        identifiers = [update.chunk_id for update in updates]
        if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("pending update chunk IDs must be unique and non-empty")
        try:
            with self._pool.connection() as connection, connection.transaction():
                revision = self._assert_building(connection, for_update=True)
                if revision["embedding_config_fingerprint"] != embedding_config_fingerprint:
                    raise StorageError("pending update fingerprint is stale")
                self._pool.register_vector(connection)
                for update in updates:
                    if not update.content_hash or (update.vector is None) == (
                        update.failure is None
                    ):
                        raise ValueError("pending update requires exactly one vector or failure")
                    row = connection.execute(
                        """
                        SELECT content_hash, embedding_state
                        FROM vault_rag.revision_chunks
                        WHERE revision_id = %s AND chunk_id = %s
                        """,
                        (self._revision_uuid, update.chunk_id),
                    ).fetchone()
                    if (
                        row is None
                        or row["embedding_state"] != EmbeddingState.PENDING.value
                        or row["content_hash"] != update.content_hash
                    ):
                        raise StorageError("pending update targets a stale chunk")
                    if update.vector is not None:
                        vector = update.vector
                        if (
                            vector.chunk_id != update.chunk_id
                            or vector.config_fingerprint != embedding_config_fingerprint
                            or not vector.observed_fingerprint
                            or vector.observed_fingerprint
                            != observed_fingerprint(embedding_config_fingerprint, vector.dimensions)
                        ):
                            raise ValueError("pending vector identity is invalid")
                        raw = encode_vector(vector.vector, dimensions=vector.dimensions)
                        normalized = np.frombuffer(raw, dtype=VECTOR_DTYPE).copy()
                        connection.execute(
                            """
                            INSERT INTO vault_rag.embeddings(
                                content_hash, configuration_fingerprint,
                                observed_fingerprint, dimensions, embedding
                            ) VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (
                                content_hash, configuration_fingerprint,
                                observed_fingerprint
                            ) DO NOTHING
                            """,
                            (
                                update.content_hash,
                                vector.config_fingerprint,
                                vector.observed_fingerprint,
                                vector.dimensions,
                                Vector(normalized),
                            ),
                        )
                        connection.execute(
                            """
                            UPDATE vault_rag.revision_chunks SET
                                embedding_state = 'ready',
                                embedding_config_fingerprint = %s,
                                observed_fingerprint = %s,
                                dimensions = %s
                            WHERE revision_id = %s AND chunk_id = %s
                            """,
                            (
                                vector.config_fingerprint,
                                vector.observed_fingerprint,
                                vector.dimensions,
                                self._revision_uuid,
                                update.chunk_id,
                            ),
                        )
                        connection.execute(
                            "DELETE FROM vault_rag.embedding_queue "
                            "WHERE revision_id = %s AND chunk_id = %s",
                            (self._revision_uuid, update.chunk_id),
                        )
                    else:
                        failure = cast(PendingEmbedding, update.failure)
                        if (
                            failure.chunk_id != update.chunk_id
                            or not failure.category
                            or not failure.message
                            or failure.attempted_at.tzinfo is None
                            or failure.attempted_at.utcoffset() is None
                        ):
                            raise ValueError("pending failure is invalid")
                        connection.execute(
                            """
                            UPDATE vault_rag.embedding_queue SET
                                category = %s, message = %s,
                                attempt_count = attempt_count + 1,
                                last_attempted_at = %s,
                                next_eligible_at = %s
                            WHERE revision_id = %s AND chunk_id = %s
                            """,
                            (
                                failure.category[:100],
                                failure.message[:_MAX_DIAGNOSTIC_LENGTH],
                                failure.attempted_at.astimezone(UTC),
                                failure.attempted_at.astimezone(UTC),
                                self._revision_uuid,
                                update.chunk_id,
                            ),
                        )
        except (TypeError, ValueError) as exc:
            raise StorageError(
                f"could not update pending embeddings for {vault_id}",
                details={"vault_id": vault_id},
            ) from exc
        except StorageError:
            raise
        except Error as exc:
            raise StorageError(
                f"could not update pending embeddings for {vault_id}",
                details={"vault_id": vault_id},
            ) from exc

    def reset_vaults(self, vault_ids: tuple[str, ...]) -> None:
        self._validate_vault_scope(vault_ids)
        try:
            with self._pool.connection() as connection, connection.transaction():
                self._assert_building(connection, for_update=True)
                connection.execute(
                    """
                    UPDATE vault_rag.vault_revisions SET
                        manifest_fingerprint = NULL, parser_fingerprint = NULL,
                        chunker_fingerprint = NULL,
                        embedding_config_fingerprint = NULL,
                        observed_fingerprint = NULL
                    WHERE revision_id = %s
                    """,
                    (self._revision_uuid,),
                )
                connection.execute(
                    "DELETE FROM vault_rag.revision_sources WHERE revision_id = %s",
                    (self._revision_uuid,),
                )
        except Error as exc:
            raise StorageError("could not reset vault index state") from exc

    def requeue_disabled_chunks(
        self,
        source_keys: tuple[tuple[str, str], ...],
        attempted_at: datetime,
    ) -> tuple[RequeueResult, ...]:
        if attempted_at.tzinfo is None or attempted_at.utcoffset() is None:
            raise ValueError("requeue timestamp must be timezone-aware")
        if not source_keys:
            return ()
        if any(
            vault_id != self._vault_id or not relative_path
            for vault_id, relative_path in source_keys
        ) or len(set(source_keys)) != len(source_keys):
            raise ValueError("requeue source keys must be unique and match the revision")
        try:
            with self._pool.connection() as connection, connection.transaction():
                self._assert_building(connection, for_update=True)
                paths = [relative_path for _vault_id, relative_path in source_keys]
                rows = connection.execute(
                    """
                    UPDATE vault_rag.revision_chunks SET embedding_state = 'pending'
                    WHERE revision_id = %s AND source_path = ANY(%s)
                        AND embedding_state = 'disabled'
                    RETURNING chunk_id
                    """,
                    (self._revision_uuid, paths),
                ).fetchall()
                for row in rows:
                    connection.execute(
                        """
                        INSERT INTO vault_rag.embedding_queue(
                            revision_id, chunk_id, category, message,
                            last_attempted_at, next_eligible_at
                        ) VALUES (
                            %s, %s, 'policy_transition',
                            'semantic embedding re-enabled', %s, %s
                        )
                        """,
                        (
                            self._revision_uuid,
                            row["chunk_id"],
                            attempted_at.astimezone(UTC),
                            attempted_at.astimezone(UTC),
                        ),
                    )
        except Error as exc:
            raise StorageError("could not requeue disabled embeddings") from exc
        return () if not rows else (RequeueResult(self._vault_id, len(rows)),)

    def suppress_source(
        self,
        vault_id: str,
        relative_path: str,
        content_hash: str,
        diagnostic: str,
    ) -> None:
        if vault_id != self._vault_id or not content_hash:
            raise ValueError("suppressed source identity is invalid")
        try:
            with self._pool.connection() as connection, connection.transaction():
                self._assert_building(connection, for_update=True)
                connection.execute(
                    "DELETE FROM vault_rag.revision_chunks "
                    "WHERE revision_id = %s AND source_path = %s",
                    (self._revision_uuid, relative_path),
                )
                connection.execute(
                    """
                    UPDATE vault_rag.revision_sources SET parse_state = 'suppressed',
                        diagnostic = %s, content_hash = %s
                    WHERE revision_id = %s AND relative_path = %s
                    """,
                    (
                        diagnostic[:_MAX_DIAGNOSTIC_LENGTH],
                        content_hash,
                        self._revision_uuid,
                        relative_path,
                    ),
                )
        except Error as exc:
            raise StorageError(
                f"could not suppress source {vault_id}/{relative_path}",
                details={"vault_id": vault_id, "path": relative_path},
            ) from exc

    def delete_sources(
        self,
        keys: tuple[tuple[str, str], ...],
        *,
        fingerprints: tuple[VaultFingerprint, ...] = (),
        disable_semantics_for: tuple[str, ...] = (),
        reconciled_vault_ids: tuple[str, ...] | None = None,
    ) -> None:
        if any(vault_id != self._vault_id for vault_id, _path in keys):
            raise ValueError("source delete is outside the revision vault")
        if any(vault_id != self._vault_id for vault_id in disable_semantics_for):
            raise ValueError("semantic disable is outside the revision vault")
        if reconciled_vault_ids is not None and any(
            vault_id != self._vault_id for vault_id in reconciled_vault_ids
        ):
            raise ValueError("reconciled vault is outside the revision")
        try:
            with self._pool.connection() as connection, connection.transaction():
                self._assert_building(connection, for_update=True)
                for _vault_id, relative_path in keys:
                    connection.execute(
                        "DELETE FROM vault_rag.revision_sources "
                        "WHERE revision_id = %s AND relative_path = %s",
                        (self._revision_uuid, relative_path),
                    )
                if disable_semantics_for:
                    connection.execute(
                        "DELETE FROM vault_rag.embedding_queue WHERE revision_id = %s",
                        (self._revision_uuid,),
                    )
                    connection.execute(
                        """
                        UPDATE vault_rag.revision_chunks SET
                            embedding_state = 'disabled',
                            embedding_config_fingerprint = NULL,
                            observed_fingerprint = NULL,
                            dimensions = NULL
                        WHERE revision_id = %s
                        """,
                        (self._revision_uuid,),
                    )
                for fingerprint in fingerprints:
                    if fingerprint.vault_id != self._vault_id:
                        raise ValueError("vault fingerprint is outside the revision")
                    connection.execute(
                        """
                        UPDATE vault_rag.vault_revisions SET
                            manifest_fingerprint = %s, parser_fingerprint = %s,
                            chunker_fingerprint = %s,
                            embedding_config_fingerprint = %s
                        WHERE revision_id = %s
                        """,
                        (
                            fingerprint.manifest_fingerprint,
                            fingerprint.parser_fingerprint,
                            fingerprint.chunker_fingerprint,
                            fingerprint.embedding_config_fingerprint,
                            self._revision_uuid,
                        ),
                    )
        except Error as exc:
            raise StorageError("could not finalize source reconciliation") from exc

    def pending_chunks(self, vault_ids: tuple[str, ...]) -> tuple[StoredChunk, ...]:
        self._validate_vault_scope(vault_ids)
        try:
            with self._pool.connection() as connection:
                rows = connection.execute(
                    self._stored_chunk_select()
                    + sql.SQL(
                        " WHERE rc.revision_id = %s AND rs.parse_state = 'active' "
                        "AND rc.embedding_state = 'pending' "
                        "ORDER BY rs.relative_path, rc.ordinal"
                    ),
                    (self._revision_uuid,),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not load pending chunks") from exc
        return tuple(PostgresStore._stored_chunk(row) for row in rows)

    def snapshot(self, vault_ids: tuple[str, ...]) -> IndexSnapshot:
        self._validate_vault_scope(vault_ids)
        try:
            with self._pool.connection() as connection:
                revision = connection.execute(
                    "SELECT * FROM vault_rag.vault_revisions WHERE revision_id = %s",
                    (self._revision_uuid,),
                ).fetchone()
                if revision is None:
                    raise StorageError("PostgreSQL revision is missing")
                counts = self._counts(connection)
                vector_rows = connection.execute(
                    """
                    SELECT DISTINCT observed_fingerprint, dimensions,
                        embedding_config_fingerprint
                    FROM vault_rag.revision_chunks
                    WHERE revision_id = %s AND embedding_state = 'ready'
                    ORDER BY observed_fingerprint, dimensions,
                        embedding_config_fingerprint
                    """,
                    (self._revision_uuid,),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not inspect PostgreSQL revision") from exc
        return IndexSnapshot(
            source_count=counts[0],
            chunk_count=counts[1],
            ready_count=counts[2],
            pending_count=counts[3],
            vector_bytes=counts[4],
            indexed_at=cast(datetime, revision["started_at"]),
            schema_version=SCHEMA_VERSION,
            manifest_fingerprint=cast(str | None, revision["manifest_fingerprint"]),
            parser_fingerprint=cast(str | None, revision["parser_fingerprint"]),
            chunker_fingerprint=cast(str | None, revision["chunker_fingerprint"]),
            embedding_config_fingerprint=cast(str | None, revision["embedding_config_fingerprint"]),
            observed_fingerprints=tuple(
                sorted(
                    {
                        cast(str, row["observed_fingerprint"])
                        for row in vector_rows
                        if row["observed_fingerprint"] is not None
                    }
                )
            ),
            vector_dimensions=tuple(
                sorted(
                    {
                        cast(int, row["dimensions"])
                        for row in vector_rows
                        if row["dimensions"] is not None
                    }
                )
            ),
            vector_config_fingerprints=tuple(
                sorted(
                    {
                        cast(str, row["embedding_config_fingerprint"])
                        for row in vector_rows
                        if row["embedding_config_fingerprint"] is not None
                    }
                )
            ),
        )

    def _promote_on(
        self,
        connection: DatabaseConnection,
        advisory_key: int,
        *,
        lexical_complete: bool,
        fully_reconciled: bool,
    ) -> None:
        if not lexical_complete:
            raise StorageError("lexically incomplete revision cannot be promoted")
        try:
            with connection.transaction():
                held = connection.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM pg_locks
                        WHERE locktype = 'advisory'
                          AND pid = pg_backend_pid()
                          AND granted
                          AND classid =
                              ((%s::bigint >> 32) & 4294967295)::oid
                          AND objid = (%s::bigint & 4294967295)::oid
                          AND objsubid = 1
                    ) AS held
                    """,
                    (advisory_key, advisory_key),
                ).fetchone()
                if held is None or not bool(held["held"]):
                    raise StorageError("PostgreSQL reconciliation lease was lost")
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 1))",
                    (self._vault_id,),
                )
                revision = self._assert_building(connection, for_update=True)
                vault = connection.execute(
                    "SELECT active_revision_id FROM vault_rag.vaults "
                    "WHERE vault_id = %s FOR UPDATE",
                    (self._vault_id,),
                ).fetchone()
                if vault is None:
                    raise StorageError("PostgreSQL vault state is missing")
                if vault["active_revision_id"] != self._expected_active_revision_id:
                    raise StorageError("active revision changed during build")
                required = (
                    revision["manifest_fingerprint"],
                    revision["parser_fingerprint"],
                    revision["chunker_fingerprint"],
                    revision["embedding_config_fingerprint"],
                )
                if any(value is None for value in required):
                    raise StorageError("revision fingerprints are incomplete")
                counts = self._counts(connection)
                state = "active_degraded" if counts[3] else "active"
                if self._expected_active_revision_id is not None:
                    connection.execute(
                        """
                        UPDATE vault_rag.vault_revisions SET state = 'superseded',
                            superseded_at = now()
                        WHERE revision_id = %s AND state IN ('active', 'active_degraded')
                        """,
                        (self._expected_active_revision_id,),
                    )
                connection.execute(
                    """
                    UPDATE vault_rag.vault_revisions SET state = %s,
                        lexical_complete = true, fully_reconciled = %s,
                        source_count = %s, chunk_count = %s, ready_count = %s,
                        pending_count = %s, vector_bytes = %s,
                        completed_at = now(), promoted_at = now()
                    WHERE revision_id = %s AND state = 'building'
                    """,
                    (state, fully_reconciled, *counts, self._revision_uuid),
                )
                connection.execute(
                    """
                    UPDATE vault_rag.vaults SET active_revision_id = %s,
                        attempted_sha = %s,
                        reconciled_sha = CASE WHEN %s THEN %s ELSE reconciled_sha END,
                        attempted_at = now(),
                        reconciled_at = CASE WHEN %s THEN now() ELSE reconciled_at END,
                        last_reconciliation_attempt_at = now(),
                        last_reconciliation_success_at = CASE WHEN %s THEN now()
                            ELSE last_reconciliation_success_at END,
                        semantic_degradation_category = CASE WHEN %s > 0
                            THEN 'semantic_pending' ELSE NULL END,
                        semantic_degradation_message = NULL,
                        sync_degradation_category = NULL,
                        sync_degradation_message = NULL,
                        updated_at = now()
                    WHERE vault_id = %s
                    """,
                    (
                        self._revision_uuid,
                        self._commit_sha,
                        fully_reconciled,
                        self._commit_sha,
                        fully_reconciled,
                        fully_reconciled,
                        counts[3],
                        self._vault_id,
                    ),
                )
        except StorageError:
            raise
        except Error as exc:
            raise StorageError("could not promote PostgreSQL revision") from exc

    def fail(self, category: str, message: str) -> None:
        if not category or not message:
            raise ValueError("revision failure requires category and message")
        try:
            with self._pool.connection() as connection, connection.transaction():
                row = connection.execute(
                    """
                    UPDATE vault_rag.vault_revisions SET state = 'failed',
                        diagnostics = %s, completed_at = now()
                    WHERE revision_id = %s AND state = 'building'
                    RETURNING revision_id
                    """,
                    (
                        Jsonb(
                            [
                                {
                                    "category": category[:100],
                                    "message": message[:_MAX_DIAGNOSTIC_LENGTH],
                                }
                            ]
                        ),
                        self._revision_uuid,
                    ),
                ).fetchone()
                if row is None:
                    raise StorageError("only a building revision can fail")
        except StorageError:
            raise
        except Error as exc:
            raise StorageError("could not fail PostgreSQL revision") from exc

    @staticmethod
    def _copy_active_revision(
        connection: DatabaseConnection,
        active_revision_id: UUID,
        new_revision_id: UUID,
    ) -> None:
        connection.execute(
            """
            INSERT INTO vault_rag.revision_sources(
                revision_id, relative_path, folded_path, source_kind,
                content_hash, source_blob_hash, size_bytes, mtime_ns, parse_state,
                diagnostic, indexed_at, metadata_json
            )
            SELECT %s, relative_path, folded_path, source_kind, content_hash,
                source_blob_hash, size_bytes, mtime_ns, parse_state, diagnostic,
                indexed_at, metadata_json
            FROM vault_rag.revision_sources WHERE revision_id = %s
            """,
            (new_revision_id, active_revision_id),
        )
        connection.execute(
            """
            INSERT INTO vault_rag.revision_chunks(
                revision_id, chunk_id, source_path, source_blob_hash, ordinal,
                title, heading_json, start_line, end_line, body, embedding_text,
                token_count, metadata_json, content_hash, embedding_state,
                embedding_config_fingerprint, observed_fingerprint, dimensions,
                search_document
            )
            SELECT %s, chunk_id, source_path, source_blob_hash, ordinal, title,
                heading_json, start_line, end_line, body, embedding_text, token_count,
                metadata_json, content_hash, embedding_state,
                embedding_config_fingerprint, observed_fingerprint, dimensions,
                search_document
            FROM vault_rag.revision_chunks WHERE revision_id = %s
            """,
            (new_revision_id, active_revision_id),
        )
        connection.execute(
            """
            INSERT INTO vault_rag.embedding_queue(
                revision_id, chunk_id, category, message, attempt_count,
                last_attempted_at, next_eligible_at, terminal
            )
            SELECT %s, chunk_id, category, message, attempt_count,
                last_attempted_at, next_eligible_at, terminal
            FROM vault_rag.embedding_queue WHERE revision_id = %s
            """,
            (new_revision_id, active_revision_id),
        )

    def _assert_building(
        self,
        connection: DatabaseConnection,
        *,
        for_update: bool = False,
    ) -> Mapping[str, Any]:
        suffix = " FOR UPDATE" if for_update else ""
        row = connection.execute(
            "SELECT * FROM vault_rag.vault_revisions WHERE revision_id = %s" + suffix,
            (self._revision_uuid,),
        ).fetchone()
        if row is None or row["vault_id"] != self._vault_id or row["state"] != "building":
            raise StorageError("PostgreSQL revision is not writable")
        return row

    def _validate_vault_scope(self, vault_ids: tuple[str, ...]) -> None:
        validate_vault_ids(vault_ids)
        if set(vault_ids) != {self._vault_id}:
            raise ValueError("revision store only supports its bound vault")

    def _counts(self, connection: DatabaseConnection) -> tuple[int, int, int, int, int]:
        row = connection.execute(
            """
            SELECT COUNT(DISTINCT rs.relative_path) AS source_count,
                COUNT(rc.chunk_id) AS chunk_count,
                COUNT(rc.chunk_id) FILTER (
                    WHERE rc.embedding_state = 'ready'
                ) AS ready_count,
                COUNT(rc.chunk_id) FILTER (
                    WHERE rc.embedding_state = 'pending'
                ) AS pending_count,
                COALESCE(SUM(rc.dimensions * 4) FILTER (
                    WHERE rc.embedding_state = 'ready'
                ), 0) AS vector_bytes
            FROM vault_rag.revision_sources AS rs
            LEFT JOIN vault_rag.revision_chunks AS rc
                ON rc.revision_id = rs.revision_id
                AND rc.source_path = rs.relative_path
            WHERE rs.revision_id = %s AND rs.parse_state = 'active'
            """,
            (self._revision_uuid,),
        ).fetchone()
        if row is None:
            return (0, 0, 0, 0, 0)
        return (
            cast(int, row["source_count"]),
            cast(int, row["chunk_count"]),
            cast(int, row["ready_count"]),
            cast(int, row["pending_count"]),
            cast(int, row["vector_bytes"]),
        )

    @staticmethod
    def _stored_chunk_select() -> sql.Composable:
        return sql.SQL(
            """
            SELECT rc.chunk_id, vr.vault_id, rs.relative_path, rs.source_kind,
                rs.source_blob_hash, rc.ordinal, rc.title, rc.heading_json,
                rc.start_line, rc.end_line, rc.body, rc.embedding_text,
                rc.token_count, rc.metadata_json,
                rc.content_hash AS chunk_content_hash, rc.embedding_state,
                q.category, q.message, q.last_attempted_at
            FROM vault_rag.revision_chunks AS rc
            JOIN vault_rag.vault_revisions AS vr ON vr.revision_id = rc.revision_id
            JOIN vault_rag.revision_sources AS rs
                ON rs.revision_id = rc.revision_id
                AND rs.relative_path = rc.source_path
            LEFT JOIN vault_rag.embedding_queue AS q
                ON q.revision_id = rc.revision_id AND q.chunk_id = rc.chunk_id
            """
        )
