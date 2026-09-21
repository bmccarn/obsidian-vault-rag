"""Vault-scoped hybrid retrieval and hash-verified source reads."""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol, cast

import numpy as np  # pyright: ignore[reportMissingImports]

from vault_rag.config import ResolvedProfile
from vault_rag.domain import DegradedState, JsonValue, LineRange, SourceKind, SourceRef
from vault_rag.errors import SecurityError, StaleSourceError
from vault_rag.ingest.lines import split_source_lines
from vault_rag.ingest.markdown import parse_markdown
from vault_rag.ingest.models import DiscoveredSource, SourceSection
from vault_rag.security import folded_path_key, secure_relative_path
from vault_rag.storage import (
    IdentifierCandidate,
    LexicalRequest,
    StorageFilters,
    StoredChunk,
)
from vault_rag.storage.ports import QueryStore
from vault_rag.storage.records import DenseCandidate, DenseRequest, validate_relative_prefix

from .identifiers import IdentifierMatch, build_fts_query, recognize_identifiers

_MAX_OUTPUT_CHARACTERS = 1_800
_MAX_QUERY_CHARACTERS = 10_000
_MAX_FILTER_JSON_CHARACTERS = 20_000
_MAX_SEARCH_LIMIT = 100
_RRF_K = 60


class QueryEmbeddingClient(Protocol):
    """The retrieval-only surface of an embedding provider."""

    def embed_query(self, text: str, expected_dimensions: int) -> np.ndarray: ...


class _ExactFields(Protocol):
    @property
    def relative_path(self) -> str: ...

    @property
    def title(self) -> str: ...

    @property
    def metadata(self) -> Mapping[str, JsonValue]: ...


def _freeze_json(value: JsonValue) -> object:
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    return value


def _thaw_json(value: object) -> JsonValue:
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, Mapping):
        return {cast(str, key): _thaw_json(item) for key, item in value.items()}
    return cast(JsonValue, value)


def _validate_path_prefix(prefix: str) -> str:
    if not prefix or prefix.startswith("/") or "\\" in prefix:
        raise ValueError("path prefix must be a non-empty POSIX-relative path")
    normalized = unicodedata.normalize("NFC", prefix).rstrip("/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("path prefix must be a non-empty POSIX-relative path")
    return normalized


@dataclass(frozen=True, slots=True, init=False)
class SearchFilters:
    """Validated, copied user filters which compile to storage-only filters."""

    vault_ids: tuple[str, ...]
    path_prefix: str | None
    source_kind: SourceKind | None
    frontmatter: Mapping[str, object]

    def __init__(
        self,
        vault_ids: Sequence[str] = (),
        path_prefix: str | None = None,
        source_kind: SourceKind | str | None = None,
        frontmatter: Mapping[str, JsonValue] | None = None,
    ) -> None:
        copied_vault_ids = tuple(vault_ids)
        if any(not isinstance(vault_id, str) or not vault_id for vault_id in copied_vault_ids):
            raise ValueError("filter vault IDs must be non-empty text")
        if len(set(copied_vault_ids)) != len(copied_vault_ids):
            raise ValueError("filter vault IDs must not contain duplicates")
        normalized_prefix = None if path_prefix is None else _validate_path_prefix(path_prefix)
        try:
            normalized_kind = None if source_kind is None else SourceKind(source_kind)
        except ValueError as exc:
            raise ValueError("unsupported source kind filter") from exc
        copied_frontmatter = dict(frontmatter or {})
        if any(not isinstance(key, str) or not key or "\x00" in key for key in copied_frontmatter):
            raise ValueError("frontmatter filter names must be non-empty text")
        try:
            encoded = json.dumps(
                copied_frontmatter,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            decoded = cast(dict[str, JsonValue], json.loads(encoded))
        except (TypeError, ValueError) as exc:
            raise ValueError("frontmatter filters must contain JSON values") from exc
        if len(encoded) > _MAX_FILTER_JSON_CHARACTERS:
            raise ValueError("frontmatter filters are too large")
        object.__setattr__(self, "vault_ids", copied_vault_ids)
        object.__setattr__(self, "path_prefix", normalized_prefix)
        object.__setattr__(self, "source_kind", normalized_kind)
        object.__setattr__(
            self,
            "frontmatter",
            MappingProxyType({key: _freeze_json(value) for key, value in decoded.items()}),
        )

    def to_storage(
        self,
        profile_vault_ids: tuple[str, ...],
        allowed_frontmatter_fields: frozenset[str],
    ) -> StorageFilters:
        """Compile copied values after checking the bound profile and manifest fields."""
        foreign = sorted(set(self.vault_ids).difference(profile_vault_ids))
        if foreign:
            rendered = ", ".join(foreign)
            raise ValueError(f"filter vault IDs are outside the bound profile: {rendered}")
        unsupported = sorted(set(self.frontmatter).difference(allowed_frontmatter_fields))
        if unsupported:
            raise ValueError(
                "frontmatter filters are not configured for the profile: " + ", ".join(unsupported)
            )
        return StorageFilters(
            vault_ids=self.vault_ids,
            path_prefix=self.path_prefix,
            source_kind=self.source_kind,
            frontmatter={key: _thaw_json(value) for key, value in self.frontmatter.items()},
        )


class SearchMode(StrEnum):
    """Candidate generators used for one retrieval request."""

    LEXICAL = "lexical"
    DENSE = "dense"
    HYBRID = "hybrid"


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    filters: SearchFilters = field(default_factory=SearchFilters)
    limit: int = 10
    mode: SearchMode = SearchMode.HYBRID

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("query must be text")
        if len(self.query) > _MAX_QUERY_CHARACTERS:
            raise ValueError("query is too long")
        if not isinstance(self.filters, SearchFilters):
            raise TypeError("filters must be SearchFilters")
        if isinstance(self.limit, bool) or not 1 <= self.limit <= _MAX_SEARCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {_MAX_SEARCH_LIMIT}")
        try:
            normalized_mode = SearchMode(self.mode)
        except ValueError as exc:
            raise ValueError("unsupported search mode") from exc
        object.__setattr__(self, "mode", normalized_mode)


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    lexical_rank: int | None
    lexical_score: float | None
    dense_rank: int | None
    dense_score: float | None
    fused_score: float
    exact_identifier: bool


@dataclass(frozen=True, slots=True)
class Continuation:
    remaining_characters: int
    next_offset: int
    next_line: int
    next_character: int


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    ref: SourceRef
    text: str
    metadata: Mapping[str, JsonValue]
    scores: ScoreComponents
    continuation: Continuation | None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    hits: tuple[SearchHit, ...]
    degraded: DegradedState
    elapsed_ms: float
    index_age: float | None


@dataclass(frozen=True, slots=True)
class ReadRequest:
    path: str
    vault_id: str | None = None
    heading: str | tuple[str, ...] | None = None
    lines: LineRange | None = None
    expected_source_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("read path must be non-empty text")
        if self.vault_id is not None and not self.vault_id:
            raise ValueError("vault_id must be non-empty when supplied")
        if self.heading is not None and self.lines is not None:
            raise ValueError("heading and lines are mutually exclusive")
        if isinstance(self.heading, str):
            if not self.heading.strip():
                raise ValueError("heading must not be empty")
        elif self.heading is not None and (
            not self.heading or any(not part for part in self.heading)
        ):
            raise ValueError("heading breadcrumb must contain non-empty parts")
        if self.expected_source_hash is not None and (
            not isinstance(self.expected_source_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.expected_source_hash) is None
        ):
            raise ValueError("expected_source_hash must be a lowercase sha256 hash")


@dataclass(frozen=True, slots=True)
class ReadResponse:
    ref: SourceRef
    text: str
    continuation: Continuation | None


def rrf(rank: int) -> float:
    """Return equal-weight one-based reciprocal-rank contribution."""
    if rank < 1:
        raise ValueError("rank must be one-based")
    return 1.0 / (_RRF_K + rank)


def _bounded_text(text: str, start_line: int) -> tuple[str, Continuation | None]:
    if len(text) <= _MAX_OUTPUT_CHARACTERS:
        return text, None
    boundary = _MAX_OUTPUT_CHARACTERS
    if text[boundary - 1 : boundary + 1] == "\r\n":
        boundary -= 1
    excerpt = text[:boundary]
    excerpt_lines = split_source_lines(excerpt)
    last_line = excerpt_lines[-1] if excerpt_lines else ""
    completed_lines = sum(1 for line in excerpt_lines if line.endswith(("\n", "\r")))
    ends_on_boundary = last_line.endswith(("\n", "\r"))
    return excerpt, Continuation(
        remaining_characters=len(text) - boundary,
        next_offset=boundary,
        next_line=start_line + completed_lines,
        next_character=1 if ends_on_boundary else len(last_line) + 1,
    )


def _metadata_scalars(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(item for nested in value.values() for item in _metadata_scalars(nested))
    if isinstance(value, list):
        return tuple(item for nested in value for item in _metadata_scalars(nested))
    if value is None or isinstance(value, bool):
        return ()
    if isinstance(value, str):
        return (unicodedata.normalize("NFC", value).strip(),)
    return (str(value),)


def _normalized_identifier_value(kind: str, raw: str) -> str | None:
    value = raw.strip()
    if kind == "jira":
        return value.upper()
    if kind == "git":
        lowered = value.casefold()
        for prefix in ("sha:", "commit:"):
            if lowered.startswith(prefix):
                lowered = lowered[len(prefix) :]
        return lowered
    if kind == "pr":
        compact = "".join(value.split())
        return compact.casefold()
    if kind == "path":
        return unicodedata.normalize("NFC", value)
    return None


def _is_exact(chunk: _ExactFields, identifiers: tuple[IdentifierMatch, ...]) -> bool:
    if not identifiers:
        return False
    scalars = (
        unicodedata.normalize("NFC", chunk.title).strip(),
        *(scalar for value in chunk.metadata.values() for scalar in _metadata_scalars(value)),
    )
    for identifier in identifiers:
        if identifier.kind == "path" and chunk.relative_path == identifier.value:
            return True
        expected = _normalized_identifier_value(identifier.kind, identifier.value)
        if expected is not None and any(
            _normalized_identifier_value(identifier.kind, scalar) == expected for scalar in scalars
        ):
            return True
        if identifier.kind == "pr" and "#" in identifier.value:
            repository, number = identifier.value.rsplit("#", 1)
            repo_values = _metadata_scalars(chunk.metadata.get("repo"))
            pr_values = _metadata_scalars(chunk.metadata.get("pr"))
            if any(value.casefold() == repository for value in repo_values) and number in pr_values:
                return True
    return False


@dataclass(frozen=True, slots=True)
class _PreparedDenseQuery:
    dimensions: int | None = None
    vector: np.ndarray | None = None
    degraded_reason: str | None = None


class RetrievalService:
    """Search and read only within one resolved profile allowlist."""

    def __init__(
        self,
        profile: ResolvedProfile,
        store: QueryStore,
        embedding: QueryEmbeddingClient,
        observed_fingerprint: str,
        *,
        observed_vector_dimensions: tuple[int, ...] | None = None,
        semantic_unavailable_reason: str | None = None,
        coverage_degraded_reason: str | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not observed_fingerprint:
            raise ValueError("observed fingerprint must be non-empty")
        self._profile = profile
        self._store = store
        self._embedding = embedding
        self._observed_fingerprint = observed_fingerprint
        self._semantic_unavailable_reason = semantic_unavailable_reason
        self._coverage_degraded_reason = coverage_degraded_reason
        self._observed_vector_dimensions = observed_vector_dimensions
        self._clock = clock
        self._timer = timer
        self._roots = {vault.manifest.id: vault.root for vault in profile.vaults}
        if len(self._roots) != len(profile.vaults):
            raise ValueError("profile vault IDs must be unique")
        self._vault_ids = tuple(self._roots)
        self._frontmatter_fields = frozenset(
            field_name
            for vault in profile.vaults
            for field_name in vault.manifest.metadata.frontmatter_fields
        )

    @property
    def vault_ids(self) -> tuple[str, ...]:
        """Return the explicit bound vault allowlist for validation-only consumers."""
        return self._vault_ids

    def search(self, request: SearchRequest) -> SearchResponse:
        """Prepare remote semantics before executing candidates on a pinned store."""
        started = self._timer()
        storage_filters = request.filters.to_storage(self._vault_ids, self._frontmatter_fields)
        prepared_dense = self._prepare_dense_query(request, storage_filters)
        with self._store.consistent_read() as store:
            return self._search(request, store, storage_filters, prepared_dense, started)

    def _prepare_dense_query(
        self,
        request: SearchRequest,
        storage_filters: StorageFilters,
    ) -> _PreparedDenseQuery:
        dense_enabled = request.mode in {SearchMode.DENSE, SearchMode.HYBRID}
        if not dense_enabled or not request.query.strip():
            return _PreparedDenseQuery()
        if not self._profile.semantic_enabled:
            return _PreparedDenseQuery(degraded_reason="semantic_disabled_by_policy")
        if self._semantic_unavailable_reason is not None:
            return _PreparedDenseQuery(degraded_reason=self._semantic_unavailable_reason)

        dimensions_seen = self._observed_vector_dimensions
        if dimensions_seen is None:
            with self._store.consistent_read() as pinned:
                dimensions_seen = pinned.vector_scope(
                    self._vault_ids,
                    self._observed_fingerprint,
                    storage_filters,
                ).dimensions
        if not dimensions_seen:
            return _PreparedDenseQuery(degraded_reason="no_compatible_vectors")
        if len(dimensions_seen) != 1:
            return _PreparedDenseQuery(degraded_reason="incompatible_vector_dimensions")
        dimensions = dimensions_seen[0]
        try:
            query_vector = np.asarray(
                self._embedding.embed_query(request.query, dimensions), dtype=np.float32
            )
            if (
                query_vector.ndim != 1
                or query_vector.size != dimensions
                or not np.isfinite(query_vector).all()
            ):
                raise ValueError("invalid query vector")
            norm = float(np.linalg.norm(query_vector))
            if not np.isfinite(norm) or norm <= 0:
                raise ValueError("invalid query vector")
            normalized = np.asarray(query_vector / norm, dtype=np.float32)
        except Exception:  # provider and query-vector validation failures degrade uniformly
            return _PreparedDenseQuery(degraded_reason="query_embedding_failed")
        return _PreparedDenseQuery(dimensions, normalized)

    def _search(
        self,
        request: SearchRequest,
        store: QueryStore,
        storage_filters: StorageFilters,
        prepared_dense: _PreparedDenseQuery,
        started: float,
    ) -> SearchResponse:
        candidate_limit = max(request.limit * 4, 20)
        lexical_enabled = request.mode in {SearchMode.LEXICAL, SearchMode.HYBRID}
        dense_enabled = request.mode in {SearchMode.DENSE, SearchMode.HYBRID}
        identifiers = recognize_identifiers(request.query) if lexical_enabled else ()
        exact_candidates: tuple[IdentifierCandidate, ...] = ()
        if identifiers:
            exact_candidates = tuple(
                candidate
                for candidate in store.identifier_candidates(self._vault_ids, storage_filters)
                if _is_exact(candidate, identifiers)
            )
        fts_query = build_fts_query(request.query) if lexical_enabled else ""
        lexical = (
            store.lexical_search(
                LexicalRequest(self._vault_ids, fts_query, storage_filters, candidate_limit)
            )
            if fts_query
            else ()
        )
        lexical_ranks = {
            candidate.chunk_id: (rank, candidate.score)
            for rank, candidate in enumerate(lexical, start=1)
        }

        dense: tuple[DenseCandidate, ...] = ()
        degraded = DegradedState(
            prepared_dense.degraded_reason is not None,
            prepared_dense.degraded_reason,
        )
        if prepared_dense.vector is not None and prepared_dense.dimensions is not None:
            dimensions_seen = store.vector_scope(
                self._vault_ids,
                self._observed_fingerprint,
                storage_filters,
            ).dimensions
            if not dimensions_seen:
                degraded = DegradedState(True, "no_compatible_vectors")
            elif dimensions_seen != (prepared_dense.dimensions,):
                degraded = DegradedState(True, "incompatible_vector_dimensions")
            else:
                dense = store.dense_search(
                    DenseRequest(
                        self._vault_ids,
                        self._observed_fingerprint,
                        storage_filters,
                        candidate_limit,
                    ),
                    prepared_dense.vector,
                )
                if not dense:
                    degraded = DegradedState(True, "no_compatible_vectors")

        dense_ranks = {
            candidate.chunk_id: (rank, candidate.score)
            for rank, candidate in enumerate(dense, start=1)
        }
        exact_ids = tuple(candidate.chunk_id for candidate in exact_candidates)
        candidate_ids = tuple(dict.fromkeys((*exact_ids, *lexical_ranks, *dense_ranks)))
        chunks = store.chunks_by_ids(candidate_ids, vault_ids=self._vault_ids)
        scored: list[tuple[StoredChunk, ScoreComponents]] = []
        for chunk_id in candidate_ids:
            chunk = chunks.get(chunk_id)
            if chunk is None:
                continue
            lexical_component = lexical_ranks.get(chunk_id)
            dense_component = dense_ranks.get(chunk_id)
            fused_score = sum(
                rrf(component[0])
                for component in (lexical_component, dense_component)
                if component is not None
            )
            scored.append(
                (
                    chunk,
                    ScoreComponents(
                        lexical_rank=None if lexical_component is None else lexical_component[0],
                        lexical_score=None if lexical_component is None else lexical_component[1],
                        dense_rank=None if dense_component is None else dense_component[0],
                        dense_score=None if dense_component is None else dense_component[1],
                        fused_score=fused_score,
                        exact_identifier=_is_exact(chunk, identifiers),
                    ),
                )
            )
        scored.sort(
            key=lambda item: (
                not item[1].exact_identifier,
                -item[1].fused_score,
                item[0].vault_id,
                item[0].relative_path,
                item[0].lines.start,
                item[0].id,
            )
        )
        hits: list[SearchHit] = []
        for chunk, score_components in scored[: request.limit]:
            text, continuation = _bounded_text(chunk.text, chunk.lines.start)
            hits.append(
                SearchHit(
                    chunk_id=chunk.id,
                    ref=chunk.ref,
                    text=text,
                    metadata=MappingProxyType(dict(chunk.metadata)),
                    scores=score_components,
                    continuation=continuation,
                )
            )
        if (
            dense_enabled
            and self._coverage_degraded_reason is not None
            and degraded.reason
            in {
                None,
                "no_compatible_vectors",
            }
        ):
            degraded = DegradedState(True, self._coverage_degraded_reason)
        snapshot = store.snapshot(self._vault_ids)
        index_age = None
        if snapshot.indexed_at is not None:
            index_age = max(0.0, (self._clock() - snapshot.indexed_at).total_seconds())
        elapsed_ms = max(0.0, (self._timer() - started) * 1000.0)
        return SearchResponse(tuple(hits), degraded, elapsed_ms, index_age)

    def read(self, request: ReadRequest) -> ReadResponse:
        """Read current strict UTF-8 bytes only after active-index hash verification."""
        if request.vault_id is None:
            if len(self._vault_ids) != 1:
                raise ValueError("vault_id is required for a multi-vault profile")
            vault_id = self._vault_ids[0]
        else:
            vault_id = request.vault_id
        root = self._roots.get(vault_id)
        if root is None:
            raise ValueError("vault_id is outside the bound profile")
        if root.exists() or Path(request.path).is_absolute():
            relative_path = secure_relative_path(root, Path(request.path))
        else:
            validate_relative_prefix(request.path)
            relative_path = unicodedata.normalize("NFC", PurePosixPath(request.path).as_posix())
        provenance = self._store.source_provenance(vault_id, relative_path)
        if provenance is None:
            raise ValueError("source is not active in the bound profile index")
        if provenance.content is not None:
            raw = provenance.content
        else:
            source_path = root / secure_relative_path(root, Path(request.path))
            try:
                raw = source_path.read_bytes()
            except OSError as exc:
                raise SecurityError("could not read source beneath registered vault") from exc
        current_hash = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        if current_hash != provenance.source_hash:
            raise StaleSourceError(
                "source bytes changed after indexing",
                details={"vault_id": vault_id, "path": relative_path},
            )
        if (
            request.expected_source_hash is not None
            and current_hash != request.expected_source_hash
        ):
            raise StaleSourceError(
                "source hash does not match caller expectation",
                details={"vault_id": vault_id, "path": relative_path},
            )
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("indexed source is not strict UTF-8") from exc
        source_lines = split_source_lines(text)
        if not source_lines:
            raise ValueError("source has no readable lines")

        heading: tuple[str, ...] = ()
        selected_lines: LineRange
        if request.heading is not None:
            sections: tuple[SourceSection, ...] = ()
            if provenance.source_kind is SourceKind.MARKDOWN:
                parsed = parse_markdown(
                    DiscoveredSource(
                        vault_id=vault_id,
                        root=root,
                        relative_path=relative_path,
                        folded_path=folded_path_key(relative_path),
                        kind=provenance.source_kind,
                        text=text,
                        content_hash=provenance.source_hash,
                        size_bytes=len(raw),
                        mtime_ns=0,
                    )
                )
                sections = parsed.sections
            if isinstance(request.heading, tuple):
                candidates = tuple(
                    section for section in sections if section.heading == request.heading
                )
            else:
                candidates = tuple(
                    section
                    for section in sections
                    if section.heading and section.heading[-1] == request.heading
                )
            if len(candidates) != 1:
                evidence = tuple(
                    f"{' > '.join(candidate.heading) or '<preamble>'} "
                    f"(L{candidate.lines.start}-L{candidate.lines.end})"
                    for candidate in candidates[:10]
                )
                if not candidates:
                    raise ValueError("heading did not match an indexed breadcrumb")
                rendered = "; ".join(evidence)[:800]
                raise ValueError(f"heading is ambiguous; candidates: {rendered}")
            heading = candidates[0].heading
            selected_lines = candidates[0].lines
        elif request.lines is not None:
            selected_lines = request.lines
        else:
            selected_lines = LineRange(1, len(source_lines))
        if selected_lines.end > len(source_lines):
            raise ValueError("line range exceeds the current source")
        selected_text = "".join(source_lines[selected_lines.start - 1 : selected_lines.end])
        bounded, continuation = _bounded_text(selected_text, selected_lines.start)
        return ReadResponse(
            ref=SourceRef(
                vault_id=vault_id,
                path=relative_path,
                heading=heading,
                lines=selected_lines,
                source_hash=provenance.source_hash,
            ),
            text=bounded,
            continuation=continuation,
        )
