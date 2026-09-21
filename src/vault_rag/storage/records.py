"""Immutable storage requests, records, and validation helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import PurePosixPath

import numpy as np  # pyright: ignore[reportMissingImports]

from vault_rag.domain import EmbeddingState, JsonValue, LineRange, SourceKind, SourceRef
from vault_rag.ingest.chunker import ChunkRecord
from vault_rag.ingest.models import DiscoveredSource


def canonical_json(value: object) -> str:
    """Return deterministic JSON suitable for durable equality checks."""
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def validate_vault_ids(vault_ids: tuple[str, ...]) -> None:
    if not vault_ids or any(not vault_id for vault_id in vault_ids):
        raise ValueError("at least one non-empty vault id is required")


def validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be positive")


def validate_relative_prefix(prefix: str) -> None:
    if not prefix or prefix.startswith("/") or "\\" in prefix:
        raise ValueError("path prefix must be a non-empty POSIX-relative path")
    path = PurePosixPath(prefix)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("path prefix must be a non-empty POSIX-relative path")


@dataclass(frozen=True, slots=True)
class VectorRecord:
    chunk_id: str
    dimensions: int
    config_fingerprint: str
    observed_fingerprint: str
    vector: np.ndarray


@dataclass(frozen=True, slots=True)
class VectorScope:
    count: int
    dimensions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IdentifierCandidate:
    chunk_id: str
    vault_id: str
    relative_path: str
    title: str
    metadata: Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    vault_id: str
    relative_path: str
    source_kind: SourceKind
    source_hash: str
    indexed_at: datetime
    content: bytes | None = None
    byte_length: int | None = None

    def __post_init__(self) -> None:
        if (self.content is None) != (self.byte_length is None):
            raise ValueError("source bytes and byte length must be supplied together")
        if self.content is not None and self.byte_length != len(self.content):
            raise ValueError("source byte length does not match its payload")


@dataclass(frozen=True, slots=True)
class PendingEmbedding:
    chunk_id: str
    category: str
    message: str
    attempted_at: datetime


@dataclass(frozen=True, slots=True)
class SourceRevision:
    source: DiscoveredSource
    chunks: tuple[ChunkRecord, ...]
    vectors: tuple[VectorRecord, ...]
    pending: tuple[PendingEmbedding, ...]
    manifest_fingerprint: str
    parser_fingerprint: str
    chunker_fingerprint: str
    embedding_config_fingerprint: str
    update_vault_fingerprint: bool = True


@dataclass(frozen=True, slots=True)
class VaultFingerprint:
    vault_id: str
    manifest_fingerprint: str
    parser_fingerprint: str
    chunker_fingerprint: str
    embedding_config_fingerprint: str


@dataclass(frozen=True, slots=True)
class ActiveRevision:
    """One immutable, lexically usable revision selected by a vault pointer."""

    vault_id: str
    revision_id: str
    commit_sha: str
    manifest: Mapping[str, JsonValue]
    state: str
    source_count: int
    chunk_count: int
    ready_count: int
    pending_count: int
    vector_bytes: int
    manifest_fingerprint: str
    parser_fingerprint: str
    chunker_fingerprint: str
    embedding_config_fingerprint: str
    promoted_at: datetime
    configured_ref: str
    configured_sha: str | None = None
    checkout_sha: str | None = None
    checkout_at: datetime | None = None
    configured_at: datetime | None = None
    fetched_sha: str | None = None
    fetched_at: datetime | None = None
    attempted_sha: str | None = None
    attempted_at: datetime | None = None
    reconciled_sha: str | None = None
    reconciled_at: datetime | None = None
    fully_reconciled: bool = False
    sync_degradation_category: str | None = None
    sync_degraded_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PendingEmbeddingUpdate:
    chunk_id: str
    content_hash: str
    vector: VectorRecord | None = None
    failure: PendingEmbedding | None = None


@dataclass(frozen=True, slots=True)
class RequeueResult:
    vault_id: str
    chunk_count: int


@dataclass(frozen=True, slots=True)
class StoredChunk:
    id: str
    vault_id: str
    relative_path: str
    source_kind: SourceKind
    source_hash: str
    ordinal: int
    title: str
    heading: tuple[str, ...]
    lines: LineRange
    text: str
    embedding_text: str
    token_count: int
    metadata: Mapping[str, JsonValue]
    content_hash: str
    embedding_state: EmbeddingState
    pending: PendingEmbedding | None = None

    @property
    def chunk_id(self) -> str:
        return self.id

    @property
    def ref(self) -> SourceRef:
        return SourceRef(
            vault_id=self.vault_id,
            path=self.relative_path,
            heading=self.heading,
            lines=self.lines,
            source_hash=self.source_hash,
        )


@dataclass(frozen=True, slots=True)
class StorageFilters:
    vault_ids: tuple[str, ...] = ()
    path_prefix: str | None = None
    source_kind: SourceKind | None = None
    frontmatter: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if any(not vault_id for vault_id in self.vault_ids):
            raise ValueError("filter vault ids must be non-empty")
        if self.path_prefix is not None:
            validate_relative_prefix(self.path_prefix)
        for key in self.frontmatter:
            if not key or "\x00" in key:
                raise ValueError("frontmatter filter names must be non-empty")
        try:
            canonical_json(dict(self.frontmatter))
        except (TypeError, ValueError) as exc:
            raise ValueError("frontmatter filters must contain canonical JSON values") from exc


@dataclass(frozen=True, slots=True)
class LexicalRequest:
    vault_ids: tuple[str, ...]
    fts_query: str
    filters: StorageFilters
    limit: int

    def __post_init__(self) -> None:
        validate_vault_ids(self.vault_ids)
        validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class LexicalCandidate:
    chunk_id: str
    vault_id: str
    relative_path: str
    source_kind: SourceKind
    source_hash: str
    title: str
    heading: tuple[str, ...]
    lines: LineRange
    text: str
    metadata: Mapping[str, JsonValue]
    score: float

    @property
    def id(self) -> str:
        return self.chunk_id

    @property
    def ref(self) -> SourceRef:
        return SourceRef(
            vault_id=self.vault_id,
            path=self.relative_path,
            heading=self.heading,
            lines=self.lines,
            source_hash=self.source_hash,
        )


@dataclass(frozen=True, slots=True)
class VectorLoadRequest:
    vault_ids: tuple[str, ...]
    observed_fingerprint: str
    filters: StorageFilters
    limit: int

    def __post_init__(self) -> None:
        validate_vault_ids(self.vault_ids)
        if not self.observed_fingerprint:
            raise ValueError("observed fingerprint must be non-empty")
        validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class DenseRequest:
    vault_ids: tuple[str, ...]
    observed_fingerprint: str
    filters: StorageFilters
    limit: int

    def __post_init__(self) -> None:
        validate_vault_ids(self.vault_ids)
        if not self.observed_fingerprint:
            raise ValueError("observed fingerprint must be non-empty")
        validate_limit(self.limit)


@dataclass(frozen=True, slots=True)
class DenseCandidate:
    chunk_id: str
    score: float


@dataclass(frozen=True, slots=True)
class IndexSnapshot:
    source_count: int
    chunk_count: int
    ready_count: int
    pending_count: int
    vector_bytes: int
    indexed_at: datetime | None
    schema_version: int
    manifest_fingerprint: str | None
    parser_fingerprint: str | None
    chunker_fingerprint: str | None
    embedding_config_fingerprint: str | None
    observed_fingerprints: tuple[str, ...]
    vector_dimensions: tuple[int, ...]
    vector_config_fingerprints: tuple[str, ...]

    @property
    def sources(self) -> int:
        return self.source_count

    @property
    def chunks(self) -> int:
        return self.chunk_count

    @property
    def ready(self) -> int:
        return self.ready_count

    @property
    def pending(self) -> int:
        return self.pending_count
