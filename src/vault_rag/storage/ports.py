"""Backend-neutral storage contracts consumed by indexing and retrieval."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np  # pyright: ignore[reportMissingImports]

from vault_rag.domain import JsonValue
from vault_rag.ingest.chunker import ChunkRecord

from .records import (
    ActiveRevision,
    DenseCandidate,
    DenseRequest,
    IdentifierCandidate,
    IndexSnapshot,
    LexicalCandidate,
    LexicalRequest,
    PendingEmbeddingUpdate,
    RequeueResult,
    SourceProvenance,
    SourceRevision,
    StorageFilters,
    StoredChunk,
    VaultFingerprint,
    VectorRecord,
    VectorScope,
)


@runtime_checkable
class IndexStore(Protocol):
    """Persistence operations required by one incremental indexing build."""

    def initialize(self) -> None: ...

    def source_hashes(self, vault_ids: tuple[str, ...]) -> dict[tuple[str, str], str]: ...

    def replace_source(
        self, revision: SourceRevision, *, replacing_path: str | None = None
    ) -> None: ...

    def record_vault_fingerprint(self, fingerprint: VaultFingerprint) -> None: ...

    def update_pending_embeddings(
        self,
        vault_id: str,
        embedding_config_fingerprint: str,
        updates: tuple[PendingEmbeddingUpdate, ...],
    ) -> None: ...

    def reset_vaults(self, vault_ids: tuple[str, ...]) -> None: ...

    def requeue_disabled_chunks(
        self,
        source_keys: tuple[tuple[str, str], ...],
        attempted_at: datetime,
    ) -> tuple[RequeueResult, ...]: ...

    def suppress_source(
        self,
        vault_id: str,
        relative_path: str,
        content_hash: str,
        diagnostic: str,
    ) -> None: ...

    def delete_sources(
        self,
        keys: tuple[tuple[str, str], ...],
        *,
        fingerprints: tuple[VaultFingerprint, ...] = (),
        disable_semantics_for: tuple[str, ...] = (),
        reconciled_vault_ids: tuple[str, ...] | None = None,
    ) -> None: ...
    def reusable_vectors(
        self, chunks: tuple[ChunkRecord, ...], embedding_config_fingerprint: str
    ) -> dict[str, VectorRecord]: ...
    def pending_chunks(self, vault_ids: tuple[str, ...]) -> tuple[StoredChunk, ...]: ...

    def snapshot(self, vault_ids: tuple[str, ...]) -> IndexSnapshot: ...


@runtime_checkable
class AtomicReplaceIndexStore(IndexStore, Protocol):
    """Local store that can publish a rebuilt database with one atomic rename."""

    def replacement_store(self, path: Path) -> AtomicReplaceIndexStore: ...

    @property
    def path(self) -> Path: ...

    def prepare_for_atomic_replace(self) -> None: ...


@runtime_checkable
class QueryStore(Protocol):
    """Bounded query operations required by retrieval and status surfaces."""

    def initialize(self) -> None: ...

    @property
    def database_location(self) -> str: ...

    def consistent_read(self) -> AbstractContextManager[QueryStore]: ...

    def identifier_candidates(
        self,
        vault_ids: tuple[str, ...],
        filters: StorageFilters,
    ) -> tuple[IdentifierCandidate, ...]: ...

    def lexical_search(self, request: LexicalRequest) -> tuple[LexicalCandidate, ...]: ...

    def dense_search(
        self,
        request: DenseRequest,
        query_vector: np.ndarray,
    ) -> tuple[DenseCandidate, ...]: ...

    def vector_scope(
        self,
        vault_ids: tuple[str, ...],
        observed_fingerprint: str,
        filters: StorageFilters,
    ) -> VectorScope: ...

    def chunks_by_ids(
        self,
        ids: tuple[str, ...],
        *,
        vault_ids: tuple[str, ...],
    ) -> dict[str, StoredChunk]: ...

    def source_provenance(
        self,
        vault_id: str,
        relative_path: str,
    ) -> SourceProvenance | None: ...

    def snapshot(self, vault_ids: tuple[str, ...]) -> IndexSnapshot: ...


@runtime_checkable
class SnapshotQueryStore(QueryStore, Protocol):
    """Query store able to expose active revision identity on its current read."""

    def active_revisions(self, vault_ids: tuple[str, ...]) -> tuple[ActiveRevision, ...]: ...

    def snapshots(self, vault_ids: tuple[str, ...]) -> Mapping[str, IndexSnapshot]: ...


@runtime_checkable
class Store(IndexStore, QueryStore, Protocol):
    """Combined backend surface returned by application factories."""


@runtime_checkable
class RevisionBuildStore(IndexStore, Protocol):
    """One invisible PostgreSQL revision under construction."""

    @property
    def revision_id(self) -> str: ...

    def fail(self, category: str, message: str) -> None: ...


@runtime_checkable
class RevisionStore(QueryStore, Protocol):
    """Backend capable of starting immutable per-vault revisions."""

    def begin_revision(
        self,
        vault_id: str,
        configured_ref: str,
        commit_sha: str,
        *,
        manifest: Mapping[str, JsonValue] | None = None,
    ) -> RevisionBuildStore: ...
