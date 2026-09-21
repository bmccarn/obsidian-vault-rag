"""Persistent SQLite index storage contracts."""

from .ports import AtomicReplaceIndexStore, IndexStore, QueryStore, Store
from .records import (
    DenseCandidate,
    DenseRequest,
    IdentifierCandidate,
    IndexSnapshot,
    LexicalCandidate,
    LexicalRequest,
    PendingEmbedding,
    PendingEmbeddingUpdate,
    RequeueResult,
    SourceProvenance,
    SourceRevision,
    StorageFilters,
    StoredChunk,
    VaultFingerprint,
    VectorLoadRequest,
    VectorRecord,
    VectorScope,
)
from .schema import SCHEMA_VERSION
from .sqlite import SQLiteStore

__all__ = [
    "SCHEMA_VERSION",
    "AtomicReplaceIndexStore",
    "DenseCandidate",
    "DenseRequest",
    "IdentifierCandidate",
    "IndexSnapshot",
    "IndexStore",
    "LexicalCandidate",
    "LexicalRequest",
    "PendingEmbedding",
    "PendingEmbeddingUpdate",
    "QueryStore",
    "RequeueResult",
    "SQLiteStore",
    "SourceProvenance",
    "SourceRevision",
    "StorageFilters",
    "Store",
    "StoredChunk",
    "VaultFingerprint",
    "VectorLoadRequest",
    "VectorRecord",
    "VectorScope",
]
