"""Vault-scoped hybrid retrieval contracts."""

from .identifiers import (
    IDENTIFIER_SCHEMA_VERSION,
    IdentifierMatch,
    build_fts_query,
    recognize_identifiers,
)
from .service import (
    Continuation,
    ReadRequest,
    ReadResponse,
    RetrievalService,
    ScoreComponents,
    SearchFilters,
    SearchHit,
    SearchMode,
    SearchRequest,
    SearchResponse,
    rrf,
)

__all__ = [
    "IDENTIFIER_SCHEMA_VERSION",
    "Continuation",
    "IdentifierMatch",
    "ReadRequest",
    "ReadResponse",
    "RetrievalService",
    "ScoreComponents",
    "SearchFilters",
    "SearchHit",
    "SearchMode",
    "SearchRequest",
    "SearchResponse",
    "build_fts_query",
    "recognize_identifiers",
    "rrf",
]
