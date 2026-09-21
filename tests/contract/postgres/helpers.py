from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]

from vault_rag.domain import JsonValue, LineRange, SourceKind
from vault_rag.ingest.chunker import ChunkRecord
from vault_rag.ingest.models import DiscoveredSource
from vault_rag.storage import PendingEmbedding, SourceRevision, VectorRecord


def source(
    text: str,
    *,
    vault_id: str = "vault-a",
    path: str = "notes/example.md",
) -> DiscoveredSource:
    payload = text.encode("utf-8")
    return DiscoveredSource(
        vault_id=vault_id,
        root=Path("/unused"),
        relative_path=path,
        folded_path=path.casefold(),
        kind=SourceKind.MARKDOWN,
        text=text,
        content_hash=f"sha256:{hashlib.sha256(payload).hexdigest()}",
        size_bytes=len(payload),
        mtime_ns=123,
    )


def chunk(
    discovered: DiscoveredSource,
    *,
    chunk_id: str = "chunk-a",
    text: str | None = None,
    metadata: dict[str, JsonValue] | None = None,
) -> ChunkRecord:
    body = discovered.text if text is None else text
    return ChunkRecord(
        id=chunk_id,
        vault_id=discovered.vault_id,
        relative_path=discovered.relative_path,
        ordinal=0,
        title="Example",
        heading=("Heading",),
        lines=LineRange(1, 1),
        text=body,
        embedding_text=f"Title: Example\n\n{body}",
        token_count=3,
        metadata=metadata or {"aliases": ["Alias"], "status": "active", "priority": 2},
        content_hash=f"sha256:{hashlib.sha256(body.encode('utf-8')).hexdigest()}",
    )


def revision(
    text: str,
    *,
    vault_id: str = "vault-a",
    path: str = "notes/example.md",
    vector_values: tuple[float, ...] | None = (3.0, 4.0),
    pending: bool = False,
    metadata: dict[str, JsonValue] | None = None,
) -> SourceRevision:
    discovered = source(text, vault_id=vault_id, path=path)
    record = chunk(discovered, metadata=metadata)
    vectors = (
        ()
        if vector_values is None
        else (
            VectorRecord(
                chunk_id=record.id,
                dimensions=len(vector_values),
                config_fingerprint="config-v1",
                observed_fingerprint="observed-v1",
                vector=np.asarray(vector_values, dtype=np.float32),
            ),
        )
    )
    failures = (
        (
            PendingEmbedding(
                chunk_id=record.id,
                category="provider_error",
                message="temporarily unavailable",
                attempted_at=datetime(2026, 8, 8, tzinfo=UTC),
            ),
        )
        if pending
        else ()
    )
    return SourceRevision(
        source=discovered,
        chunks=(record,),
        vectors=vectors,
        pending=failures,
        manifest_fingerprint="manifest-v1",
        parser_fingerprint="parser-v1",
        chunker_fingerprint="chunker-v1",
        embedding_config_fingerprint="config-v1",
    )
