from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np  # pyright: ignore[reportMissingImports]
import psycopg  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

from tests.contract.postgres_target import (
    assert_safe_test_connection,
    validated_postgres_test_dsn,
)
from vault_rag.domain import JsonValue, LineRange, SourceKind
from vault_rag.ingest.chunker import ChunkRecord
from vault_rag.ingest.models import DiscoveredSource
from vault_rag.storage import PendingEmbedding, SourceRevision, SQLiteStore, VectorRecord
from vault_rag.storage.postgres import (
    PostgresMigrator,
    PostgresPool,
    PostgresStore,
    PostgresWorkerLease,
)
from vault_rag.storage.postgres.index_store import PostgresIndexStore


@dataclass
class StorageBackend:
    """Execute the same active-state mutation against SQLite or PostgreSQL."""

    query: Any
    postgres: bool
    generation: int = 0

    def mutate(self, change: Callable[[Any], None]) -> None:
        if not self.postgres:
            change(self.query)
            return
        self.generation += 1
        store = cast(PostgresStore, self.query)
        build = cast(
            PostgresIndexStore,
            store.begin_revision("vault-a", "refs/heads/main", f"{self.generation:040x}"),
        )
        change(build)
        lease = PostgresWorkerLease(build._pool, "sync:vault-a")
        assert lease.acquire() is True
        try:
            lease.promote(build, lexical_complete=True, fully_reconciled=True)
        finally:
            lease.release()


def source_revision(
    path: str,
    text: str,
    chunk_id: str,
    *,
    metadata: dict[str, JsonValue] | None = None,
    vector: tuple[float, ...] | None = (3.0, 4.0),
    pending: bool = False,
    root: Path = Path("/unused"),
) -> SourceRevision:
    raw = text.encode("utf-8")
    source = DiscoveredSource(
        vault_id="vault-a",
        root=root,
        relative_path=path,
        folded_path=path.casefold(),
        kind=SourceKind.MARKDOWN,
        text=text,
        content_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        size_bytes=len(raw),
        mtime_ns=123,
    )
    chunk = ChunkRecord(
        id=chunk_id,
        vault_id="vault-a",
        relative_path=path,
        ordinal=0,
        title=path,
        heading=("Heading",),
        lines=LineRange(1, 1),
        text=text,
        embedding_text=text,
        token_count=3,
        metadata=metadata or {"kind": "note"},
        content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
    )
    vectors = (
        ()
        if vector is None
        else (
            VectorRecord(
                chunk_id,
                len(vector),
                "config-v1",
                "observed-v1",
                np.asarray(vector, dtype=np.float32),
            ),
        )
    )
    pending_items = (
        ()
        if not pending
        else (
            PendingEmbedding(chunk_id, "provider", "temporary", datetime(2026, 8, 8, tzinfo=UTC)),
        )
    )
    return SourceRevision(
        source,
        (chunk,),
        vectors,
        pending_items,
        "manifest-v1",
        "parser-v1",
        "chunker-v1",
        "config-v1",
    )


@pytest.fixture(params=("sqlite", "postgres"), ids=("sqlite", "postgres"))
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StorageBackend]:
    if request.param == "sqlite":
        store = SQLiteStore(tmp_path / "index.sqlite3")
        store.initialize()
        yield StorageBackend(store, False)
        return

    dsn = validated_postgres_test_dsn()
    with psycopg.connect(dsn, autocommit=True) as connection:
        assert_safe_test_connection(connection)
        connection.execute("DROP SCHEMA IF EXISTS vault_rag CASCADE")
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")
        connection.execute("CREATE EXTENSION vector")
    pool = PostgresPool(dsn, min_size=1, max_size=4, timeout=2.0)
    pool.open(wait=True)
    try:
        PostgresMigrator(pool).apply()
        yield StorageBackend(PostgresStore(pool), True)
    finally:
        pool.close()
