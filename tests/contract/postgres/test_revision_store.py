from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.embedding.fingerprint import observed_fingerprint
from vault_rag.errors import StorageError
from vault_rag.storage import (
    SCHEMA_VERSION,
    DenseRequest,
    LexicalRequest,
    PendingEmbedding,
    PendingEmbeddingUpdate,
    SourceRevision,
    StorageFilters,
    VaultFingerprint,
    VectorRecord,
)
from vault_rag.storage.ports import IndexStore, QueryStore
from vault_rag.storage.postgres import (
    PostgresMigrator,
    PostgresPool,
    PostgresStore,
    PostgresWorkerLease,
)
from vault_rag.storage.postgres.index_store import PostgresIndexStore

from .helpers import chunk, revision, source

SHA_A = "a" * 40
SHA_B = "b" * 40
SHA_C = "c" * 40


@pytest.fixture
def store(postgres_pool: PostgresPool) -> PostgresStore:
    PostgresMigrator(postgres_pool).apply()
    result = PostgresStore(postgres_pool)
    result.initialize()
    return result


def _promote(
    build: PostgresIndexStore,
    *,
    lexical_complete: bool,
    fully_reconciled: bool,
) -> None:
    lease = PostgresWorkerLease(build._pool, f"sync:{build._vault_id}")
    assert lease.acquire() is True
    try:
        lease.promote(
            build,
            lexical_complete=lexical_complete,
            fully_reconciled=fully_reconciled,
        )
    finally:
        lease.release()


def test_build_is_invisible_until_atomic_promotion(store: PostgresStore) -> None:
    build = store.begin_revision("vault-a", "refs/heads/main", SHA_A)
    assert isinstance(build, IndexStore)
    assert isinstance(store, QueryStore)

    assert build.snapshot(("vault-a",)).schema_version == SCHEMA_VERSION
    build.replace_source(revision("old searchable text"))

    assert store.snapshot(("vault-a",)).chunk_count == 0
    assert (
        store.lexical_search(LexicalRequest(("vault-a",), '"searchable"', StorageFilters(), 10))
        == ()
    )

    _promote(build, lexical_complete=True, fully_reconciled=True)

    snapshot = store.snapshot(("vault-a",))
    lexical = store.lexical_search(
        LexicalRequest(("vault-a",), '"searchable"', StorageFilters(), 10)
    )
    dense = store.dense_search(
        DenseRequest(("vault-a",), "observed-v1", StorageFilters(), 10),
        np.asarray([3.0, 4.0], dtype=np.float32),
    )

    assert snapshot.source_count == 1
    assert snapshot.chunk_count == snapshot.ready_count == 1
    assert snapshot.pending_count == 0
    assert snapshot.manifest_fingerprint == "manifest-v1"
    assert [candidate.chunk_id for candidate in lexical] == ["chunk-a"]
    assert dense[0].chunk_id == "chunk-a"
    assert dense[0].score == pytest.approx(1.0)
    assert snapshot.schema_version == SCHEMA_VERSION
    assert store.chunks_by_ids(("chunk-a",), vault_ids=("vault-a",))["chunk-a"].text == (
        "old searchable text"
    )
    provenance = store.source_provenance("vault-a", "notes/example.md")
    assert provenance is not None
    assert provenance.source_hash.startswith("sha256:")


def test_reusable_vectors_round_trip_pgvector_text_from_postgresql_18(
    store: PostgresStore,
) -> None:
    seeded = revision("cached vector")
    vector = replace(
        seeded.vectors[0],
        observed_fingerprint=observed_fingerprint("config-v1", 2),
    )
    seeded = replace(seeded, vectors=(vector,))
    first = cast(PostgresIndexStore, store.begin_revision("vault-a", "refs/heads/main", SHA_A))
    first.replace_source(seeded)
    _promote(first, lexical_complete=True, fully_reconciled=True)

    second = store.begin_revision("vault-a", "refs/heads/main", SHA_B)
    reusable = second.reusable_vectors(seeded.chunks, "config-v1")
    assert tuple(reusable) == (seeded.chunks[0].content_hash,)

    np.testing.assert_allclose(
        reusable[seeded.chunks[0].content_hash].vector,
        np.asarray([0.6, 0.8], dtype=np.float32),
    )


def test_failed_build_preserves_prior_active_revision(store: PostgresStore) -> None:
    first = store.begin_revision("vault-a", "refs/heads/main", SHA_A)
    first.replace_source(revision("old active text"))
    _promote(first, lexical_complete=True, fully_reconciled=True)

    second = store.begin_revision("vault-a", "refs/heads/main", SHA_B)
    second.replace_source(revision("new hidden text"))

    before_failure = store.chunks_by_ids(("chunk-a",), vault_ids=("vault-a",))["chunk-a"]
    second.fail("embedding_error", "provider unavailable")
    after_failure = store.chunks_by_ids(("chunk-a",), vault_ids=("vault-a",))["chunk-a"]

    assert before_failure.text == "old active text"
    assert after_failure.text == "old active text"


def test_promotion_rejects_stale_expected_active_revision(store: PostgresStore) -> None:
    first = store.begin_revision("vault-a", "refs/heads/main", SHA_A)
    first.replace_source(revision("first"))
    _promote(first, lexical_complete=True, fully_reconciled=True)

    competing_a = store.begin_revision("vault-a", "refs/heads/main", SHA_B)
    competing_b = store.begin_revision("vault-a", "refs/heads/main", SHA_C)
    competing_a.replace_source(revision("winner"))
    competing_b.replace_source(revision("loser"))

    _promote(competing_a, lexical_complete=True, fully_reconciled=True)
    with pytest.raises(StorageError, match="active revision changed"):
        _promote(competing_b, lexical_complete=True, fully_reconciled=True)

    active = store.chunks_by_ids(("chunk-a",), vault_ids=("vault-a",))["chunk-a"]
    assert active.text == "winner"


def test_new_build_copies_unchanged_active_content(store: PostgresStore) -> None:
    first = store.begin_revision("vault-a", "refs/heads/main", SHA_A)
    first.replace_source(revision("unchanged"))
    _promote(first, lexical_complete=True, fully_reconciled=True)

    second = store.begin_revision("vault-a", "refs/heads/main", SHA_B)
    assert second.source_hashes(("vault-a",))
    _promote(second, lexical_complete=True, fully_reconciled=True)

    active = store.chunks_by_ids(("chunk-a",), vault_ids=("vault-a",))["chunk-a"]
    assert active.text == "unchanged"


def test_suppressed_source_records_observed_hash_without_rewriting_blob(
    store: PostgresStore,
    postgres_pool: PostgresPool,
) -> None:
    first = store.begin_revision("vault-a", "refs/heads/main", SHA_A)
    first.replace_source(revision("last usable bytes"))
    _promote(first, lexical_complete=True, fully_reconciled=True)
    original = store.source_provenance("vault-a", "notes/example.md")
    assert original is not None

    failing_hash = f"sha256:{'f' * 64}"
    second = store.begin_revision("vault-a", "refs/heads/main", SHA_B)
    second.suppress_source("vault-a", "notes/example.md", failing_hash, "parse failure")

    assert second.source_hashes(("vault-a",)) == {("vault-a", "notes/example.md"): failing_hash}
    with postgres_pool.connection() as connection:
        row = connection.execute(
            """
            SELECT content_hash, source_blob_hash, parse_state
            FROM vault_rag.revision_sources
            WHERE revision_id = %s AND relative_path = 'notes/example.md'
            """,
            (second.revision_id,),
        ).fetchone()
    assert row is not None
    assert row["content_hash"] == failing_hash
    assert row["source_blob_hash"] == original.source_hash
    assert row["parse_state"] == "suppressed"


def test_pending_retry_promotes_success_and_refreshes_failure_atomically(
    store: PostgresStore,
) -> None:
    discovered = source("pending chunks")
    first = chunk(discovered, chunk_id="first", text="first")
    second = replace(chunk(discovered, chunk_id="second", text="second"), ordinal=1)
    attempted_at = datetime(2026, 8, 6, tzinfo=UTC)
    build = store.begin_revision("vault-a", "refs/heads/main", SHA_A)
    build.replace_source(
        SourceRevision(
            source=discovered,
            chunks=(first, second),
            vectors=(),
            pending=(
                PendingEmbedding(first.id, "transport", "old first", attempted_at),
                PendingEmbedding(second.id, "transport", "old second", attempted_at),
            ),
            manifest_fingerprint="manifest-v1",
            parser_fingerprint="parser-v1",
            chunker_fingerprint="chunker-v1",
            embedding_config_fingerprint="config-v1",
        )
    )
    refreshed_at = datetime(2026, 8, 7, tzinfo=UTC)

    build.update_pending_embeddings(
        "vault-a",
        "config-v1",
        (
            PendingEmbeddingUpdate(
                first.id,
                first.content_hash,
                vector=VectorRecord(
                    chunk_id=first.id,
                    dimensions=2,
                    config_fingerprint="config-v1",
                    observed_fingerprint=observed_fingerprint("config-v1", 2),
                    vector=np.asarray([3.0, 4.0], dtype=np.float32),
                ),
            ),
            PendingEmbeddingUpdate(
                second.id,
                second.content_hash,
                failure=PendingEmbedding(
                    second.id,
                    "http",
                    "still unavailable",
                    refreshed_at,
                ),
            ),
        ),
    )
    _promote(build, lexical_complete=True, fully_reconciled=False)

    ready = store.chunks_by_ids((first.id,), vault_ids=("vault-a",))[first.id]
    assert ready.embedding_state.value == "ready"
    pending = store.pending_chunks(("vault-a",))
    assert [item.id for item in pending] == [second.id]
    assert pending[0].pending == PendingEmbedding(
        second.id,
        "http",
        "still unavailable",
        refreshed_at,
    )


def test_requeue_disabled_source_makes_its_chunks_pending(store: PostgresStore) -> None:
    discovered = source("disabled chunk")
    record = chunk(discovered)
    build = store.begin_revision("vault-a", "refs/heads/main", SHA_A)
    build.replace_source(
        SourceRevision(
            source=discovered,
            chunks=(record,),
            vectors=(),
            pending=(),
            manifest_fingerprint="manifest-v1",
            parser_fingerprint="parser-v1",
            chunker_fingerprint="chunker-v1",
            embedding_config_fingerprint="config-v1",
        )
    )

    result = build.requeue_disabled_chunks(
        (("vault-a", discovered.relative_path),),
        datetime(2026, 8, 7, tzinfo=UTC),
    )
    _promote(build, lexical_complete=True, fully_reconciled=False)

    assert result[0].vault_id == "vault-a"
    assert result[0].chunk_count == 1
    assert [item.id for item in store.pending_chunks(("vault-a",))] == [record.id]


def test_empty_vault_fingerprint_is_promoted_without_sources(store: PostgresStore) -> None:
    build = store.begin_revision("empty-vault", "refs/heads/main", SHA_A)
    build.record_vault_fingerprint(
        VaultFingerprint(
            "empty-vault",
            "manifest-empty",
            "parser-v1",
            "chunker-v1",
            "config-v1",
        )
    )
    _promote(build, lexical_complete=True, fully_reconciled=True)

    snapshot = store.snapshot(("empty-vault",))
    assert snapshot.manifest_fingerprint == "manifest-empty"
    assert snapshot.source_count == 0
    assert snapshot.chunk_count == 0


def test_revision_build_rejects_stale_or_malformed_pending_operations(
    store: PostgresStore,
) -> None:
    pending_revision = revision("pending", vector_values=None, pending=True)
    record = pending_revision.chunks[0]
    build = store.begin_revision("vault-a", "refs/heads/main", SHA_A)
    build.replace_source(pending_revision)
    valid_failure = PendingEmbedding(
        record.id,
        "transport",
        "still unavailable",
        datetime(2026, 8, 7, tzinfo=UTC),
    )
    valid_update = PendingEmbeddingUpdate(
        record.id,
        record.content_hash,
        failure=valid_failure,
    )

    build.update_pending_embeddings("vault-a", "config-v1", ())
    with pytest.raises(ValueError, match="vault id"):
        build.update_pending_embeddings("other", "config-v1", (valid_update,))
    with pytest.raises(ValueError, match="unique"):
        build.update_pending_embeddings(
            "vault-a",
            "config-v1",
            (valid_update, valid_update),
        )
    with pytest.raises(StorageError, match="fingerprint is stale"):
        build.update_pending_embeddings("vault-a", "stale-config", (valid_update,))
    with pytest.raises(StorageError, match="stale chunk"):
        build.update_pending_embeddings(
            "vault-a",
            "config-v1",
            (
                PendingEmbeddingUpdate(
                    "missing",
                    record.content_hash,
                    failure=replace(valid_failure, chunk_id="missing"),
                ),
            ),
        )
    with pytest.raises(StorageError, match="could not update pending"):
        build.update_pending_embeddings(
            "vault-a",
            "config-v1",
            (
                PendingEmbeddingUpdate(
                    record.id,
                    record.content_hash,
                    vector=VectorRecord(
                        chunk_id=record.id,
                        dimensions=2,
                        config_fingerprint="config-v1",
                        observed_fingerprint="wrong",
                        vector=np.asarray([3.0, 4.0], dtype=np.float32),
                    ),
                ),
            ),
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        build.requeue_disabled_chunks(
            (("vault-a", "notes/example.md"),),
            datetime(2026, 8, 7),
        )
    with pytest.raises(ValueError, match="match the revision"):
        build.requeue_disabled_chunks(
            (("other", "notes/example.md"),),
            datetime(2026, 8, 7, tzinfo=UTC),
        )
    with pytest.raises(ValueError, match="identity"):
        build.suppress_source("other", "notes/example.md", record.content_hash, "failure")
    with pytest.raises(ValueError, match="does not match"):
        build.record_vault_fingerprint(
            VaultFingerprint(
                "other",
                "manifest-v1",
                "parser-v1",
                "chunker-v1",
                "config-v1",
            )
        )
    with pytest.raises(ValueError, match="non-empty"):
        build.record_vault_fingerprint(
            VaultFingerprint("vault-a", "", "parser-v1", "chunker-v1", "config-v1")
        )


def test_revision_build_rejects_malformed_source_revision_records(store: PostgresStore) -> None:
    build = store.begin_revision("vault-a", "refs/heads/main", SHA_A)
    ready = revision("ready")
    record = ready.chunks[0]
    vector = ready.vectors[0]
    second = replace(record, id="second", ordinal=1)
    attempted_at = datetime(2026, 8, 7, tzinfo=UTC)
    pending = PendingEmbedding(record.id, "transport", "unavailable", attempted_at)

    malformed = (
        replace(ready, manifest_fingerprint=""),
        replace(ready, chunks=(record, record)),
        replace(ready, chunks=(record, replace(second, ordinal=record.ordinal))),
        replace(ready, chunks=(replace(record, vault_id="other"),)),
        replace(ready, chunks=(replace(record, ordinal=-1),)),
        replace(ready, vectors=(replace(vector, chunk_id="missing"),)),
        replace(ready, vectors=(replace(vector, config_fingerprint="other"),)),
        replace(ready, vectors=(replace(vector, observed_fingerprint=""),)),
        replace(
            ready,
            chunks=(record, second),
            vectors=(
                vector,
                replace(
                    vector,
                    chunk_id=second.id,
                    dimensions=3,
                    vector=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                ),
            ),
        ),
        replace(ready, vectors=(), pending=(replace(pending, chunk_id="missing"),)),
        replace(ready, pending=(pending,)),
        replace(ready, vectors=(), pending=(replace(pending, category=""),)),
        replace(
            ready,
            vectors=(),
            pending=(replace(pending, attempted_at=datetime(2026, 8, 7)),),
        ),
    )

    for invalid in malformed:
        with pytest.raises(StorageError, match="invalid source revision"):
            build.replace_source(invalid)
