import hashlib
import inspect
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.domain import JsonValue, LineRange, SourceKind  # type: ignore[import-untyped]
from vault_rag.embedding import observed_fingerprint  # type: ignore[import-untyped]
from vault_rag.errors import RebuildRequiredError, StorageError  # type: ignore[import-untyped]
from vault_rag.ingest.chunker import ChunkRecord  # type: ignore[import-untyped]
from vault_rag.ingest.models import DiscoveredSource  # type: ignore[import-untyped]
from vault_rag.storage import (  # type: ignore[import-untyped]
    LexicalRequest,
    PendingEmbedding,
    PendingEmbeddingUpdate,
    RequeueResult,
    SourceRevision,
    SQLiteStore,
    StorageFilters,
    VaultFingerprint,
    VectorLoadRequest,
    VectorRecord,
)
from vault_rag.storage.codecs import decode_vector, encode_vector  # type: ignore[import-untyped]
from vault_rag.storage.records import (  # type: ignore[import-untyped]
    DenseRequest,
)
from vault_rag.storage.records import (
    StorageFilters as RecordsStorageFilters,
)
from vault_rag.storage.records import (
    VectorRecord as RecordsVectorRecord,
)
from vault_rag.storage.sqlite import (  # type: ignore[import-untyped]
    StorageFilters as SQLiteStorageFilters,
)
from vault_rag.storage.sqlite import (  # type: ignore[import-untyped]
    VectorRecord as SQLiteVectorRecord,
)


def discovered_source(
    text: str = "old text",
    *,
    vault_id: str = "vault-a",
    path: str = "notes/example.md",
    folded_path: str | None = None,
) -> DiscoveredSource:
    raw = text.encode()
    return DiscoveredSource(
        vault_id=vault_id,
        root=Path("/vault"),
        relative_path=path,
        folded_path=folded_path or path.casefold(),
        kind=SourceKind.MARKDOWN,
        text=text,
        content_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        size_bytes=len(raw),
        mtime_ns=123,
    )


def chunk(
    source: DiscoveredSource,
    *,
    chunk_id: str = "chunk-a",
    ordinal: int = 0,
    text: str | None = None,
    metadata: dict[str, JsonValue] | None = None,
) -> ChunkRecord:
    body = source.text if text is None else text
    return ChunkRecord(
        id=chunk_id,
        vault_id=source.vault_id,
        relative_path=source.relative_path,
        ordinal=ordinal,
        title="Example",
        heading=("Heading",),
        lines=LineRange(1, 1),
        text=body,
        embedding_text=f"Title: Example\n\n{body}",
        token_count=3,
        metadata=metadata or {"status": "active", "priority": 2},
        content_hash=f"sha256:{hashlib.sha256(body.encode()).hexdigest()}",
    )


def vector(
    chunk_id: str = "chunk-a",
    values: tuple[float, ...] = (3.0, 4.0),
    *,
    config_fingerprint: str = "config-v1",
    observed_fingerprint: str = "observed-v1",
) -> VectorRecord:
    return VectorRecord(
        chunk_id=chunk_id,
        dimensions=len(values),
        config_fingerprint=config_fingerprint,
        observed_fingerprint=observed_fingerprint,
        vector=np.asarray(values, dtype=np.float64),
    )


def revision(
    source: DiscoveredSource | None = None,
    *,
    chunks: tuple[ChunkRecord, ...] | None = None,
    vectors: tuple[VectorRecord, ...] | None = None,
    pending: tuple[PendingEmbedding, ...] = (),
) -> SourceRevision:
    source = source or discovered_source()
    source_chunks = chunks or (chunk(source),)
    source_vectors = vectors if vectors is not None else (vector(source_chunks[0].id),)
    return SourceRevision(
        source=source,
        chunks=source_chunks,
        vectors=source_vectors,
        pending=pending,
        manifest_fingerprint="manifest-v1",
        parser_fingerprint="parser-v1",
        chunker_fingerprint="chunker-v1",
        embedding_config_fingerprint="config-v1",
    )


def test_sqlite_reexports_storage_record_types() -> None:
    assert SQLiteVectorRecord is RecordsVectorRecord is VectorRecord
    assert SQLiteStorageFilters is RecordsStorageFilters is StorageFilters


def test_vector_codecs_normalize_and_validate_payloads() -> None:
    encoded = encode_vector(np.asarray([3.0, 4.0]), dimensions=2)
    decoded = decode_vector(encoded, dimensions=2)

    assert decoded.dtype == np.dtype("float32")
    assert np.allclose(decoded, np.asarray([0.6, 0.8], dtype=np.float32))
    assert decoded.flags.writeable is False
    with pytest.raises(StorageError, match="dimensions"):
        decode_vector(encoded[:-1], dimensions=2)
    with pytest.raises(ValueError, match="non-zero"):
        encode_vector(np.asarray([0.0, 0.0]), dimensions=2)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return tmp_path / "index.sqlite3"


@pytest.fixture
def store(database: Path) -> SQLiteStore:
    result = SQLiteStore(database)
    result.initialize()
    return result


def test_initialize_enables_required_sqlite_features(database: Path, store: SQLiteStore) -> None:
    del store
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert (
            connection.execute("SELECT sql FROM sqlite_master WHERE name = 'fts_chunks'")
            .fetchone()[0]
            .startswith("CREATE VIRTUAL TABLE")
        )
        assert connection.execute("SELECT fts5('fts5 available')").fetchone()[0] is None
        vault_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(vaults)").fetchall()
        }
        assert {
            "vault_id",
            "manifest_fingerprint",
            "parser_fingerprint",
            "chunker_fingerprint",
            "embedding_config_fingerprint",
        } <= vault_columns
    finally:
        connection.close()
    assert SQLiteStore(database).foreign_keys_enabled() is True


def test_initialize_rejects_unknown_schema_version(database: Path) -> None:
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA user_version = 99")
    connection.close()
    with pytest.raises(RebuildRequiredError):
        SQLiteStore(database).initialize()


def test_initialize_rejects_pre_fix_schema_v1_vault_shape(database: Path) -> None:
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE vaults (vault_id TEXT PRIMARY KEY) STRICT")
    connection.execute("PRAGMA user_version = 1")
    connection.close()

    with pytest.raises(RebuildRequiredError, match=r"vaults.*missing required columns") as error:
        SQLiteStore(database).initialize()

    assert error.value.details == {
        "table": "vaults",
        "missing_columns": [
            "chunker_fingerprint",
            "embedding_config_fingerprint",
            "manifest_fingerprint",
            "parser_fingerprint",
            "updated_at",
        ],
    }
    connection = sqlite3.connect(database)
    try:
        assert {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        } == {"vaults"}
        assert [row[1] for row in connection.execute("PRAGMA table_info(vaults)")] == ["vault_id"]
    finally:
        connection.close()


def test_replace_source_is_atomic(store: SQLiteStore) -> None:
    initial = revision()
    store.replace_source(initial)
    changed = replace(initial, chunks=(initial.chunks[0], initial.chunks[0]))
    with pytest.raises(StorageError):
        store.replace_source(changed)
    assert (
        store.chunks_by_ids((initial.chunks[0].id,), vault_ids=("vault-a",))[
            initial.chunks[0].id
        ].text
        == "old text"
    )


def test_replace_removes_old_fts_vector_and_pending_rows(store: SQLiteStore) -> None:
    source = discovered_source("shared old text")
    old_chunk = chunk(source, chunk_id="old-chunk")
    store.replace_source(
        revision(
            source,
            chunks=(old_chunk,),
            vectors=(),
            pending=(
                PendingEmbedding(
                    chunk_id=old_chunk.id,
                    category="transport",
                    message="temporarily unavailable",
                    attempted_at=datetime(2026, 8, 6, tzinfo=UTC),
                ),
            ),
        )
    )
    changed_source = discovered_source("fresh text")
    fresh_chunk = chunk(changed_source, chunk_id="fresh-chunk")
    store.replace_source(revision(changed_source, chunks=(fresh_chunk,)))

    assert store.chunks_by_ids(("old-chunk",), vault_ids=("vault-a",)) == {}
    assert store.pending_chunks(("vault-a",)) == ()
    assert (
        store.load_vectors(
            VectorLoadRequest(
                vault_ids=("vault-a",),
                observed_fingerprint="observed-v1",
                filters=StorageFilters(),
                limit=10,
            )
        )[0].chunk_id
        == "fresh-chunk"
    )
    assert (
        store.lexical_search(
            LexicalRequest(
                vault_ids=("vault-a",), fts_query='"shared"', filters=StorageFilters(), limit=10
            )
        )
        == ()
    )


def test_pending_state_has_no_vector_and_persists_error(store: SQLiteStore) -> None:
    source = discovered_source("not embedded")
    source_chunk = chunk(source)
    attempted_at = datetime(2026, 8, 6, 12, 30, tzinfo=UTC)
    store.replace_source(
        revision(
            source,
            chunks=(source_chunk,),
            vectors=(),
            pending=(
                PendingEmbedding(
                    source_chunk.id,
                    "timeout",
                    "endpoint unavailable",
                    attempted_at,
                ),
            ),
        )
    )
    assert (
        store.load_vectors(VectorLoadRequest(("vault-a",), "observed-v1", StorageFilters(), 10))
        == ()
    )
    pending = store.pending_chunks(("vault-a",))
    assert len(pending) == 1
    assert pending[0].embedding_state.value == "pending"
    assert pending[0].pending is not None
    assert pending[0].pending.attempted_at == attempted_at


def test_lexical_filters_apply_before_rank_and_limit(store: SQLiteStore) -> None:
    source_a = discovered_source("shared phrase", vault_id="vault-a", path="a.md")
    source_b = discovered_source("shared phrase", vault_id="vault-b", path="projects/b.md")
    store.replace_source(revision(source_a, chunks=(chunk(source_a, chunk_id="a"),)))
    store.replace_source(
        revision(
            source_b,
            chunks=(
                chunk(
                    source_b,
                    chunk_id="b",
                    metadata={"status": "active", "priority": 2},
                ),
            ),
        )
    )
    request = LexicalRequest(
        vault_ids=("vault-a", "vault-b"),
        fts_query='"shared"',
        filters=StorageFilters(
            vault_ids=("vault-b",),
            path_prefix="projects/",
            source_kind=SourceKind.MARKDOWN,
            frontmatter={"status": "active", "priority": 2},
        ),
        limit=1,
    )
    result = store.lexical_search(request)
    assert len(result) == 1
    assert result[0].vault_id == "vault-b"
    assert result[0].chunk_id == "b"


def test_selected_frontmatter_values_are_searchable_and_null_is_filterable(
    store: SQLiteStore,
) -> None:
    source = discovered_source("ordinary body")
    source_chunk = chunk(
        source,
        metadata={"ticket": "DIS-123", "owner": None, "aliases": ["special-alias"]},
    )
    store.replace_source(revision(source, chunks=(source_chunk,)))

    result = store.lexical_search(
        LexicalRequest(
            ("vault-a",),
            '"DIS" AND "123"',
            StorageFilters(frontmatter={"owner": None}),
            5,
        )
    )
    assert result[0].chunk_id == source_chunk.id
    assert store.lexical_search(LexicalRequest(("vault-a",), '"special"', StorageFilters(), 5))


def test_lexical_query_is_bound_and_malformed_queries_are_bounded(store: SQLiteStore) -> None:
    store.replace_source(revision())
    assert store.lexical_search(
        LexicalRequest(("vault-a",), '"old" OR "text"', StorageFilters(), 5)
    )
    with pytest.raises(StorageError) as error:
        store.lexical_search(LexicalRequest(("vault-a",), '" OR 1=1 --', StorageFilters(), 5))
    assert "SELECT" not in str(error.value)
    assert "old text" not in str(error.value)


def test_vector_codec_normalizes_float32_and_suppresses_stale_fingerprint(
    store: SQLiteStore,
) -> None:
    store.replace_source(revision())
    loaded = store.load_vectors(VectorLoadRequest(("vault-a",), "observed-v1", StorageFilters(), 5))
    assert loaded[0].vector.dtype == np.float32
    np.testing.assert_allclose(loaded[0].vector, np.array([0.6, 0.8], dtype=np.float32))
    assert store.load_vectors(VectorLoadRequest(("vault-a",), "stale", StorageFilters(), 5)) == ()


@pytest.mark.parametrize(
    "bad_vector",
    [
        VectorRecord("chunk-a", 3, "config-v1", "observed-v1", np.array([1.0, 2.0])),
        VectorRecord("chunk-a", 2, "config-v1", "observed-v1", np.array([0.0, 0.0])),
        VectorRecord("chunk-a", 2, "config-v1", "observed-v1", np.array([np.nan, 1.0])),
        VectorRecord("chunk-a", 2, "other-config", "observed-v1", np.array([1.0, 2.0])),
    ],
)
def test_replace_rejects_invalid_vectors(store: SQLiteStore, bad_vector: VectorRecord) -> None:
    with pytest.raises(StorageError):
        store.replace_source(revision(vectors=(bad_vector,)))
    assert store.snapshot(("vault-a",)).source_count == 0


def test_same_observed_fingerprint_cannot_mix_embedding_configurations(
    store: SQLiteStore,
) -> None:
    source_a = discovered_source("first source", path="a.md")
    chunk_a = chunk(source_a, chunk_id="first")
    store.replace_source(revision(source_a, chunks=(chunk_a,)))
    before_snapshot = store.snapshot(("vault-a",))
    before_hashes = store.source_hashes(("vault-a",))
    with sqlite3.connect(store.path) as connection:
        before_state = connection.execute("SELECT * FROM index_state").fetchall()
        before_vaults = connection.execute("SELECT * FROM vaults").fetchall()

    source_b = discovered_source("second source", path="b.md")
    chunk_b = chunk(source_b, chunk_id="second")
    incompatible = replace(
        revision(source_b, chunks=(chunk_b,)),
        vectors=(
            vector(
                chunk_b.id,
                config_fingerprint="config-v2",
                observed_fingerprint="observed-v1",
            ),
        ),
        embedding_config_fingerprint="config-v2",
    )
    with pytest.raises(StorageError):
        store.replace_source(incompatible)

    assert store.snapshot(("vault-a",)) == before_snapshot
    assert store.source_hashes(("vault-a",)) == before_hashes
    assert store.chunks_by_ids((chunk_b.id,), vault_ids=("vault-a",)) == {}
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT * FROM index_state").fetchall() == before_state
        assert connection.execute("SELECT * FROM vaults").fetchall() == before_vaults


def test_same_observed_fingerprint_cannot_mix_dimensions(store: SQLiteStore) -> None:
    source = discovered_source("two chunks")
    first = chunk(source, chunk_id="first", ordinal=0, text="first")
    second = chunk(source, chunk_id="second", ordinal=1, text="second")
    with pytest.raises(StorageError):
        store.replace_source(
            revision(
                source,
                chunks=(first, second),
                vectors=(
                    vector("first", (1.0, 2.0)),
                    vector("second", (1.0, 2.0, 3.0)),
                ),
            )
        )


def test_delete_and_suppress_remove_all_searchable_dependents(store: SQLiteStore) -> None:
    source_a = discovered_source("deletable", path="a.md")
    source_b = discovered_source("suppressible", path="b.md")
    store.replace_source(revision(source_a, chunks=(chunk(source_a, chunk_id="a"),)))
    store.replace_source(revision(source_b, chunks=(chunk(source_b, chunk_id="b"),)))
    store.suppress_source("vault-a", "b.md", "sha256:broken", "parse failure")
    store.delete_sources((("vault-a", "a.md"),))
    snapshot = store.snapshot(("vault-a",))
    assert snapshot.source_count == 0
    assert snapshot.chunk_count == 0
    assert (
        store.lexical_search(
            LexicalRequest(("vault-a",), '"deletable" OR "suppressible"', StorageFilters(), 10)
        )
        == ()
    )
    assert store.source_hashes(("vault-a",)) == {("vault-a", "b.md"): "sha256:broken"}


def test_case_folded_paths_are_unique_per_vault(store: SQLiteStore) -> None:
    upper = discovered_source("one", path="Notes/A.md", folded_path="notes/a.md")
    lower = discovered_source("two", path="notes/a.md", folded_path="notes/a.md")
    store.replace_source(revision(upper, chunks=(chunk(upper, chunk_id="upper"),)))
    with pytest.raises(StorageError):
        store.replace_source(revision(lower, chunks=(chunk(lower, chunk_id="lower"),)))
    assert store.source_hashes(("vault-a",)) == {("vault-a", "Notes/A.md"): upper.content_hash}


def test_snapshot_fingerprints_are_vault_scoped_and_require_agreement(
    database: Path, store: SQLiteStore
) -> None:
    source_a = discovered_source("alpha", vault_id="vault-a", path="a.md")
    source_b = discovered_source("beta", vault_id="vault-b", path="b.md")
    source_c = discovered_source("gamma", vault_id="vault-c", path="c.md")
    store.replace_source(revision(source_a, chunks=(chunk(source_a, chunk_id="a"),)))
    store.replace_source(
        replace(
            revision(source_b, chunks=(chunk(source_b, chunk_id="b"),), vectors=()),
            manifest_fingerprint="manifest-v2",
            parser_fingerprint="parser-v2",
            chunker_fingerprint="chunker-v2",
            embedding_config_fingerprint="config-v2",
        )
    )
    store.replace_source(revision(source_c, chunks=(chunk(source_c, chunk_id="c"),), vectors=()))

    snapshot_a = store.snapshot(("vault-a",))
    assert snapshot_a.manifest_fingerprint == "manifest-v1"
    assert snapshot_a.parser_fingerprint == "parser-v1"
    assert snapshot_a.chunker_fingerprint == "chunker-v1"
    assert snapshot_a.embedding_config_fingerprint == "config-v1"

    disagreeing = store.snapshot(("vault-a", "vault-b"))
    assert disagreeing.manifest_fingerprint is None
    assert disagreeing.parser_fingerprint is None
    assert disagreeing.chunker_fingerprint is None
    assert disagreeing.embedding_config_fingerprint is None

    agreeing = store.snapshot(("vault-a", "vault-c"))
    assert agreeing.manifest_fingerprint == "manifest-v1"
    assert agreeing.parser_fingerprint == "parser-v1"
    assert agreeing.chunker_fingerprint == "chunker-v1"
    assert agreeing.embedding_config_fingerprint == "config-v1"

    missing = store.snapshot(("vault-a", "missing-vault"))
    assert missing.manifest_fingerprint is None
    assert missing.embedding_config_fingerprint is None

    reopened = SQLiteStore(database)
    reopened.initialize()
    assert reopened.snapshot(("vault-a",)) == snapshot_a
    assert reopened.snapshot(("vault-a", "vault-b")) == disagreeing


def test_snapshot_vector_bytes_are_scoped_to_requested_vaults(store: SQLiteStore) -> None:
    """Including vectors outside the scope would overstate a vault's index footprint."""
    source_a = discovered_source("alpha", vault_id="vault-a", path="a.md")
    source_b = discovered_source("beta", vault_id="vault-b", path="b.md")
    store.replace_source(
        revision(
            source_a,
            chunks=(chunk(source_a, chunk_id="a"),),
            vectors=(vector("a", (1.0, 2.0)),),
        )
    )
    store.replace_source(
        revision(
            source_b,
            chunks=(chunk(source_b, chunk_id="b"),),
            vectors=(vector("b", (3.0, 4.0, 5.0), observed_fingerprint="observed-v2"),),
        )
    )

    assert store.snapshot(("empty-vault",)).vector_bytes == 0
    assert store.snapshot(("vault-a",)).vector_bytes == 8
    assert store.snapshot(("vault-a", "vault-b")).vector_bytes == 20


def test_snapshot_missing_schema_state_raises_typed_error(store: SQLiteStore) -> None:
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM index_state")
    with pytest.raises(StorageError, match="schema state is missing"):
        store.snapshot(("vault-a",))


def test_chunks_by_ids_requires_and_applies_vault_allowlist(store: SQLiteStore) -> None:
    source_a = discovered_source("alpha secret", vault_id="vault-a", path="a.md")
    source_b = discovered_source("beta secret", vault_id="vault-b", path="b.md")
    chunk_a = chunk(source_a, chunk_id="known-a")
    chunk_b = chunk(source_b, chunk_id="known-b")
    store.replace_source(revision(source_a, chunks=(chunk_a,)))
    store.replace_source(revision(source_b, chunks=(chunk_b,)))

    hydrated = store.chunks_by_ids(
        (chunk_a.id, chunk_b.id),
        vault_ids=("vault-a",),
    )
    assert set(hydrated) == {chunk_a.id}
    assert hydrated[chunk_a.id].vault_id == "vault-a"
    with pytest.raises(ValueError):
        store.chunks_by_ids((chunk_a.id,), vault_ids=())


def test_metadata_and_fingerprints_are_canonical_and_durable(
    database: Path, store: SQLiteStore
) -> None:
    source = discovered_source()
    source_chunk = chunk(source, metadata={"z": 1, "a": ["x", 2]})
    store.replace_source(revision(source, chunks=(source_chunk,)))

    connection = sqlite3.connect(database)
    metadata_json = connection.execute("SELECT metadata_json FROM chunks").fetchone()[0]
    connection.close()
    assert metadata_json == json.dumps(
        {"a": ["x", 2], "z": 1}, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )

    reopened = SQLiteStore(database)
    reopened.initialize()
    snapshot = reopened.snapshot(("vault-a",))
    assert snapshot.schema_version == 1
    assert snapshot.manifest_fingerprint == "manifest-v1"
    assert snapshot.parser_fingerprint == "parser-v1"
    assert snapshot.chunker_fingerprint == "chunker-v1"
    assert snapshot.embedding_config_fingerprint == "config-v1"
    assert snapshot.ready_count == 1


def test_suppress_source_atomically_records_failing_content_hash(
    store: SQLiteStore,
) -> None:
    source = discovered_source("previously good")
    store.replace_source(revision(source))

    store.suppress_source("vault-a", source.relative_path, "sha256:failing-bytes", "parse failure")

    assert store.source_hashes(("vault-a",)) == {
        ("vault-a", source.relative_path): "sha256:failing-bytes"
    }
    with sqlite3.connect(store.path) as connection:
        row = connection.execute(
            "SELECT parse_state, diagnostic FROM sources WHERE vault_id = ? AND relative_path = ?",
            ("vault-a", source.relative_path),
        ).fetchone()
    assert row == ("suppressed", "parse failure")


def test_requeue_disabled_chunks_is_vault_scoped_and_removes_stale_vectors(
    store: SQLiteStore,
) -> None:
    source_a = discovered_source("alpha", vault_id="vault-a", path="a.md")
    source_b = discovered_source("beta", vault_id="vault-b", path="b.md")
    chunk_a = chunk(source_a, chunk_id="a", ordinal=0, text="alpha first")
    chunk_a_second = chunk(source_a, chunk_id="a-second", ordinal=1, text="alpha second")
    chunk_b = chunk(source_b, chunk_id="b")
    store.replace_source(revision(source_a, chunks=(chunk_a, chunk_a_second), vectors=()))
    store.replace_source(revision(source_b, chunks=(chunk_b,), vectors=()))
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO chunk_vectors("
            "chunk_id, dimensions, embedding_config_fingerprint, "
            "observed_fingerprint, vector) VALUES (?, ?, ?, ?, ?)",
            (
                "a",
                2,
                "config-v1",
                observed_fingerprint("config-v1", 2),
                np.array([1, 0], dtype="<f4").tobytes(),
            ),
        )
    attempted_at = datetime(2026, 8, 8, tzinfo=UTC)

    requeued = store.requeue_disabled_chunks((("vault-a", "a.md"),), attempted_at)

    assert requeued == (RequeueResult("vault-a", 2),)
    pending = store.pending_chunks(("vault-a",))
    assert [item.id for item in pending] == ["a", "a-second"]
    assert all(
        item.pending is not None
        and item.pending.category == "policy_transition"
        and item.pending.message == "semantic embedding re-enabled"
        and item.pending.attempted_at == attempted_at
        for item in pending
    )
    stored_b = store.chunks_by_ids(("b",), vault_ids=("vault-b",))["b"]
    assert stored_b.embedding_state.value == "disabled"
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM chunk_vectors WHERE chunk_id = 'a'"
            ).fetchone()[0]
            == 0
        )


def test_requeue_disabled_chunks_returns_deterministic_per_vault_counts(
    store: SQLiteStore,
) -> None:
    source_a = discovered_source("alpha", vault_id="vault-a", path="a.md")
    source_b = discovered_source("beta", vault_id="vault-b", path="b.md")
    store.replace_source(revision(source_a, chunks=(chunk(source_a, chunk_id="a"),), vectors=()))
    store.replace_source(revision(source_b, chunks=(chunk(source_b, chunk_id="b"),), vectors=()))

    result = store.requeue_disabled_chunks(
        (("vault-b", "b.md"), ("vault-a", "a.md")),
        datetime(2026, 8, 8, tzinfo=UTC),
    )

    assert result == (RequeueResult("vault-a", 1), RequeueResult("vault-b", 1))


def test_requeue_disabled_chunks_rolls_back_every_state_on_failure(
    store: SQLiteStore,
) -> None:
    source = discovered_source("two disabled")
    first = chunk(source, chunk_id="first", ordinal=0, text="first")
    second = chunk(source, chunk_id="second", ordinal=1, text="second")
    store.replace_source(revision(source, chunks=(first, second), vectors=()))
    stale_vector = np.array([1, 0], dtype="<f4").tobytes()
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO chunk_vectors("
            "chunk_id, dimensions, embedding_config_fingerprint, "
            "observed_fingerprint, vector) VALUES (?, ?, ?, ?, ?)",
            (
                "first",
                2,
                "config-v1",
                observed_fingerprint("config-v1", 2),
                stale_vector,
            ),
        )
        connection.execute(
            """
            CREATE TRIGGER fail_second_requeue
            BEFORE UPDATE OF embedding_state ON chunks
            WHEN NEW.chunk_id = 'second' AND NEW.embedding_state = 'pending'
            BEGIN
                SELECT RAISE(ABORT, 'synthetic requeue failure');
            END
            """
        )

    with pytest.raises(StorageError):
        store.requeue_disabled_chunks(
            (("vault-a", source.relative_path),),
            datetime(2026, 8, 8, tzinfo=UTC),
        )

    stored = store.chunks_by_ids(("first", "second"), vault_ids=("vault-a",))
    assert {item.embedding_state.value for item in stored.values()} == {"disabled"}
    assert store.pending_chunks(("vault-a",)) == ()
    with sqlite3.connect(store.path) as connection:
        assert (
            connection.execute(
                "SELECT vector FROM chunk_vectors WHERE chunk_id = 'first'"
            ).fetchone()[0]
            == stale_vector
        )


def test_replace_source_can_defer_vault_fingerprint_until_finalization(
    store: SQLiteStore,
) -> None:
    initial = revision()
    store.replace_source(initial)
    changed_source = discovered_source("changed bytes")
    changed_chunk = chunk(changed_source)
    changed = replace(
        revision(changed_source, chunks=(changed_chunk,)),
        manifest_fingerprint="manifest-v2",
    )

    store.replace_source(replace(changed, update_vault_fingerprint=False))

    snapshot = store.snapshot(("vault-a",))
    assert snapshot.manifest_fingerprint == "manifest-v1"
    assert store.source_hashes(("vault-a",)) == {
        ("vault-a", changed_source.relative_path): changed_source.content_hash
    }

    store.replace_source(changed)
    assert store.snapshot(("vault-a",)).manifest_fingerprint == "manifest-v2"


def test_pending_retry_promotes_success_and_refreshes_failure_atomically(
    store: SQLiteStore,
) -> None:
    source = discovered_source("pending chunks")
    first = chunk(source, chunk_id="first", ordinal=0, text="first")
    second = chunk(source, chunk_id="second", ordinal=1, text="second")
    attempted_at = datetime(2026, 8, 6, tzinfo=UTC)
    store.replace_source(
        revision(
            source,
            chunks=(first, second),
            vectors=(),
            pending=(
                PendingEmbedding(first.id, "transport", "old first", attempted_at),
                PendingEmbedding(second.id, "transport", "old second", attempted_at),
            ),
        )
    )
    refreshed_at = datetime(2026, 8, 7, tzinfo=UTC)

    store.update_pending_embeddings(
        "vault-a",
        "config-v1",
        (
            PendingEmbeddingUpdate(
                first.id,
                first.content_hash,
                vector=vector(
                    first.id,
                    observed_fingerprint=observed_fingerprint("config-v1", 2),
                ),
            ),
            PendingEmbeddingUpdate(
                second.id,
                second.content_hash,
                failure=PendingEmbedding(second.id, "http", "still unavailable", refreshed_at),
            ),
        ),
    )

    assert (
        store.chunks_by_ids((first.id,), vault_ids=("vault-a",))[first.id].embedding_state.value
        == "ready"
    )
    pending = store.pending_chunks(("vault-a",))
    assert [item.id for item in pending] == [second.id]
    assert pending[0].pending == PendingEmbedding(
        second.id, "http", "still unavailable", refreshed_at
    )


def test_pending_retry_rejects_stale_content_without_partial_promotion(
    store: SQLiteStore,
) -> None:
    source = discovered_source("pending chunks")
    first = chunk(source, chunk_id="first", ordinal=0, text="first")
    second = chunk(source, chunk_id="second", ordinal=1, text="second")
    attempted_at = datetime(2026, 8, 6, tzinfo=UTC)
    store.replace_source(
        revision(
            source,
            chunks=(first, second),
            vectors=(),
            pending=(
                PendingEmbedding(first.id, "transport", "old first", attempted_at),
                PendingEmbedding(second.id, "transport", "old second", attempted_at),
            ),
        )
    )

    with pytest.raises(StorageError):
        store.update_pending_embeddings(
            "vault-a",
            "config-v1",
            (
                PendingEmbeddingUpdate(
                    first.id,
                    first.content_hash,
                    vector=vector(
                        first.id,
                        observed_fingerprint=observed_fingerprint("config-v1", 2),
                    ),
                ),
                PendingEmbeddingUpdate(
                    second.id,
                    "stale-hash",
                    vector=vector(
                        second.id,
                        observed_fingerprint=observed_fingerprint("config-v1", 2),
                    ),
                ),
            ),
        )

    assert [item.id for item in store.pending_chunks(("vault-a",))] == ["first", "second"]


def test_empty_vault_fingerprint_is_recorded_and_survives_deletion(
    store: SQLiteStore,
) -> None:
    fingerprint = VaultFingerprint(
        "empty-vault", "manifest-empty", "parser-v1", "chunker-v1", "config-v1"
    )
    store.record_vault_fingerprint(fingerprint)
    snapshot = store.snapshot(("empty-vault",))
    assert snapshot.manifest_fingerprint == "manifest-empty"
    assert snapshot.source_count == 0
    assert snapshot.indexed_at is not None

    source = discovered_source(vault_id="empty-vault", path="temporary.md")
    source_chunk = chunk(source, chunk_id="temporary")
    store.replace_source(
        replace(
            revision(source, chunks=(source_chunk,), vectors=()),
            manifest_fingerprint="manifest-empty",
        )
    )
    store.delete_sources((("empty-vault", "temporary.md"),))

    assert store.snapshot(("empty-vault",)).manifest_fingerprint == "manifest-empty"


def test_case_only_replacement_is_one_atomic_source_revision(store: SQLiteStore) -> None:
    upper = discovered_source("old", path="Notes/A.md", folded_path="notes/a.md")
    lower = discovered_source("new", path="notes/a.md", folded_path="notes/a.md")
    store.replace_source(revision(upper, chunks=(chunk(upper, chunk_id="upper"),)))

    store.replace_source(
        revision(lower, chunks=(chunk(lower, chunk_id="lower"),)),
        replacing_path="Notes/A.md",
    )

    assert store.source_hashes(("vault-a",)) == {("vault-a", "notes/a.md"): lower.content_hash}
    assert store.chunks_by_ids(("upper",), vault_ids=("vault-a",)) == {}


def test_prepare_for_atomic_replace_removes_wal_sidecars(
    database: Path, store: SQLiteStore
) -> None:
    store.replace_source(revision())

    store.prepare_for_atomic_replace()

    assert not Path(f"{database}-wal").exists()
    assert not Path(f"{database}-shm").exists()
    assert SQLiteStore(database).source_hashes(("vault-a",))


def test_public_query_api_has_no_raw_sql_arguments() -> None:
    for method in (
        SQLiteStore.source_hashes,
        SQLiteStore.lexical_search,
        SQLiteStore.load_vectors,
        SQLiteStore.chunks_by_ids,
        SQLiteStore.pending_chunks,
        SQLiteStore.snapshot,
    ):
        names = set(inspect.signature(method).parameters)
        assert names.isdisjoint({"sql", "where", "order_by", "join"})


def test_requests_and_filters_validate_untrusted_values() -> None:
    with pytest.raises(ValueError):
        LexicalRequest((), '"x"', StorageFilters(), 1)
    with pytest.raises(ValueError):
        VectorLoadRequest(("vault-a",), "observed", StorageFilters(), 0)
    with pytest.raises(ValueError):
        StorageFilters(path_prefix="../escape")
    with pytest.raises(ValueError):
        StorageFilters(frontmatter={"bad": float("nan")})


def test_path_prefix_is_a_posix_component_prefix(store: SQLiteStore) -> None:
    project = discovered_source("shared", path="projects/example.md")
    projection = discovered_source("shared", path="projects-old/example.md")
    store.replace_source(revision(project, chunks=(chunk(project, chunk_id="project"),)))
    store.replace_source(revision(projection, chunks=(chunk(projection, chunk_id="projection"),)))

    result = store.lexical_search(
        LexicalRequest(
            ("vault-a",),
            '"shared"',
            StorageFilters(path_prefix="projects"),
            10,
        )
    )

    assert [candidate.chunk_id for candidate in result] == ["project"]


def test_explicit_json_null_filter_requires_key_presence(store: SQLiteStore) -> None:
    present = discovered_source("shared", path="present.md")
    missing = discovered_source("shared", path="missing.md")
    store.replace_source(
        revision(present, chunks=(chunk(present, chunk_id="present", metadata={"owner": None}),))
    )
    store.replace_source(
        revision(missing, chunks=(chunk(missing, chunk_id="missing", metadata={}),))
    )

    result = store.lexical_search(
        LexicalRequest(
            ("vault-a",),
            '"shared"',
            StorageFilters(frontmatter={"owner": None}),
            10,
        )
    )

    assert [candidate.chunk_id for candidate in result] == ["present"]


def test_chunks_by_ids_batches_large_hydration_and_remains_vault_scoped(
    store: SQLiteStore,
) -> None:
    source = discovered_source("body")
    chunks = tuple(
        chunk(source, chunk_id=f"chunk-{index}", ordinal=index) for index in range(1_100)
    )
    store.replace_source(revision(source, chunks=chunks, vectors=()))

    result = store.chunks_by_ids(tuple(record.id for record in chunks), vault_ids=("vault-a",))

    assert len(result) == 1_100
    assert store.chunks_by_ids(("chunk-1",), vault_ids=("vault-b",)) == {}


def test_snapshot_indexed_at_uses_committed_vault_reconciliation_state(
    store: SQLiteStore,
) -> None:
    store.replace_source(revision())
    assert store.snapshot(("vault-a",)).indexed_at is None

    reconciliation = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
    later_source_change = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE vaults SET updated_at = ? WHERE vault_id = ?",
            (reconciliation.isoformat(), "vault-a"),
        )
        connection.execute(
            "UPDATE sources SET indexed_at = ?",
            (later_source_change.isoformat(),),
        )

    assert store.snapshot(("vault-a",)).indexed_at == reconciliation
    store.replace_source(revision(discovered_source("later source write")))
    assert store.snapshot(("vault-a",)).indexed_at == reconciliation


def test_snapshot_reconciliation_is_oldest_parsed_time_and_requires_every_vault(
    database: Path, store: SQLiteStore
) -> None:
    for vault_id in ("vault-a", "vault-b"):
        store.record_vault_fingerprint(
            VaultFingerprint(vault_id, "manifest", "parser", "chunker", "config")
        )
    older = datetime(2026, 8, 9, 23, 30, tzinfo=timezone(timedelta(hours=14)))
    newer = datetime(2026, 8, 9, 12, 0, tzinfo=timezone(timedelta(hours=-10)))
    assert older < newer
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE vaults SET updated_at = ? WHERE vault_id = 'vault-a'",
            (older.isoformat(),),
        )
        connection.execute(
            "UPDATE vaults SET updated_at = ? WHERE vault_id = 'vault-b'",
            (newer.isoformat(),),
        )

    assert store.snapshot(("vault-a", "vault-b")).indexed_at == older
    assert SQLiteStore(database).snapshot(("vault-a", "vault-b")).indexed_at == older
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE vaults SET updated_at = NULL WHERE vault_id = 'vault-b'")
    assert store.snapshot(("vault-a", "vault-b")).indexed_at is None
    assert store.snapshot(("vault-a", "missing")).indexed_at is None


def test_failed_final_reconciliation_keeps_previous_vault_timestamp(
    store: SQLiteStore,
) -> None:
    fingerprint = VaultFingerprint("vault-a", "manifest-v1", "parser-v1", "chunker-v1", "config-v1")
    store.record_vault_fingerprint(fingerprint)
    before = store.snapshot(("vault-a",)).indexed_at
    assert before is not None
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_reconciliation_time
            BEFORE UPDATE OF updated_at ON vaults
            BEGIN
                SELECT RAISE(ABORT, 'synthetic finalization failure');
            END
            """
        )

    with pytest.raises(StorageError):
        store.delete_sources((), fingerprints=(fingerprint,))

    assert store.snapshot(("vault-a",)).indexed_at == before


def test_initialize_rejects_schema_v1_vaults_without_reconciliation_timestamp(
    database: Path,
) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE vaults (
                vault_id TEXT PRIMARY KEY,
                manifest_fingerprint TEXT NOT NULL,
                parser_fingerprint TEXT NOT NULL,
                chunker_fingerprint TEXT NOT NULL,
                embedding_config_fingerprint TEXT NOT NULL
            ) STRICT
            """
        )
        connection.execute("PRAGMA user_version = 1")

    with pytest.raises(RebuildRequiredError) as error:
        SQLiteStore(database).initialize()
    assert error.value.details["missing_columns"] == ["updated_at"]


def test_vector_scope_and_source_provenance_are_filter_and_vault_scoped(
    store: SQLiteStore,
) -> None:
    source = discovered_source("body", path="projects/example.md")
    store.replace_source(revision(source))

    scope = store.vector_scope(("vault-a",), "observed-v1", StorageFilters(path_prefix="projects"))
    foreign = store.vector_scope(("vault-b",), "observed-v1", StorageFilters())
    provenance = store.source_provenance("vault-a", "projects/example.md")

    assert scope.count == 1
    assert scope.dimensions == (2,)
    assert foreign.count == 0
    assert provenance is not None
    assert provenance.source_hash == source.content_hash
    assert provenance.source_kind is SourceKind.MARKDOWN
    assert store.source_provenance("vault-b", "projects/example.md") is None
    assert store.snapshot(("vault-a",)).indexed_at is None
    store.delete_sources(
        (),
        fingerprints=(
            VaultFingerprint("vault-a", "manifest-v1", "parser-v1", "chunker-v1", "config-v1"),
        ),
    )
    assert store.snapshot(("vault-a",)).indexed_at is not None


def test_complete_vector_load_has_no_limit_and_applies_scope_filters(
    store: SQLiteStore,
) -> None:
    for index in range(25):
        source = discovered_source("body", path=f"projects/{index:02}.md")
        source_chunk = chunk(source, chunk_id=f"chunk-{index}")
        store.replace_source(
            revision(
                source,
                chunks=(source_chunk,),
                vectors=(vector(source_chunk.id, (float(index + 1), 1.0)),),
            )
        )
    excluded = discovered_source("body", path="other/excluded.md")
    excluded_chunk = chunk(excluded, chunk_id="excluded")
    store.replace_source(revision(excluded, chunks=(excluded_chunk,)))

    records = store.load_vectors_complete(
        ("vault-a",), "observed-v1", StorageFilters(path_prefix="projects")
    )

    assert len(records) == 25
    assert records[-1].chunk_id == "chunk-24"
    assert all(record.observed_fingerprint == "observed-v1" for record in records)


def test_complete_vector_load_includes_true_tail_after_early_concurrent_inserts(
    store: SQLiteStore,
) -> None:
    for path, chunk_id, values in (
        ("zz0.md", "old-worse", (0.0, 1.0)),
        ("zzz-best.md", "true-tail-best", (1.0, 0.0)),
    ):
        source = discovered_source("body", path=path)
        source_chunk = chunk(source, chunk_id=chunk_id)
        store.replace_source(
            revision(source, chunks=(source_chunk,), vectors=(vector(chunk_id, values),))
        )
    stale_scope = store.vector_scope(("vault-a",), "observed-v1", StorageFilters())
    assert stale_scope.count == 2

    writer = SQLiteStore(store.path)
    for index in range(3):
        source = discovered_source("body", path=f"aa{index}.md")
        source_chunk = chunk(source, chunk_id=f"inserted-{index}")
        writer.replace_source(
            revision(
                source,
                chunks=(source_chunk,),
                vectors=(vector(source_chunk.id, (1.0, 1.0)),),
            )
        )

    stale_limited = store.load_vectors(
        VectorLoadRequest(("vault-a",), "observed-v1", StorageFilters(), stale_scope.count)
    )
    complete = store.load_vectors_complete(("vault-a",), "observed-v1", StorageFilters())

    assert "true-tail-best" not in {record.chunk_id for record in stale_limited}
    assert "true-tail-best" in {record.chunk_id for record in complete}
    assert len(complete) == 5


def test_dense_search_ranks_the_complete_scope_before_limiting(store: SQLiteStore) -> None:
    for path, chunk_id, values in (
        ("zz0.md", "old-worse", (0.0, 1.0)),
        ("zzz-best.md", "true-tail-best", (1.0, 0.0)),
        ("aa0.md", "stable-tie-first", (1.0, 1.0)),
        ("aa1.md", "stable-tie-second", (1.0, 1.0)),
    ):
        source = discovered_source("body", path=path)
        source_chunk = chunk(source, chunk_id=chunk_id)
        store.replace_source(
            revision(source, chunks=(source_chunk,), vectors=(vector(chunk_id, values),))
        )

    candidates = store.dense_search(
        DenseRequest(("vault-a",), "observed-v1", StorageFilters(), limit=3),
        np.asarray((1.0, 0.0), dtype=np.float32),
    )

    assert tuple(candidate.chunk_id for candidate in candidates) == (
        "true-tail-best",
        "stable-tie-first",
        "stable-tie-second",
    )
    assert candidates[0].score == pytest.approx(1.0)


def test_identifier_candidate_scan_is_lightweight_filter_and_vault_scoped(
    store: SQLiteStore,
) -> None:
    included = discovered_source("body secret", path="projects/included.md")
    excluded = discovered_source("body", path="other/excluded.md")
    foreign = discovered_source("body", vault_id="vault-b", path="projects/foreign.md")
    store.replace_source(
        revision(
            included,
            chunks=(
                chunk(
                    included,
                    chunk_id="included",
                    metadata={"ticket": "DIS-123", "owner": "active"},
                ),
            ),
        )
    )
    store.replace_source(
        revision(
            excluded,
            chunks=(chunk(excluded, chunk_id="excluded", metadata={"owner": "active"}),),
        )
    )
    store.replace_source(
        revision(
            foreign,
            chunks=(chunk(foreign, chunk_id="foreign", metadata={"owner": "active"}),),
        )
    )

    candidates = store.identifier_candidates(
        ("vault-a",),
        StorageFilters(path_prefix="projects", frontmatter={"owner": "active"}),
    )

    assert [candidate.chunk_id for candidate in candidates] == ["included"]
    assert candidates[0].title == "Example"
    assert candidates[0].metadata["ticket"] == "DIS-123"
    assert not hasattr(candidates[0], "text")
