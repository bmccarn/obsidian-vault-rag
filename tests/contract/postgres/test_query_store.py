from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.domain import JsonValue, SourceKind
from vault_rag.storage import DenseRequest, LexicalRequest, StorageFilters
from vault_rag.storage.ports import RevisionBuildStore
from vault_rag.storage.postgres import (
    PostgresMigrator,
    PostgresPool,
    PostgresStore,
    PostgresWorkerLease,
)
from vault_rag.storage.postgres.index_store import PostgresIndexStore

from .helpers import revision


def _promote(build: RevisionBuildStore) -> None:
    native = cast(PostgresIndexStore, build)
    lease = PostgresWorkerLease(native._pool, f"sync:{native._vault_id}")
    assert lease.acquire() is True
    try:
        lease.promote(native, lexical_complete=True, fully_reconciled=True)
    finally:
        lease.release()


def _active_store(postgres_pool: PostgresPool) -> PostgresStore:
    PostgresMigrator(postgres_pool).apply()
    store = PostgresStore(postgres_pool)
    build = store.begin_revision("vault-a", "main", "a" * 40)
    build.replace_source(
        revision(
            "alpha beta searchable body",
            path="projects/alpha.md",
            metadata={
                "aliases": ["Roadmap"],
                "tags": ["search"],
                "status": "active",
                "priority": 2,
                "jira": ["VR-123"],
            },
        )
    )
    _promote(build)
    return store


def test_active_revision_supports_bounded_lexical_and_identifier_queries(
    postgres_pool: PostgresPool,
) -> None:
    store = _active_store(postgres_pool)
    filters = StorageFilters(
        vault_ids=("vault-a",),
        path_prefix="projects",
        source_kind=SourceKind.MARKDOWN,
        frontmatter={"status": "active", "priority": 2},
    )

    lexical = store.lexical_search(LexicalRequest(("vault-a",), "alpha & beta", filters, limit=5))
    identifiers = store.identifier_candidates(("vault-a",), filters)

    assert [(candidate.chunk_id, candidate.vault_id) for candidate in lexical] == [
        ("chunk-a", "vault-a")
    ]
    assert lexical[0].score > 0.0
    assert identifiers[0].metadata["jira"] == ["VR-123"]
    assert (
        store.lexical_search(
            LexicalRequest(
                ("vault-a",),
                "alpha",
                StorageFilters(path_prefix="other"),
                limit=5,
            )
        )
        == ()
    )


def test_active_revision_supports_exact_cosine_search_and_vector_scope(
    postgres_pool: PostgresPool,
) -> None:
    store = _active_store(postgres_pool)
    request = DenseRequest(
        ("vault-a",),
        "observed-v1",
        StorageFilters(frontmatter={"status": "active"}),
        limit=5,
    )

    dense = store.dense_search(request, np.asarray([3.0, 4.0], dtype=np.float32))
    scope = store.vector_scope(("vault-a",), "observed-v1", StorageFilters())

    assert [(candidate.chunk_id, candidate.score) for candidate in dense] == [
        ("chunk-a", pytest.approx(1.0))
    ]
    assert scope.count == 1
    assert scope.dimensions == (2,)


def test_active_revision_returns_hash_verified_source_provenance(
    postgres_pool: PostgresPool,
) -> None:
    store = _active_store(postgres_pool)

    chunks = store.chunks_by_ids(("chunk-a",), vault_ids=("vault-a",))
    provenance = store.source_provenance("vault-a", "projects/alpha.md")
    snapshot = store.snapshot(("vault-a",))

    assert chunks["chunk-a"].text == "alpha beta searchable body"
    assert provenance is not None
    assert provenance.content == b"alpha beta searchable body"
    assert provenance.byte_length == len(provenance.content)
    assert provenance.source_hash.startswith("sha256:")
    assert snapshot.source_count == 1
    assert snapshot.chunk_count == 1
    assert snapshot.ready_count == 1
    assert snapshot.pending_count == 0
    assert snapshot.vector_dimensions == (2,)


def test_building_revision_is_invisible_to_all_query_surfaces(
    postgres_pool: PostgresPool,
) -> None:
    store = _active_store(postgres_pool)
    build = store.begin_revision("vault-a", "main", "b" * 40)
    build.replace_source(revision("unpublished replacement", path="projects/alpha.md"))

    lexical = store.lexical_search(
        LexicalRequest(("vault-a",), "unpublished", StorageFilters(), limit=5)
    )
    active = store.source_provenance("vault-a", "projects/alpha.md")

    assert lexical == ()
    assert active is not None
    assert active.content == b"alpha beta searchable body"


def test_consistent_read_pins_one_active_revision_during_promotion(
    postgres_pool: PostgresPool,
) -> None:
    store = _active_store(postgres_pool)

    with store.consistent_read() as pinned:
        lexical = pinned.lexical_search(
            LexicalRequest(("vault-a",), "alpha", StorageFilters(), limit=5)
        )
        replacement = store.begin_revision("vault-a", "main", "b" * 40)
        replacement.replace_source(revision("replacement body", path="projects/alpha.md"))
        _promote(replacement)
        chunks = pinned.chunks_by_ids(
            tuple(candidate.chunk_id for candidate in lexical),
            vault_ids=("vault-a",),
        )

    assert chunks["chunk-a"].text == "alpha beta searchable body"
    current = store.chunks_by_ids(("chunk-a",), vault_ids=("vault-a",))
    assert current["chunk-a"].text == "replacement body"


def test_frontmatter_filters_require_exact_json_values_before_limit(
    postgres_pool: PostgresPool,
) -> None:
    store = PostgresStore(postgres_pool)
    PostgresMigrator(postgres_pool).apply()
    build = store.begin_revision("vault-a", "main", "c" * 40)
    records: tuple[tuple[str, str, dict[str, JsonValue]], ...] = (
        ("a-superset-array.md", "superset-array", {"value": ["a", "b"]}),
        ("b-exact-array.md", "exact-array", {"value": ["a"]}),
        ("c-superset-map.md", "superset-map", {"value": {"child": 1, "extra": True}}),
        ("d-exact-map.md", "exact-map", {"value": {"child": 1}}),
        ("e-string.md", "string", {"value": "one"}),
        ("f-number.md", "number", {"value": 2}),
        ("g-bool.md", "bool", {"value": True}),
        ("h-null.md", "null", {"value": None}),
    )
    for path, chunk_id, metadata in records:
        record = revision(
            "common searchable text", path=path, metadata=metadata, vector_values=None
        )
        build.replace_source(replace(record, chunks=(replace(record.chunks[0], id=chunk_id),)))
    _promote(build)

    cases: tuple[tuple[dict[str, JsonValue], str], ...] = (
        ({"value": ["a"]}, "exact-array"),
        ({"value": {"child": 1}}, "exact-map"),
        ({"value": "one"}, "string"),
        ({"value": 2}, "number"),
        ({"value": True}, "bool"),
        ({"value": None}, "null"),
    )
    for frontmatter, expected_chunk_id in cases:
        candidates = store.lexical_search(
            LexicalRequest(
                ("vault-a",),
                "common",
                StorageFilters(frontmatter=frontmatter),
                limit=1,
            )
        )

        assert [candidate.chunk_id for candidate in candidates] == [expected_chunk_id]


def test_query_surfaces_handle_empty_missing_and_invalid_inputs(
    postgres_pool: PostgresPool,
) -> None:
    store = _active_store(postgres_pool)
    request = DenseRequest(("vault-a",), "observed-v1", StorageFilters(), limit=5)

    assert (
        store.lexical_search(LexicalRequest(("vault-a",), "   ", StorageFilters(), limit=5)) == ()
    )
    assert store.chunks_by_ids((), vault_ids=("vault-a",)) == {}
    assert store.source_provenance("vault-a", "projects/missing.md") is None
    with pytest.raises(ValueError, match="non-empty"):
        store.vector_scope(("vault-a",), "", StorageFilters())
    with pytest.raises(ValueError, match="one-dimensional"):
        store.dense_search(request, np.asarray([[1.0, 2.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-zero"):
        store.dense_search(request, np.asarray([0.0, 0.0], dtype=np.float32))
