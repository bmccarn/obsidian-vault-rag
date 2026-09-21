from __future__ import annotations

from datetime import UTC, datetime

import numpy as np  # pyright: ignore[reportMissingImports]
from backend_fixture import StorageBackend, source_revision

from vault_rag.domain import JsonValue
from vault_rag.storage import DenseRequest, LexicalRequest, StorageFilters


def test_backend_filters_before_limit_with_exact_typed_frontmatter(backend: StorageBackend) -> None:
    """A containment predicate or a post-limit filter returns the wrong first candidate."""
    records = (
        source_revision(
            "a-array-superset.md", "common query", "array-superset", metadata={"value": ["a", "b"]}
        ),
        source_revision(
            "b-array-exact.md", "common query", "array-exact", metadata={"value": ["a"]}
        ),
        source_revision(
            "c-map-superset.md",
            "common query",
            "map-superset",
            metadata={"value": {"child": 1, "extra": True}},
        ),
        source_revision(
            "d-map-exact.md", "common query", "map-exact", metadata={"value": {"child": 1}}
        ),
        source_revision("e-null.md", "common query", "null", metadata={"value": None}),
        source_revision("f-bool.md", "common query", "bool", metadata={"value": True}),
        source_revision("g-number.md", "common query", "number", metadata={"value": 2}),
        source_revision("h-string.md", "common query", "string", metadata={"value": "two"}),
    )

    def replace_all(store: object) -> None:
        for record in records:
            store.replace_source(record)  # type: ignore[union-attr]

    backend.mutate(replace_all)

    cases: tuple[tuple[dict[str, JsonValue], str], ...] = (
        ({"value": ["a"]}, "array-exact"),
        ({"value": {"child": 1}}, "map-exact"),
        ({"value": None}, "null"),
        ({"value": True}, "bool"),
        ({"value": 2}, "number"),
        ({"value": "two"}, "string"),
    )
    for frontmatter, expected in cases:
        result = backend.query.lexical_search(
            LexicalRequest(("vault-a",), "common", StorageFilters(frontmatter=frontmatter), limit=1)
        )
        assert [candidate.chunk_id for candidate in result] == [expected]


def test_backend_identifier_lexical_dense_and_provenance_queries_share_active_state(
    backend: StorageBackend,
) -> None:
    """Candidate generators and source reads must refer to the same active revision."""
    backend.mutate(
        lambda store: store.replace_source(
            source_revision(
                "projects/alpha.md",
                "alpha beta searchable",
                "alpha",
                metadata={"jira": ["VR-123"], "status": "active"},
            )
        )
    )

    filters = StorageFilters(frontmatter={"status": "active"})
    identifiers = backend.query.identifier_candidates(("vault-a",), filters)
    lexical = backend.query.lexical_search(LexicalRequest(("vault-a",), "alpha", filters, limit=5))
    dense = backend.query.dense_search(
        DenseRequest(("vault-a",), "observed-v1", filters, limit=5),
        np.asarray([3.0, 4.0], dtype=np.float32),
    )
    provenance = backend.query.source_provenance("vault-a", "projects/alpha.md")

    assert [item.chunk_id for item in identifiers] == ["alpha"]
    assert [item.chunk_id for item in lexical] == ["alpha"]
    assert [item.chunk_id for item in dense] == ["alpha"]
    assert provenance is not None
    assert provenance.source_hash == (
        "sha256:2f0fd7b7436bbbe550c1c5a751fab6825847985f8a8e599c8b297763b23987ad"
    )
    assert provenance.byte_length in {None, len(b"alpha beta searchable")}


def test_backend_pending_disabled_and_requeued_embedding_states_are_observable(
    backend: StorageBackend,
) -> None:
    """A backend that drops a transition changes coverage and retry behavior."""
    backend.mutate(
        lambda store: store.replace_source(
            source_revision("notes/pending.md", "pending", "pending", vector=None, pending=True)
        )
    )
    assert [item.id for item in backend.query.pending_chunks(("vault-a",))] == ["pending"]

    backend.mutate(
        lambda store: store.replace_source(
            source_revision("notes/disabled.md", "disabled", "disabled", vector=None)
        )
    )

    def requeue_disabled(store: object) -> None:
        store.requeue_disabled_chunks(  # type: ignore[union-attr]
            (("vault-a", "notes/disabled.md"),), datetime(2026, 8, 8, tzinfo=UTC)
        )

    backend.mutate(requeue_disabled)
    assert [item.id for item in backend.query.pending_chunks(("vault-a",))] == [
        "disabled",
        "pending",
    ]
