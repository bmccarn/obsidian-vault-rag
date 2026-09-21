from __future__ import annotations

import pytest  # pyright: ignore[reportMissingImports]
from backend_fixture import StorageBackend, source_revision

from vault_rag.errors import StorageError
from vault_rag.storage import LexicalRequest, StorageFilters


def test_backend_replaces_suppresses_and_deletes_active_sources(backend: StorageBackend) -> None:
    """A stale source must never survive a shared active-state transition."""
    backend.mutate(
        lambda store: store.replace_source(source_revision("notes/a.md", "old body", "old"))
    )
    assert backend.query.chunks_by_ids(("old",), vault_ids=("vault-a",))["old"].text == "old body"

    backend.mutate(
        lambda store: store.replace_source(source_revision("notes/a.md", "replacement body", "new"))
    )
    replacement = backend.query.chunks_by_ids(("old", "new"), vault_ids=("vault-a",))
    assert tuple(replacement) == ("new",)
    assert replacement["new"].text == "replacement body"

    backend.mutate(
        lambda store: store.suppress_source(
            "vault-a", "notes/a.md", "sha256:" + "f" * 64, "parser failure"
        )
    )
    assert backend.query.source_provenance("vault-a", "notes/a.md") is None
    assert (
        backend.query.lexical_search(
            LexicalRequest(("vault-a",), "replacement", StorageFilters(), limit=5)
        )
        == ()
    )

    backend.mutate(lambda store: store.delete_sources((("vault-a", "notes/a.md"),)))
    assert backend.query.snapshot(("vault-a",)).source_count == 0


def test_backend_rejects_folded_path_collisions(backend: StorageBackend) -> None:
    """Case-folded identity is durable, rather than an accidental filesystem detail."""
    backend.mutate(
        lambda store: store.replace_source(source_revision("Notes/A.md", "first", "upper"))
    )

    with pytest.raises(StorageError):
        backend.mutate(
            lambda store: store.replace_source(source_revision("notes/a.md", "second", "lower"))
        )


def test_backend_snapshot_and_vector_scope_follow_active_fingerprint(
    backend: StorageBackend,
) -> None:
    """A wrong fingerprint must not expose active vectors from another configuration."""
    backend.mutate(
        lambda store: store.replace_source(
            source_revision("notes/vector.md", "vector body", "vector")
        )
    )

    snapshot = backend.query.snapshot(("vault-a",))
    assert (
        snapshot.source_count,
        snapshot.chunk_count,
        snapshot.ready_count,
        snapshot.pending_count,
    ) == (1, 1, 1, 0)
    assert backend.query.vector_scope(("vault-a",), "observed-v1", StorageFilters()).dimensions == (
        2,
    )
    assert backend.query.vector_scope(("vault-a",), "different", StorageFilters()).count == 0
