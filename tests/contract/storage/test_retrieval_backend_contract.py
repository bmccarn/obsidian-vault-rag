from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]
from backend_fixture import StorageBackend, source_revision  # pyright: ignore[reportMissingImports]

from vault_rag.config import EgressPolicy, ResolvedProfile, ResolvedVault, VaultManifest
from vault_rag.errors import StaleSourceError
from vault_rag.retrieval import ReadRequest, RetrievalService, SearchMode, SearchRequest
from vault_rag.storage.ports import QueryStore


class NoEmbedding:
    def embed_query(self, text: str, expected_dimensions: int) -> np.ndarray:
        raise AssertionError("read contracts must not invoke embeddings")


class ReadOnlyQuerySpy:
    """Permit only query methods; any accidental retrieval write fails the test."""

    _WRITERS = frozenset({"begin_revision", "replace_source", "reset", "promote", "fail", "mutate"})

    def __init__(self, store: Any) -> None:
        self._store = store

    def __getattr__(self, name: str) -> Any:
        if name in self._WRITERS:
            raise AssertionError(f"read path attempted writer {name}")
        return getattr(self._store, name)


def _profile(root: Path) -> ResolvedProfile:
    manifest = VaultManifest.model_validate(
        {
            "schema_version": 1,
            "id": "vault-a",
            "egress_policy": "remote-allowed",
            "include": ["**/*.md"],
            "metadata": {"frontmatter_fields": ["status"]},
        }
    )
    return ResolvedProfile(
        name="contract",
        vaults=(ResolvedVault(root=root, manifest=manifest),),
        embedding_model="test",
        api_key_env=None,
        effective_policy=EgressPolicy.REMOTE_ALLOWED,
        semantic_enabled=True,
        semantic_disabled_reason=None,
    )


def _service(backend: StorageBackend, root: Path) -> RetrievalService:
    return RetrievalService(
        _profile(root),
        cast(QueryStore, ReadOnlyQuerySpy(backend.query)),
        NoEmbedding(),
        "observed-v1",
    )


def test_backend_read_contract_hashes_before_returning_content(
    backend: StorageBackend, tmp_path: Path
) -> None:
    """Returning an excerpt before either hash check would expose stale content."""
    root = tmp_path / "vault"
    root.mkdir()
    path = root / "notes" / "read.md"
    path.parent.mkdir()
    text = "retrieval contract\n"
    path.write_text(text, encoding="utf-8")
    backend.mutate(
        lambda store: store.replace_source(
            source_revision("notes/read.md", text, "read", root=root)
        )
    )
    service = _service(backend, root)
    matching_hash = "sha256:296db4e65875b6324423c9ea399cd6ed0d852cadb3cd6e7a1adac830f7c35106"

    assert service.read(ReadRequest("notes/read.md")).text == text
    assert (
        service.read(ReadRequest("notes/read.md", expected_source_hash=matching_hash)).text == text
    )
    with pytest.raises(StaleSourceError):
        service.read(ReadRequest("notes/read.md", expected_source_hash="sha256:" + "0" * 64))

    if not backend.postgres:
        path.write_text("changed source bytes\n", encoding="utf-8")
        with pytest.raises(StaleSourceError):
            service.read(ReadRequest("notes/read.md"))


def test_backend_read_only_search_read_status_and_citations_are_identical(
    backend: StorageBackend, tmp_path: Path
) -> None:
    """A query-side write or divergent citation identity is observable API corruption."""
    root = tmp_path / "vault"
    root.mkdir()
    path = root / "notes" / "citation.md"
    path.parent.mkdir()
    text = "citation searchable\n"
    path.write_text(text, encoding="utf-8")
    backend.mutate(
        lambda store: store.replace_source(
            source_revision("notes/citation.md", text, "citation", root=root)
        )
    )
    service = _service(backend, root)

    hit = service.search(SearchRequest("citation", mode=SearchMode.LEXICAL)).hits[0]
    read = service.read(ReadRequest("notes/citation.md"))
    snapshot = ReadOnlyQuerySpy(backend.query).snapshot(("vault-a",))

    assert hit.ref.citation == read.ref.citation
    assert (
        hit.ref.vault_id,
        hit.ref.path,
        hit.ref.heading,
        hit.ref.lines,
        hit.ref.source_hash,
    ) == ("vault-a", "notes/citation.md", ("Heading",), read.ref.lines, read.ref.source_hash)
    assert snapshot.source_count == 1
