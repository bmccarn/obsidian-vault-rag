from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np  # pyright: ignore[reportMissingImports]
from backend_fixture import StorageBackend  # pyright: ignore[reportMissingImports]

from vault_rag.config import (
    EgressPolicy,
    EmbeddingConfig,
    ResolvedProfile,
    ResolvedVault,
    VaultManifest,
)
from vault_rag.embedding import EmbeddingBatch, EmbeddingFailure
from vault_rag.indexing import Indexer, IndexReport
from vault_rag.storage.postgres import PostgresStore, PostgresWorkerLease
from vault_rag.storage.postgres.index_store import PostgresIndexStore


class CountingEmbedder:
    def __init__(self) -> None:
        self.requests: list[tuple[str, ...]] = []

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.requests.append(tuple(texts))
        return EmbeddingBatch(tuple(np.asarray([1.0, 0.0], dtype=np.float32) for _ in texts), (), 2)


class FailingEmbedder(CountingEmbedder):
    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.requests.append(tuple(texts))
        return EmbeddingBatch(
            (),
            tuple(
                EmbeddingFailure(index, "temporary", "provider unavailable")
                for index in range(len(texts))
            ),
            None,
        )


class Counter:
    def count(self, text: str) -> int:
        return len(text.split())

    def take_tail(self, text: str, tokens: int) -> str:
        return text


def _profile(root: Path, *, endpoint: str = "local") -> tuple[ResolvedProfile, EmbeddingConfig]:
    manifest = VaultManifest.model_validate(
        {
            "schema_version": 1,
            "id": "vault-a",
            "egress_policy": "remote-allowed",
            "include": ["**/*.md"],
        }
    )
    profile = ResolvedProfile(
        name="reuse",
        vaults=(ResolvedVault(root=root, manifest=manifest),),
        embedding_model="embed",
        api_key_env=None,
        effective_policy=EgressPolicy.REMOTE_ALLOWED,
        semantic_enabled=True,
        semantic_disabled_reason=None,
    )
    config = EmbeddingConfig.model_validate(
        {
            "base_url": (
                "https://embedding.example.test/v1"
                if endpoint == "remote"
                else "http://127.0.0.1:4000/v1"
            ),
            "model_env": "MODEL",
            "endpoint_class": endpoint,
            "target_min_tokens": 10,
            "target_max_tokens": 30,
            "overlap_tokens": 2,
            "max_input_tokens": 100,
        }
    )
    return profile, config


def _run(
    backend: StorageBackend,
    profile: ResolvedProfile,
    config: EmbeddingConfig,
    embedder: CountingEmbedder,
    generation: int,
    *,
    rebuild: bool = False,
) -> IndexReport:
    if not backend.postgres:
        return Indexer(profile, config, embedder, backend.query, counter=Counter()).run(
            rebuild=rebuild
        )
    build = cast(
        PostgresIndexStore,
        backend.query.begin_revision("vault-a", "refs/heads/main", f"{generation:040x}"),
    )
    report = Indexer(profile, config, embedder, build, counter=Counter()).run(rebuild=rebuild)
    lease = PostgresWorkerLease(build._pool, "sync:vault-a")
    assert lease.acquire()
    try:
        lease.promote(build, lexical_complete=True, fully_reconciled=True)
    finally:
        lease.release()
    return report


def _durable_embeddings(backend: StorageBackend) -> int:
    if not backend.postgres:
        return 0
    store = cast(PostgresStore, backend.query)
    with store._pool.connection() as connection:
        row = connection.execute("SELECT count(*) AS count FROM vault_rag.embeddings").fetchone()
    assert row is not None
    return cast(int, row["count"])


def test_duplicate_content_is_embedded_once_and_reused_across_renames(
    backend: StorageBackend, tmp_path: Path
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    content = "# Same\n\nidentical embedding body\n"
    (root / "first.md").write_text(content, encoding="utf-8")
    (root / "duplicate.md").write_text(content, encoding="utf-8")
    profile, config = _profile(root)
    embedder = CountingEmbedder()

    first = _run(backend, profile, config, embedder, 1)

    assert first.embedding_requests == 1
    assert first.embedding_failures == 0
    assert first.ready_chunks == 2
    assert [len(request) for request in embedder.requests] == [1]
    assert _durable_embeddings(backend) == (1 if backend.postgres else 0)

    (root / "first.md").rename(root / "renamed.md")
    renamed = _run(backend, profile, config, embedder, 2)

    assert renamed.embedding_requests == 0
    assert renamed.embedding_failures == 0
    assert renamed.ready_chunks == 2
    assert [len(request) for request in embedder.requests] == [1]
    assert _durable_embeddings(backend) == (1 if backend.postgres else 0)


def test_duplicate_pending_chunks_retry_once_and_account_each_chunk(
    backend: StorageBackend, tmp_path: Path
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    content = "# Same\n\nidentical embedding body\n"
    (root / "first.md").write_text(content, encoding="utf-8")
    (root / "duplicate.md").write_text(content, encoding="utf-8")
    profile, config = _profile(root)
    failing = FailingEmbedder()

    failed = _run(backend, profile, config, failing, 1)

    assert failed.embedding_requests == 1
    assert failed.embedding_failures == 2
    assert failed.pending_chunks == 2
    assert [len(request) for request in failing.requests] == [1]

    recovered_embedder = CountingEmbedder()
    recovered = _run(backend, profile, config, recovered_embedder, 2)

    assert recovered.embedding_requests == 1
    assert recovered.embedding_failures == 0
    assert recovered.pending_chunks == 0
    assert recovered.ready_chunks == 2
    assert [len(request) for request in recovered_embedder.requests] == [1]


def test_embedding_configuration_mismatch_misses_reuse(
    backend: StorageBackend, tmp_path: Path
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    content = "# Same\n\nidentical embedding body\n"
    (root / "first.md").write_text(content, encoding="utf-8")
    (root / "duplicate.md").write_text(content, encoding="utf-8")
    profile, config = _profile(root)
    embedder = CountingEmbedder()
    first = _run(backend, profile, config, embedder, 1)
    assert first.embedding_requests == 1

    changed_profile, changed_config = _profile(root, endpoint="remote")
    changed = _run(backend, changed_profile, changed_config, embedder, 2, rebuild=True)

    assert changed.embedding_requests == 1
    assert [len(request) for request in embedder.requests] == [1, 1]
    assert _durable_embeddings(backend) == (2 if backend.postgres else 0)
