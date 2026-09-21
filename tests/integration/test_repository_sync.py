from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import httpx

from vault_rag.app import AppFactory
from vault_rag.retrieval import ReadRequest, SearchMode, SearchRequest
from vault_rag.service.config import RepositoryConfig, ServiceConfig, ServiceEmbeddingConfig
from vault_rag.service.coordinator import RepositorySyncCoordinator
from vault_rag.service.git import ManagedGit
from vault_rag.service.locking import LifecycleLock
from vault_rag.service.state import SyncStateStore


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(("git", *args), cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def _commit(worktree: Path, text: str) -> str:
    (worktree / "note.md").write_text(text, encoding="utf-8")
    _git(worktree, "add", "note.md")
    _git(worktree, "commit", "-m", "update note")
    _git(worktree, "push", "origin", "main")
    return _git(worktree, "rev-parse", "HEAD")


def _service_config(remote: Path) -> ServiceConfig:
    # Production config permits only HTTPS. Integration uses the validated model's
    # derived behavior with a synthetic local remote, exercising ManagedGit itself.
    repository = RepositoryConfig.model_construct(url=remote.as_uri(), ref="refs/heads/main")
    return ServiceConfig.model_construct(
        schema_version=1,
        sync_interval=__import__("datetime").timedelta(hours=1),
        repositories={"alpha": repository},
        profiles={},
        embedding=ServiceEmbeddingConfig(
            base_url="http://127.0.0.1:9000/v1", model_env="EMBED_MODEL", endpoint_class="local"
        ),
    )


def test_repository_sync_reconciles_real_git_sqlite_and_survives_restart(tmp_path: Path) -> None:
    """Breaking activation, state reuse, or lexical fallback makes this end-to-end path fail."""
    bare = tmp_path / "remote.git"
    worktree = tmp_path / "author"
    _git(tmp_path, "init", "--bare", str(bare))
    _git(tmp_path, "init", str(worktree))
    _git(worktree, "config", "user.name", "Test Author")
    _git(worktree, "config", "user.email", "author@example.test")
    (worktree / ".vault-rag.toml").write_text(
        'schema_version = 1\nid = "alpha"\negress_policy = "local-only"\ninclude = ["**/*.md"]\n',
        encoding="utf-8",
    )
    _git(worktree, "add", ".vault-rag.toml")
    _git(worktree, "commit", "-m", "add manifest")
    _git(worktree, "branch", "-M", "main")
    _git(worktree, "remote", "add", "origin", str(bare))
    first_sha = _commit(worktree, "# Note\nfirst searchable text\n")

    calls: list[object] = []
    fail_embeddings = False

    def embed(request: httpx.Request) -> httpx.Response:
        nonlocal fail_embeddings
        calls.append(request)
        if fail_embeddings:
            return httpx.Response(503, request=request)
        count = len(json.loads(request.content)["input"])
        return httpx.Response(
            200,
            json={"data": [{"index": index, "embedding": [1.0, 0.0]} for index in range(count)]},
            request=request,
        )

    config = _service_config(bare)
    data_root = tmp_path / "data"
    registry = config.registry(data_root)
    http_client = httpx.Client(transport=httpx.MockTransport(embed))
    factory = AppFactory(
        config_path=tmp_path / "unused.toml",
        database_path=data_root / "index.sqlite3",
        registry=registry,
        environ={"EMBED_MODEL": "synthetic"},
        http_client=http_client,
    )
    state_store = SyncStateStore(data_root / "sync-state")
    coordinator = RepositorySyncCoordinator(
        config, factory, state_store, LifecycleLock(), ManagedGit()
    )

    coordinator.run_once()
    profile = factory.resolve_profile("_reconcile_alpha")
    first = factory.retrieval(profile).search(SearchRequest("searchable", mode=SearchMode.LEXICAL))
    assert first.hits and "first searchable text" in first.hits[0].text
    assert (
        factory.retrieval(profile)
        .read(ReadRequest(path="note.md"))
        .text.endswith("first searchable text\n")
    )
    assert state_store.read("alpha", "refs/heads/main").state.reconciled_sha == first_sha

    second_sha = _commit(worktree, "# Note\nsecond searchable text\n")
    coordinator.run_once()
    profile = factory.resolve_profile("_reconcile_alpha")
    second = factory.retrieval(profile).search(SearchRequest("second", mode=SearchMode.LEXICAL))
    assert second.hits and "second searchable text" in second.hits[0].text
    assert state_store.read("alpha", "refs/heads/main").state.reconciled_sha == second_sha

    indexed_calls = len(calls)
    coordinator.run_once()
    assert len(calls) == indexed_calls

    third_sha = _commit(worktree, "# Note\nthird lexical text\n")
    fail_embeddings = True
    coordinator.run_once()
    state = state_store.read("alpha", "refs/heads/main").state
    assert state.report is not None and state.report.embedding_failures > 0
    fail_embeddings = False
    profile = factory.resolve_profile("_reconcile_alpha")
    degraded = factory.retrieval(profile).search(SearchRequest("third"))
    assert degraded.hits and "third lexical text" in degraded.hits[0].text
    assert degraded.degraded.semantic_search is True
    assert degraded.degraded.reason == "pending_embeddings"
    assert state_store.read("alpha", "refs/heads/main").state.reconciled_sha == third_sha

    factory.close()
    restarted_calls: list[object] = []
    restarted_http = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: (
                restarted_calls.append(request)
                or httpx.Response(200, json={"data": []}, request=request)
            )
        )
    )
    restarted = AppFactory(
        config_path=tmp_path / "unused.toml",
        database_path=data_root / "index.sqlite3",
        registry=registry,
        environ={"EMBED_MODEL": "synthetic"},
        http_client=restarted_http,
    )
    restarted_coordinator = RepositorySyncCoordinator(
        config, restarted, state_store, LifecycleLock(), ManagedGit()
    )
    restarted_coordinator.run_once()
    assert restarted_calls == []

    shutil.rmtree(bare)
    restarted_coordinator.run_once()
    profile = restarted.resolve_profile("_reconcile_alpha")
    retained = restarted.retrieval(profile).search(SearchRequest("third", mode=SearchMode.LEXICAL))
    assert retained.hits and "third lexical text" in retained.hits[0].text

    restarted.close()
    http_client.close()
    restarted_http.close()


def test_invalid_first_manifest_does_not_leave_staged_checkouts(tmp_path: Path) -> None:
    bare = tmp_path / "remote.git"
    worktree = tmp_path / "author"
    _git(tmp_path, "init", "--bare", str(bare))
    _git(tmp_path, "init", str(worktree))
    _git(worktree, "config", "user.name", "Test Author")
    _git(worktree, "config", "user.email", "author@example.test")
    (worktree / ".vault-rag.toml").write_text("not valid toml = [", encoding="utf-8")
    _git(worktree, "add", ".vault-rag.toml")
    _git(worktree, "commit", "-m", "add invalid manifest")
    _git(worktree, "branch", "-M", "main")
    _git(worktree, "remote", "add", "origin", str(bare))
    _git(worktree, "push", "origin", "main")

    config = _service_config(bare)
    data_root = tmp_path / "data"
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(500, request=request))
    )
    factory = AppFactory(
        config_path=tmp_path / "unused.toml",
        database_path=data_root / "index.sqlite3",
        registry=config.registry(data_root),
        environ={"EMBED_MODEL": "synthetic"},
        http_client=http_client,
    )
    state_store = SyncStateStore(data_root / "sync-state")
    coordinator = RepositorySyncCoordinator(
        config, factory, state_store, LifecycleLock(), ManagedGit()
    )

    coordinator.run_once()
    coordinator.run_once()

    assert tuple((data_root / "repos").glob(".alpha.*")) == ()
    assert not (data_root / "repos" / "alpha").exists()
    assert (
        state_store.read("alpha", "refs/heads/main").state.sync_degraded_reason
        == "manifest_invalid"
    )
    factory.close()
    http_client.close()
