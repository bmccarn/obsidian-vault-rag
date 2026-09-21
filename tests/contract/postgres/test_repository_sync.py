from __future__ import annotations

import json
import shutil
import subprocess
from datetime import timedelta
from pathlib import Path

import httpx

from vault_rag.app import AppFactory
from vault_rag.retrieval import ReadRequest, SearchMode, SearchRequest
from vault_rag.service.config import (
    RepositoryConfig,
    ServiceConfig,
    ServiceEmbeddingConfig,
    ServiceProfileConfig,
)
from vault_rag.service.coordinator import RepositorySyncCoordinator
from vault_rag.service.facade import VaultService
from vault_rag.service.git import ManagedGit
from vault_rag.service.locking import LifecycleLock
from vault_rag.service.postgres_coordinator import PostgresServiceCoordinator
from vault_rag.service.postgres_profile import PostgresProfileResolver
from vault_rag.service.postgres_state import PostgresSyncStateStore
from vault_rag.service.worker import LeaseSession
from vault_rag.storage import SCHEMA_VERSION
from vault_rag.storage.postgres import (
    PostgresMigrator,
    PostgresPool,
    PostgresStore,
    PostgresWorkerLease,
)


def _git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(("git", *args), cwd=cwd, check=True, text=True, capture_output=True)
    return completed.stdout.strip()


def _commit(worktree: Path, text: str) -> str:
    (worktree / "note.md").write_text(text, encoding="utf-8")
    _git(worktree, "add", "note.md")
    _git(worktree, "commit", "-m", "update note")
    _git(worktree, "push", "origin", "main")
    return _git(worktree, "rev-parse", "HEAD")


def _config(remote: Path) -> ServiceConfig:
    repository = RepositoryConfig.model_construct(url=remote.as_uri(), ref="refs/heads/main")
    return ServiceConfig.model_construct(
        schema_version=1,
        sync_interval=timedelta(hours=1),
        repositories={"alpha": repository},
        profiles={"public": ServiceProfileConfig(vaults=("alpha",))},
        embedding=ServiceEmbeddingConfig(
            base_url="http://127.0.0.1:9000/v1",
            model_env="EMBED_MODEL",
            endpoint_class="local",
        ),
    )


def _lease_session(pool: PostgresPool, vault_id: str) -> LeaseSession:
    return LeaseSession(PostgresWorkerLease(pool, f"sync:{vault_id}"), timedelta(minutes=1))


def test_repository_sync_promotes_postgresql_revision_and_reads_without_checkout(
    tmp_path: Path,
    postgres_pool: PostgresPool,
) -> None:
    PostgresMigrator(postgres_pool).apply()
    bare = tmp_path / "remote.git"
    author = tmp_path / "author"
    _git(tmp_path, "init", "--bare", str(bare))
    _git(tmp_path, "init", str(author))
    _git(author, "config", "user.name", "Test Author")
    _git(author, "config", "user.email", "author@example.test")
    (author / ".vault-rag.toml").write_text(
        'schema_version = 1\nid = "alpha"\negress_policy = "local-only"\ninclude = ["**/*.md"]\n',
        encoding="utf-8",
    )
    _git(author, "add", ".vault-rag.toml")
    _git(author, "commit", "-m", "add manifest")
    _git(author, "branch", "-M", "main")
    _git(author, "remote", "add", "origin", str(bare))
    commit_sha = _commit(author, "# Note\ncentral searchable text\n")

    leases: list[PostgresWorkerLease] = []
    embedding_calls = {"count": 0}
    terminate_lock = {"pending": False}
    embedding_503 = {"pending": False}

    def embed(request: httpx.Request) -> httpx.Response:
        if terminate_lock["pending"]:
            backend_pid = leases[-1].backend_pid
            assert backend_pid is not None
            with postgres_pool.connection() as connection:
                connection.execute("SELECT pg_terminate_backend(%s)", (backend_pid,))
                connection.commit()
            terminate_lock["pending"] = False
        embedding_calls["count"] += 1
        if embedding_503["pending"]:
            return httpx.Response(503, request=request)
        count = len(json.loads(request.content)["input"])
        return httpx.Response(
            200,
            json={"data": [{"index": index, "embedding": [1.0, 0.0]} for index in range(count)]},
            request=request,
        )

    data_root = tmp_path / "data"
    config = _config(bare)
    http_client = httpx.Client(transport=httpx.MockTransport(embed))
    factory = AppFactory(
        config_path=tmp_path / "unused.toml",
        database_path=data_root / "unused.sqlite3",
        registry=config.registry(data_root),
        environ={"EMBED_MODEL": "synthetic"},
        http_client=http_client,
        store_factory=lambda: PostgresStore(postgres_pool),
    )
    state_store = PostgresSyncStateStore(postgres_pool)
    coordinator = RepositorySyncCoordinator(
        config,
        factory,
        state_store,
        LifecycleLock(),
        ManagedGit(),
        environ={"EMBED_MODEL": "synthetic"},
        lease_factory=lambda vault_id: (
            leases.append(PostgresWorkerLease(postgres_pool, f"sync:{vault_id}"))
            or LeaseSession(leases[-1], timedelta(minutes=1))
        ),
    )

    coordinator.run_once()
    assert state_store.read("alpha", "refs/heads/main").state.sync_degraded_reason is None
    profile = factory.resolve_profile("_reconcile_alpha")
    search = factory.retrieval(profile).search(SearchRequest("central", mode=SearchMode.LEXICAL))

    assert search.hits and "central searchable text" in search.hits[0].text
    assert state_store.read("alpha", "refs/heads/main").state.reconciled_sha == commit_sha

    with postgres_pool.connection() as connection:
        active_before = connection.execute(
            "SELECT active_revision_id FROM vault_rag.vaults WHERE vault_id = 'alpha'"
        ).fetchone()
    assert active_before is not None
    _commit(author, "# New\nmust not become active\n")
    terminate_lock["pending"] = True
    coordinator.run_once()

    with postgres_pool.connection() as connection:
        active_after = connection.execute(
            "SELECT active_revision_id FROM vault_rag.vaults WHERE vault_id = 'alpha'"
        ).fetchone()
    assert active_after == active_before

    # Use a new source hash so the provider failure cannot be masked by a
    # vector that the interrupted build already persisted for reuse.
    _commit(author, "# Failure\nembedding must remain pending\n")
    embedding_503["pending"] = True
    coordinator.run_once()
    coordinator.run_once()

    with postgres_pool.connection() as connection:
        degraded = connection.execute(
            """
            SELECT vr.commit_sha, vr.state
            FROM vault_rag.vaults AS v
            JOIN vault_rag.vault_revisions AS vr ON vr.revision_id = v.active_revision_id
            WHERE v.vault_id = 'alpha'
            """
        ).fetchone()
    assert degraded == {"commit_sha": _git(author, "rev-parse", "HEAD"), "state": "active_degraded"}
    profile = factory.resolve_profile("_reconcile_alpha")
    degraded_search = factory.retrieval(profile).search(
        SearchRequest("must not become active", mode=SearchMode.LEXICAL)
    )
    assert degraded_search.hits

    embedding_503["pending"] = False
    _git(author, "reset", "--hard", commit_sha)
    _git(author, "push", "--force", "origin", "main")
    coordinator.run_once()

    with postgres_pool.connection() as connection:
        revision_count_before = connection.execute(
            "SELECT count(*) AS count FROM vault_rag.vault_revisions WHERE vault_id = 'alpha'"
        ).fetchone()["count"]
        active_before_reschedule = connection.execute(
            "SELECT active_revision_id FROM vault_rag.vaults WHERE vault_id = 'alpha'"
        ).fetchone()
    embedding_calls_before = embedding_calls["count"]
    shutil.rmtree(data_root / "repos" / "alpha")

    coordinator.run_once()

    with postgres_pool.connection() as connection:
        active_after_reschedule = connection.execute(
            "SELECT active_revision_id FROM vault_rag.vaults WHERE vault_id = 'alpha'"
        ).fetchone()
        revision_count_after = connection.execute(
            "SELECT count(*) AS count FROM vault_rag.vault_revisions WHERE vault_id = 'alpha'"
        ).fetchone()["count"]
    assert active_after_reschedule == active_before_reschedule
    assert revision_count_after == revision_count_before
    assert embedding_calls["count"] == embedding_calls_before
    assert not (data_root / "repos" / "alpha").exists()
    assert state_store.read("alpha", "refs/heads/main").state.reconciled_sha == commit_sha

    shutil.rmtree(data_root / "repos" / "alpha", ignore_errors=True)
    factory.close()
    offline_remote = tmp_path / "offline.git"
    bare.rename(offline_remote)
    coordinator.run_once()
    failed_fetch_state = state_store.read("alpha", "refs/heads/main").state
    assert failed_fetch_state.reconciled_sha == commit_sha
    assert failed_fetch_state.sync_degraded_reason == "fetch_failed"
    registry = config.registry(data_root)
    api_factory = AppFactory(
        config_path=tmp_path / "unused.toml",
        database_path=data_root / "unused.sqlite3",
        registry=registry,
        environ={"EMBED_MODEL": "synthetic"},
        http_client=http_client,
        store_factory=lambda: PostgresStore(postgres_pool),
        profile_resolver=PostgresProfileResolver(
            postgres_pool,
            registry,
            {"EMBED_MODEL": "synthetic"},
        ),
    )
    api_coordinator = PostgresServiceCoordinator(postgres_pool)
    api_service = VaultService(
        config,
        api_factory,
        api_coordinator,
        api_coordinator,
        state_store,
        LifecycleLock(),
    )
    api_status = api_service.status("public")
    assert api_status["vaults"]["alpha"]["index"]["counts"]["sources"] > 0
    assert api_status["vaults"]["alpha"]["index"]["schema"]["actual"] == SCHEMA_VERSION
    assert api_service.ready().profiles["public"] is True
    assert api_service.status("public")["vaults"]["alpha"]["commit"]["reconciled_sha"] == commit_sha
    assert api_service.search(
        "public", SearchRequest("central", mode=SearchMode.LEXICAL)
    ).response.hits
    assert api_service.read(
        "public", ReadRequest(path="note.md", vault_id="alpha")
    ).response.text.endswith("central searchable text\n")
    api_profile = api_factory.resolve_profile("_reconcile_alpha")
    read = api_factory.retrieval(api_profile).read(ReadRequest(path="note.md"))

    assert read.text.endswith("central searchable text\n")
    assert not (data_root / "index.sqlite3").exists()
    api_factory.close()
    http_client.close()


def test_serialized_independent_postgresql_workers_skip_an_unchanged_commit(
    tmp_path: Path,
    postgres_pool: PostgresPool,
) -> None:
    """The second worker's fresh session must reuse the first worker's promotion."""
    PostgresMigrator(postgres_pool).apply()
    bare = tmp_path / "remote.git"
    author = tmp_path / "author"
    _git(tmp_path, "init", "--bare", str(bare))
    _git(tmp_path, "init", str(author))
    _git(author, "config", "user.name", "Test Author")
    _git(author, "config", "user.email", "author@example.test")
    (author / ".vault-rag.toml").write_text(
        'schema_version = 1\nid = "alpha"\negress_policy = "local-only"\ninclude = ["**/*.md"]\n',
        encoding="utf-8",
    )
    _git(author, "add", ".vault-rag.toml")
    _git(author, "commit", "-m", "add manifest")
    _git(author, "branch", "-M", "main")
    _git(author, "remote", "add", "origin", str(bare))
    commit_sha = _commit(author, "# Note\none stable commit\n")
    config = _config(bare)
    factories: list[AppFactory] = []
    clients: list[httpx.Client] = []

    def worker(data_root: Path) -> RepositorySyncCoordinator:
        def embed(request: httpx.Request) -> httpx.Response:
            count = len(json.loads(request.content)["input"])
            return httpx.Response(
                200,
                json={
                    "data": [{"index": index, "embedding": [1.0, 0.0]} for index in range(count)]
                },
                request=request,
            )

        client = httpx.Client(transport=httpx.MockTransport(embed))
        clients.append(client)
        factory = AppFactory(
            config_path=tmp_path / "unused.toml",
            database_path=data_root / "unused.sqlite3",
            registry=config.registry(data_root),
            environ={"EMBED_MODEL": "synthetic"},
            http_client=client,
            store_factory=lambda: PostgresStore(postgres_pool),
        )
        factories.append(factory)
        return RepositorySyncCoordinator(
            config,
            factory,
            PostgresSyncStateStore(postgres_pool),
            LifecycleLock(),
            ManagedGit(),
            environ={"EMBED_MODEL": "synthetic"},
            lease_factory=lambda vault_id: _lease_session(postgres_pool, vault_id),
        )

    try:
        worker(tmp_path / "worker-a-data").run_once()
        with postgres_pool.connection() as connection:
            first = connection.execute(
                """
                SELECT count(*) AS revisions, max(commit_sha) AS commit_sha
                FROM vault_rag.vault_revisions
                WHERE vault_id = 'alpha'
                """
            ).fetchone()
        assert first == {"revisions": 1, "commit_sha": commit_sha}

        worker(tmp_path / "worker-b-data").run_once()
        with postgres_pool.connection() as connection:
            second = connection.execute(
                """
                SELECT count(*) AS revisions, max(commit_sha) AS commit_sha
                FROM vault_rag.vault_revisions
                WHERE vault_id = 'alpha'
                """
            ).fetchone()
        assert second == {"revisions": 1, "commit_sha": commit_sha}
    finally:
        for factory in factories:
            factory.close()
        for client in clients:
            client.close()
