from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from psycopg import OperationalError  # pyright: ignore[reportMissingImports]

import vault_rag.app as app_module
import vault_rag.service.runtime as runtime_module
from vault_rag.config import EgressPolicy, VaultManifest
from vault_rag.errors import ConfigError
from vault_rag.service.http import create_app
from vault_rag.service.postgres_coordinator import PostgresServiceCoordinator
from vault_rag.service.runtime import build_api_runtime, build_worker_runtime, migrate_database
from vault_rag.service.worker import WorkerRuntime
from vault_rag.storage.ports import RevisionBuildStore
from vault_rag.storage.postgres import (
    POSTGRES_SCHEMA_VERSION,
    PostgresMigrator,
    PostgresPool,
    PostgresStore,
    PostgresWorkerLease,
)

from .helpers import revision


def _service_config(path: Path) -> None:
    path.write_text(
        """
schemaVersion: 2
syncInterval: 5m
repositories:
  vault-a:
    url: https://example.test/vault-a.git
    ref: refs/heads/main
profiles:
  default:
    vaults: [vault-a]
embedding:
  baseUrl: http://127.0.0.1:9000/v1
  modelEnv: EMBED_MODEL
  endpointClass: local
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  pool:
    minSize: 1
    maxSize: 4
    timeout: 2s
  statementTimeout: 5s
""".lstrip(),
        encoding="utf-8",
    )


def test_migrate_and_build_split_postgresql_runtimes(
    tmp_path: Path,
    clean_postgres: str,
) -> None:
    config_path = tmp_path / "service.yaml"
    data_root = tmp_path / "data"
    _service_config(config_path)
    environ = {
        "VAULT_RAG_DATABASE_URL": clean_postgres,
        "EMBED_MODEL": "synthetic",
    }

    assert migrate_database(config_path, environ) == tuple(range(1, POSTGRES_SCHEMA_VERSION + 1))

    api = build_api_runtime(config_path, data_root, environ)
    try:
        assert isinstance(api.service._coordinator, PostgresServiceCoordinator)
        assert isinstance(api.service._checkout_inspector, PostgresServiceCoordinator)
        api.start()
    finally:
        api.close()

    worker = build_worker_runtime(config_path, data_root, environ)
    assert isinstance(worker, WorkerRuntime)
    worker.close()

    pool = PostgresPool(clean_postgres, min_size=1, max_size=1, timeout=2.0)
    pool.open(wait=True)
    try:
        with pool.connection() as connection:
            version = connection.execute(
                "SELECT max(version) AS version FROM vault_rag.schema_migrations"
            ).fetchone()
        assert version is not None and version["version"] == POSTGRES_SCHEMA_VERSION
    finally:
        pool.close()


def test_api_only_postgresql_construction_avoids_local_runtime_components(
    tmp_path: Path, clean_postgres: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """API-only PostgreSQL assembly must not create checkout, SQLite, or index work."""
    config_path = tmp_path / "service.yaml"
    _service_config(config_path)
    environ = {
        "VAULT_RAG_DATABASE_URL": clean_postgres,
        "EMBED_MODEL": "synthetic",
    }
    assert migrate_database(config_path, environ) == tuple(range(1, POSTGRES_SCHEMA_VERSION + 1))

    def forbidden(*_args: object, **_kwargs: object) -> object:
        pytest.fail("API-only PostgreSQL runtime instantiated a local component")

    monkeypatch.setattr(app_module, "SQLiteStore", forbidden)
    monkeypatch.setattr(app_module, "Indexer", forbidden)
    monkeypatch.setattr(runtime_module, "SyncStateStore", forbidden)
    monkeypatch.setattr(runtime_module, "ManagedGit", forbidden)

    api = build_api_runtime(config_path, tmp_path / "data", environ)
    api.close()


def test_split_entrypoints_reject_sqlite_service_configuration(tmp_path: Path) -> None:
    config_path = tmp_path / "service.yaml"
    config_path.write_text(
        """
schemaVersion: 1
syncInterval: 5m
repositories:
  vault-a:
    url: https://example.test/vault-a.git
    ref: refs/heads/main
profiles:
  default:
    vaults: [vault-a]
embedding:
  baseUrl: http://127.0.0.1:9000/v1
  modelEnv: EMBED_MODEL
  endpointClass: local
""".lstrip(),
        encoding="utf-8",
    )
    environ = {"EMBED_MODEL": "synthetic"}

    with pytest.raises(ConfigError, match="API process requires PostgreSQL"):
        build_api_runtime(config_path, tmp_path / "api", environ)
    with pytest.raises(ConfigError, match="worker process requires PostgreSQL"):
        build_worker_runtime(config_path, tmp_path / "worker", environ)
    with pytest.raises(ConfigError, match="migration requires PostgreSQL"):
        migrate_database(config_path, environ)


class _FaultingPool:
    """Inject a deterministic outage at the checked-out PostgreSQL boundary."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool
        self.unavailable = False

    def connection(self, timeout: float | None = None) -> Any:
        if self.unavailable:
            raise OperationalError("deterministic database outage")
        return self._pool.connection(timeout=timeout)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pool, name)


def _promote(pool: PostgresPool, revision_store: RevisionBuildStore) -> None:
    lease = PostgresWorkerLease(pool, "sync:vault-a")
    assert lease.acquire() is True
    try:
        lease.promote(revision_store, lexical_complete=True, fully_reconciled=True)
    finally:
        lease.release()


def _seed_ready_revision(dsn: str) -> None:
    pool = PostgresPool(dsn, min_size=1, max_size=2, timeout=2)
    pool.open(wait=True)
    try:
        PostgresMigrator(pool).apply()
        manifest = VaultManifest(
            schema_version=1,
            id="vault-a",
            egress_policy=EgressPolicy.LOCAL_ONLY,
            include=("**/*.md",),
        )
        store = PostgresStore(pool)
        revision_store = store.begin_revision(
            "vault-a",
            "refs/heads/main",
            "a" * 40,
            manifest=manifest.model_dump(mode="json"),
        )
        revision_store.replace_source(revision("ready database content"))
        _promote(pool, revision_store)
    finally:
        pool.close()


def test_split_runtimes_report_database_loss_then_recover_without_reconstruction(
    tmp_path: Path, clean_postgres: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing runtime health checks or silently recreating pools hides a real outage."""
    config_path = tmp_path / "service.yaml"
    _service_config(config_path)
    environ = {
        "VAULT_RAG_DATABASE_URL": clean_postgres,
        "EMBED_MODEL": "synthetic",
    }
    _seed_ready_revision(clean_postgres)
    original_pool = runtime_module._postgres_pool
    faulting_pools: list[_FaultingPool] = []

    def build_faulting_pool(*args: Any, **kwargs: Any) -> PostgresPool:
        pool = original_pool(*args, **kwargs)
        fault = _FaultingPool(pool._pool)
        pool._pool = cast(Any, fault)
        faulting_pools.append(fault)
        return pool

    monkeypatch.setattr(runtime_module, "_postgres_pool", build_faulting_pool)
    api = build_api_runtime(config_path, tmp_path / "api", environ)
    worker = build_worker_runtime(config_path, tmp_path / "worker", environ)
    api_id = id(api)
    worker_id = id(worker)
    api_pool = api.storage_pool
    assert api_pool is not None
    try:
        with TestClient(create_app(api)) as client:
            assert client.get("/health/live").json() == {"live": True}
            assert client.get("/health/ready").status_code == 200
            assert worker.live() is True
            assert worker.ready() is True

            for fault in faulting_pools:
                fault.unavailable = True

            assert client.get("/health/live").json() == {"live": True}
            assert client.get("/health/ready").status_code == 503
            assert worker.live() is True
            assert worker.ready() is False

            for fault in faulting_pools:
                fault.unavailable = False

            assert client.get("/health/ready").status_code == 200
            assert worker.ready() is True
            rendered = api.metrics.render().decode()
            assert "vault_rag_pool_connection_failure_total 1.0" in rendered
            assert "vault_rag_pool_reconnect_total 1.0" in rendered
            assert id(api) == api_id
            assert id(worker) == worker_id
            assert api.storage_pool is api_pool
    finally:
        worker.close()
