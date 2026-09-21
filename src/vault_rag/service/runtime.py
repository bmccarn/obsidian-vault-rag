"""Production composition and lifecycle ownership for service processes."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from threading import RLock

import httpx

from vault_rag.app import AppFactory
from vault_rag.config import ResolvedProfile
from vault_rag.errors import ConfigError, StorageError
from vault_rag.storage.ports import QueryStore
from vault_rag.storage.postgres import (
    CleanupPolicy,
    CleanupResult,
    PostgresMigrator,
    PostgresPool,
    PostgresRevisionCleaner,
    PostgresStore,
    PostgresWorkerLease,
)

from .config import (
    MCPTransportConfig,
    ServiceConfig,
    StorageBackend,
    load_service_config,
    materialize_secret_files,
)
from .coordinator import RepositorySyncCoordinator
from .facade import CheckoutInspectorPort, SyncCoordinatorPort, VaultService
from .git import ManagedGit
from .locking import LifecycleLock
from .observability import ServiceMetrics, ServiceObservability, StructuredEvents
from .postgres_coordinator import PostgresServiceCoordinator
from .postgres_profile import PostgresProfileResolver
from .postgres_state import PostgresSyncStateStore
from .state import SyncStateStore, SyncStateStorePort
from .worker import LeaseSession, WorkerRuntime

_WORKER_HEARTBEAT_INTERVAL = timedelta(seconds=20)


@dataclass(slots=True)
class ServiceRuntime:
    """Own process-scoped HTTP service resources and close them once."""

    service: VaultService
    metrics: ServiceMetrics
    http_client: httpx.Client
    events: StructuredEvents = field(
        default_factory=lambda: StructuredEvents(logging.getLogger("vault_rag.service"))
    )
    storage_pool: PostgresPool | None = None
    mcp_config: MCPTransportConfig = field(default_factory=MCPTransportConfig)
    _closed: bool = field(default=False, init=False, repr=False)
    _lifecycle_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def start(self) -> None:
        """Start the configured lifecycle; API-only coordinators deliberately do nothing."""
        with self._lifecycle_lock:
            if self._closed:
                return
        self.service.start()

    def close(self) -> None:
        """Close service, storage, and external HTTP resources in dependency order."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.service.close()
        finally:
            try:
                if self.storage_pool is not None:
                    self.storage_pool.close()
            finally:
                self.http_client.close()


def _postgres_pool(
    config: ServiceConfig,
    environ: Mapping[str, str],
    *,
    min_size: int | None = None,
    max_size: int | None = None,
    metrics: ServiceMetrics | None = None,
) -> PostgresPool:
    database_url_env = config.storage.database_url_env
    if database_url_env is None or not environ.get(database_url_env):
        raise ConfigError("PostgreSQL database credential is unavailable")
    pool = PostgresPool(
        environ[database_url_env],
        min_size=config.storage.pool.min_size if min_size is None else min_size,
        max_size=config.storage.pool.max_size if max_size is None else max_size,
        timeout=config.storage.pool.timeout.total_seconds(),
        statement_timeout=config.storage.statement_timeout.total_seconds(),
        connect_timeout=config.storage.connect_timeout.total_seconds(),
        lock_timeout=config.storage.lock_timeout.total_seconds(),
        idle_transaction_timeout=config.storage.idle_transaction_timeout.total_seconds(),
        allow_insecure_transport=config.storage.allow_insecure_transport,
        metrics=metrics,
    )
    try:
        pool.open(wait=True)
    except Exception:
        pool.close()
        raise
    return pool


def _postgres_store_factory(pool: PostgresPool) -> Callable[[], QueryStore]:
    def create_store() -> PostgresStore:
        return PostgresStore(pool)

    return create_store


def _build_service_runtime(
    config_path: Path,
    data_root: Path,
    environ: Mapping[str, str],
    *,
    api_only: bool,
) -> ServiceRuntime:
    config = load_service_config(config_path)
    if api_only and config.storage.backend is not StorageBackend.POSTGRESQL:
        raise ConfigError("the API process requires PostgreSQL storage")
    captured_environ = materialize_secret_files(config, environ)
    resolved_data_root = Path(data_root)
    http_client = httpx.Client()
    metrics = ServiceMetrics()
    storage_pool: PostgresPool | None = None
    try:
        store_factory: Callable[[], QueryStore] | None
        state_store: SyncStateStorePort
        profile_resolver: Callable[[str], ResolvedProfile] | None = None
        if config.storage.backend is StorageBackend.POSTGRESQL:
            postgres_pool = _postgres_pool(config, captured_environ, metrics=metrics)
            storage_pool = postgres_pool
            PostgresMigrator(postgres_pool).check()
            store_factory = _postgres_store_factory(postgres_pool)
            state_store = PostgresSyncStateStore(postgres_pool)
            profile_resolver = PostgresProfileResolver(
                postgres_pool,
                config.registry(resolved_data_root),
                captured_environ,
            )
        else:
            store_factory = None
            state_store = SyncStateStore(resolved_data_root / "sync-state")

        factory = AppFactory(
            config_path=Path(config_path),
            database_path=resolved_data_root / "index.sqlite3",
            registry=config.registry(resolved_data_root),
            environ=captured_environ,
            http_client=http_client,
            store_factory=store_factory,
            profile_resolver=profile_resolver,
        )
        lock = LifecycleLock()
        events = StructuredEvents(logging.getLogger("vault_rag.service"))
        observability = ServiceObservability(metrics, events)
        lifecycle: SyncCoordinatorPort
        checkout_inspector: CheckoutInspectorPort
        if api_only:
            assert storage_pool is not None
            lifecycle = PostgresServiceCoordinator(storage_pool)
            checkout_inspector = lifecycle
        else:
            coordinator = RepositorySyncCoordinator(
                config,
                factory,
                state_store,
                lock,
                ManagedGit(),
                environ=captured_environ,
                observer=observability,
            )
            lifecycle = coordinator
            checkout_inspector = coordinator
        service = VaultService(
            config,
            factory,
            lifecycle,
            checkout_inspector,
            state_store,
            lock,
            metrics=metrics,
        )
        return ServiceRuntime(
            service,
            metrics,
            http_client,
            events,
            storage_pool,
            mcp_config=config.mcp,
        )
    except Exception:
        if storage_pool is not None:
            storage_pool.close()
        http_client.close()
        raise


def build_runtime(config_path: Path, data_root: Path, environ: Mapping[str, str]) -> ServiceRuntime:
    """Build the combined single-process service for the configured backend."""
    return _build_service_runtime(config_path, data_root, environ, api_only=False)


def build_api_runtime(
    config_path: Path,
    data_root: Path,
    environ: Mapping[str, str],
) -> ServiceRuntime:
    """Build a PostgreSQL HTTP runtime that never owns Git or indexing work."""
    return _build_service_runtime(config_path, data_root, environ, api_only=True)


def build_worker_runtime(
    config_path: Path,
    data_root: Path,
    environ: Mapping[str, str],
) -> WorkerRuntime:
    """Build a PostgreSQL worker that coordinates reconciliation per vault."""
    config = load_service_config(config_path)
    if config.storage.backend is not StorageBackend.POSTGRESQL:
        raise ConfigError("the worker process requires PostgreSQL storage")
    captured_environ = materialize_secret_files(config, environ)
    resolved_data_root = Path(data_root)
    http_client = httpx.Client()
    metrics = ServiceMetrics()
    pool: PostgresPool | None = None
    factory: AppFactory | None = None
    try:
        pool = _postgres_pool(config, captured_environ, metrics=metrics)
        state_store = PostgresSyncStateStore(pool)
        factory = AppFactory(
            config_path=Path(config_path),
            database_path=resolved_data_root / "unused.sqlite3",
            registry=config.registry(resolved_data_root),
            environ=captured_environ,
            http_client=http_client,
            store_factory=_postgres_store_factory(pool),
        )
        events = StructuredEvents(logging.getLogger("vault_rag.worker"))

        def lease_factory(vault_id: str) -> LeaseSession:
            assert pool is not None
            lease = PostgresWorkerLease(pool, f"sync:{vault_id}")
            return LeaseSession(lease, _WORKER_HEARTBEAT_INTERVAL)

        coordinator = RepositorySyncCoordinator(
            config,
            factory,
            state_store,
            LifecycleLock(),
            ManagedGit(),
            environ=captured_environ,
            observer=ServiceObservability(metrics, events),
            lease_factory=lease_factory,
        )

        cleanup_callback: Callable[[], None] | None = None
        if config.storage.cleanup.enabled:
            cleanup_policy = CleanupPolicy(
                enabled=True,
                keep_promoted=config.storage.cleanup.keep_promoted,
                min_age=config.storage.cleanup.min_age,
                batch_size=config.storage.cleanup.batch_size,
            )
            cleaner = PostgresRevisionCleaner(pool)

            def run_cleanup() -> None:
                for vault_id in config.repositories:
                    cleaner.run(vault_id, cleanup_policy)

            cleanup_callback = run_cleanup
        requests = PostgresServiceCoordinator(pool)

        def close_resources() -> None:
            assert factory is not None and pool is not None
            try:
                factory.close()
            finally:
                try:
                    pool.close()
                finally:
                    http_client.close()

        def worker_ready() -> bool:
            assert pool is not None
            try:
                PostgresMigrator(pool).check()
            except StorageError:
                return False
            return True

        return WorkerRuntime(
            coordinator,
            poll_interval=config.sync_interval,
            request_queue=requests,
            cleanup=cleanup_callback,
            close_resources=close_resources,
            health_check=worker_ready,
        )
    except Exception:
        if factory is not None:
            factory.close()
        if pool is not None:
            pool.close()
        http_client.close()
        raise


def migrate_database(config_path: Path, environ: Mapping[str, str]) -> tuple[int, ...]:
    """Apply application-owned PostgreSQL migrations under the database advisory lock."""
    config = load_service_config(config_path)
    if config.storage.backend is not StorageBackend.POSTGRESQL:
        raise ConfigError("database migration requires PostgreSQL storage")
    captured_environ = materialize_secret_files(config, environ)
    pool = _postgres_pool(config, captured_environ, min_size=0, max_size=1)
    try:
        return PostgresMigrator(pool, owner_role=config.storage.owner_role).apply()
    finally:
        pool.close()


def check_database(config_path: Path, environ: Mapping[str, str]) -> tuple[int, ...]:
    """Verify that the configured PostgreSQL schema is complete and compatible."""
    config = load_service_config(config_path)
    if config.storage.backend is not StorageBackend.POSTGRESQL:
        raise ConfigError("database check requires PostgreSQL storage")
    captured_environ = materialize_secret_files(config, environ)
    pool = _postgres_pool(config, captured_environ, min_size=0, max_size=1)
    try:
        return PostgresMigrator(pool).check()
    finally:
        pool.close()


def cleanup_database(config_path: Path, environ: Mapping[str, str], vault_id: str) -> CleanupResult:
    """Run an explicitly requested bounded cleanup pass for one vault."""
    config = load_service_config(config_path)
    if config.storage.backend is not StorageBackend.POSTGRESQL:
        raise ConfigError("database cleanup requires PostgreSQL storage")
    captured_environ = materialize_secret_files(config, environ)
    pool = _postgres_pool(config, captured_environ, min_size=0, max_size=2)
    cleanup = config.storage.cleanup
    policy = CleanupPolicy(
        enabled=True,
        keep_promoted=cleanup.keep_promoted,
        min_age=cleanup.min_age,
        batch_size=cleanup.batch_size,
    )
    try:
        PostgresMigrator(pool).check()
        return PostgresRevisionCleaner(pool).run(vault_id, policy)
    finally:
        pool.close()
