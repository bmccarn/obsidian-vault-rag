"""Bounded Psycopg connection pool with secret-safe diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import AbstractContextManager, contextmanager, suppress
from ipaddress import ip_address
from time import perf_counter
from typing import Any, Protocol, cast

from pgvector.psycopg import register_vector  # type: ignore[import-untyped]
from psycopg import Connection, Error, OperationalError  # pyright: ignore[reportMissingImports]
from psycopg.conninfo import conninfo_to_dict  # pyright: ignore[reportMissingImports]
from psycopg.rows import dict_row  # pyright: ignore[reportMissingImports]
from psycopg_pool import ConnectionPool, PoolTimeout  # pyright: ignore[reportMissingImports]

from vault_rag.errors import StorageError

type DatabaseRow = dict[str, Any]
type DatabaseConnection = Connection[DatabaseRow]
type PoolFactory = Callable[..., _Pool]


class PoolMetrics(Protocol):
    """Small fixed-label metrics surface used by PostgreSQL pool paths."""

    def observe_pool(self, *, size: int, used: int, idle: int, wait: int) -> None: ...

    def observe_pool_acquire(self, elapsed_ms: int | float) -> None: ...

    def pool_reconnected(self) -> None: ...

    def pool_connection_failed(self) -> None: ...

    def observe_postgres_operation(
        self, operation: str, outcome: str, elapsed_ms: int | float
    ) -> None: ...

    def observe_lock(self, event: str, elapsed_ms: int | float = 0) -> None: ...

    def observe_migration(
        self, operation: str, outcome: str, *, version: int, elapsed_ms: int | float
    ) -> None: ...

    def observe_cleanup(self, outcome: str, *, deleted: int, elapsed_ms: int | float) -> None: ...


class _Pool(Protocol):
    def open(self) -> None: ...

    def wait(self, timeout: float) -> None: ...

    def connection(
        self, timeout: float | None = None
    ) -> AbstractContextManager[DatabaseConnection]: ...

    def close(self) -> None: ...


def _database_location(conninfo: str) -> str:
    try:
        fields = conninfo_to_dict(conninfo)
    except Error:
        raise ValueError("PostgreSQL DSN is invalid") from None
    host = fields.get("host") or "localhost"
    port = fields.get("port") or "5432"
    database = fields.get("dbname") or "postgres"
    return f"postgresql://{host}:{port}/{database}"


def _is_loopback_endpoint(value: object, *, allow_hostname: bool) -> bool:
    """Return true only when every libpq endpoint is unquestionably local."""
    endpoints = str(value).split(",")
    for endpoint in endpoints:
        normalized = endpoint.strip().strip("[]").lower()
        if allow_hostname and normalized == "localhost":
            continue
        if allow_hostname and normalized.startswith("/"):
            continue
        try:
            if ip_address(normalized).is_loopback:
                continue
        except ValueError:
            pass
        return False
    return True


def _requires_secure_transport(conninfo: str) -> bool:
    """Reject unauthenticated network transport unless configuration says otherwise."""
    try:
        fields = conninfo_to_dict(conninfo)
    except Error:
        raise ValueError("PostgreSQL DSN is invalid") from None
    endpoints: list[tuple[object, bool]] = []
    if fields.get("host") is not None:
        endpoints.append((fields["host"], True))
    if fields.get("hostaddr") is not None:
        endpoints.append((fields["hostaddr"], False))
    if not endpoints:
        return False
    if all(
        _is_loopback_endpoint(value, allow_hostname=allow_hostname)
        for value, allow_hostname in endpoints
    ):
        return False
    return str(fields.get("sslmode") or "").lower() not in {"require", "verify-ca", "verify-full"}


class PostgresPool:
    """Explicitly opened, bounded pool shared by one process."""

    def __init__(
        self,
        conninfo: str,
        *,
        min_size: int,
        max_size: int,
        timeout: float,
        statement_timeout: float = 5.0,
        connect_timeout: float = 5.0,
        lock_timeout: float = 5.0,
        idle_transaction_timeout: float = 30.0,
        allow_insecure_transport: bool = False,
        pool_factory: PoolFactory | None = None,
        metrics: PoolMetrics | None = None,
    ) -> None:
        if (
            isinstance(min_size, bool)
            or isinstance(max_size, bool)
            or min_size < 0
            or max_size < 1
            or min_size > max_size
        ):
            raise ValueError("PostgreSQL pool bounds are invalid")
        if isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("PostgreSQL pool timeout must be positive")
        for name, value in (
            ("statement", statement_timeout),
            ("connect", connect_timeout),
            ("lock", lock_timeout),
            ("idle transaction", idle_transaction_timeout),
        ):
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"PostgreSQL {name} timeout must be positive")
        if not conninfo:
            raise ValueError("PostgreSQL DSN must be non-empty")
        if not allow_insecure_transport and _requires_secure_transport(conninfo):
            raise ValueError(
                "PostgreSQL non-loopback connections require secure transport; "
                "set sslmode=require, verify-ca, or verify-full"
            )

        self._database_location = _database_location(conninfo)
        self._timeout = float(timeout)
        self._max_size = max_size
        self._metrics = metrics
        self._opened = False
        self._was_unavailable = False
        statement_timeout_ms = max(1, round(statement_timeout * 1000))
        connect_timeout_seconds = max(1, round(connect_timeout))
        lock_timeout_ms = max(1, round(lock_timeout * 1000))
        idle_transaction_timeout_ms = max(1, round(idle_transaction_timeout * 1000))
        if pool_factory is None:
            self._pool = cast(
                _Pool,
                ConnectionPool(
                    conninfo,
                    min_size=min_size,
                    max_size=max_size,
                    timeout=self._timeout,
                    kwargs={
                        "row_factory": dict_row,
                        "connect_timeout": connect_timeout_seconds,
                        "options": (
                            f"-c statement_timeout={statement_timeout_ms}"
                            f" -c lock_timeout={lock_timeout_ms}"
                            " -c idle_in_transaction_session_timeout="
                            f"{idle_transaction_timeout_ms}"
                        ),
                    },
                    open=False,
                    name="vault-rag",
                ),
            )
        else:
            self._pool = pool_factory(
                conninfo,
                min_size=min_size,
                max_size=max_size,
                timeout=self._timeout,
                kwargs={
                    "row_factory": dict_row,
                    "connect_timeout": connect_timeout_seconds,
                    "options": (
                        f"-c statement_timeout={statement_timeout_ms}"
                        f" -c lock_timeout={lock_timeout_ms}"
                        " -c idle_in_transaction_session_timeout="
                        f"{idle_transaction_timeout_ms}"
                    ),
                },
                open=False,
                name="vault-rag",
            )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(database_location={self.database_location!r})"

    @property
    def database_location(self) -> str:
        return self._database_location

    def stats(self) -> Mapping[str, int]:
        """Return a small secret-free snapshot of supported pool statistics."""
        stats_getter = getattr(self._pool, "get_stats", None)
        if not callable(stats_getter):
            raise StorageError("PostgreSQL pool statistics are unavailable")
        raw_stats = stats_getter()
        if not isinstance(raw_stats, Mapping):
            raise StorageError("PostgreSQL pool statistics are unavailable")
        return {
            key: max(0, int(value))
            for key, value in raw_stats.items()
            if isinstance(key, str) and isinstance(value, int)
        }

    def open(self, *, wait: bool = False) -> None:
        if self._opened:
            return
        try:
            self._pool.open()
            if wait:
                self._pool.wait(self._timeout)
        except (Error, PoolTimeout) as exc:
            self._pool.close()
            self._pool_failed()
            raise StorageError("could not open PostgreSQL connection pool") from exc
        self._opened = True
        if self._was_unavailable and self._metrics is not None:
            with suppress(Exception):
                self._metrics.pool_reconnected()
        self._was_unavailable = False
        self._observe_pool()

    def close(self) -> None:
        if not self._opened:
            return
        self._pool.close()
        self._opened = False
        self._observe_pool()

    @contextmanager
    def connection(self) -> Generator[DatabaseConnection]:
        if not self._opened:
            self._pool_failed()
            raise StorageError("PostgreSQL connection pool is not open")
        started = perf_counter()
        try:
            with self._pool.connection(timeout=self._timeout) as connection:
                self._observe_pool_acquire((perf_counter() - started) * 1000)
                if self._was_unavailable and self._metrics is not None:
                    with suppress(Exception):
                        self._metrics.pool_reconnected()
                self._was_unavailable = False
                yield connection
        except StorageError:
            raise
        except (OperationalError, PoolTimeout) as exc:
            self._pool_failed()
            raise StorageError("PostgreSQL connection is unavailable") from exc
        finally:
            self._observe_pool()

    def checkout(self) -> AbstractContextManager[DatabaseConnection]:
        """Check out one connection for a caller-managed session lifetime."""
        return self.connection()

    def observe_operation(self, operation: str, outcome: str, elapsed_ms: float) -> None:
        """Forward a pre-allowlisted adapter operation without SQL or exception details."""
        if self._metrics is not None:
            with suppress(Exception):
                self._metrics.observe_postgres_operation(operation, outcome, elapsed_ms)

    def observe_lock(self, event: str, elapsed_ms: float = 0) -> None:
        """Forward a bounded worker lock event without lease identity."""
        if self._metrics is not None:
            with suppress(Exception):
                self._metrics.observe_lock(event, elapsed_ms)

    def observe_migration(
        self, operation: str, outcome: str, *, version: int, elapsed_ms: float
    ) -> None:
        """Forward a bounded schema health observation."""
        if self._metrics is not None:
            with suppress(Exception):
                self._metrics.observe_migration(
                    operation, outcome, version=version, elapsed_ms=elapsed_ms
                )

    def observe_cleanup(self, outcome: str, *, deleted: int, elapsed_ms: float) -> None:
        """Forward one bounded cleanup result."""
        if self._metrics is not None:
            with suppress(Exception):
                self._metrics.observe_cleanup(outcome, deleted=deleted, elapsed_ms=elapsed_ms)

    def _pool_failed(self) -> None:
        self._was_unavailable = True
        if self._metrics is not None:
            with suppress(Exception):
                self._metrics.pool_connection_failed()

    def _observe_pool_acquire(self, elapsed_ms: float) -> None:
        if self._metrics is not None:
            with suppress(Exception):
                self._metrics.observe_pool_acquire(elapsed_ms)

    def _observe_pool(self) -> None:
        if self._metrics is None:
            return
        stats_getter = getattr(self._pool, "get_stats", None)
        raw_stats = stats_getter() if callable(stats_getter) else {}
        stats = cast(Mapping[str, int], raw_stats) if isinstance(raw_stats, Mapping) else {}
        size = int(stats.get("pool_size", self._max_size)) if self._opened else 0
        idle = int(stats.get("pool_available", 0)) if self._opened else 0
        waiting = int(stats.get("requests_waiting", 0)) if self._opened else 0
        with suppress(Exception):
            self._metrics.observe_pool(
                size=size,
                used=max(0, size - idle),
                idle=idle,
                wait=waiting,
            )

    @staticmethod
    def register_vector(connection: DatabaseConnection) -> None:
        """Install pgvector codecs after the extension migration exists."""
        try:
            register_vector(connection)
        except Error as exc:
            raise StorageError("PostgreSQL vector type is unavailable") from exc
