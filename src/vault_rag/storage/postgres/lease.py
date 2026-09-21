"""Session advisory locks for per-vault PostgreSQL reconciliation."""

from __future__ import annotations

from contextlib import AbstractContextManager
from hashlib import sha256
from threading import RLock
from time import perf_counter

from psycopg import Error  # pyright: ignore[reportMissingImports]

from vault_rag.errors import StorageError
from vault_rag.storage.ports import RevisionBuildStore

from .pool import DatabaseConnection, PostgresPool

_LOCK_NAMESPACE = b"vault-rag:worker-lock:v1"


def _lock_key(lease_name: str) -> int:
    """Derive a stable signed PostgreSQL advisory-lock key for one vault."""
    digest = sha256(_LOCK_NAMESPACE + b"\0" + lease_name.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=True)


class PostgresWorkerLease:
    """Dedicated PostgreSQL session advisory lock for one reconciliation pass."""

    def __init__(self, pool: PostgresPool, lease_name: str) -> None:
        if not lease_name or len(lease_name) > 100:
            raise ValueError("lease name is invalid")
        self._pool = pool
        self._lease_name = lease_name
        self._lock_key = _lock_key(lease_name)
        self._connection_context: AbstractContextManager[DatabaseConnection] | None = None
        self._connection: DatabaseConnection | None = None
        self._backend_pid: int | None = None
        self._lock = RLock()

    @property
    def backend_pid(self) -> int | None:
        """Return the retained lock session's PostgreSQL backend PID."""
        return self._backend_pid

    def acquire(self) -> bool:
        with self._lock:
            if self._connection is not None:
                return self.refresh()

            started = perf_counter()
            connection_context = self._pool.checkout()
            try:
                connection = connection_context.__enter__()
            except StorageError:
                self._observe("loss")
                raise
            try:
                row = connection.execute(
                    """
                    SELECT pg_backend_pid() AS backend_pid,
                           pg_try_advisory_lock(%s) AS acquired
                    """,
                    (self._lock_key,),
                ).fetchone()
                connection.commit()
            except Error as exc:
                self._return_connection(connection_context, exc)
                self._observe("loss")
                raise StorageError("could not acquire PostgreSQL worker lease") from exc

            self._observe("wait", (perf_counter() - started) * 1000)
            if row is None or not bool(row["acquired"]):
                self._return_connection(connection_context)
                self._observe("busy")
                return False

            self._connection_context = connection_context
            self._connection = connection
            self._backend_pid = int(row["backend_pid"])
            return True

    def refresh(self) -> bool:
        with self._lock:
            connection = self._connection
            connection_context = self._connection_context
            backend_pid = self._backend_pid
            if connection is None or connection_context is None or backend_pid is None:
                return False

            try:
                row = connection.execute(
                    """
                    SELECT pg_backend_pid() AS backend_pid,
                           EXISTS (
                               SELECT 1
                               FROM pg_locks
                               WHERE locktype = 'advisory'
                                 AND pid = pg_backend_pid()
                                 AND granted
                                 AND classid =
                                     ((%s::bigint >> 32) & 4294967295)::oid
                                 AND objid =
                                     (%s::bigint & 4294967295)::oid
                                 AND objsubid = 1
                           ) AS held
                    """,
                    (self._lock_key, self._lock_key),
                ).fetchone()
                connection.commit()
            except Error as exc:
                self._observe("loss")
                self._clear_connection()
                self._return_connection(connection_context, exc)
                return False

            held = row is not None and int(row["backend_pid"]) == backend_pid and bool(row["held"])
            if not held:
                self._observe("loss")
                self._clear_connection()
                self._return_connection(connection_context)
            return held

    def promote(
        self,
        revision: RevisionBuildStore,
        *,
        lexical_complete: bool,
        fully_reconciled: bool,
    ) -> None:
        with self._lock:
            connection = self._connection
            if connection is None:
                raise StorageError("PostgreSQL reconciliation lease was lost")
            from .index_store import PostgresIndexStore

            if not isinstance(revision, PostgresIndexStore):
                raise StorageError("PostgreSQL reconciliation requires a PostgreSQL revision")
            revision._promote_on(
                connection,
                self._lock_key,
                lexical_complete=lexical_complete,
                fully_reconciled=fully_reconciled,
            )

    def release(self) -> None:
        with self._lock:
            connection = self._connection
            connection_context = self._connection_context
            self._clear_connection()
            if connection is None or connection_context is None:
                return

            try:
                row = connection.execute(
                    "SELECT pg_advisory_unlock(%s) AS released", (self._lock_key,)
                ).fetchone()
                if row is None or not bool(row["released"]):
                    raise StorageError("PostgreSQL worker lease was not held by its session")
                connection.commit()
            except Error as exc:
                self._return_connection(connection_context, exc)
                raise StorageError("could not release PostgreSQL worker lease") from exc
            except StorageError as exc:
                self._return_connection(connection_context, exc)
                raise
            self._return_connection(connection_context)

    def _observe(self, event: str, elapsed_ms: float = 0) -> None:
        observer = getattr(self._pool, "observe_lock", None)
        if callable(observer):
            try:
                observer(event, elapsed_ms)
            except Exception:
                return

    def _clear_connection(self) -> None:
        self._connection_context = None
        self._connection = None
        self._backend_pid = None

    @staticmethod
    def _return_connection(
        connection_context: AbstractContextManager[DatabaseConnection],
        error: BaseException | None = None,
    ) -> None:
        try:
            if error is None:
                connection_context.__exit__(None, None, None)
            else:
                connection_context.__exit__(type(error), error, error.__traceback__)
        except (Error, StorageError):
            # A failed backend is already unusable; psycopg-pool discards it.
            return
