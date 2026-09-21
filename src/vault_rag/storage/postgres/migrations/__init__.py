"""Application-owned, checksum-verified PostgreSQL migrations."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib.resources import files
from time import perf_counter
from typing import LiteralString, cast

from psycopg import Error, sql  # pyright: ignore[reportMissingImports]

from vault_rag.errors import StorageError
from vault_rag.storage.postgres.pool import DatabaseConnection, PostgresPool

POSTGRES_SCHEMA_VERSION = 3
_MIGRATION_NAME = re.compile(r"^(?P<version>[0-9]{4})_[a-z0-9_]+\.sql$")
_MIGRATION_LOCK_SQL = "SELECT pg_advisory_lock(hashtextextended('vault-rag-migrations', 0))"
_MIGRATION_UNLOCK_SQL = "SELECT pg_advisory_unlock(hashtextextended('vault-rag-migrations', 0))"
_POSTGRES_ROLE_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


@dataclass(frozen=True, slots=True)
class _Migration:
    version: int
    name: str
    checksum: str
    sql: str


def _load_migrations() -> tuple[_Migration, ...]:
    migrations: list[_Migration] = []
    for resource in files(__package__).iterdir():
        match = _MIGRATION_NAME.fullmatch(resource.name)
        if match is None:
            continue
        payload = resource.read_bytes()
        migrations.append(
            _Migration(
                version=int(match.group("version")),
                name=resource.name,
                checksum=hashlib.sha256(payload).hexdigest(),
                sql=payload.decode("utf-8"),
            )
        )
    migrations.sort(key=lambda migration: migration.version)
    versions = [migration.version for migration in migrations]
    expected = list(range(1, POSTGRES_SCHEMA_VERSION + 1))
    if versions != expected:
        raise StorageError("packaged PostgreSQL migration history is incomplete")
    return tuple(migrations)


class PostgresMigrator:
    """Apply immutable migrations while holding one database-wide session lock."""

    def __init__(self, pool: PostgresPool, *, owner_role: str | None = None) -> None:
        if owner_role is not None and _POSTGRES_ROLE_PATTERN.fullmatch(owner_role) is None:
            raise ValueError("owner_role must be a lowercase PostgreSQL identifier")
        self._pool = pool
        self._owner_role = owner_role

    def apply(self) -> tuple[int, ...]:
        started = perf_counter()
        migrations = _load_migrations()
        applied_now: list[int] = []
        with self._pool.connection() as connection:
            locked = False
            try:
                connection.execute(_MIGRATION_LOCK_SQL)
                connection.commit()
                locked = True
                if self._owner_role is not None:
                    statement = sql.SQL("SET ROLE {}").format(sql.Identifier(self._owner_role))
                    connection.execute(statement)
                    connection.commit()
                self._require_pgvector(connection)
                connection.commit()
                self._bootstrap_history(connection)
                applied = self._read_applied(connection)
                connection.commit()
                self._validate_history(migrations, applied)
                for migration in migrations[len(applied) :]:
                    with connection.transaction():
                        connection.execute(
                            sql.SQL(cast(LiteralString, migration.sql))  # type: ignore[redundant-cast]
                        )
                        connection.execute(
                            "INSERT INTO vault_rag.schema_migrations"
                            "(version, checksum, applied_at) VALUES (%s, %s, now())",
                            (migration.version, migration.checksum),
                        )
                    applied_now.append(migration.version)
            except StorageError:
                self._observe("apply", "failure", 0, started)
                raise
            except Error as exc:
                self._observe("apply", "failure", 0, started)
                raise StorageError("could not apply PostgreSQL migrations") from exc
            finally:
                if locked:
                    try:
                        connection.execute(_MIGRATION_UNLOCK_SQL)
                        connection.commit()
                    except Error:
                        connection.rollback()
        applied_versions = tuple(applied_now)
        self._observe(
            "apply",
            "success",
            applied_versions[-1] if applied_versions else POSTGRES_SCHEMA_VERSION,
            started,
        )
        return applied_versions

    def check(self) -> tuple[int, ...]:
        """Verify compatibility without creating or changing database objects."""
        started = perf_counter()
        migrations = _load_migrations()
        with self._pool.connection() as connection:
            try:
                self._require_pgvector(connection)
                required = connection.execute(
                    """
                    SELECT
                        to_regclass('vault_rag.schema_migrations') AS migrations,
                        to_regclass('vault_rag.vaults') AS vaults,
                        to_regclass('vault_rag.vault_revisions') AS revisions,
                        to_regclass('vault_rag.source_blobs') AS blobs,
                        to_regclass('vault_rag.embeddings') AS embeddings,
                        to_regclass('vault_rag.revision_sources') AS sources,
                        to_regclass('vault_rag.revision_chunks') AS chunks,
                        to_regclass('vault_rag.embedding_queue') AS queue,
                        to_regclass('vault_rag.sync_requests') AS requests
                    """
                ).fetchone()
                if required is None or any(value is None for value in required.values()):
                    raise StorageError("PostgreSQL schema objects are missing")
                applied = self._read_applied(connection)
                self._validate_history(migrations, applied)
                if len(applied) != len(migrations):
                    raise StorageError("PostgreSQL schema migrations are pending")
                versions = tuple(version for version, _checksum in applied)
                self._observe("check", "success", versions[-1] if versions else 0, started)
                return versions
            except StorageError:
                self._observe("check", "failure", 0, started)
                raise
            except Error as exc:
                self._observe("check", "failure", 0, started)
                raise StorageError("could not check PostgreSQL schema compatibility") from exc

    def _observe(self, operation: str, outcome: str, version: int, started: float) -> None:
        observer = getattr(self._pool, "observe_migration", None)
        if callable(observer):
            try:
                observer(
                    operation,
                    outcome,
                    version=version,
                    elapsed_ms=(perf_counter() - started) * 1000,
                )
            except Exception:
                return

    @staticmethod
    def _require_pgvector(connection: DatabaseConnection) -> None:
        installed = connection.execute(
            "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS installed"
        ).fetchone()
        if installed is None or not installed["installed"]:
            raise StorageError("required pgvector extension is not installed")

    @staticmethod
    def _bootstrap_history(connection: DatabaseConnection) -> None:
        with connection.transaction():
            connection.execute("CREATE SCHEMA IF NOT EXISTS vault_rag")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vault_rag.schema_migrations (
                    version integer PRIMARY KEY CHECK (version > 0),
                    checksum text NOT NULL CHECK (checksum ~ '^[0-9a-f]{64}$'),
                    applied_at timestamptz NOT NULL
                )
                """
            )

    @staticmethod
    def _read_applied(connection: DatabaseConnection) -> tuple[tuple[int, str], ...]:
        rows = connection.execute(
            "SELECT version, checksum FROM vault_rag.schema_migrations ORDER BY version"
        ).fetchall()
        return tuple((cast(int, row["version"]), cast(str, row["checksum"])) for row in rows)

    @staticmethod
    def _validate_history(
        migrations: tuple[_Migration, ...],
        applied: tuple[tuple[int, str], ...],
    ) -> None:
        if len(applied) > len(migrations):
            raise StorageError("PostgreSQL schema is newer than this application")
        expected_versions = tuple(range(1, len(applied) + 1))
        if tuple(version for version, _checksum in applied) != expected_versions:
            raise StorageError("PostgreSQL migration history has missing versions")
        for migration, (version, checksum) in zip(migrations, applied, strict=False):
            if migration.version != version:
                raise StorageError("PostgreSQL migration history is incompatible")
            if migration.checksum != checksum:
                raise StorageError(
                    "PostgreSQL migration checksum mismatch",
                    details={"version": version},
                )
