from __future__ import annotations

from threading import Barrier, Thread
from traceback import format_exception
from typing import LiteralString, cast

import pytest  # pyright: ignore[reportMissingImports]
from psycopg import sql  # pyright: ignore[reportMissingImports]
from psycopg.conninfo import conninfo_to_dict  # pyright: ignore[reportMissingImports]

from tests.contract.postgres_target import (
    _TARGET_OVERRIDE_ENV as _IMPLEMENTED_TARGET_OVERRIDE_ENV,
)
from tests.contract.postgres_target import (
    assert_safe_test_connection,
    assert_safe_test_target,
    postgres_test_dsn,
)
from vault_rag.errors import StorageError
from vault_rag.storage.postgres.migrations import (
    POSTGRES_SCHEMA_VERSION,
    PostgresMigrator,
    _load_migrations,
)
from vault_rag.storage.postgres.pool import PostgresPool

_REQUIRED_TABLES = {
    "embedding_queue",
    "embeddings",
    "revision_chunks",
    "revision_sources",
    "schema_migrations",
    "source_blobs",
    "sync_requests",
    "vault_revisions",
    "vaults",
}

_EXPECTED_TARGET_OVERRIDE_ENV = (
    "PGDATABASE",
    "PGHOST",
    "PGHOSTADDR",
    "PGOPTIONS",
    "PGPORT",
    "PGSERVICE",
    "PGSERVICEFILE",
    "PGSYSCONFDIR",
)


@pytest.fixture(autouse=True)
def _clear_postgres_target_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _IMPLEMENTED_TARGET_OVERRIDE_ENV:
        monkeypatch.delenv(name, raising=False)


def test_destructive_postgres_fixture_target_environment_contract_is_complete() -> None:
    assert _IMPLEMENTED_TARGET_OVERRIDE_ENV == _EXPECTED_TARGET_OVERRIDE_ENV


def test_destructive_postgres_fixture_refuses_non_test_target() -> None:
    with pytest.raises(RuntimeError, match="test target"):
        assert_safe_test_target("postgresql://vault_rag:password@127.0.0.1:5432/production")


@pytest.mark.parametrize(
    "dsn",
    (
        ("postgresql://vault_rag:password@127.0.0.1:5432/vault_rag_test?hostaddr=198.51.100.10"),
        (
            "postgresql://vault_rag:password@127.0.0.1:5432/vault_rag_test"
            "?options=-c%20search_path%3Devil,pg_catalog"
        ),
        "postgresql://vault_rag:password@127.0.0.1:5432/vault_rag_test?service=unsafe",
        "postgresql://vault_rag:password@localhost:5432/vault_rag_test",
        "postgresql://vault_rag:password@127.0.0.1/vault_rag_test",
        "postgresql:///vault_rag_test?port=5432",
    ),
)
def test_destructive_postgres_fixture_rejects_ambiguous_connection_target(dsn: str) -> None:
    with pytest.raises(RuntimeError, match="test target"):
        assert_safe_test_target(dsn)


@pytest.mark.parametrize("env_name", _EXPECTED_TARGET_OVERRIDE_ENV)
def test_destructive_postgres_fixture_rejects_libpq_target_environment(
    env_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(env_name, "unsafe")

    with pytest.raises(RuntimeError, match=env_name):
        assert_safe_test_target("postgresql://vault_rag:password@127.0.0.1:5432/vault_rag_test")


def test_destructive_postgres_fixture_verifies_connected_database() -> None:
    class WrongDatabaseConnection:
        def execute(self, _query: str) -> WrongDatabaseConnection:
            return self

        def fetchone(self) -> tuple[str]:
            return ("vault_rag",)

    with pytest.raises(RuntimeError, match="connected database"):
        assert_safe_test_connection(WrongDatabaseConnection())


def test_destructive_postgres_fixture_qualifies_connected_database_check() -> None:
    queries: list[str] = []

    class TestDatabaseConnection:
        def execute(self, query: str) -> TestDatabaseConnection:
            queries.append(query)
            return self

        def fetchone(self) -> tuple[str]:
            return ("vault_rag_test",)

    assert_safe_test_connection(TestDatabaseConnection())

    assert queries == ["SELECT pg_catalog.current_database()"]


def test_destructive_postgres_fixture_does_not_chain_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = "fixture-secret"

    def fail_to_parse(_dsn: str) -> dict[str, str]:
        raise ValueError(f"invalid DSN containing {password}")

    monkeypatch.setattr("tests.contract.postgres_target.conninfo_to_dict", fail_to_parse)

    with pytest.raises(RuntimeError) as raised:
        assert_safe_test_target("invalid")

    rendered = "".join(format_exception(raised.type, raised.value, raised.tb))
    assert password not in rendered


def test_postgres_dsn_skips_when_optional_and_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VAULT_RAG_TEST_POSTGRES_DSN", raising=False)
    monkeypatch.delenv("VAULT_RAG_REQUIRE_POSTGRES", raising=False)

    with pytest.raises(pytest.skip.Exception, match="not configured"):
        postgres_test_dsn()


def test_postgres_dsn_fails_when_required_and_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VAULT_RAG_TEST_POSTGRES_DSN", raising=False)
    monkeypatch.setenv("VAULT_RAG_REQUIRE_POSTGRES", "1")

    with pytest.raises(pytest.fail.Exception, match="required"):
        postgres_test_dsn()


def test_migrations_create_required_schema_and_are_idempotent(
    postgres_pool: PostgresPool,
) -> None:
    migrator = PostgresMigrator(postgres_pool)

    assert migrator.apply() == tuple(range(1, POSTGRES_SCHEMA_VERSION + 1))
    assert migrator.apply() == ()

    with postgres_pool.connection() as connection:
        tables = {
            row["tablename"]
            for row in connection.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'vault_rag'"
            ).fetchall()
        }
        vector_extension = connection.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        versions = connection.execute(
            "SELECT version FROM vault_rag.schema_migrations ORDER BY version"
        ).fetchall()

    assert tables >= _REQUIRED_TABLES
    assert vector_extension is not None
    assert [row["version"] for row in versions] == list(range(1, POSTGRES_SCHEMA_VERSION + 1))
    assert migrator.check() == tuple(range(1, POSTGRES_SCHEMA_VERSION + 1))


def test_concurrent_migration_runners_apply_each_version_once(postgres_pool: PostgresPool) -> None:
    barrier = Barrier(2)
    applied: list[tuple[int, ...]] = []

    def migrate() -> None:
        barrier.wait()
        applied.append(PostgresMigrator(postgres_pool).apply())

    first = Thread(target=migrate)
    second = Thread(target=migrate)
    first.start()
    second.start()
    first.join()
    second.join()

    assert sorted(applied) == [(), tuple(range(1, POSTGRES_SCHEMA_VERSION + 1))]


def test_second_migration_removes_the_legacy_worker_lease_table(
    postgres_pool: PostgresPool,
) -> None:
    """Mutating migration 0001 would reject deployed databases by checksum."""
    first, second, third = _load_migrations()
    with postgres_pool.connection() as connection, connection.transaction():
        PostgresMigrator._bootstrap_history(connection)
        connection.execute(sql.SQL(cast(LiteralString, first.sql)))
        connection.execute(
            "INSERT INTO vault_rag.schema_migrations(version, checksum, applied_at) "
            "VALUES (%s, %s, now())",
            (first.version, first.checksum),
        )
        assert connection.execute(
            "SELECT to_regclass('vault_rag.worker_leases') IS NOT NULL AS has_table"
        ).fetchone() == {"has_table": True}

    assert PostgresMigrator(postgres_pool).apply() == (second.version, third.version)
    with postgres_pool.connection() as connection:
        assert connection.execute(
            "SELECT to_regclass('vault_rag.worker_leases') IS NOT NULL AS has_table"
        ).fetchone() == {"has_table": False}


def test_third_migration_adds_cleanup_reference_indexes(
    postgres_pool: PostgresPool,
) -> None:
    PostgresMigrator(postgres_pool).apply()

    with postgres_pool.connection() as connection:
        indexes = {
            row["indexname"]
            for row in connection.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'vault_rag' AND indexname = ANY(%s)",
                (
                    [
                        "revision_sources_source_blob_hash_idx",
                        "revision_chunks_embedding_identity_idx",
                    ],
                ),
            ).fetchall()
        }

    assert indexes == {
        "revision_sources_source_blob_hash_idx",
        "revision_chunks_embedding_identity_idx",
    }


def test_migrations_reject_changed_applied_sql(postgres_pool: PostgresPool) -> None:
    migrator = PostgresMigrator(postgres_pool)
    migrator.apply()
    with postgres_pool.connection() as connection:
        connection.execute(
            "UPDATE vault_rag.schema_migrations SET checksum = %s WHERE version = 1",
            ("0" * 64,),
        )

    with pytest.raises(StorageError, match="checksum"):
        migrator.apply()


def test_migrations_do_not_expose_credentials_in_errors(clean_postgres: str) -> None:
    password = conninfo_to_dict(clean_postgres).get("password")
    pool = PostgresPool(clean_postgres, min_size=1, max_size=1, timeout=2.0)
    pool.open(wait=True)
    try:
        with pool.connection() as connection:
            connection.execute("CREATE SCHEMA vault_rag")
            connection.execute(
                "CREATE TABLE vault_rag.schema_migrations(version integer PRIMARY KEY)"
            )
        with pytest.raises(StorageError) as raised:
            PostgresMigrator(pool).apply()
    finally:
        pool.close()
    rendered = "".join(format_exception(raised.type, raised.value, raised.tb))
    if password:
        for marker in (
            f":{password}@",
            f"password={password}",
            f"password='{password}'",
            f'password="{password}"',
        ):
            assert marker not in rendered
    assert clean_postgres not in rendered


def test_migrations_set_configured_object_owner(postgres_pool: PostgresPool) -> None:
    with postgres_pool.connection() as connection:
        row = connection.execute("SELECT current_user AS role").fetchone()
    assert row is not None
    owner_role = str(row["role"])

    PostgresMigrator(postgres_pool, owner_role=owner_role).apply()

    with postgres_pool.connection() as connection:
        owners = connection.execute(
            """
            SELECT pg_get_userbyid(namespace.nspowner) AS schema_owner,
                   pg_get_userbyid(relation.relowner) AS table_owner
            FROM pg_namespace AS namespace
            JOIN pg_class AS relation ON relation.relnamespace = namespace.oid
            WHERE namespace.nspname = 'vault_rag'
              AND relation.relname = 'vaults'
            """
        ).fetchone()
    assert owners == {"schema_owner": owner_role, "table_owner": owner_role}


def test_schema_check_rejects_any_missing_required_table(
    postgres_pool: PostgresPool,
) -> None:
    migrator = PostgresMigrator(postgres_pool)
    migrator.apply()
    with postgres_pool.connection() as connection:
        connection.execute("DROP TABLE vault_rag.source_blobs CASCADE")

    with pytest.raises(StorageError, match="schema objects"):
        migrator.check()


def test_migration_requires_preinstalled_pgvector(postgres_pool: PostgresPool) -> None:
    with postgres_pool.connection() as connection:
        connection.execute("DROP EXTENSION IF EXISTS vector CASCADE")
    try:
        with pytest.raises(StorageError, match="pgvector extension"):
            PostgresMigrator(postgres_pool).apply()
    finally:
        with postgres_pool.connection() as connection:
            connection.execute("CREATE EXTENSION vector")


def test_each_migration_commits_before_a_later_migration_fails(
    postgres_pool: PostgresPool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vault_rag.storage.postgres.migrations as migrations_module

    first = migrations_module._load_migrations()
    broken_version = POSTGRES_SCHEMA_VERSION + 1
    broken = migrations_module._Migration(
        broken_version,
        f"{broken_version:04d}_broken.sql",
        "0" * 64,
        "INVALID SQL",
    )
    monkeypatch.setattr(migrations_module, "_MIGRATION_UNLOCK_SQL", "INVALID SQL")
    monkeypatch.setattr(migrations_module, "_load_migrations", lambda: (*first, broken))

    with pytest.raises(StorageError, match="could not apply"):
        PostgresMigrator(postgres_pool).apply()

    with postgres_pool.connection() as connection:
        applied = connection.execute(
            "SELECT version FROM vault_rag.schema_migrations ORDER BY version"
        ).fetchall()
    assert [row["version"] for row in applied] == list(range(1, POSTGRES_SCHEMA_VERSION + 1))
