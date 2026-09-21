from __future__ import annotations

from collections.abc import Iterator

import psycopg  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]

from tests.contract.postgres_target import (
    assert_safe_test_connection,
    validated_postgres_test_dsn,
)
from vault_rag.storage.postgres.pool import PostgresPool


@pytest.fixture
def postgres_dsn() -> str:
    return validated_postgres_test_dsn()


@pytest.fixture
def clean_postgres(postgres_dsn: str) -> Iterator[str]:
    with psycopg.connect(postgres_dsn, autocommit=True) as connection:
        assert_safe_test_connection(connection)
        connection.execute("DROP SCHEMA IF EXISTS vault_rag CASCADE")
        connection.execute("DROP SCHEMA public CASCADE")
        connection.execute("CREATE SCHEMA public")
        connection.execute("CREATE EXTENSION vector")
    yield postgres_dsn


@pytest.fixture
def postgres_pool(clean_postgres: str) -> Iterator[PostgresPool]:
    pool = PostgresPool(clean_postgres, min_size=1, max_size=4, timeout=2.0)
    pool.open(wait=True)
    try:
        yield pool
    finally:
        pool.close()
