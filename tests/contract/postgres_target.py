from __future__ import annotations

import os
from ipaddress import ip_address
from typing import Any

import psycopg  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]
from psycopg.conninfo import conninfo_to_dict  # pyright: ignore[reportMissingImports]

_TARGET_OVERRIDE_ENV = (
    "PGDATABASE",
    "PGHOST",
    "PGHOSTADDR",
    "PGOPTIONS",
    "PGPORT",
    "PGSERVICE",
    "PGSERVICEFILE",
    "PGSYSCONFDIR",
)
_TEST_DATABASE = "vault_rag_test"


def postgres_test_dsn() -> str:
    """Return the configured test DSN, failing closed in PostgreSQL CI jobs."""
    dsn = os.environ.get("VAULT_RAG_TEST_POSTGRES_DSN")
    if dsn is not None:
        return dsn
    if os.environ.get("VAULT_RAG_REQUIRE_POSTGRES") == "1":
        pytest.fail("VAULT_RAG_TEST_POSTGRES_DSN is required by this test run")
    pytest.skip("VAULT_RAG_TEST_POSTGRES_DSN is not configured")
    raise AssertionError("pytest.skip returned unexpectedly")


def assert_safe_test_target(dsn: str) -> None:
    """Require an explicit loopback DSN for the dedicated test database."""
    overridden = next((name for name in _TARGET_OVERRIDE_ENV if name in os.environ), None)
    if overridden is not None:
        raise RuntimeError(f"PostgreSQL test target is invalid: unset {overridden}")
    try:
        fields = conninfo_to_dict(dsn)
        raw_host = fields.get("host")
        raw_database = fields.get("dbname")
        raw_port = fields.get("port")
        if (
            raw_host is None
            or raw_database is None
            or raw_port is None
            or fields.get("hostaddr") is not None
            or fields.get("options") is not None
            or fields.get("service") is not None
        ):
            raise ValueError("ambiguous PostgreSQL test target")
        host = str(raw_host).strip("[]").lower()
        is_loopback = ip_address(host).is_loopback
        database = str(raw_database)
    except (ValueError, psycopg.Error):
        raise RuntimeError("PostgreSQL test target is invalid") from None
    if not is_loopback or database != _TEST_DATABASE:
        raise RuntimeError("refusing destructive cleanup against a non-test target")


def validated_postgres_test_dsn() -> str:
    """Return a configured DSN only after destructive-target validation."""
    dsn = postgres_test_dsn()
    assert_safe_test_target(dsn)
    return dsn


def assert_safe_test_connection(connection: Any) -> None:
    """Verify the connected database before issuing destructive fixture SQL."""
    row = connection.execute("SELECT pg_catalog.current_database()").fetchone()
    if row is None or str(row[0]) != _TEST_DATABASE:
        raise RuntimeError("refusing destructive cleanup against the connected database")
