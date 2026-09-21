"""Real-PostgreSQL contract tests for the deterministic exact-search scale harness."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import psycopg  # pyright: ignore[reportMissingImports]
import pytest  # pyright: ignore[reportMissingImports]
from typer.testing import CliRunner

from tests.contract.postgres_target import (
    assert_safe_test_connection,
    validated_postgres_test_dsn,
)
from vault_rag.cli import create_app
from vault_rag.evaluation import ScaleCase, run_postgres_scale
from vault_rag.storage.postgres import PostgresMigrator, PostgresPool


@pytest.fixture
def postgres_dsn() -> str:
    return validated_postgres_test_dsn()


@pytest.fixture
def postgres_pool(postgres_dsn: str) -> Iterator[PostgresPool]:
    with psycopg.connect(postgres_dsn, autocommit=True) as connection:
        assert_safe_test_connection(connection)
        connection.execute("DROP SCHEMA IF EXISTS vault_rag CASCADE")
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    pool = PostgresPool(postgres_dsn, min_size=1, max_size=4, timeout=5.0)
    pool.open(wait=True)
    PostgresMigrator(pool).apply()
    try:
        yield pool
    finally:
        pool.close()


def _physical_index_bytes(pool: PostgresPool) -> int:
    with pool.connection() as connection:
        row = connection.execute(
            """
            SELECT COALESCE(sum(pg_relation_size(indexrelid)), 0) AS index_size_bytes
            FROM pg_stat_user_indexes
            WHERE schemaname = 'vault_rag'
            """
        ).fetchone()
    assert row is not None
    return int(row["index_size_bytes"])


def test_scale_case_rejects_unbounded_or_invalid_values() -> None:
    for kwargs in (
        {"vectors": 0, "concurrency": 1, "query_count": 1, "seed": 1},
        {"vectors": 9, "concurrency": 0, "query_count": 1, "seed": 1},
        {"vectors": 9, "concurrency": 1, "query_count": 0, "seed": 1},
        {"vectors": 100_001, "concurrency": 1, "query_count": 1, "seed": 1},
    ):
        with pytest.raises(ValueError):
            ScaleCase(**kwargs)


def test_small_runs_are_deterministic_exact_finite_and_sanitized(
    postgres_pool: PostgresPool,
) -> None:
    case = ScaleCase(vectors=8, concurrency=3, query_count=12, seed=17)

    first = run_postgres_scale(postgres_pool, case)
    second = run_postgres_scale(postgres_pool, case)

    assert first.dataset_id == second.dataset_id
    assert first.revision_identity == second.revision_identity
    assert tuple(result.query_id for result in first.results) == tuple(
        result.query_id for result in second.results
    )
    assert tuple(result.result_ids for result in first.results) == tuple(
        result.result_ids for result in second.results
    )
    assert tuple(result.target_id for result in first.results) == tuple(
        result.target_id for result in second.results
    )
    assert first.errors == second.errors == 0
    assert first.no_hnsw is second.no_hnsw is True
    assert first.completed_queries == case.query_count
    for result in first.results:
        assert result.result_ids[0] == result.target_id
        assert result.scores == tuple(sorted(result.scores, reverse=True))
        assert result.scores[0] == pytest.approx(1.0, abs=1e-5)
    numeric_values = (
        first.p50_latency_ms,
        first.p95_latency_ms,
        first.p99_latency_ms,
        first.throughput_per_second,
        first.cpu_user_seconds,
        first.cpu_system_seconds,
        first.rss_bytes,
        first.postgres_operation_latency_ms,
        first.postgres_operation_rows,
        first.pool_waits,
        first.pool_connections,
        first.index_size_bytes,
        first.promotion_duration_ms,
        *(result.latency_ms for result in first.results),
    )
    assert all(math.isfinite(value) and value >= 0.0 for value in numeric_values)
    assert all(math.isfinite(score) for result in first.results for score in result.scores)
    serialized = json.dumps(first.model_dump(mode="json"), sort_keys=True).lower()
    for forbidden in ("scale-source", "query_vector", "postgresql://", "password", "secret", "dsn"):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "arguments",
    (
        ["--vectors", "9", "--concurrency", "1"],
        ["--vectors", "10000", "--concurrency", "2"],
    ),
)
def test_operator_cli_rejects_non_matrix_values_without_opening_a_pool(
    arguments: list[str],
) -> None:
    result = CliRunner().invoke(create_app(), ["postgres-scale", *arguments])

    assert result.exit_code != 0
    assert "10" in result.output


def test_pool_waits_are_query_scoped_cumulative_delta(
    postgres_pool: PostgresPool,
    postgres_dsn: str,
) -> None:
    del postgres_pool
    tight_pool = PostgresPool(postgres_dsn, min_size=1, max_size=1, timeout=5.0)
    tight_pool.open(wait=True)
    try:

        def acquire_once() -> None:
            with tight_pool.connection():
                return

        executor = ThreadPoolExecutor(max_workers=2)
        try:
            with tight_pool.connection():
                futures = (executor.submit(acquire_once), executor.submit(acquire_once))
            for future in futures:
                future.result()
        finally:
            executor.shutdown()
        queued_before = tight_pool.stats().get("requests_queued")
        assert queued_before is not None and queued_before > 0

        report = run_postgres_scale(
            tight_pool,
            ScaleCase(vectors=3, concurrency=3, query_count=9, seed=31),
        )

        queued_after = tight_pool.stats().get("requests_queued")
        assert queued_after is not None
        assert report.pool_waits == queued_after - queued_before
        assert report.pool_waits > 0
    finally:
        tight_pool.close()


def test_index_size_is_run_delta_not_total_schema_size(postgres_pool: PostgresPool) -> None:
    run_postgres_scale(
        postgres_pool,
        ScaleCase(vectors=3, concurrency=1, query_count=1, seed=37),
    )
    before = _physical_index_bytes(postgres_pool)

    report = run_postgres_scale(
        postgres_pool,
        ScaleCase(vectors=4, concurrency=1, query_count=1, seed=41),
    )
    after = _physical_index_bytes(postgres_pool)

    assert report.index_size_bytes == after - before
    assert after > report.index_size_bytes
