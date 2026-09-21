from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from traceback import format_exception
from typing import Any, cast

import pytest  # pyright: ignore[reportMissingImports]
from psycopg import Error  # pyright: ignore[reportMissingImports]
from psycopg_pool import PoolTimeout  # pyright: ignore[reportMissingImports]

from vault_rag.errors import StorageError
from vault_rag.service.observability import ServiceMetrics
from vault_rag.storage.postgres.pool import PostgresPool


class FakePool:
    def __init__(self, conninfo: str, **kwargs: Any) -> None:
        self.conninfo = conninfo
        self.kwargs = kwargs
        self.open_calls = 0
        self.wait_calls: list[float] = []
        self.close_calls = 0
        self.connection_calls: list[float | None] = []

    def open(self) -> None:
        self.open_calls += 1

    def wait(self, timeout: float) -> None:
        self.wait_calls.append(timeout)

    def connection(self, timeout: float | None = None) -> Any:
        self.connection_calls.append(timeout)
        return nullcontext("connection")

    def close(self) -> None:
        self.close_calls += 1


def test_pool_is_bounded_explicitly_opened_and_never_repr_exposes_credentials() -> None:
    created: list[FakePool] = []

    def factory(conninfo: str, **kwargs: Any) -> FakePool:
        pool = FakePool(conninfo, **kwargs)
        created.append(pool)
        return pool

    dsn = "postgresql://vault_rag:top-secret@db.internal:5432/vault_rag?sslmode=require"
    pool = PostgresPool(
        dsn,
        min_size=2,
        max_size=7,
        timeout=1.5,
        statement_timeout=2.25,
        connect_timeout=3.0,
        lock_timeout=4.0,
        idle_transaction_timeout=6.0,
        pool_factory=factory,
    )

    assert created[0].kwargs["open"] is False
    assert created[0].kwargs["min_size"] == 2
    assert created[0].kwargs["max_size"] == 7
    assert created[0].kwargs["timeout"] == 1.5
    assert created[0].kwargs["kwargs"]["connect_timeout"] == 3
    assert created[0].kwargs["kwargs"]["options"] == (
        "-c statement_timeout=2250 -c lock_timeout=4000 -c idle_in_transaction_session_timeout=6000"
    )
    assert "top-secret" not in repr(pool)
    assert dsn not in repr(pool)
    assert pool.database_location == "postgresql://db.internal:5432/vault_rag"

    pool.open(wait=True)
    checkout = pool.checkout()
    assert checkout.__enter__() == "connection"
    assert checkout.__exit__(None, None, None) is False
    pool.close()

    assert created[0].open_calls == 1
    assert created[0].wait_calls == [1.5]
    assert created[0].connection_calls == [1.5]
    assert created[0].close_calls == 1

    pool.close()
    pool.open(wait=True)
    pool.open(wait=True)
    pool.close()
    pool.close()
    assert created[0].open_calls == 2
    assert created[0].close_calls == 2


def test_pool_rejects_unusable_dsn_and_access_before_open() -> None:
    with pytest.raises(ValueError):
        PostgresPool("", min_size=1, max_size=1, timeout=1)
    with pytest.raises(ValueError):
        PostgresPool("postgresql://[", min_size=1, max_size=1, timeout=1)

    pool = PostgresPool(
        "postgresql://127.0.0.1/vault_rag",
        min_size=1,
        max_size=1,
        timeout=1,
        pool_factory=FakePool,
    )
    pool.close()
    with pytest.raises(StorageError, match="not open"), pool.connection():
        raise AssertionError("unreachable")


def test_pool_invalid_dsn_does_not_chain_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    password = "pool-secret"

    def fail_to_parse(_dsn: str) -> dict[str, str]:
        raise Error(f"invalid DSN containing {password}")

    monkeypatch.setattr("vault_rag.storage.postgres.pool.conninfo_to_dict", fail_to_parse)

    with pytest.raises(ValueError) as raised:
        PostgresPool("invalid", min_size=1, max_size=1, timeout=1)

    rendered = "".join(format_exception(raised.type, raised.value, raised.tb))
    assert password not in rendered


def test_pool_translates_open_connection_and_vector_registration_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingOpenPool(FakePool):
        def open(self) -> None:
            raise PoolTimeout

    failing_open = PostgresPool(
        "postgresql://127.0.0.1/vault_rag",
        min_size=1,
        max_size=1,
        timeout=1,
        pool_factory=FailingOpenPool,
    )
    with pytest.raises(StorageError, match="could not open"):
        failing_open.open()

    class FailingContext(AbstractContextManager[object]):
        def __enter__(self) -> object:
            raise PoolTimeout

        def __exit__(self, *_args: object) -> None:
            return None

    class FailingConnectionPool(FakePool):
        def connection(self, timeout: float | None = None) -> AbstractContextManager[object]:
            self.connection_calls.append(timeout)
            return FailingContext()

    failing_connection = PostgresPool(
        "postgresql://127.0.0.1/vault_rag",
        min_size=1,
        max_size=1,
        timeout=1,
        pool_factory=cast(Any, FailingConnectionPool),
    )
    failing_connection.open()
    with pytest.raises(StorageError, match="unavailable"), failing_connection.connection():
        raise AssertionError("unreachable")

    def fail_registration(_connection: object) -> None:
        raise Error("missing vector type")

    monkeypatch.setattr("vault_rag.storage.postgres.pool.register_vector", fail_registration)
    with pytest.raises(StorageError, match="vector type"):
        PostgresPool.register_vector(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("min_size", "max_size", "timeout", "statement_timeout"),
    [
        (-1, 1, 1.0, 1.0),
        (2, 1, 1.0, 1.0),
        (1, 1, 0.0, 1.0),
        (True, 1, 1.0, 1.0),
        (1, 1, 1.0, 0.0),
    ],
)
def test_pool_rejects_invalid_bounds(
    min_size: int, max_size: int, timeout: float, statement_timeout: float
) -> None:
    with pytest.raises(ValueError):
        PostgresPool(
            "postgresql://127.0.0.1/vault_rag",
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            statement_timeout=statement_timeout,
        )


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql://db.internal/vault_rag",
        "postgresql://db.internal/vault_rag?sslmode=disable",
        "postgresql://db.internal/vault_rag?sslmode=prefer",
        "postgresql:///vault_rag?hostaddr=10.23.4.5",
        "postgresql://localhost/vault_rag?hostaddr=10.23.4.5",
    ],
)
def test_pool_rejects_insecure_non_loopback_transport_without_acknowledgement(dsn: str) -> None:
    with pytest.raises(ValueError, match="secure transport"):
        PostgresPool(dsn, min_size=1, max_size=1, timeout=1, pool_factory=FakePool)


def test_pool_allows_loopback_or_explicit_insecure_transport() -> None:
    PostgresPool(
        "postgresql://127.0.0.1/vault_rag",
        min_size=1,
        max_size=1,
        timeout=1,
        pool_factory=FakePool,
    )
    PostgresPool(
        "postgresql:///vault_rag?hostaddr=127.0.0.1",
        min_size=1,
        max_size=1,
        timeout=1,
        pool_factory=FakePool,
    )
    PostgresPool(
        "postgresql://db.internal/vault_rag",
        min_size=1,
        max_size=1,
        timeout=1,
        allow_insecure_transport=True,
        pool_factory=FakePool,
    )


def test_pool_metrics_change_through_open_checkout_and_connection_failure() -> None:
    class MeasuredPool(FakePool):
        def get_stats(self) -> dict[str, int]:
            return {"pool_size": 2, "pool_available": 1, "requests_waiting": 0}

    metrics = ServiceMetrics()
    pool = PostgresPool(
        "postgresql://127.0.0.1/vault_rag",
        min_size=1,
        max_size=2,
        timeout=1,
        pool_factory=MeasuredPool,
        metrics=metrics,
    )
    pool.open()
    with pool.connection():
        checked_out = metrics.render().decode()
    pool.close()
    with pytest.raises(StorageError), pool.connection():
        pass

    rendered = metrics.render().decode()
    assert "vault_rag_pool_used 1.0" in checked_out
    assert "vault_rag_pool_size 0.0" in rendered
    assert "vault_rag_pool_acquire_duration_seconds_count 1.0" in rendered
    assert "vault_rag_pool_connection_failure_total 1.0" in rendered


@pytest.mark.parametrize(
    "pool_factory",
    (
        FakePool,
        type(
            "NonMappingStatsPool",
            (FakePool,),
            {"get_stats": lambda _self: cast(object, [])},
        ),
    ),
)
def test_pool_stats_fails_closed_without_a_mapping_stats_surface(pool_factory: Any) -> None:
    pool = PostgresPool(
        "postgresql://vault_rag:top-secret@127.0.0.1/vault_rag",
        min_size=1,
        max_size=1,
        timeout=1,
        pool_factory=pool_factory,
    )

    with pytest.raises(StorageError, match="statistics are unavailable") as raised:
        pool.stats()

    assert "top-secret" not in str(raised.value)
    assert "postgresql://" not in str(raised.value)
