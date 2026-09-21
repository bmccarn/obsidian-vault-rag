from __future__ import annotations

from vault_rag.storage.postgres import PostgresMigrator, PostgresPool, PostgresWorkerLease


def test_worker_lease_prevents_a_second_session_for_the_same_vault(
    postgres_pool: PostgresPool,
) -> None:
    """A missing session advisory lock would let concurrent workers build one vault."""
    PostgresMigrator(postgres_pool).apply()
    first = PostgresWorkerLease(postgres_pool, "sync:alpha")
    second = PostgresWorkerLease(postgres_pool, "sync:alpha")

    assert first.acquire() is True
    assert second.acquire() is False
    assert first.refresh() is True

    first.release()
    assert second.acquire() is True
    second.release()


def test_worker_leases_for_different_vaults_progress_concurrently(
    postgres_pool: PostgresPool,
) -> None:
    """Hashing every vault to one key would serialize independent reconciliation."""
    PostgresMigrator(postgres_pool).apply()
    alpha = PostgresWorkerLease(postgres_pool, "sync:alpha")
    beta = PostgresWorkerLease(postgres_pool, "sync:beta")

    assert alpha.acquire() is True
    assert beta.acquire() is True
    assert alpha.refresh() is True
    assert beta.refresh() is True

    alpha.release()
    beta.release()


def test_terminated_lock_backend_fences_stale_session_and_allows_replacement(
    postgres_pool: PostgresPool,
) -> None:
    """Continuing after a lost database session could promote a stale build."""
    PostgresMigrator(postgres_pool).apply()
    stale = PostgresWorkerLease(postgres_pool, "sync:alpha")
    replacement = PostgresWorkerLease(postgres_pool, "sync:alpha")

    assert stale.acquire() is True
    assert stale.backend_pid is not None
    with postgres_pool.connection() as connection:
        terminated = connection.execute(
            "SELECT pg_terminate_backend(%s) AS terminated", (stale.backend_pid,)
        ).fetchone()
        connection.commit()
    assert terminated is not None and terminated["terminated"] is True

    assert stale.refresh() is False
    assert replacement.acquire() is True

    stale.release()
    assert replacement.refresh() is True
    replacement.release()


def test_worker_lease_release_is_idempotent_and_session_scoped(
    postgres_pool: PostgresPool,
) -> None:
    """A non-owner or repeated release must not unlock the current lock holder."""
    PostgresMigrator(postgres_pool).apply()
    first = PostgresWorkerLease(postgres_pool, "sync:alpha")
    second = PostgresWorkerLease(postgres_pool, "sync:alpha")

    assert first.acquire() is True
    second.release()
    second.release()

    assert second.acquire() is False
    assert first.refresh() is True

    first.release()
    first.release()
    assert second.acquire() is True
    second.release()
