from __future__ import annotations

from threading import Barrier, Thread
from uuid import uuid4

from vault_rag.service.postgres_coordinator import PostgresServiceCoordinator
from vault_rag.storage.postgres import PostgresMigrator, PostgresPool


def test_sync_requests_are_coalesced_claimed_and_completed(
    postgres_pool: PostgresPool,
) -> None:
    PostgresMigrator(postgres_pool).apply()
    coordinator = PostgresServiceCoordinator(postgres_pool)

    coordinator.request_sync()
    coordinator.request_sync()

    with postgres_pool.connection() as connection:
        pending = connection.execute(
            "SELECT count(*) AS count FROM vault_rag.sync_requests WHERE state = 'pending'"
        ).fetchone()
    assert pending is not None and pending["count"] == 1

    request = coordinator.claim_request()
    assert request is not None
    assert coordinator.claim_request() is None

    coordinator.complete_request(request, succeeded=True)
    with postgres_pool.connection() as connection:
        completed = connection.execute(
            "SELECT state, outcome FROM vault_rag.sync_requests WHERE request_id = %s",
            (request.request_id,),
        ).fetchone()
    assert completed == {"state": "completed", "outcome": "reconciled"}


def test_two_sessions_exclusively_claim_one_sync_request(postgres_pool: PostgresPool) -> None:
    PostgresMigrator(postgres_pool).apply()
    PostgresServiceCoordinator(postgres_pool).request_sync()
    barrier = Barrier(2)
    claims: list[object] = []

    def claim() -> None:
        barrier.wait()
        claims.append(PostgresServiceCoordinator(postgres_pool).claim_request())

    first = Thread(target=claim)
    second = Thread(target=claim)
    first.start()
    second.start()
    first.join()
    second.join()

    assert sum(claim is not None for claim in claims) == 1


def test_reclaimed_sync_request_fences_stale_completion(
    postgres_pool: PostgresPool,
) -> None:
    PostgresMigrator(postgres_pool).apply()
    coordinator = PostgresServiceCoordinator(postgres_pool)
    coordinator.request_sync()
    stale_claim = coordinator.claim_request()
    assert stale_claim is not None

    with postgres_pool.connection() as connection, connection.transaction():
        connection.execute(
            "UPDATE vault_rag.sync_requests "
            "SET claimed_at = now() - interval '6 minutes' "
            "WHERE request_id = %s",
            (stale_claim.request_id,),
        )

    current_claim = coordinator.claim_request()
    assert current_claim is not None
    assert current_claim.request_id == stale_claim.request_id
    assert current_claim.claim_token != stale_claim.claim_token

    coordinator.complete_request(stale_claim, succeeded=True)
    with postgres_pool.connection() as connection:
        still_claimed = connection.execute(
            "SELECT state, claim_token FROM vault_rag.sync_requests WHERE request_id = %s",
            (current_claim.request_id,),
        ).fetchone()
    assert still_claimed == {
        "state": "claimed",
        "claim_token": current_claim.claim_token,
    }

    coordinator.complete_request(current_claim, succeeded=True)


def test_service_coordinator_reports_active_database_revision(
    postgres_pool: PostgresPool,
) -> None:
    PostgresMigrator(postgres_pool).apply()
    sha = "a" * 40
    with postgres_pool.connection() as connection, connection.transaction():
        revision_id = connection.execute(
            """
            INSERT INTO vault_rag.vaults(vault_id, configured_ref)
            VALUES ('vault-a', 'refs/heads/main')
            RETURNING vault_id
            """
        ).fetchone()
        assert revision_id is not None
        revision_id = uuid4()
        active = connection.execute(
            """
            INSERT INTO vault_rag.vault_revisions(revision_id, vault_id, commit_sha, state)
            VALUES (%s, 'vault-a', %s, 'building')
            RETURNING revision_id
            """,
            (revision_id, sha),
        ).fetchone()
        assert active is not None
        connection.execute(
            "UPDATE vault_rag.vaults SET active_revision_id = %s WHERE vault_id = 'vault-a'",
            (active["revision_id"],),
        )

    assert PostgresServiceCoordinator(postgres_pool).head("vault-a") == sha
    assert PostgresServiceCoordinator(postgres_pool).head("missing") is None
