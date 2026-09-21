from __future__ import annotations

from datetime import UTC, datetime

from vault_rag.service.postgres_state import PostgresSyncStateStore
from vault_rag.service.state import IndexDiagnosticState, IndexReportState, VaultSyncState
from vault_rag.storage.postgres import PostgresMigrator, PostgresPool


def test_postgresql_sync_state_round_trips_bounded_progress(
    postgres_pool: PostgresPool,
) -> None:
    PostgresMigrator(postgres_pool).apply()
    state_store = PostgresSyncStateStore(postgres_pool)
    now = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
    state = VaultSyncState(
        vault_id="vault-a",
        configured_ref="refs/heads/main",
        configured_sha="a" * 40,
        fetched_sha="a" * 40,
        checkout_sha="a" * 40,
        attempted_sha="a" * 40,
        reconciled_sha="a" * 40,
        configured_at=now,
        fetched_at=now,
        checkout_at=now,
        attempted_at=now,
        reconciled_at=now,
        report=IndexReportState(
            total_sources=1,
            added_sources=1,
            changed_sources=0,
            unchanged_sources=0,
            deleted_sources=0,
            ready_chunks=2,
            pending_chunks=0,
            parse_failures=0,
            embedding_failures=0,
            embedding_requests=1,
            blocking_failures=0,
            diagnostics=(
                IndexDiagnosticState(
                    vault_id="vault-a",
                    path="notes/a.md",
                    category="notice",
                    message="indexed",
                ),
            ),
            elapsed_ms=15,
        ),
    )

    state_store.write(state)
    reread = PostgresSyncStateStore(postgres_pool).read("vault-a", "refs/heads/main")

    assert reread.warning is None
    assert reread.state == state


def test_postgresql_sync_state_fails_closed_on_configured_ref_mismatch(
    postgres_pool: PostgresPool,
) -> None:
    PostgresMigrator(postgres_pool).apply()
    state_store = PostgresSyncStateStore(postgres_pool)
    state_store.write(
        VaultSyncState(
            vault_id="vault-a",
            configured_ref="refs/heads/main",
            fetched_sha="a" * 40,
        )
    )

    reread = state_store.read("vault-a", "refs/heads/replacement")

    assert reread.state.configured_ref == "refs/heads/replacement"
    assert reread.state.fetched_sha is None
    assert reread.state.sync_degraded_reason == "state_invalid"
    assert reread.warning == "stored synchronization state is invalid"
