"""PostgreSQL-backed synchronization state shared by API and worker processes."""

from __future__ import annotations

from typing import Any

from psycopg import Error  # pyright: ignore[reportMissingImports]
from psycopg.types.json import Jsonb  # pyright: ignore[reportMissingImports]

from vault_rag.errors import StorageError
from vault_rag.service.state import StateRead, VaultSyncState
from vault_rag.storage.postgres.pool import PostgresPool


class PostgresSyncStateStore:
    """Persist bounded per-vault lifecycle state in PostgreSQL."""

    checkout_independent = True

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    def read(self, vault_id: str, configured_ref: str) -> StateRead:
        initial = VaultSyncState(vault_id=vault_id, configured_ref=configured_ref)
        try:
            with self._pool.connection() as connection:
                row = connection.execute(
                    """
                    SELECT v.*, vr.commit_sha AS active_commit_sha
                    FROM vault_rag.vaults AS v
                    LEFT JOIN vault_rag.vault_revisions AS vr
                        ON vr.revision_id = v.active_revision_id
                    WHERE v.vault_id = %s
                    """,
                    (vault_id,),
                ).fetchone()
        except Error as exc:
            raise StorageError("could not load PostgreSQL synchronization state") from exc
        if row is None:
            return StateRead(state=initial)
        if row["configured_ref"] != configured_ref:
            return self._invalid(initial)
        try:
            report = row["index_report"]
            active_commit_sha = row["active_commit_sha"]
            state = VaultSyncState.model_validate(
                {
                    "vault_id": vault_id,
                    "configured_ref": configured_ref,
                    "configured_sha": row["configured_sha"],
                    "fetched_sha": row["fetched_sha"],
                    "checkout_sha": active_commit_sha or row["checkout_sha"],
                    "attempted_sha": row["attempted_sha"],
                    "reconciled_sha": active_commit_sha or row["reconciled_sha"],
                    "configured_at": row["configured_at"],
                    "fetched_at": row["fetched_at"],
                    "checkout_at": row["checkout_at"],
                    "attempted_at": row["attempted_at"],
                    "reconciled_at": row["reconciled_at"],
                    "report": report,
                    "sync_degraded_reason": row["sync_degradation_category"],
                    "sync_degraded_at": row["sync_degraded_at"],
                }
            )
        except (TypeError, ValueError):
            return self._invalid(initial)
        return StateRead(state=state)

    def write(self, state: VaultSyncState) -> None:
        report: dict[str, Any] | None = (
            None if state.report is None else state.report.model_dump(mode="json")
        )
        try:
            with self._pool.connection() as connection, connection.transaction():
                connection.execute(
                    """
                    INSERT INTO vault_rag.vaults(
                        vault_id, configured_ref, configured_sha, fetched_sha,
                        checkout_sha, attempted_sha, reconciled_sha, configured_at,
                        fetched_at, checkout_at, attempted_at, reconciled_at,
                        index_report, sync_degradation_category,
                        sync_degradation_message, sync_degraded_at,
                        last_fetch_attempt_at, last_fetch_success_at,
                        last_reconciliation_attempt_at,
                        last_reconciliation_success_at, updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, NULL, %s, %s, %s, %s, %s, now()
                    )
                    ON CONFLICT (vault_id) DO UPDATE SET
                        configured_ref = excluded.configured_ref,
                        configured_sha = excluded.configured_sha,
                        fetched_sha = excluded.fetched_sha,
                        checkout_sha = excluded.checkout_sha,
                        attempted_sha = excluded.attempted_sha,
                        reconciled_sha = excluded.reconciled_sha,
                        configured_at = excluded.configured_at,
                        fetched_at = excluded.fetched_at,
                        checkout_at = excluded.checkout_at,
                        attempted_at = excluded.attempted_at,
                        reconciled_at = excluded.reconciled_at,
                        index_report = excluded.index_report,
                        sync_degradation_category = excluded.sync_degradation_category,
                        sync_degradation_message = NULL,
                        sync_degraded_at = excluded.sync_degraded_at,
                        last_fetch_attempt_at = excluded.last_fetch_attempt_at,
                        last_fetch_success_at = excluded.last_fetch_success_at,
                        last_reconciliation_attempt_at =
                            excluded.last_reconciliation_attempt_at,
                        last_reconciliation_success_at =
                            excluded.last_reconciliation_success_at,
                        updated_at = now()
                    """,
                    (
                        state.vault_id,
                        state.configured_ref,
                        state.configured_sha,
                        state.fetched_sha,
                        state.checkout_sha,
                        state.attempted_sha,
                        state.reconciled_sha,
                        state.configured_at,
                        state.fetched_at,
                        state.checkout_at,
                        state.attempted_at,
                        state.reconciled_at,
                        None if report is None else Jsonb(report),
                        state.sync_degraded_reason,
                        state.sync_degraded_at,
                        state.configured_at or state.sync_degraded_at,
                        state.fetched_at,
                        state.attempted_at,
                        state.reconciled_at,
                    ),
                )
        except Error as exc:
            raise StorageError("could not persist PostgreSQL synchronization state") from exc

    @staticmethod
    def _invalid(initial: VaultSyncState) -> StateRead:
        return StateRead(
            state=initial.model_copy(update={"sync_degraded_reason": "state_invalid"}),
            warning="stored synchronization state is invalid",
        )
