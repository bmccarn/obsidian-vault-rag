"""Database coordination shared by PostgreSQL API and worker processes."""

from __future__ import annotations

from uuid import UUID, uuid4

from psycopg import Error  # pyright: ignore[reportMissingImports]

from vault_rag.errors import StorageError
from vault_rag.storage.postgres.pool import PostgresPool

from .worker import ClaimedSyncRequest


class PostgresServiceCoordinator:
    """Queue sync requests and inspect the active database revision without a checkout."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    def start(self) -> None:
        """API processes have no background synchronization lifecycle."""

    def close(self) -> None:
        """The owning runtime closes the shared pool."""

    def request_sync(self) -> None:
        """Coalesce an all-vault reconciliation request in PostgreSQL."""
        try:
            with self._pool.connection() as connection, connection.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended('vault-rag-sync-request', 0))"
                )
                connection.execute(
                    """
                    INSERT INTO vault_rag.sync_requests(request_id, vault_id, state)
                    SELECT %s, NULL, 'pending'
                    WHERE NOT EXISTS (
                        SELECT 1 FROM vault_rag.sync_requests
                        WHERE vault_id IS NULL AND state IN ('pending', 'claimed')
                    )
                    """,
                    (uuid4(),),
                )
        except Error as exc:
            raise StorageError("could not queue PostgreSQL synchronization request") from exc

    def claim_request(self) -> ClaimedSyncRequest | None:
        """Claim the oldest request with a generation-like ownership token."""
        try:
            with self._pool.connection() as connection, connection.transaction():
                row = connection.execute(
                    """
                    WITH candidate AS (
                        SELECT request_id FROM vault_rag.sync_requests
                        WHERE state = 'pending'
                           OR (state = 'claimed' AND claimed_at <= now() - interval '5 minutes')
                        ORDER BY created_at, request_id
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE vault_rag.sync_requests AS request
                    SET state = 'claimed', claimed_at = now(), claim_token = %s,
                        outcome = NULL, completed_at = NULL
                    FROM candidate
                    WHERE request.request_id = candidate.request_id
                    RETURNING request.request_id, request.claim_token
                    """,
                    (uuid4(),),
                ).fetchone()
        except Error as exc:
            raise StorageError("could not claim PostgreSQL synchronization request") from exc
        if row is None:
            return None
        return ClaimedSyncRequest(
            UUID(str(row["request_id"])),
            UUID(str(row["claim_token"])),
        )

    def complete_request(self, request: ClaimedSyncRequest, *, succeeded: bool) -> None:
        """Record bounded completion for one request owned by the active worker."""
        state = "completed" if succeeded else "failed"
        outcome = "reconciled" if succeeded else "worker_failed"
        try:
            with self._pool.connection() as connection, connection.transaction():
                connection.execute(
                    """
                    UPDATE vault_rag.sync_requests
                    SET state = %s, outcome = %s, completed_at = now(), claim_token = NULL
                    WHERE request_id = %s AND claim_token = %s AND state = 'claimed'
                    """,
                    (state, outcome, request.request_id, request.claim_token),
                )
        except Error as exc:
            raise StorageError("could not complete PostgreSQL synchronization request") from exc
        # A stale owner completing after a request was reclaimed is an expected no-op.

    def head(self, vault_id: str) -> str | None:
        """Return the active immutable revision commit used by database retrieval."""
        try:
            with self._pool.connection() as connection:
                row = connection.execute(
                    """
                    SELECT revision.commit_sha
                    FROM vault_rag.vaults AS vault
                    JOIN vault_rag.vault_revisions AS revision
                      ON revision.revision_id = vault.active_revision_id
                    WHERE vault.vault_id = %s
                    """,
                    (vault_id,),
                ).fetchone()
        except Error as exc:
            raise StorageError("could not inspect active PostgreSQL revision") from exc
        return None if row is None else str(row["commit_sha"])
