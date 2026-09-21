"""Bounded PostgreSQL revision and content garbage collection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from time import monotonic

from psycopg import Error  # pyright: ignore[reportMissingImports]

from vault_rag.errors import StorageError

from .lease import PostgresWorkerLease
from .pool import DatabaseConnection, PostgresPool


@dataclass(frozen=True, slots=True)
class CleanupPolicy:
    """Explicit, conservative retention settings for revision cleanup."""

    enabled: bool = False
    keep_promoted: int = 3
    min_age: timedelta = timedelta(days=7)
    batch_size: int = 100

    def __post_init__(self) -> None:
        if isinstance(self.keep_promoted, bool) or self.keep_promoted < 0:
            raise ValueError("cleanup keep_promoted must be non-negative")
        if self.min_age <= timedelta(0):
            raise ValueError("cleanup min_age must be positive")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError("cleanup batch_size must be positive")


@dataclass(frozen=True, slots=True)
class CleanupResult:
    """Counts from one bounded cleanup pass."""

    revisions_deleted: int
    blobs_deleted: int
    embeddings_deleted: int
    elapsed_ms: int


class PostgresRevisionCleaner:
    """Delete only retention-eligible data while excluding active identities."""

    def __init__(self, pool: PostgresPool) -> None:
        self._pool = pool

    def run(self, vault_id: str, policy: CleanupPolicy) -> CleanupResult:
        """Run one bounded pass under the vault's reconciliation advisory lock."""
        started = monotonic()
        if not policy.enabled:
            result = self._result(0, 0, 0, started)
            self._observe("success", result)
            return result

        lease = PostgresWorkerLease(self._pool, f"sync:{vault_id}")
        if not lease.acquire():
            result = self._result(0, 0, 0, started)
            self._observe("success", result)
            return result
        try:
            with self._pool.connection() as connection, connection.transaction():
                revisions_deleted = self._delete_revisions(connection, vault_id, policy)
                blobs_deleted = self._delete_blobs(connection, policy.batch_size)
                embeddings_deleted = self._delete_embeddings(connection, policy.batch_size)
            result = self._result(revisions_deleted, blobs_deleted, embeddings_deleted, started)
            self._observe("success", result)
            return result
        except Error as exc:
            self._observe("failure", self._result(0, 0, 0, started))
            raise StorageError("could not clean PostgreSQL revision data") from exc
        finally:
            lease.release()

    @staticmethod
    def _result(
        revisions_deleted: int,
        blobs_deleted: int,
        embeddings_deleted: int,
        started: float,
    ) -> CleanupResult:
        return CleanupResult(
            revisions_deleted=revisions_deleted,
            blobs_deleted=blobs_deleted,
            embeddings_deleted=embeddings_deleted,
            elapsed_ms=round((monotonic() - started) * 1000),
        )

    def _observe(self, outcome: str, result: CleanupResult) -> None:
        observer = getattr(self._pool, "observe_cleanup", None)
        if callable(observer):
            try:
                observer(
                    outcome,
                    deleted=result.revisions_deleted
                    + result.blobs_deleted
                    + result.embeddings_deleted,
                    elapsed_ms=result.elapsed_ms,
                )
            except Exception:
                return

    @staticmethod
    def _delete_revisions(
        connection: DatabaseConnection, vault_id: str, policy: CleanupPolicy
    ) -> int:
        rows = connection.execute(
            """
            WITH ranked_promoted AS (
                SELECT revision_id,
                       row_number() OVER (
                           PARTITION BY vault_id ORDER BY promoted_at DESC, revision_id DESC
                       ) AS promoted_rank
                FROM vault_rag.vault_revisions
                WHERE vault_id = %s AND promoted_at IS NOT NULL
            ), candidates AS (
                SELECT revision.revision_id,
                       COALESCE(
                           revision.superseded_at, revision.completed_at, revision.started_at
                       ) AS eligible_at
                FROM vault_rag.vault_revisions AS revision
                LEFT JOIN ranked_promoted ON ranked_promoted.revision_id = revision.revision_id
                JOIN vault_rag.vaults AS vault ON vault.vault_id = revision.vault_id
                WHERE revision.vault_id = %s
                  AND revision.revision_id IS DISTINCT FROM vault.active_revision_id
                  AND COALESCE(
                      revision.superseded_at, revision.completed_at, revision.started_at
                  ) < now() - %s::interval
                  AND (ranked_promoted.promoted_rank IS NULL OR ranked_promoted.promoted_rank > %s)
                  AND revision.state IN ('building', 'failed', 'superseded')
                ORDER BY eligible_at ASC, revision.revision_id ASC
                LIMIT %s
            )
            DELETE FROM vault_rag.vault_revisions AS revision
            USING candidates
            WHERE revision.revision_id = candidates.revision_id
            RETURNING revision.revision_id
            """,
            (vault_id, vault_id, policy.min_age, policy.keep_promoted, policy.batch_size),
        ).fetchall()
        return len(rows)

    @staticmethod
    def _delete_blobs(connection: DatabaseConnection, batch_size: int) -> int:
        rows = connection.execute(
            """
            WITH candidates AS (
                SELECT blob.content_hash
                FROM vault_rag.source_blobs AS blob
                WHERE NOT EXISTS (
                    SELECT 1 FROM vault_rag.revision_sources AS source
                    WHERE source.source_blob_hash = blob.content_hash
                )
                ORDER BY blob.created_at ASC, blob.content_hash ASC
                LIMIT %s
            )
            DELETE FROM vault_rag.source_blobs AS blob
            USING candidates
            WHERE blob.content_hash = candidates.content_hash
            RETURNING blob.content_hash
            """,
            (batch_size,),
        ).fetchall()
        return len(rows)

    @staticmethod
    def _delete_embeddings(connection: DatabaseConnection, batch_size: int) -> int:
        rows = connection.execute(
            """
            WITH candidates AS (
                SELECT embedding.content_hash, embedding.configuration_fingerprint,
                       embedding.observed_fingerprint
                FROM vault_rag.embeddings AS embedding
                WHERE NOT EXISTS (
                    SELECT 1 FROM vault_rag.revision_chunks AS chunk
                    WHERE chunk.content_hash = embedding.content_hash
                      AND chunk.embedding_config_fingerprint = embedding.configuration_fingerprint
                      AND chunk.observed_fingerprint = embedding.observed_fingerprint
                )
                ORDER BY embedding.created_at ASC, embedding.content_hash ASC,
                         embedding.configuration_fingerprint ASC, embedding.observed_fingerprint ASC
                LIMIT %s
            )
            DELETE FROM vault_rag.embeddings AS embedding
            USING candidates
            WHERE embedding.content_hash = candidates.content_hash
              AND embedding.configuration_fingerprint = candidates.configuration_fingerprint
              AND embedding.observed_fingerprint = candidates.observed_fingerprint
            RETURNING embedding.content_hash
            """,
            (batch_size,),
        ).fetchall()
        return len(rows)
