from __future__ import annotations

from datetime import timedelta
from uuid import UUID, uuid4

from vault_rag.storage.postgres.cleanup import CleanupPolicy, PostgresRevisionCleaner
from vault_rag.storage.postgres.lease import PostgresWorkerLease
from vault_rag.storage.postgres.migrations import PostgresMigrator
from vault_rag.storage.postgres.pool import PostgresPool


def _revision(
    pool: PostgresPool,
    vault_id: str,
    revision_id: UUID,
    state: str,
    *,
    age_days: int,
    promoted: bool = False,
) -> None:
    with pool.connection() as connection:
        connection.execute(
            """
            INSERT INTO vault_rag.vault_revisions(
                revision_id, vault_id, commit_sha, state, manifest_fingerprint,
                parser_fingerprint, chunker_fingerprint, embedding_config_fingerprint,
                started_at, completed_at, promoted_at, superseded_at
            ) VALUES (
                %s, %s, %s, %s, 'manifest', 'parser', 'chunker', 'embedding',
                now() - make_interval(days => %s),
                CASE WHEN %s THEN now() - make_interval(days => %s) ELSE NULL END,
                CASE WHEN %s THEN now() - make_interval(days => %s) ELSE NULL END,
                CASE WHEN %s THEN now() - make_interval(days => %s) ELSE NULL END
            )
            """,
            (
                revision_id,
                vault_id,
                "a" * 40,
                state,
                age_days,
                state != "building",
                age_days,
                promoted,
                age_days,
                promoted,
                age_days,
            ),
        )


def _source_and_embedding(pool: PostgresPool, revision_id: UUID, content_hash: str) -> None:
    config_fingerprint = "config"
    observed_fingerprint = "observed"
    with pool.connection() as connection:
        connection.execute(
            """
            INSERT INTO vault_rag.source_blobs(content_hash, byte_length, encoding, content)
            VALUES (%s, 1, 'utf-8', %s)
            ON CONFLICT (content_hash) DO NOTHING
            """,
            (content_hash, b"x"),
        )
        connection.execute(
            """
            INSERT INTO vault_rag.embeddings(
                content_hash, configuration_fingerprint, observed_fingerprint,
                dimensions, embedding
            ) VALUES (%s, %s, %s, 2, '[0.1,0.2]'::vector)
            ON CONFLICT DO NOTHING
            """,
            (content_hash, config_fingerprint, observed_fingerprint),
        )
        connection.execute(
            """
            INSERT INTO vault_rag.revision_sources(
                revision_id, relative_path, folded_path, source_kind, content_hash,
                source_blob_hash, size_bytes, mtime_ns, parse_state, indexed_at
            ) VALUES (%s, 'note.md', 'note.md', 'markdown', %s, %s, 1, 1, 'active', now())
            """,
            (revision_id, content_hash, content_hash),
        )
        connection.execute(
            """
            INSERT INTO vault_rag.revision_chunks(
                revision_id, chunk_id, source_path, source_blob_hash, ordinal, title,
                heading_json, start_line, end_line, body, embedding_text, token_count,
                metadata_json, content_hash, embedding_state, embedding_config_fingerprint,
                observed_fingerprint, dimensions, search_document
            ) VALUES (
                %s, 'chunk', 'note.md', %s, 0, 'Title', '[]'::jsonb, 1, 1, 'x', 'x', 1,
                '{}'::jsonb, %s, 'ready', %s, %s, 2, to_tsvector('simple', 'x')
            )
            """,
            (
                revision_id,
                content_hash,
                content_hash,
                config_fingerprint,
                observed_fingerprint,
            ),
        )


def test_cleanup_is_disabled_without_explicit_policy(postgres_pool: PostgresPool) -> None:
    PostgresMigrator(postgres_pool).apply()
    revision_id = uuid4()
    with postgres_pool.connection() as connection:
        connection.execute(
            "INSERT INTO vault_rag.vaults(vault_id, configured_ref) VALUES ('vault', 'main')"
        )
    _revision(postgres_pool, "vault", revision_id, "failed", age_days=8)

    result = PostgresRevisionCleaner(postgres_pool).run("vault", CleanupPolicy())

    assert result.revisions_deleted == 0
    with postgres_pool.connection() as connection:
        assert connection.execute(
            "SELECT state FROM vault_rag.vault_revisions WHERE revision_id = %s", (revision_id,)
        ).fetchone() == {"state": "failed"}


def test_cleanup_batches_eligible_revisions_and_collects_only_unreferenced_data(
    postgres_pool: PostgresPool,
) -> None:
    PostgresMigrator(postgres_pool).apply()
    active, retained, failed = uuid4(), uuid4(), uuid4()
    shared_hash = "sha256:" + "1" * 64
    abandoned_hash = "sha256:" + "2" * 64
    with postgres_pool.connection() as connection:
        connection.execute(
            "INSERT INTO vault_rag.vaults(vault_id, configured_ref) VALUES ('vault', 'main')"
        )
    _revision(postgres_pool, "vault", active, "active", age_days=30, promoted=True)
    with postgres_pool.connection() as connection:
        connection.execute(
            "UPDATE vault_rag.vaults SET active_revision_id = %s WHERE vault_id = 'vault'",
            (active,),
        )
    _revision(postgres_pool, "vault", retained, "superseded", age_days=20, promoted=True)
    _revision(postgres_pool, "vault", failed, "failed", age_days=20)
    _source_and_embedding(postgres_pool, active, shared_hash)
    _source_and_embedding(postgres_pool, retained, shared_hash)
    _source_and_embedding(postgres_pool, failed, abandoned_hash)

    result = PostgresRevisionCleaner(postgres_pool).run(
        "vault",
        CleanupPolicy(enabled=True, keep_promoted=1, min_age=timedelta(days=7), batch_size=1),
    )

    assert result.revisions_deleted == 1
    assert result.blobs_deleted == 1
    assert result.embeddings_deleted == 1
    with postgres_pool.connection() as connection:
        states = connection.execute(
            "SELECT state FROM vault_rag.vault_revisions ORDER BY state"
        ).fetchall()
        assert [row["state"] for row in states] == ["active", "superseded"]
        assert connection.execute(
            "SELECT count(*) AS count FROM vault_rag.source_blobs WHERE content_hash = %s",
            (shared_hash,),
        ).fetchone() == {"count": 1}
        assert connection.execute(
            "SELECT count(*) AS count FROM vault_rag.embeddings WHERE content_hash = %s",
            (shared_hash,),
        ).fetchone() == {"count": 1}


def test_cleanup_reclaims_abandoned_build_only_after_it_acquires_vault_lease(
    postgres_pool: PostgresPool,
) -> None:
    PostgresMigrator(postgres_pool).apply()
    abandoned = uuid4()
    with postgres_pool.connection() as connection:
        connection.execute(
            "INSERT INTO vault_rag.vaults(vault_id, configured_ref) VALUES ('vault', 'main')"
        )
    _revision(postgres_pool, "vault", abandoned, "building", age_days=8)
    policy = CleanupPolicy(enabled=True, min_age=timedelta(days=7))
    lease = PostgresWorkerLease(postgres_pool, "sync:vault")
    assert lease.acquire()
    try:
        assert PostgresRevisionCleaner(postgres_pool).run("vault", policy).revisions_deleted == 0
    finally:
        lease.release()

    assert PostgresRevisionCleaner(postgres_pool).run("vault", policy).revisions_deleted == 1
    with postgres_pool.connection() as connection:
        assert connection.execute(
            "SELECT count(*) AS count FROM vault_rag.vault_revisions WHERE revision_id = %s",
            (abandoned,),
        ).fetchone() == {"count": 0}
