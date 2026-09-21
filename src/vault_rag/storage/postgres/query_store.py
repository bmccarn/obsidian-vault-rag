"""Active-revision PostgreSQL retrieval queries."""

from __future__ import annotations

import math
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from time import perf_counter
from types import MappingProxyType
from typing import Any, cast

import numpy as np  # pyright: ignore[reportMissingImports]
from pgvector import Vector  # type: ignore[import-untyped]
from psycopg import Error  # pyright: ignore[reportMissingImports]
from psycopg.types.json import Jsonb  # pyright: ignore[reportMissingImports]

from vault_rag.domain import EmbeddingState, JsonValue, LineRange, SourceKind
from vault_rag.errors import StorageError
from vault_rag.storage.ports import QueryStore, RevisionBuildStore, SnapshotQueryStore
from vault_rag.storage.records import (
    ActiveRevision,
    DenseCandidate,
    DenseRequest,
    IdentifierCandidate,
    IndexSnapshot,
    LexicalCandidate,
    LexicalRequest,
    PendingEmbedding,
    SourceProvenance,
    StorageFilters,
    StoredChunk,
    VectorScope,
    validate_relative_prefix,
    validate_vault_ids,
)
from vault_rag.storage.schema import SCHEMA_VERSION

from .migrations import PostgresMigrator
from .pool import DatabaseConnection, PostgresPool

_MAX_IDENTIFIER_CANDIDATES = 10_000


@dataclass(frozen=True, slots=True)
class DatabaseState:
    """Bounded aggregates from the active PostgreSQL revision and sync queue."""

    revision_count: int
    pending_count: int
    pending_age_seconds: float
    sync_queue_depth: int
    sync_queue_age_seconds: float
    dense_mode: bool


def _json_mapping(value: object) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise StorageError("stored chunk metadata is invalid")
    return cast(dict[str, JsonValue], value)


def _heading(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(part, str) for part in value):
        raise StorageError("stored chunk heading is invalid")
    return tuple(cast(list[str], value))


def _like_prefix(prefix: str) -> str:
    escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}/%"


def _query_filters(
    vault_ids: tuple[str, ...],
    filters: StorageFilters,
) -> tuple[str, tuple[object, ...]]:
    validate_vault_ids(vault_ids)
    conditions = ["v.vault_id = ANY(%s)"]
    parameters: list[object] = [list(vault_ids)]
    if filters.vault_ids:
        conditions.append("v.vault_id = ANY(%s)")
        parameters.append(list(filters.vault_ids))
    if filters.path_prefix is not None:
        prefix = filters.path_prefix.rstrip("/")
        conditions.append("(rs.relative_path = %s OR rs.relative_path LIKE %s ESCAPE E'\\\\')")
        parameters.extend((prefix, _like_prefix(prefix)))
    if filters.source_kind is not None:
        conditions.append("rs.source_kind = %s")
        parameters.append(filters.source_kind.value)
    for key, value in sorted(filters.frontmatter.items()):
        conditions.append("(rc.metadata_json ? %s AND (rc.metadata_json -> %s) = %s)")
        parameters.extend((key, key, Jsonb(value)))
    return " AND ".join(conditions), tuple(parameters)


class PostgresStore(SnapshotQueryStore):
    """Query active immutable revisions and start isolated revision builds."""

    def __init__(
        self,
        pool: PostgresPool,
        *,
        connection: DatabaseConnection | None = None,
    ) -> None:
        self._pool = pool
        self._connection = connection

    @property
    def database_location(self) -> str:
        return self._pool.database_location

    @contextmanager
    def consistent_read(self) -> Generator[QueryStore]:
        """Pin every query in one request to one repeatable-read snapshot."""
        if self._connection is not None:
            yield self
            return
        try:
            with self._pool.connection() as connection, connection.transaction():
                connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                yield type(self)(self._pool, connection=connection)
        except Error as exc:
            raise StorageError("could not hold a consistent PostgreSQL read") from exc

    @contextmanager
    def _read_connection(self, operation: str) -> Generator[DatabaseConnection]:
        started = perf_counter()
        outcome = "failure"
        try:
            if self._connection is not None:
                yield self._connection
            else:
                with self._pool.connection() as connection:
                    yield connection
            outcome = "success"
        finally:
            observer = getattr(self._pool, "observe_operation", None)
            if callable(observer):
                with suppress(Exception):
                    observer(operation, outcome, (perf_counter() - started) * 1000)

    def initialize(self) -> None:
        PostgresMigrator(self._pool).check()

    def active_revisions(self, vault_ids: tuple[str, ...]) -> tuple[ActiveRevision, ...]:
        """Return lexically usable active revisions on this store's read connection."""
        validate_vault_ids(vault_ids)
        try:
            with self._read_connection("active_revisions") as connection:
                rows = connection.execute(
                    """
                    SELECT
                        v.vault_id, v.configured_ref, v.configured_sha, v.configured_at,
                        v.fetched_sha, v.fetched_at, v.checkout_sha, v.checkout_at,
                        v.attempted_sha, v.attempted_at, v.reconciled_sha,
                        v.reconciled_at, v.sync_degradation_category, v.sync_degraded_at,
                        vr.revision_id, vr.commit_sha, vr.manifest_json, vr.state,
                        vr.fully_reconciled, vr.source_count, vr.chunk_count, vr.ready_count,
                        vr.pending_count, vr.vector_bytes, vr.manifest_fingerprint,
                        vr.parser_fingerprint, vr.chunker_fingerprint,
                        vr.embedding_config_fingerprint, vr.promoted_at
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.vault_revisions AS vr
                        ON vr.revision_id = v.active_revision_id
                    WHERE v.vault_id = ANY(%s)
                        AND vr.state IN ('active', 'active_degraded')
                        AND vr.lexical_complete
                    ORDER BY v.vault_id
                    """,
                    (list(vault_ids),),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not load active PostgreSQL revisions") from exc
        return tuple(self._active_revision(row) for row in rows)

    def begin_revision(
        self,
        vault_id: str,
        configured_ref: str,
        commit_sha: str,
        *,
        manifest: Mapping[str, JsonValue] | None = None,
    ) -> RevisionBuildStore:
        from .index_store import PostgresIndexStore

        return PostgresIndexStore.begin(
            self._pool,
            vault_id=vault_id,
            configured_ref=configured_ref,
            commit_sha=commit_sha,
            manifest=manifest,
        )

    def identifier_candidates(
        self,
        vault_ids: tuple[str, ...],
        filters: StorageFilters,
    ) -> tuple[IdentifierCandidate, ...]:
        conditions, parameters = _query_filters(vault_ids, filters)
        try:
            with self._read_connection("identifier_candidates") as connection:
                rows = connection.execute(
                    f"""
                        SELECT rc.chunk_id, v.vault_id, rs.relative_path, rc.title,
                        rc.metadata_json
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.revision_chunks AS rc
                        ON rc.revision_id = v.active_revision_id
                    JOIN vault_rag.revision_sources AS rs
                        ON rs.revision_id = rc.revision_id
                        AND rs.relative_path = rc.source_path
                    WHERE rs.parse_state = 'active' AND {conditions}
                    ORDER BY v.vault_id, rs.relative_path, rc.start_line, rc.chunk_id
                    LIMIT %s
                    """,  # nosec B608 - conditions are internally compiled  # pyright: ignore[reportArgumentType]
                    (*parameters, _MAX_IDENTIFIER_CANDIDATES),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not load identifier candidates") from exc
        return tuple(
            IdentifierCandidate(
                chunk_id=cast(str, row["chunk_id"]),
                vault_id=cast(str, row["vault_id"]),
                relative_path=cast(str, row["relative_path"]),
                title=cast(str, row["title"]),
                metadata=_json_mapping(row["metadata_json"]),
            )
            for row in rows
        )

    def lexical_search(self, request: LexicalRequest) -> tuple[LexicalCandidate, ...]:
        if not request.fts_query.strip():
            return ()
        conditions, parameters = _query_filters(request.vault_ids, request.filters)
        try:
            with self._read_connection("lexical_search") as connection:
                rows = connection.execute(
                    f"""
                        WITH query AS (
                        SELECT websearch_to_tsquery('simple', %s) AS value
                    )
                    SELECT rc.chunk_id, v.vault_id, rs.relative_path, rs.source_kind,
                        rs.source_blob_hash, rc.title, rc.heading_json, rc.start_line,
                        rc.end_line, rc.body, rc.metadata_json,
                        ts_rank_cd(rc.search_document, query.value) AS lexical_score
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.revision_chunks AS rc
                        ON rc.revision_id = v.active_revision_id
                    JOIN vault_rag.revision_sources AS rs
                        ON rs.revision_id = rc.revision_id
                        AND rs.relative_path = rc.source_path
                    CROSS JOIN query
                    WHERE rs.parse_state = 'active'
                        AND rc.search_document @@ query.value AND {conditions}
                    ORDER BY lexical_score DESC, v.vault_id, rs.relative_path,
                        rc.start_line, rc.chunk_id
                    LIMIT %s
                    """,  # nosec B608 - conditions are internally compiled  # pyright: ignore[reportArgumentType]
                    (request.fts_query, *parameters, request.limit),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not execute lexical query") from exc
        return tuple(self._lexical_candidate(row) for row in rows)

    def dense_search(
        self,
        request: DenseRequest,
        query_vector: np.ndarray,
    ) -> tuple[DenseCandidate, ...]:
        normalized = np.asarray(query_vector, dtype=np.float32)
        if normalized.ndim != 1 or not np.isfinite(normalized).all():
            raise ValueError("query vector must be a finite one-dimensional array")
        norm = float(np.linalg.norm(normalized))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("query vector must have a finite non-zero norm")
        normalized = np.asarray(normalized / norm, dtype=np.float32)
        dimensions = int(normalized.size)
        conditions, parameters = _query_filters(request.vault_ids, request.filters)
        try:
            with self._read_connection("dense_search") as connection:
                self._pool.register_vector(connection)
                value = Vector(normalized)
                rows = connection.execute(
                    f"""
                        SELECT rc.chunk_id, 1 - (e.embedding <=> %s) AS score
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.revision_chunks AS rc
                        ON rc.revision_id = v.active_revision_id
                    JOIN vault_rag.revision_sources AS rs
                        ON rs.revision_id = rc.revision_id
                        AND rs.relative_path = rc.source_path
                    JOIN vault_rag.embeddings AS e
                        ON e.content_hash = rc.content_hash
                        AND e.configuration_fingerprint = rc.embedding_config_fingerprint
                        AND e.observed_fingerprint = rc.observed_fingerprint
                    WHERE rs.parse_state = 'active' AND rc.embedding_state = 'ready'
                        AND rc.observed_fingerprint = %s AND rc.dimensions = %s
                        AND e.dimensions = %s AND {conditions}
                    ORDER BY e.embedding <=> %s, v.vault_id, rs.relative_path,
                        rc.start_line, rc.chunk_id
                    LIMIT %s
                    """,  # nosec B608 - conditions are internally compiled  # pyright: ignore[reportArgumentType]
                    (
                        value,
                        request.observed_fingerprint,
                        dimensions,
                        dimensions,
                        *parameters,
                        value,
                        request.limit,
                    ),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not execute dense query") from exc
        return tuple(
            DenseCandidate(cast(str, row["chunk_id"]), cast(float, row["score"])) for row in rows
        )

    def vector_scope(
        self,
        vault_ids: tuple[str, ...],
        observed_fingerprint: str,
        filters: StorageFilters,
    ) -> VectorScope:
        if not observed_fingerprint:
            raise ValueError("observed fingerprint must be non-empty")
        conditions, parameters = _query_filters(vault_ids, filters)
        try:
            with self._read_connection("vector_scope") as connection:
                rows = connection.execute(
                    f"""
                        SELECT rc.dimensions, COUNT(*) AS vector_count
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.revision_chunks AS rc
                        ON rc.revision_id = v.active_revision_id
                    JOIN vault_rag.revision_sources AS rs
                        ON rs.revision_id = rc.revision_id
                        AND rs.relative_path = rc.source_path
                    WHERE rs.parse_state = 'active' AND rc.embedding_state = 'ready'
                        AND rc.observed_fingerprint = %s AND {conditions}
                    GROUP BY rc.dimensions ORDER BY rc.dimensions
                    """,  # nosec B608 - conditions are internally compiled  # pyright: ignore[reportArgumentType]
                    (observed_fingerprint, *parameters),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not inspect vector scope") from exc
        return VectorScope(
            count=sum(cast(int, row["vector_count"]) for row in rows),
            dimensions=tuple(cast(int, row["dimensions"]) for row in rows),
        )

    def chunks_by_ids(
        self,
        ids: tuple[str, ...],
        *,
        vault_ids: tuple[str, ...],
    ) -> dict[str, StoredChunk]:
        validate_vault_ids(vault_ids)
        if not ids:
            return {}
        unique_ids = tuple(dict.fromkeys(ids))
        try:
            with self._read_connection("chunks_by_ids") as connection:
                rows = connection.execute(
                    self._stored_chunk_select()
                    + " WHERE v.vault_id = ANY(%s) AND rc.chunk_id = ANY(%s) "
                    "AND rs.parse_state = 'active'",  # pyright: ignore[reportArgumentType]
                    (list(vault_ids), list(unique_ids)),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not load chunks") from exc
        return {record.id: record for record in map(self._stored_chunk, rows)}

    def source_provenance(
        self,
        vault_id: str,
        relative_path: str,
    ) -> SourceProvenance | None:
        validate_vault_ids((vault_id,))
        validate_relative_prefix(relative_path)
        try:
            with self._read_connection("source_provenance") as connection:
                row = connection.execute(
                    """
                    SELECT rs.source_kind, rs.source_blob_hash, rs.indexed_at,
                        sb.byte_length, sb.content
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.revision_sources AS rs
                        ON rs.revision_id = v.active_revision_id
                    JOIN vault_rag.source_blobs AS sb
                        ON sb.content_hash = rs.source_blob_hash
                    WHERE v.vault_id = %s AND rs.relative_path = %s
                        AND rs.parse_state = 'active'
                    """,
                    (vault_id, relative_path),
                ).fetchone()
        except Error as exc:
            raise StorageError("could not load source provenance") from exc
        if row is None:
            return None
        return SourceProvenance(
            vault_id=vault_id,
            relative_path=relative_path,
            source_kind=SourceKind(cast(str, row["source_kind"])),
            source_hash=cast(str, row["source_blob_hash"]),
            indexed_at=cast(datetime, row["indexed_at"]),
            content=bytes(row["content"]),
            byte_length=cast(int, row["byte_length"]),
        )

    def pending_chunks(self, vault_ids: tuple[str, ...]) -> tuple[StoredChunk, ...]:
        validate_vault_ids(vault_ids)
        try:
            with self._read_connection("pending_chunks") as connection:
                rows = connection.execute(
                    self._stored_chunk_select()
                    + " WHERE v.vault_id = ANY(%s) AND rs.parse_state = 'active' "
                    "AND rc.embedding_state = 'pending' "
                    "ORDER BY v.vault_id, rs.relative_path, rc.ordinal",  # pyright: ignore[reportArgumentType]
                    (list(vault_ids),),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not load pending chunks") from exc
        return tuple(self._stored_chunk(row) for row in rows)

    def snapshots(self, vault_ids: tuple[str, ...]) -> Mapping[str, IndexSnapshot]:
        """Return each requested active-vault snapshot with two set-based queries."""
        validate_vault_ids(vault_ids)
        try:
            with self._read_connection("snapshots") as connection:
                rows = connection.execute(
                    """
                    SELECT v.vault_id, vr.*
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.vault_revisions AS vr
                      ON vr.revision_id = v.active_revision_id
                    WHERE v.vault_id = ANY(%s)
                    ORDER BY v.vault_id
                    """,
                    (list(vault_ids),),
                ).fetchall()
                vector_rows = connection.execute(
                    """
                    SELECT DISTINCT v.vault_id, rc.observed_fingerprint, rc.dimensions,
                      rc.embedding_config_fingerprint
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.revision_chunks AS rc
                      ON rc.revision_id = v.active_revision_id
                    WHERE v.vault_id = ANY(%s) AND rc.embedding_state = 'ready'
                    ORDER BY v.vault_id, rc.observed_fingerprint, rc.dimensions,
                      rc.embedding_config_fingerprint
                    """,
                    (list(vault_ids),),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not inspect PostgreSQL index") from exc
        by_vault = {str(row["vault_id"]): row for row in rows}
        if set(by_vault) != set(vault_ids):
            raise StorageError("configured PostgreSQL profile has no complete active revision set")
        vectors: dict[str, list[Mapping[str, Any]]] = {vault_id: [] for vault_id in vault_ids}
        for row in vector_rows:
            vectors[str(row["vault_id"])].append(row)
        result: dict[str, IndexSnapshot] = {}
        for vault_id in vault_ids:
            row = by_vault[vault_id]
            ready_vectors = vectors[vault_id]
            result[vault_id] = IndexSnapshot(
                source_count=cast(int, row["source_count"]),
                chunk_count=cast(int, row["chunk_count"]),
                ready_count=cast(int, row["ready_count"]),
                pending_count=cast(int, row["pending_count"]),
                vector_bytes=cast(int, row["vector_bytes"]),
                indexed_at=cast(datetime, row["promoted_at"]),
                schema_version=SCHEMA_VERSION,
                manifest_fingerprint=cast(str, row["manifest_fingerprint"]),
                parser_fingerprint=cast(str, row["parser_fingerprint"]),
                chunker_fingerprint=cast(str, row["chunker_fingerprint"]),
                embedding_config_fingerprint=cast(str, row["embedding_config_fingerprint"]),
                observed_fingerprints=tuple(
                    sorted(
                        {
                            cast(str, item["observed_fingerprint"])
                            for item in ready_vectors
                            if item["observed_fingerprint"] is not None
                        }
                    )
                ),
                vector_dimensions=tuple(
                    sorted(
                        {
                            cast(int, item["dimensions"])
                            for item in ready_vectors
                            if item["dimensions"] is not None
                        }
                    )
                ),
                vector_config_fingerprints=tuple(
                    sorted(
                        {
                            cast(str, item["embedding_config_fingerprint"])
                            for item in ready_vectors
                            if item["embedding_config_fingerprint"] is not None
                        }
                    )
                ),
            )
        return MappingProxyType(result)

    def database_state(self, vault_id: str) -> DatabaseState:
        """Return one vault's bounded state and the shared pending-sync aggregate."""
        validate_vault_ids((vault_id,))
        try:
            with self._read_connection("database_state") as connection:
                row = connection.execute(
                    """
                    SELECT
                        (
                            SELECT count(*)
                            FROM vault_rag.vault_revisions AS revision
                            WHERE revision.vault_id = %s
                        ) AS revision_count,
                        (
                            SELECT count(*)
                            FROM vault_rag.vaults AS vault
                            JOIN vault_rag.revision_chunks AS chunk
                              ON chunk.revision_id = vault.active_revision_id
                            WHERE vault.vault_id = %s
                              AND chunk.embedding_state = 'pending'
                        ) AS pending_count,
                        COALESCE((
                            SELECT EXTRACT(EPOCH FROM now() - MIN(queue.last_attempted_at))
                            FROM vault_rag.vaults AS vault
                            JOIN vault_rag.embedding_queue AS queue
                              ON queue.revision_id = vault.active_revision_id
                            WHERE vault.vault_id = %s AND NOT queue.terminal
                        ), 0) AS pending_age_seconds,
                        (
                            SELECT count(*)
                            FROM vault_rag.sync_requests AS request
                            WHERE request.state = 'pending'
                        ) AS sync_queue_depth,
                        COALESCE((
                            SELECT EXTRACT(EPOCH FROM now() - MIN(request.created_at))
                            FROM vault_rag.sync_requests AS request
                            WHERE request.state = 'pending'
                        ), 0) AS sync_queue_age_seconds,
                        EXISTS(
                            SELECT 1
                            FROM vault_rag.vaults AS vault
                            JOIN vault_rag.revision_chunks AS chunk
                              ON chunk.revision_id = vault.active_revision_id
                            WHERE vault.vault_id = %s
                              AND chunk.embedding_state = 'ready'
                        ) AS dense_mode
                    """,
                    (vault_id, vault_id, vault_id, vault_id),
                ).fetchone()
        except Error as exc:
            raise StorageError("could not inspect PostgreSQL database state") from exc
        if row is None:
            raise StorageError("PostgreSQL database state query returned no result")
        return DatabaseState(
            revision_count=max(0, int(row["revision_count"])),
            pending_count=max(0, int(row["pending_count"])),
            pending_age_seconds=max(0.0, float(row["pending_age_seconds"])),
            sync_queue_depth=max(0, int(row["sync_queue_depth"])),
            sync_queue_age_seconds=max(0.0, float(row["sync_queue_age_seconds"])),
            dense_mode=bool(row["dense_mode"]),
        )

    def snapshot(self, vault_ids: tuple[str, ...]) -> IndexSnapshot:
        validate_vault_ids(vault_ids)
        try:
            with self._read_connection("snapshot") as connection:
                rows = connection.execute(
                    """
                    SELECT vr.*
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.vault_revisions AS vr
                        ON vr.revision_id = v.active_revision_id
                    WHERE v.vault_id = ANY(%s)
                    ORDER BY v.vault_id
                    """,
                    (list(vault_ids),),
                ).fetchall()
                vector_rows = connection.execute(
                    """
                    SELECT DISTINCT rc.observed_fingerprint, rc.dimensions,
                        rc.embedding_config_fingerprint
                    FROM vault_rag.vaults AS v
                    JOIN vault_rag.revision_chunks AS rc
                        ON rc.revision_id = v.active_revision_id
                    WHERE v.vault_id = ANY(%s) AND rc.embedding_state = 'ready'
                    ORDER BY rc.observed_fingerprint, rc.dimensions,
                        rc.embedding_config_fingerprint
                    """,
                    (list(vault_ids),),
                ).fetchall()
        except Error as exc:
            raise StorageError("could not inspect PostgreSQL index") from exc
        expected = len(set(vault_ids))
        observed = tuple(
            sorted(
                {
                    cast(str, row["observed_fingerprint"])
                    for row in vector_rows
                    if row["observed_fingerprint"] is not None
                }
            )
        )
        dimensions = tuple(
            sorted(
                {
                    cast(int, row["dimensions"])
                    for row in vector_rows
                    if row["dimensions"] is not None
                }
            )
        )
        configs = tuple(
            sorted(
                {
                    cast(str, row["embedding_config_fingerprint"])
                    for row in vector_rows
                    if row["embedding_config_fingerprint"] is not None
                }
            )
        )
        return IndexSnapshot(
            source_count=sum(cast(int, row["source_count"]) for row in rows),
            chunk_count=sum(cast(int, row["chunk_count"]) for row in rows),
            ready_count=sum(cast(int, row["ready_count"]) for row in rows),
            pending_count=sum(cast(int, row["pending_count"]) for row in rows),
            vector_bytes=sum(cast(int, row["vector_bytes"]) for row in rows),
            indexed_at=self._oldest_timestamp(rows, "promoted_at", expected),
            schema_version=SCHEMA_VERSION,
            manifest_fingerprint=self._agreed(rows, "manifest_fingerprint", expected),
            parser_fingerprint=self._agreed(rows, "parser_fingerprint", expected),
            chunker_fingerprint=self._agreed(rows, "chunker_fingerprint", expected),
            embedding_config_fingerprint=self._agreed(
                rows, "embedding_config_fingerprint", expected
            ),
            observed_fingerprints=observed,
            vector_dimensions=dimensions,
            vector_config_fingerprints=configs,
        )

    @staticmethod
    def _active_revision(row: Mapping[str, Any]) -> ActiveRevision:
        manifest = _json_mapping(row["manifest_json"])
        required_text = (
            "vault_id",
            "commit_sha",
            "state",
            "manifest_fingerprint",
            "parser_fingerprint",
            "chunker_fingerprint",
            "embedding_config_fingerprint",
            "configured_ref",
        )
        if any(not isinstance(row[key], str) or not row[key] for key in required_text):
            raise StorageError("stored active revision identity is invalid")
        if not str(row["revision_id"]):
            raise StorageError("stored active revision identity is invalid")
        promoted_at = row["promoted_at"]
        if not isinstance(promoted_at, datetime):
            raise StorageError("stored active revision timestamp is invalid")
        counts = ("source_count", "chunk_count", "ready_count", "pending_count", "vector_bytes")
        if any(
            not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0
            for key in counts
        ):
            raise StorageError("stored active revision counts are invalid")
        return ActiveRevision(
            vault_id=cast(str, row["vault_id"]),
            revision_id=str(row["revision_id"]),
            commit_sha=cast(str, row["commit_sha"]),
            manifest=manifest,
            state=cast(str, row["state"]),
            source_count=cast(int, row["source_count"]),
            chunk_count=cast(int, row["chunk_count"]),
            ready_count=cast(int, row["ready_count"]),
            pending_count=cast(int, row["pending_count"]),
            vector_bytes=cast(int, row["vector_bytes"]),
            manifest_fingerprint=cast(str, row["manifest_fingerprint"]),
            parser_fingerprint=cast(str, row["parser_fingerprint"]),
            chunker_fingerprint=cast(str, row["chunker_fingerprint"]),
            embedding_config_fingerprint=cast(str, row["embedding_config_fingerprint"]),
            promoted_at=promoted_at,
            configured_ref=cast(str, row["configured_ref"]),
            configured_sha=cast(str | None, row["configured_sha"]),
            checkout_sha=cast(str | None, row["checkout_sha"]),
            checkout_at=cast(datetime | None, row["checkout_at"]),
            configured_at=cast(datetime | None, row["configured_at"]),
            fetched_sha=cast(str | None, row["fetched_sha"]),
            fetched_at=cast(datetime | None, row["fetched_at"]),
            attempted_sha=cast(str | None, row["attempted_sha"]),
            attempted_at=cast(datetime | None, row["attempted_at"]),
            reconciled_sha=cast(str | None, row["reconciled_sha"]),
            reconciled_at=cast(datetime | None, row["reconciled_at"]),
            fully_reconciled=cast(bool, row["fully_reconciled"]),
            sync_degradation_category=cast(str | None, row["sync_degradation_category"]),
            sync_degraded_at=cast(datetime | None, row["sync_degraded_at"]),
        )

    @staticmethod
    def _agreed(rows: Sequence[Mapping[str, Any]], key: str, expected: int) -> str | None:
        if len(rows) != expected:
            return None
        values = {cast(str, row[key]) for row in rows if row[key] is not None}
        return next(iter(values)) if len(values) == 1 else None

    @staticmethod
    def _oldest_timestamp(
        rows: Sequence[Mapping[str, Any]], key: str, expected: int
    ) -> datetime | None:
        if len(rows) != expected:
            return None
        values = [cast(datetime, row[key]) for row in rows if row[key] is not None]
        return min(values) if len(values) == expected else None

    @staticmethod
    def _lexical_candidate(row: Mapping[str, Any]) -> LexicalCandidate:
        return LexicalCandidate(
            chunk_id=cast(str, row["chunk_id"]),
            vault_id=cast(str, row["vault_id"]),
            relative_path=cast(str, row["relative_path"]),
            source_kind=SourceKind(cast(str, row["source_kind"])),
            source_hash=cast(str, row["source_blob_hash"]),
            title=cast(str, row["title"]),
            heading=_heading(row["heading_json"]),
            lines=LineRange(cast(int, row["start_line"]), cast(int, row["end_line"])),
            text=cast(str, row["body"]),
            metadata=_json_mapping(row["metadata_json"]),
            score=cast(float, row["lexical_score"]),
        )

    @staticmethod
    def _stored_chunk_select() -> str:
        return """
            SELECT rc.chunk_id, v.vault_id, rs.relative_path, rs.source_kind,
                rs.source_blob_hash, rc.ordinal, rc.title, rc.heading_json,
                rc.start_line, rc.end_line, rc.body, rc.embedding_text,
                rc.token_count, rc.metadata_json,
                rc.content_hash AS chunk_content_hash, rc.embedding_state,
                q.category, q.message, q.last_attempted_at
            FROM vault_rag.vaults AS v
            JOIN vault_rag.revision_chunks AS rc
                ON rc.revision_id = v.active_revision_id
            JOIN vault_rag.revision_sources AS rs
                ON rs.revision_id = rc.revision_id
                AND rs.relative_path = rc.source_path
            LEFT JOIN vault_rag.embedding_queue AS q
                ON q.revision_id = rc.revision_id AND q.chunk_id = rc.chunk_id
        """

    @staticmethod
    def _stored_chunk(row: Mapping[str, Any]) -> StoredChunk:
        pending: PendingEmbedding | None = None
        if row["category"] is not None:
            pending = PendingEmbedding(
                chunk_id=cast(str, row["chunk_id"]),
                category=cast(str, row["category"]),
                message=cast(str, row["message"]),
                attempted_at=cast(datetime, row["last_attempted_at"]),
            )
        return StoredChunk(
            id=cast(str, row["chunk_id"]),
            vault_id=cast(str, row["vault_id"]),
            relative_path=cast(str, row["relative_path"]),
            source_kind=SourceKind(cast(str, row["source_kind"])),
            source_hash=cast(str, row["source_blob_hash"]),
            ordinal=cast(int, row["ordinal"]),
            title=cast(str, row["title"]),
            heading=_heading(row["heading_json"]),
            lines=LineRange(cast(int, row["start_line"]), cast(int, row["end_line"])),
            text=cast(str, row["body"]),
            embedding_text=cast(str, row["embedding_text"]),
            token_count=cast(int, row["token_count"]),
            metadata=_json_mapping(row["metadata_json"]),
            content_hash=cast(str, row["chunk_content_hash"]),
            embedding_state=EmbeddingState(cast(str, row["embedding_state"])),
            pending=pending,
        )
