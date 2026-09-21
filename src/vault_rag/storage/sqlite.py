"""Transactional SQLite storage for lexical and dense retrieval data."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np  # pyright: ignore[reportMissingImports]

from vault_rag.domain import EmbeddingState, JsonValue, LineRange, SourceKind
from vault_rag.embedding.fingerprint import observed_fingerprint
from vault_rag.errors import RebuildRequiredError, StorageError
from vault_rag.ingest.chunker import ChunkRecord

from .codecs import decode_vector, encode_vector
from .records import (
    DenseCandidate,
    DenseRequest,
    IdentifierCandidate,
    IndexSnapshot,
    LexicalCandidate,
    LexicalRequest,
    PendingEmbedding,
    PendingEmbeddingUpdate,
    RequeueResult,
    SourceProvenance,
    SourceRevision,
    StorageFilters,
    StoredChunk,
    VaultFingerprint,
    VectorLoadRequest,
    VectorRecord,
    VectorScope,
    canonical_json,
    validate_relative_prefix,
    validate_vault_ids,
)
from .schema import REQUIRED_COLUMNS, REQUIRED_OBJECTS, SCHEMA_STATEMENTS, SCHEMA_VERSION

_MAX_DIAGNOSTIC_LENGTH = 1_000


def _parse_json_mapping(raw: str) -> Mapping[str, JsonValue]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise StorageError("stored chunk metadata is invalid")
    return cast(dict[str, JsonValue], value)


def _parse_heading(raw: str) -> tuple[str, ...]:
    value = json.loads(raw)
    if not isinstance(value, list) or any(not isinstance(part, str) for part in value):
        raise StorageError("stored chunk heading is invalid")
    return tuple(cast(list[str], value))


class SQLiteStore:
    """A connection-per-operation SQLite store with explicit transactions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def database_location(self) -> str:
        return str(self.path)

    @contextmanager
    def consistent_read(self) -> Generator[SQLiteStore]:
        """Use the service lifecycle lock as SQLite's request-level snapshot boundary."""
        yield self

    def replacement_store(self, path: Path) -> SQLiteStore:
        """Create an empty sibling adapter for atomic local rebuilds."""
        return type(self)(path)

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self.path,
                timeout=5.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
            journal_mode = cast(str, connection.execute("PRAGMA journal_mode = WAL").fetchone()[0])
            if journal_mode.casefold() != "wal":
                raise StorageError("SQLite WAL mode is unavailable")
        except StorageError:
            if connection is not None:
                connection.close()
            raise
        except sqlite3.Error as exc:
            if connection is not None:
                connection.close()
            raise StorageError("could not open SQLite index") from exc
        return connection

    def initialize(self) -> None:
        """Create schema version 1 or reject a database from another version."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            version = cast(int, connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, SCHEMA_VERSION}:
                raise RebuildRequiredError(
                    f"index schema version {version} is incompatible with version {SCHEMA_VERSION}",
                    details={"stored_version": version, "required_version": SCHEMA_VERSION},
                )
            if version == 0:
                existing = {
                    cast(str, row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if existing:
                    raise RebuildRequiredError("unversioned index database requires a rebuild")
            else:
                self._validate_required_columns(connection)
            connection.execute("BEGIN IMMEDIATE")
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)  # nosec B608 - schema constants only
            connection.execute(
                "INSERT INTO index_state(singleton, schema_version) VALUES (1, ?) "
                "ON CONFLICT(singleton) DO NOTHING",
                (SCHEMA_VERSION,),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            objects = {
                cast(str, row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
            if not REQUIRED_OBJECTS.issubset(objects):
                raise StorageError("SQLite index schema is incomplete")
            self._validate_required_columns(connection)
            connection.commit()
        except (RebuildRequiredError, StorageError):
            if connection.in_transaction:
                connection.rollback()
            raise
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise StorageError("could not initialize SQLite index") from exc
        finally:
            connection.close()

    @staticmethod
    def _validate_required_columns(connection: sqlite3.Connection) -> None:
        for table_name, required_columns in REQUIRED_COLUMNS.items():
            stored_columns = {
                cast(str, row[0])
                for row in connection.execute(
                    "SELECT name FROM pragma_table_info(?)",
                    (table_name,),
                )
            }
            missing_columns = sorted(required_columns - stored_columns)
            if missing_columns:
                missing_details: list[JsonValue] = []
                missing_details.extend(missing_columns)
                details: dict[str, JsonValue] = {
                    "table": table_name,
                    "missing_columns": missing_details,
                }
                raise RebuildRequiredError(
                    f"index table {table_name} is missing required columns; rebuild required",
                    details=details,
                )

    def foreign_keys_enabled(self) -> bool:
        """Report the connection-local foreign-key setting for diagnostics."""
        connection = self._connect()
        try:
            return cast(int, connection.execute("PRAGMA foreign_keys").fetchone()[0]) == 1
        finally:
            connection.close()

    def fts5_available(self) -> bool:
        """Report whether the initialized database can execute an FTS5 function."""
        connection = self._connect()
        try:
            connection.execute("SELECT fts5('vault-rag diagnostic')").fetchone()
            return True
        except sqlite3.Error:
            return False
        finally:
            connection.close()

    def source_hashes(self, vault_ids: tuple[str, ...]) -> dict[tuple[str, str], str]:
        validate_vault_ids(vault_ids)
        placeholders = ",".join("?" for _ in vault_ids)
        connection = self._connect()
        try:
            rows = connection.execute(  # nosec B608 - placeholders are generated only
                "SELECT vault_id, relative_path, content_hash FROM sources "
                f"WHERE vault_id IN ({placeholders}) ORDER BY vault_id, relative_path",
                vault_ids,
            )
            return {
                (cast(str, row["vault_id"]), cast(str, row["relative_path"])): cast(
                    str, row["content_hash"]
                )
                for row in rows
            }
        except sqlite3.Error as exc:
            raise StorageError("could not load source hashes") from exc
        finally:
            connection.close()

    def reusable_vectors(
        self, chunks: tuple[ChunkRecord, ...], embedding_config_fingerprint: str
    ) -> dict[str, VectorRecord]:
        hashes = tuple(dict.fromkeys(chunk.content_hash for chunk in chunks))
        if not hashes:
            return {}
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT c.content_hash, v.chunk_id, v.dimensions,
                    v.embedding_config_fingerprint, v.observed_fingerprint, v.vector
                FROM chunk_vectors AS v
                JOIN chunks AS c ON c.chunk_id = v.chunk_id
                JOIN sources AS s ON s.source_id = c.source_id
                WHERE s.parse_state = 'active' AND c.embedding_state = 'ready'
                    AND v.embedding_config_fingerprint = ?
                    AND c.content_hash IN (SELECT value FROM json_each(?))
                """,
                (embedding_config_fingerprint, canonical_json(list(hashes))),
            ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError("could not load reusable vectors") from exc
        finally:
            connection.close()
        grouped: dict[str, list[VectorRecord]] = {}
        for row in rows:
            grouped.setdefault(cast(str, row["content_hash"]), []).append(self._vector_record(row))
        return {
            content_hash: vectors[0]
            for content_hash, vectors in grouped.items()
            if len(
                {
                    (item.dimensions, item.config_fingerprint, item.observed_fingerprint)
                    for item in vectors
                }
            )
            == 1
            and vectors[0].config_fingerprint == embedding_config_fingerprint
            and vectors[0].observed_fingerprint
            == observed_fingerprint(embedding_config_fingerprint, vectors[0].dimensions)
            and np.asarray(vectors[0].vector).size == vectors[0].dimensions
        }

    def replace_source(
        self, revision: SourceRevision, *, replacing_path: str | None = None
    ) -> None:
        """Atomically activate one complete parsed source revision.

        ``replacing_path`` is reserved for a discovered case-only rename whose old
        folded-path row would otherwise prevent the add from committing before the
        reconciliation delete phase.
        """
        try:
            encoded_vectors = self._validate_revision(revision)
        except (TypeError, ValueError) as exc:
            identity = f"{revision.source.vault_id}/{revision.source.relative_path}"
            raise StorageError(
                f"invalid source revision for {identity}",
                details={
                    "vault_id": revision.source.vault_id,
                    "path": revision.source.relative_path,
                },
            ) from exc

        source = revision.source
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if revision.update_vault_fingerprint:
                connection.execute(
                    """
                    INSERT INTO vaults(
                        vault_id, manifest_fingerprint, parser_fingerprint,
                        chunker_fingerprint, embedding_config_fingerprint
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(vault_id) DO UPDATE SET
                        manifest_fingerprint = excluded.manifest_fingerprint,
                        parser_fingerprint = excluded.parser_fingerprint,
                        chunker_fingerprint = excluded.chunker_fingerprint,
                        embedding_config_fingerprint = excluded.embedding_config_fingerprint
                    """,
                    (
                        source.vault_id,
                        revision.manifest_fingerprint,
                        revision.parser_fingerprint,
                        revision.chunker_fingerprint,
                        revision.embedding_config_fingerprint,
                    ),
                )
            else:
                stored_fingerprint = connection.execute(
                    "SELECT parser_fingerprint, chunker_fingerprint, "
                    "embedding_config_fingerprint FROM vaults WHERE vault_id = ?",
                    (source.vault_id,),
                ).fetchone()
                if stored_fingerprint is None or (
                    stored_fingerprint["parser_fingerprint"] != revision.parser_fingerprint
                    or stored_fingerprint["chunker_fingerprint"] != revision.chunker_fingerprint
                    or stored_fingerprint["embedding_config_fingerprint"]
                    != revision.embedding_config_fingerprint
                ):
                    raise ValueError("deferred vault fingerprint is missing or incompatible")
            now = datetime.now(UTC).isoformat()
            if replacing_path is not None:
                replaced = connection.execute(
                    "SELECT source_id, folded_path FROM sources "
                    "WHERE vault_id = ? AND relative_path = ?",
                    (source.vault_id, replacing_path),
                ).fetchone()
                if replaced is not None:
                    if (
                        replacing_path == source.relative_path
                        or cast(str, replaced["folded_path"]) != source.folded_path
                    ):
                        raise ValueError("replacement path is not a case-only source rename")
                    replaced_source_id = cast(int, replaced["source_id"])
                    self._delete_source_dependents(connection, replaced_source_id)
                    connection.execute(
                        "DELETE FROM sources WHERE source_id = ?", (replaced_source_id,)
                    )
            connection.execute(
                """
                INSERT INTO sources(
                    vault_id, relative_path, folded_path, source_kind, content_hash,
                    size_bytes, mtime_ns, parse_state, diagnostic, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', NULL, ?)
                ON CONFLICT(vault_id, relative_path) DO UPDATE SET
                    folded_path = excluded.folded_path,
                    source_kind = excluded.source_kind,
                    content_hash = excluded.content_hash,
                    size_bytes = excluded.size_bytes,
                    mtime_ns = excluded.mtime_ns,
                    parse_state = 'active',
                    diagnostic = NULL,
                    indexed_at = excluded.indexed_at
                """,
                (
                    source.vault_id,
                    source.relative_path,
                    source.folded_path,
                    source.kind.value,
                    source.content_hash,
                    source.size_bytes,
                    source.mtime_ns,
                    now,
                ),
            )
            source_id = cast(
                int,
                connection.execute(
                    "SELECT source_id FROM sources WHERE vault_id = ? AND relative_path = ?",
                    (source.vault_id, source.relative_path),
                ).fetchone()[0],
            )
            self._delete_source_dependents(connection, source_id)

            vector_ids = {vector.chunk_id for vector in revision.vectors}
            pending_by_id = {pending.chunk_id: pending for pending in revision.pending}
            for chunk_record in revision.chunks:
                if chunk_record.id in vector_ids:
                    state = EmbeddingState.READY
                elif chunk_record.id in pending_by_id:
                    state = EmbeddingState.PENDING
                else:
                    state = EmbeddingState.DISABLED
                metadata_json = canonical_json(dict(chunk_record.metadata))
                connection.execute(
                    """
                    INSERT INTO chunks(
                        chunk_id, source_id, ordinal, title, heading_json, start_line,
                        end_line, body, embedding_text, token_count, metadata_json,
                        content_hash, embedding_state
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk_record.id,
                        source_id,
                        chunk_record.ordinal,
                        chunk_record.title,
                        canonical_json(list(chunk_record.heading)),
                        chunk_record.lines.start,
                        chunk_record.lines.end,
                        chunk_record.text,
                        chunk_record.embedding_text,
                        chunk_record.token_count,
                        metadata_json,
                        chunk_record.content_hash,
                        state.value,
                    ),
                )
                identifiers = self._fts_identifiers(chunk_record.metadata)
                aliases = self._fts_metadata(chunk_record.metadata, "aliases")
                tags = self._fts_metadata(chunk_record.metadata, "tags")
                connection.execute(
                    "INSERT INTO fts_chunks("
                    "chunk_id, path, title, heading, identifiers, aliases, tags, body"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk_record.id,
                        source.relative_path,
                        chunk_record.title,
                        " > ".join(chunk_record.heading),
                        identifiers,
                        aliases,
                        tags,
                        chunk_record.text,
                    ),
                )

            for vector_record in revision.vectors:
                stored_vector_config = connection.execute(
                    """
                    SELECT dimensions, embedding_config_fingerprint
                    FROM chunk_vectors
                    WHERE observed_fingerprint = ?
                    LIMIT 1
                    """,
                    (vector_record.observed_fingerprint,),
                ).fetchone()
                if stored_vector_config is not None and (
                    stored_vector_config["dimensions"] != vector_record.dimensions
                    or stored_vector_config["embedding_config_fingerprint"]
                    != vector_record.config_fingerprint
                ):
                    raise ValueError(
                        "observed vector fingerprint has inconsistent dimensions or configuration"
                    )
                connection.execute(
                    """
                    INSERT INTO chunk_vectors(
                        chunk_id, dimensions, embedding_config_fingerprint,
                        observed_fingerprint, vector
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        vector_record.chunk_id,
                        vector_record.dimensions,
                        vector_record.config_fingerprint,
                        vector_record.observed_fingerprint,
                        encoded_vectors[vector_record.chunk_id],
                    ),
                )
            for pending_record in revision.pending:
                connection.execute(
                    "INSERT INTO embedding_queue(chunk_id, category, message, attempted_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        pending_record.chunk_id,
                        pending_record.category,
                        pending_record.message[:_MAX_DIAGNOSTIC_LENGTH],
                        pending_record.attempted_at.astimezone(UTC).isoformat(),
                    ),
                )
            if revision.update_vault_fingerprint:
                connection.execute(
                    """
                    UPDATE index_state SET
                        schema_version = ?, manifest_fingerprint = ?, parser_fingerprint = ?,
                        chunker_fingerprint = ?, embedding_config_fingerprint = ?
                    WHERE singleton = 1
                    """,
                    (
                        SCHEMA_VERSION,
                        revision.manifest_fingerprint,
                        revision.parser_fingerprint,
                        revision.chunker_fingerprint,
                        revision.embedding_config_fingerprint,
                    ),
                )
            connection.commit()
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            raise StorageError(
                f"could not replace source {source.vault_id}/{source.relative_path}",
                details={"vault_id": source.vault_id, "path": source.relative_path},
            ) from exc
        finally:
            connection.close()

    def record_vault_fingerprint(self, fingerprint: VaultFingerprint) -> None:
        """Record successful vault reconciliation even when it contains no sources."""
        values = (
            fingerprint.vault_id,
            fingerprint.manifest_fingerprint,
            fingerprint.parser_fingerprint,
            fingerprint.chunker_fingerprint,
            fingerprint.embedding_config_fingerprint,
        )
        if any(not value for value in values):
            raise ValueError("vault fingerprint values must be non-empty")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            reconciled_at = datetime.now(UTC).isoformat()
            connection.execute(  # nosec B608 - static parameterized statement
                """
                INSERT INTO vaults(
                    vault_id, manifest_fingerprint, parser_fingerprint,
                    chunker_fingerprint, embedding_config_fingerprint, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(vault_id) DO UPDATE SET
                    manifest_fingerprint = excluded.manifest_fingerprint,
                    parser_fingerprint = excluded.parser_fingerprint,
                    chunker_fingerprint = excluded.chunker_fingerprint,
                    embedding_config_fingerprint = excluded.embedding_config_fingerprint,
                    updated_at = excluded.updated_at
                """,
                (*values, reconciled_at),
            )
            connection.execute(
                """
                UPDATE index_state SET
                    schema_version = ?, manifest_fingerprint = ?, parser_fingerprint = ?,
                    chunker_fingerprint = ?, embedding_config_fingerprint = ?, updated_at = ?
                WHERE singleton = 1
                """,
                (
                    SCHEMA_VERSION,
                    fingerprint.manifest_fingerprint,
                    fingerprint.parser_fingerprint,
                    fingerprint.chunker_fingerprint,
                    fingerprint.embedding_config_fingerprint,
                    reconciled_at,
                ),
            )
            connection.commit()
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise StorageError(
                f"could not record vault fingerprint for {fingerprint.vault_id}",
                details={"vault_id": fingerprint.vault_id},
            ) from exc
        finally:
            connection.close()

    def update_pending_embeddings(
        self,
        vault_id: str,
        embedding_config_fingerprint: str,
        updates: tuple[PendingEmbeddingUpdate, ...],
    ) -> None:
        """Atomically promote or refresh active pending chunks after an embedding retry."""
        if not vault_id or not embedding_config_fingerprint:
            raise ValueError("vault id and embedding fingerprint must be non-empty")
        if not updates:
            return
        identifiers = [update.chunk_id for update in updates]
        if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("pending update chunk IDs must be unique and non-empty")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            configured = connection.execute(
                "SELECT embedding_config_fingerprint FROM vaults WHERE vault_id = ?",
                (vault_id,),
            ).fetchone()
            if (
                configured is None
                or configured["embedding_config_fingerprint"] != embedding_config_fingerprint
            ):
                raise ValueError("pending update fingerprint is stale")

            encoded: dict[str, bytes] = {}
            dimensions: int | None = None
            for update in updates:
                if not update.content_hash or (update.vector is None) == (update.failure is None):
                    raise ValueError("pending update requires exactly one vector or failure")
                row = connection.execute(
                    """
                    SELECT c.content_hash, c.embedding_state, s.vault_id, s.parse_state
                    FROM chunks AS c
                    JOIN sources AS s ON s.source_id = c.source_id
                    WHERE c.chunk_id = ?
                    """,
                    (update.chunk_id,),
                ).fetchone()
                if (
                    row is None
                    or row["vault_id"] != vault_id
                    or row["parse_state"] != "active"
                    or row["embedding_state"] != EmbeddingState.PENDING.value
                    or row["content_hash"] != update.content_hash
                ):
                    raise ValueError("pending update targets a stale chunk")

                if update.vector is not None:
                    vector = update.vector
                    if (
                        vector.chunk_id != update.chunk_id
                        or vector.config_fingerprint != embedding_config_fingerprint
                        or not vector.observed_fingerprint
                        or isinstance(vector.dimensions, bool)
                        or vector.dimensions < 1
                        or vector.observed_fingerprint
                        != observed_fingerprint(embedding_config_fingerprint, vector.dimensions)
                    ):
                        raise ValueError("pending vector identity is invalid")
                    array = np.asarray(vector.vector)
                    if (
                        array.ndim != 1
                        or array.size != vector.dimensions
                        or not np.issubdtype(array.dtype, np.number)
                    ):
                        raise ValueError("pending vector dimensions are invalid")
                    try:
                        encoded_vector = encode_vector(array, dimensions=vector.dimensions)
                    except ValueError as exc:
                        raise ValueError("pending vector values are invalid") from exc
                    if dimensions is None:
                        dimensions = vector.dimensions
                    elif dimensions != vector.dimensions:
                        raise ValueError("pending vectors have mixed dimensions")
                    encoded[update.chunk_id] = encoded_vector
                else:
                    failure = cast(PendingEmbedding, update.failure)
                    if (
                        failure.chunk_id != update.chunk_id
                        or not failure.category
                        or not failure.message
                        or failure.attempted_at.tzinfo is None
                        or failure.attempted_at.utcoffset() is None
                    ):
                        raise ValueError("pending failure is invalid")

            if dimensions is not None:
                stored_dimensions = {
                    cast(int, row[0])
                    for row in connection.execute(
                        "SELECT DISTINCT dimensions FROM chunk_vectors "
                        "WHERE embedding_config_fingerprint = ?",
                        (embedding_config_fingerprint,),
                    )
                }
                if stored_dimensions and stored_dimensions != {dimensions}:
                    raise ValueError("pending vector dimensions do not match active vectors")

            for update in updates:
                if update.vector is not None:
                    vector = update.vector
                    connection.execute(
                        """
                        INSERT INTO chunk_vectors(
                            chunk_id, dimensions, embedding_config_fingerprint,
                            observed_fingerprint, vector
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            update.chunk_id,
                            vector.dimensions,
                            vector.config_fingerprint,
                            vector.observed_fingerprint,
                            encoded[update.chunk_id],
                        ),
                    )
                    connection.execute(
                        "UPDATE chunks SET embedding_state = ? WHERE chunk_id = ?",
                        (EmbeddingState.READY.value, update.chunk_id),
                    )
                    connection.execute(
                        "DELETE FROM embedding_queue WHERE chunk_id = ?", (update.chunk_id,)
                    )
                else:
                    failure = cast(PendingEmbedding, update.failure)
                    connection.execute(
                        """
                        UPDATE embedding_queue SET category = ?, message = ?, attempted_at = ?
                        WHERE chunk_id = ?
                        """,
                        (
                            failure.category,
                            failure.message[:_MAX_DIAGNOSTIC_LENGTH],
                            failure.attempted_at.astimezone(UTC).isoformat(),
                            update.chunk_id,
                        ),
                    )
            connection.commit()
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            raise StorageError(
                f"could not update pending embeddings for {vault_id}",
                details={"vault_id": vault_id},
            ) from exc
        finally:
            connection.close()

    def prepare_for_atomic_replace(self) -> None:
        """Checkpoint WAL state and leave one self-contained database file."""
        connection = self._connect()
        try:
            checkpoint = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint is None or cast(int, checkpoint[0]) != 0:
                raise StorageError("could not checkpoint SQLite index")
            journal_mode = cast(
                str, connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
            )
            if journal_mode.casefold() != "delete":
                raise StorageError("could not finalize SQLite index")
        except StorageError:
            raise
        except sqlite3.Error as exc:
            raise StorageError("could not finalize SQLite index") from exc
        finally:
            connection.close()

    def reset_vaults(self, vault_ids: tuple[str, ...]) -> None:
        """Remove one vault scope and all dependent search state atomically."""
        validate_vault_ids(vault_ids)
        vault_ids_json = canonical_json(list(vault_ids))
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            source_rows = connection.execute(
                "SELECT source_id FROM sources WHERE vault_id IN (SELECT value FROM json_each(?))",
                (vault_ids_json,),
            ).fetchall()
            for row in source_rows:
                self._delete_source_dependents(connection, cast(int, row["source_id"]))
            connection.execute(
                "DELETE FROM sources WHERE vault_id IN (SELECT value FROM json_each(?))",
                (vault_ids_json,),
            )
            connection.execute(
                "DELETE FROM vaults WHERE vault_id IN (SELECT value FROM json_each(?))",
                (vault_ids_json,),
            )
            connection.commit()
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise StorageError("could not reset vault index state") from exc
        finally:
            connection.close()

    def requeue_disabled_chunks(
        self,
        source_keys: tuple[tuple[str, str], ...],
        attempted_at: datetime,
    ) -> tuple[RequeueResult, ...]:
        """Atomically requeue active disabled chunks from explicitly allowed sources."""
        if attempted_at.tzinfo is None or attempted_at.utcoffset() is None:
            raise ValueError("requeue timestamp must be timezone-aware")
        if not source_keys:
            return ()
        if any(not vault_id or not relative_path for vault_id, relative_path in source_keys) or len(
            set(source_keys)
        ) != len(source_keys):
            raise ValueError("requeue source keys must be unique and non-empty")
        source_keys_json = canonical_json(
            [
                {"path": relative_path, "vault_id": vault_id}
                for vault_id, relative_path in source_keys
            ]
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT c.chunk_id, s.vault_id
                FROM chunks AS c
                JOIN sources AS s ON s.source_id = c.source_id
                JOIN json_each(?) AS allowed
                    ON s.vault_id = json_extract(allowed.value, '$.vault_id')
                    AND s.relative_path = json_extract(allowed.value, '$.path')
                WHERE s.parse_state = 'active' AND c.embedding_state = 'disabled'
                ORDER BY s.vault_id, s.relative_path, c.ordinal
                """,
                (source_keys_json,),
            ).fetchall()
            chunk_ids = tuple(cast(str, row["chunk_id"]) for row in rows)
            if not chunk_ids:
                connection.commit()
                return ()

            counts: dict[str, int] = {}
            for row in rows:
                vault_id = cast(str, row["vault_id"])
                counts[vault_id] = counts.get(vault_id, 0) + 1
            chunk_ids_json = canonical_json(list(chunk_ids))
            connection.execute(
                "DELETE FROM chunk_vectors WHERE chunk_id IN (SELECT value FROM json_each(?))",
                (chunk_ids_json,),
            )
            connection.execute(
                "DELETE FROM embedding_queue WHERE chunk_id IN (SELECT value FROM json_each(?))",
                (chunk_ids_json,),
            )
            connection.execute(
                "UPDATE chunks SET embedding_state = ? "
                "WHERE chunk_id IN (SELECT value FROM json_each(?))",
                (EmbeddingState.PENDING.value, chunk_ids_json),
            )
            attempted_at_text = attempted_at.astimezone(UTC).isoformat()
            connection.executemany(
                "INSERT INTO embedding_queue(chunk_id, category, message, attempted_at) "
                "VALUES (?, 'policy_transition', 'semantic embedding re-enabled', ?)",
                ((chunk_id, attempted_at_text) for chunk_id in chunk_ids),
            )
            connection.commit()
            return tuple(RequeueResult(vault_id, counts[vault_id]) for vault_id in sorted(counts))
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise StorageError("could not requeue disabled embeddings") from exc
        finally:
            connection.close()

    def suppress_source(
        self,
        vault_id: str,
        relative_path: str,
        content_hash: str,
        diagnostic: str,
    ) -> None:
        """Hide an invalid prior revision while recording its failing byte hash."""
        if not content_hash:
            raise ValueError("suppressed source content hash must be non-empty")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT source_id FROM sources WHERE vault_id = ? AND relative_path = ?",
                (vault_id, relative_path),
            ).fetchone()
            if row is not None:
                source_id = cast(int, row[0])
                self._delete_source_dependents(connection, source_id)
                connection.execute(
                    "UPDATE sources SET parse_state = 'suppressed', diagnostic = ?, "
                    "content_hash = ? WHERE source_id = ?",
                    (
                        diagnostic[:_MAX_DIAGNOSTIC_LENGTH],
                        content_hash,
                        source_id,
                    ),
                )
            connection.commit()
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise StorageError(
                f"could not suppress source {vault_id}/{relative_path}",
                details={"vault_id": vault_id, "path": relative_path},
            ) from exc
        finally:
            connection.close()

    def delete_sources(
        self,
        keys: tuple[tuple[str, str], ...],
        *,
        fingerprints: tuple[VaultFingerprint, ...] = (),
        disable_semantics_for: tuple[str, ...] = (),
        reconciled_vault_ids: tuple[str, ...] | None = None,
    ) -> None:
        """Finalize deletes, policy state, and fingerprints in one transaction."""
        if any(not vault_id for vault_id in disable_semantics_for):
            raise ValueError("disabled semantic vault IDs must be non-empty")
        reconciled = (
            {fingerprint.vault_id for fingerprint in fingerprints}
            if reconciled_vault_ids is None
            else set(reconciled_vault_ids)
        )
        if any(not vault_id for vault_id in reconciled):
            raise ValueError("reconciled vault IDs must be non-empty")
        fingerprint_ids = {fingerprint.vault_id for fingerprint in fingerprints}
        if not reconciled.issubset(fingerprint_ids):
            raise ValueError("reconciled vault IDs must have final fingerprints")
        for fingerprint in fingerprints:
            values = (
                fingerprint.vault_id,
                fingerprint.manifest_fingerprint,
                fingerprint.parser_fingerprint,
                fingerprint.chunker_fingerprint,
                fingerprint.embedding_config_fingerprint,
            )
            if any(not value for value in values):
                raise ValueError("vault fingerprint values must be non-empty")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            reconciled_at = datetime.now(UTC).isoformat()
            if disable_semantics_for:
                vault_ids_json = canonical_json(list(disable_semantics_for))
                chunk_ids = (
                    "SELECT c.chunk_id FROM chunks AS c "
                    "JOIN sources AS s ON s.source_id = c.source_id "
                    "WHERE s.vault_id IN (SELECT value FROM json_each(?))"
                )
                connection.execute(  # nosec B608 - fixed internal subquery
                    "DELETE FROM chunk_vectors WHERE chunk_id IN (" + chunk_ids + ")",
                    (vault_ids_json,),
                )
                connection.execute(  # nosec B608 - fixed internal subquery
                    "DELETE FROM embedding_queue WHERE chunk_id IN (" + chunk_ids + ")",
                    (vault_ids_json,),
                )
                connection.execute(
                    "UPDATE chunks SET embedding_state = ? WHERE source_id IN ("
                    "SELECT source_id FROM sources WHERE vault_id IN "
                    "(SELECT value FROM json_each(?)))",
                    (EmbeddingState.DISABLED.value, vault_ids_json),
                )
            for vault_id, relative_path in keys:
                row = connection.execute(
                    "SELECT source_id FROM sources WHERE vault_id = ? AND relative_path = ?",
                    (vault_id, relative_path),
                ).fetchone()
                if row is not None:
                    source_id = cast(int, row[0])
                    self._delete_source_dependents(connection, source_id)
                    connection.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
            for fingerprint in sorted(fingerprints, key=lambda item: item.vault_id):
                connection.execute(
                    """
                    INSERT INTO vaults(
                        vault_id, manifest_fingerprint, parser_fingerprint,
                        chunker_fingerprint, embedding_config_fingerprint, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(vault_id) DO UPDATE SET
                        manifest_fingerprint = excluded.manifest_fingerprint,
                        parser_fingerprint = excluded.parser_fingerprint,
                        chunker_fingerprint = excluded.chunker_fingerprint,
                        embedding_config_fingerprint = excluded.embedding_config_fingerprint,
                        updated_at = COALESCE(excluded.updated_at, vaults.updated_at)
                    """,
                    (
                        fingerprint.vault_id,
                        fingerprint.manifest_fingerprint,
                        fingerprint.parser_fingerprint,
                        fingerprint.chunker_fingerprint,
                        fingerprint.embedding_config_fingerprint,
                        reconciled_at if fingerprint.vault_id in reconciled else None,
                    ),
                )
                connection.execute(
                    """
                    UPDATE index_state SET
                        schema_version = ?, manifest_fingerprint = ?, parser_fingerprint = ?,
                        chunker_fingerprint = ?, embedding_config_fingerprint = ?,
                        updated_at = COALESCE(?, updated_at)
                    WHERE singleton = 1
                    """,
                    (
                        SCHEMA_VERSION,
                        fingerprint.manifest_fingerprint,
                        fingerprint.parser_fingerprint,
                        fingerprint.chunker_fingerprint,
                        fingerprint.embedding_config_fingerprint,
                        reconciled_at if fingerprint.vault_id in reconciled else None,
                    ),
                )
            connection.commit()
        except sqlite3.Error as exc:
            if connection.in_transaction:
                connection.rollback()
            raise StorageError("could not finalize source reconciliation") from exc
        finally:
            connection.close()

    def lexical_search(self, request: LexicalRequest) -> tuple[LexicalCandidate, ...]:
        if not request.fts_query.strip():
            return ()
        conditions, parameters = self._query_filters(
            request.vault_ids, request.filters, metadata_alias="c"
        )
        sql = f"""
            SELECT
                c.chunk_id, s.vault_id, s.relative_path, s.source_kind, s.content_hash,
                c.title, c.heading_json, c.start_line, c.end_line, c.body,
                c.metadata_json, bm25(fts_chunks) AS lexical_score
            FROM fts_chunks
            JOIN chunks AS c ON c.chunk_id = fts_chunks.chunk_id
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE fts_chunks MATCH ? AND s.parse_state = 'active' AND {conditions}
            ORDER BY lexical_score, s.vault_id, s.relative_path, c.start_line, c.chunk_id
            LIMIT ?
        """
        connection = self._connect()
        try:
            rows = connection.execute(  # nosec B608 - filters are internally compiled
                sql,
                (request.fts_query, *parameters, request.limit),
            ).fetchall()
            return tuple(self._lexical_candidate(row) for row in rows)
        except sqlite3.Error as exc:
            raise StorageError("could not execute lexical query") from exc
        finally:
            connection.close()

    def load_vectors(self, request: VectorLoadRequest) -> tuple[VectorRecord, ...]:
        conditions, parameters = self._query_filters(
            request.vault_ids, request.filters, metadata_alias="c"
        )
        sql = f"""
            SELECT
                v.chunk_id, v.dimensions, v.embedding_config_fingerprint,
                v.observed_fingerprint, v.vector
            FROM chunk_vectors AS v
            JOIN chunks AS c ON c.chunk_id = v.chunk_id
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE s.parse_state = 'active' AND c.embedding_state = 'ready'
                AND v.observed_fingerprint = ? AND {conditions}
            ORDER BY s.vault_id, s.relative_path, c.start_line, c.chunk_id
            LIMIT ?
        """
        connection = self._connect()
        try:
            rows = connection.execute(  # nosec B608 - filters are internally compiled
                sql,
                (request.observed_fingerprint, *parameters, request.limit),
            ).fetchall()
            return tuple(self._vector_record(row) for row in rows)
        except sqlite3.Error as exc:
            raise StorageError("could not load vectors") from exc
        finally:
            connection.close()

    def load_vectors_complete(
        self,
        vault_ids: tuple[str, ...],
        observed_fingerprint: str,
        filters: StorageFilters,
    ) -> tuple[VectorRecord, ...]:
        """Load every compatible vector in one scoped SQL snapshot without a limit."""
        validate_vault_ids(vault_ids)
        if not observed_fingerprint:
            raise ValueError("observed fingerprint must be non-empty")
        conditions, parameters = self._query_filters(vault_ids, filters, metadata_alias="c")
        sql = f"""
            SELECT
                v.chunk_id, v.dimensions, v.embedding_config_fingerprint,
                v.observed_fingerprint, v.vector
            FROM chunk_vectors AS v
            JOIN chunks AS c ON c.chunk_id = v.chunk_id
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE s.parse_state = 'active' AND c.embedding_state = 'ready'
                AND v.observed_fingerprint = ? AND {conditions}
            ORDER BY s.vault_id, s.relative_path, c.start_line, c.chunk_id
        """
        connection = self._connect()
        try:
            rows = connection.execute(  # nosec B608 - filters are internally compiled
                sql, (observed_fingerprint, *parameters)
            ).fetchall()
            return tuple(self._vector_record(row) for row in rows)
        except sqlite3.Error as exc:
            raise StorageError("could not load complete vector scope") from exc
        finally:
            connection.close()

    def dense_search(
        self,
        request: DenseRequest,
        query_vector: np.ndarray,
    ) -> tuple[DenseCandidate, ...]:
        """Rank the complete eligible vector scope before applying the request limit."""
        normalized = np.asarray(query_vector, dtype=np.float32)
        if normalized.ndim != 1 or not np.isfinite(normalized).all():
            raise ValueError("query vector must be a finite one-dimensional array")
        norm = float(np.linalg.norm(normalized))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("query vector must have a finite non-zero norm")
        normalized = np.asarray(normalized / norm, dtype=np.float32)

        vectors = self.load_vectors_complete(
            request.vault_ids,
            request.observed_fingerprint,
            request.filters,
        )
        compatible = tuple(
            vector
            for vector in vectors
            if vector.dimensions == normalized.size
            and vector.vector.size == normalized.size
            and np.isfinite(vector.vector).all()
        )
        if not compatible:
            return ()

        scores = np.stack([vector.vector for vector in compatible]) @ normalized
        ordered = sorted(
            range(len(compatible)),
            key=lambda index: (-float(scores[index]), index),
        )[: request.limit]
        return tuple(
            DenseCandidate(compatible[index].chunk_id, float(scores[index])) for index in ordered
        )

    def vector_scope(
        self,
        vault_ids: tuple[str, ...],
        observed_fingerprint: str,
        filters: StorageFilters,
    ) -> VectorScope:
        """Return the exact compatible-vector count and observed dimensions for a scope."""
        validate_vault_ids(vault_ids)
        if not observed_fingerprint:
            raise ValueError("observed fingerprint must be non-empty")
        conditions, parameters = self._query_filters(vault_ids, filters, metadata_alias="c")
        sql = f"""
            SELECT v.dimensions, COUNT(*) AS vector_count
            FROM chunk_vectors AS v
            JOIN chunks AS c ON c.chunk_id = v.chunk_id
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE s.parse_state = 'active' AND c.embedding_state = 'ready'
                AND v.observed_fingerprint = ? AND {conditions}
            GROUP BY v.dimensions
            ORDER BY v.dimensions
        """
        connection = self._connect()
        try:
            rows = connection.execute(  # nosec B608 - filters are internally compiled
                sql, (observed_fingerprint, *parameters)
            ).fetchall()
            return VectorScope(
                count=sum(cast(int, row["vector_count"]) for row in rows),
                dimensions=tuple(cast(int, row["dimensions"]) for row in rows),
            )
        except sqlite3.Error as exc:
            raise StorageError("could not inspect vector scope") from exc
        finally:
            connection.close()

    def identifier_candidates(
        self,
        vault_ids: tuple[str, ...],
        filters: StorageFilters,
    ) -> tuple[IdentifierCandidate, ...]:
        """Load scoped exact-match fields without hydrating chunk bodies."""
        validate_vault_ids(vault_ids)
        conditions, parameters = self._query_filters(vault_ids, filters, metadata_alias="c")
        sql = f"""
            SELECT c.chunk_id, s.vault_id, s.relative_path, c.title, c.metadata_json
            FROM chunks AS c
            JOIN sources AS s ON s.source_id = c.source_id
            WHERE s.parse_state = 'active' AND {conditions}
            ORDER BY s.vault_id, s.relative_path, c.start_line, c.chunk_id
        """
        connection = self._connect()
        try:
            rows = connection.execute(  # nosec B608 - filters are internally compiled
                sql, parameters
            ).fetchall()
            return tuple(
                IdentifierCandidate(
                    chunk_id=cast(str, row["chunk_id"]),
                    vault_id=cast(str, row["vault_id"]),
                    relative_path=cast(str, row["relative_path"]),
                    title=cast(str, row["title"]),
                    metadata=_parse_json_mapping(cast(str, row["metadata_json"])),
                )
                for row in rows
            )
        except sqlite3.Error as exc:
            raise StorageError("could not load identifier candidates") from exc
        finally:
            connection.close()

    def chunks_by_ids(
        self, ids: tuple[str, ...], *, vault_ids: tuple[str, ...]
    ) -> dict[str, StoredChunk]:
        validate_vault_ids(vault_ids)
        if not ids:
            return {}
        unique_ids = tuple(dict.fromkeys(ids))
        batch_size = 900
        vault_ids_json = canonical_json(list(vault_ids))
        connection = self._connect()
        records: dict[str, StoredChunk] = {}
        try:
            for start in range(0, len(unique_ids), batch_size):
                batch = unique_ids[start : start + batch_size]
                id_placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    self._stored_chunk_select() + f" WHERE c.chunk_id IN ({id_placeholders}) "
                    "AND s.vault_id IN (SELECT value FROM json_each(?)) "
                    "AND s.parse_state = 'active'",
                    (*batch, vault_ids_json),
                ).fetchall()
                records.update((record.id, record) for record in map(self._stored_chunk, rows))
            return records
        except sqlite3.Error as exc:
            raise StorageError("could not load chunks") from exc
        finally:
            connection.close()

    def source_provenance(self, vault_id: str, relative_path: str) -> SourceProvenance | None:
        """Load active source hash, kind, and committed timestamp without invoking FTS."""
        validate_vault_ids((vault_id,))
        validate_relative_prefix(relative_path)
        connection = self._connect()
        try:
            source = connection.execute(
                "SELECT source_kind, content_hash, indexed_at FROM sources "
                "WHERE vault_id = ? AND relative_path = ? AND parse_state = 'active'",
                (vault_id, relative_path),
            ).fetchone()
            if source is None:
                return None
            try:
                indexed_at = datetime.fromisoformat(cast(str, source["indexed_at"]))
            except ValueError as exc:
                raise StorageError("stored source timestamp is invalid") from exc
            if indexed_at.tzinfo is None or indexed_at.utcoffset() is None:
                raise StorageError("stored source timestamp is invalid")
            return SourceProvenance(
                vault_id=vault_id,
                relative_path=relative_path,
                source_kind=SourceKind(cast(str, source["source_kind"])),
                source_hash=cast(str, source["content_hash"]),
                indexed_at=indexed_at,
            )
        except sqlite3.Error as exc:
            raise StorageError("could not load source provenance") from exc
        finally:
            connection.close()

    def pending_chunks(self, vault_ids: tuple[str, ...]) -> tuple[StoredChunk, ...]:
        validate_vault_ids(vault_ids)
        placeholders = ",".join("?" for _ in vault_ids)
        connection = self._connect()
        try:
            rows = connection.execute(
                self._stored_chunk_select() + f" WHERE s.vault_id IN ({placeholders}) "
                "AND s.parse_state = 'active' AND c.embedding_state = 'pending' "
                "ORDER BY s.vault_id, s.relative_path, c.ordinal",
                vault_ids,
            ).fetchall()
            return tuple(self._stored_chunk(row) for row in rows)
        except sqlite3.Error as exc:
            raise StorageError("could not load pending chunks") from exc
        finally:
            connection.close()

    def snapshot(self, vault_ids: tuple[str, ...]) -> IndexSnapshot:
        validate_vault_ids(vault_ids)
        vault_ids_json = canonical_json(list(vault_ids))
        connection = self._connect()
        try:
            counts = connection.execute(
                """
                SELECT
                    COUNT(DISTINCT s.source_id) AS source_count,
                    COUNT(c.chunk_id) AS chunk_count,
                    COALESCE(SUM(CASE WHEN c.embedding_state = 'ready' THEN 1 ELSE 0 END), 0)
                        AS ready_count,
                    COALESCE(SUM(CASE WHEN c.embedding_state = 'pending' THEN 1 ELSE 0 END), 0)
                        AS pending_count,
                    COALESCE(SUM(LENGTH(v.vector)), 0) AS vector_bytes
                FROM sources AS s
                LEFT JOIN chunks AS c ON c.source_id = s.source_id
                LEFT JOIN chunk_vectors AS v ON v.chunk_id = c.chunk_id
                WHERE s.vault_id IN (SELECT value FROM json_each(?))
                    AND s.parse_state = 'active'
                """,
                (vault_ids_json,),
            ).fetchone()
            state = connection.execute("SELECT * FROM index_state WHERE singleton = 1").fetchone()
            if state is None:
                raise StorageError("stored index schema state is missing")
            vault_fingerprints = connection.execute(
                "SELECT manifest_fingerprint, parser_fingerprint, chunker_fingerprint, "
                "embedding_config_fingerprint, updated_at FROM vaults "
                "WHERE vault_id IN (SELECT value FROM json_each(?))",
                (vault_ids_json,),
            ).fetchall()
            expected_vault_count = len(set(vault_ids))
            vector_rows = connection.execute(
                """
                SELECT DISTINCT v.observed_fingerprint, v.dimensions,
                    v.embedding_config_fingerprint
                FROM chunk_vectors AS v
                JOIN chunks AS c ON c.chunk_id = v.chunk_id
                JOIN sources AS s ON s.source_id = c.source_id
                WHERE s.vault_id IN (SELECT value FROM json_each(?))
                    AND s.parse_state = 'active' AND c.embedding_state = 'ready'
                ORDER BY v.observed_fingerprint, v.dimensions,
                    v.embedding_config_fingerprint
                """,
                (vault_ids_json,),
            ).fetchall()
            observed = tuple(sorted({cast(str, row[0]) for row in vector_rows}))
            vector_dimensions = tuple(sorted({cast(int, row[1]) for row in vector_rows}))
            vector_configs = tuple(sorted({cast(str, row[2]) for row in vector_rows}))
            return IndexSnapshot(
                source_count=cast(int, counts["source_count"]),
                chunk_count=cast(int, counts["chunk_count"]),
                ready_count=cast(int, counts["ready_count"]),
                pending_count=cast(int, counts["pending_count"]),
                vector_bytes=cast(int, counts["vector_bytes"]),
                indexed_at=self._oldest_vault_timestamp(vault_fingerprints, expected_vault_count),
                schema_version=cast(int, state["schema_version"]),
                manifest_fingerprint=self._agreed_vault_fingerprint(
                    vault_fingerprints, "manifest_fingerprint", expected_vault_count
                ),
                parser_fingerprint=self._agreed_vault_fingerprint(
                    vault_fingerprints, "parser_fingerprint", expected_vault_count
                ),
                chunker_fingerprint=self._agreed_vault_fingerprint(
                    vault_fingerprints, "chunker_fingerprint", expected_vault_count
                ),
                embedding_config_fingerprint=self._agreed_vault_fingerprint(
                    vault_fingerprints,
                    "embedding_config_fingerprint",
                    expected_vault_count,
                ),
                observed_fingerprints=observed,
                vector_dimensions=vector_dimensions,
                vector_config_fingerprints=vector_configs,
            )
        except sqlite3.Error as exc:
            raise StorageError("could not load index snapshot") from exc
        finally:
            connection.close()

    @staticmethod
    def _optional_timestamp(raw: object) -> datetime | None:
        if raw is None:
            return None
        try:
            timestamp = datetime.fromisoformat(cast(str, raw))
        except ValueError as exc:
            raise StorageError("stored index timestamp is invalid") from exc
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise StorageError("stored index timestamp is invalid")
        return timestamp

    @classmethod
    def _oldest_vault_timestamp(
        cls, rows: Sequence[sqlite3.Row], expected_vault_count: int
    ) -> datetime | None:
        if len(rows) != expected_vault_count or any(row["updated_at"] is None for row in rows):
            return None
        timestamps = tuple(cls._optional_timestamp(row["updated_at"]) for row in rows)
        if any(timestamp is None for timestamp in timestamps):
            return None
        return min(cast(tuple[datetime, ...], timestamps))

    @staticmethod
    def _agreed_vault_fingerprint(
        rows: Sequence[sqlite3.Row], field_name: str, expected_vault_count: int
    ) -> str | None:
        if len(rows) != expected_vault_count:
            return None
        values = {cast(str, row[field_name]) for row in rows}
        return values.pop() if len(values) == 1 else None

    @staticmethod
    def _delete_source_dependents(connection: sqlite3.Connection, source_id: int) -> None:
        connection.execute(
            "DELETE FROM fts_chunks WHERE chunk_id IN ("
            "SELECT chunk_id FROM chunks WHERE source_id = ?)",
            (source_id,),
        )
        connection.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))

    @classmethod
    def _fts_identifiers(cls, metadata: Mapping[str, JsonValue]) -> str:
        return " ".join(
            cls._fts_metadata(metadata, key)
            for key in sorted(metadata)
            if key not in {"aliases", "tags"}
        )

    @staticmethod
    def _fts_metadata(metadata: Mapping[str, JsonValue], key: str) -> str:
        value = metadata.get(key)
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            return " ".join(
                item if isinstance(item, str) else canonical_json(item) for item in value
            )
        return canonical_json(value)

    @staticmethod
    def _validate_revision(revision: SourceRevision) -> dict[str, bytes]:
        source = revision.source
        fingerprints = (
            revision.manifest_fingerprint,
            revision.parser_fingerprint,
            revision.chunker_fingerprint,
            revision.embedding_config_fingerprint,
        )
        if any(not fingerprint for fingerprint in fingerprints):
            raise ValueError("revision fingerprints must be non-empty")

        chunk_ids: set[str] = set()
        ordinals: set[int] = set()
        for chunk_record in revision.chunks:
            if not chunk_record.id or chunk_record.id in chunk_ids:
                raise ValueError("chunk IDs must be unique and non-empty")
            if chunk_record.ordinal in ordinals:
                raise ValueError("chunk ordinals must be unique")
            if (
                chunk_record.vault_id != source.vault_id
                or chunk_record.relative_path != source.relative_path
            ):
                raise ValueError("chunk source identity does not match revision")
            if chunk_record.ordinal < 0 or chunk_record.token_count < 0:
                raise ValueError("chunk numeric fields must be non-negative")
            canonical_json(dict(chunk_record.metadata))
            chunk_ids.add(chunk_record.id)
            ordinals.add(chunk_record.ordinal)

        vector_ids: set[str] = set()
        dimensions_by_fingerprint: dict[str, int] = {}
        encoded: dict[str, bytes] = {}
        for vector_record in revision.vectors:
            if vector_record.chunk_id not in chunk_ids or vector_record.chunk_id in vector_ids:
                raise ValueError("vectors must map uniquely to revision chunks")
            if vector_record.config_fingerprint != revision.embedding_config_fingerprint:
                raise ValueError("vector configuration fingerprint does not match revision")
            if not vector_record.observed_fingerprint or vector_record.dimensions < 1:
                raise ValueError("vector fingerprints and dimensions must be valid")
            prior_dimension = dimensions_by_fingerprint.setdefault(
                vector_record.observed_fingerprint, vector_record.dimensions
            )
            if prior_dimension != vector_record.dimensions:
                raise ValueError("observed vector fingerprint has inconsistent dimensions")
            encoded[vector_record.chunk_id] = encode_vector(
                vector_record.vector, dimensions=vector_record.dimensions
            )
            vector_ids.add(vector_record.chunk_id)

        pending_ids: set[str] = set()
        for pending_record in revision.pending:
            if pending_record.chunk_id not in chunk_ids or pending_record.chunk_id in pending_ids:
                raise ValueError("pending rows must map uniquely to revision chunks")
            if pending_record.chunk_id in vector_ids:
                raise ValueError("a chunk cannot be both ready and pending")
            if not pending_record.category or not pending_record.message:
                raise ValueError("pending errors must include category and message")
            if (
                pending_record.attempted_at.tzinfo is None
                or pending_record.attempted_at.utcoffset() is None
            ):
                raise ValueError("pending attempt timestamps must be timezone-aware")
            pending_ids.add(pending_record.chunk_id)
        return encoded

    @staticmethod
    def _query_filters(
        vault_ids: tuple[str, ...], filters: StorageFilters, *, metadata_alias: str
    ) -> tuple[str, tuple[object, ...]]:
        conditions: list[str] = []
        parameters: list[object] = []
        placeholders = ",".join("?" for _ in vault_ids)
        conditions.append(f"s.vault_id IN ({placeholders})")
        parameters.extend(vault_ids)
        if filters.vault_ids:
            filter_placeholders = ",".join("?" for _ in filters.vault_ids)
            conditions.append(f"s.vault_id IN ({filter_placeholders})")
            parameters.extend(filters.vault_ids)
        if filters.path_prefix is not None:
            prefix = filters.path_prefix.rstrip("/")
            conditions.append(
                "(s.relative_path = ? OR substr(s.relative_path, 1, length(?) + 1) = ? || '/')"
            )
            parameters.extend((prefix, prefix, prefix))
        if filters.source_kind is not None:
            conditions.append("s.source_kind = ?")
            parameters.append(filters.source_kind.value)
        for key, value in sorted(filters.frontmatter.items()):
            escaped_key = key.replace('"', '\\"')
            json_path = f'$."{escaped_key}"'
            if value is None:
                conditions.append(f"json_type({metadata_alias}.metadata_json, ?) IS NOT NULL")
                parameters.append(json_path)
            conditions.append(
                f"json_extract({metadata_alias}.metadata_json, ?) IS json_extract(?, '$')"
            )
            parameters.extend((json_path, canonical_json(value)))
        return " AND ".join(conditions), tuple(parameters)

    @staticmethod
    def _lexical_candidate(row: sqlite3.Row) -> LexicalCandidate:
        return LexicalCandidate(
            chunk_id=cast(str, row["chunk_id"]),
            vault_id=cast(str, row["vault_id"]),
            relative_path=cast(str, row["relative_path"]),
            source_kind=SourceKind(cast(str, row["source_kind"])),
            source_hash=cast(str, row["content_hash"]),
            title=cast(str, row["title"]),
            heading=_parse_heading(cast(str, row["heading_json"])),
            lines=LineRange(cast(int, row["start_line"]), cast(int, row["end_line"])),
            text=cast(str, row["body"]),
            metadata=_parse_json_mapping(cast(str, row["metadata_json"])),
            score=cast(float, row["lexical_score"]),
        )

    @staticmethod
    def _vector_record(row: sqlite3.Row) -> VectorRecord:
        dimensions = cast(int, row["dimensions"])
        raw = cast(bytes, row["vector"])
        vector = decode_vector(raw, dimensions=dimensions)
        return VectorRecord(
            chunk_id=cast(str, row["chunk_id"]),
            dimensions=dimensions,
            config_fingerprint=cast(str, row["embedding_config_fingerprint"]),
            observed_fingerprint=cast(str, row["observed_fingerprint"]),
            vector=vector,
        )

    @staticmethod
    def _stored_chunk_select() -> str:
        return """
            SELECT
                c.chunk_id, s.vault_id, s.relative_path, s.source_kind,
                s.content_hash AS source_hash, c.ordinal, c.title, c.heading_json,
                c.start_line, c.end_line, c.body, c.embedding_text, c.token_count,
                c.metadata_json, c.content_hash AS chunk_content_hash,
                c.embedding_state, q.category, q.message, q.attempted_at
            FROM chunks AS c
            JOIN sources AS s ON s.source_id = c.source_id
            LEFT JOIN embedding_queue AS q ON q.chunk_id = c.chunk_id
        """

    @staticmethod
    def _stored_chunk(row: sqlite3.Row) -> StoredChunk:
        pending: PendingEmbedding | None = None
        if row["category"] is not None:
            pending = PendingEmbedding(
                chunk_id=cast(str, row["chunk_id"]),
                category=cast(str, row["category"]),
                message=cast(str, row["message"]),
                attempted_at=datetime.fromisoformat(cast(str, row["attempted_at"])),
            )
        return StoredChunk(
            id=cast(str, row["chunk_id"]),
            vault_id=cast(str, row["vault_id"]),
            relative_path=cast(str, row["relative_path"]),
            source_kind=SourceKind(cast(str, row["source_kind"])),
            source_hash=cast(str, row["source_hash"]),
            ordinal=cast(int, row["ordinal"]),
            title=cast(str, row["title"]),
            heading=_parse_heading(cast(str, row["heading_json"])),
            lines=LineRange(cast(int, row["start_line"]), cast(int, row["end_line"])),
            text=cast(str, row["body"]),
            embedding_text=cast(str, row["embedding_text"]),
            token_count=cast(int, row["token_count"]),
            metadata=_parse_json_mapping(cast(str, row["metadata_json"])),
            content_hash=cast(str, row["chunk_content_hash"]),
            embedding_state=EmbeddingState(cast(str, row["embedding_state"])),
            pending=pending,
        )
