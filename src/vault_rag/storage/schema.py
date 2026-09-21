"""SQLite schema versioning for the persistent retrieval index."""

SCHEMA_VERSION = 1

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS vaults (
        vault_id TEXT PRIMARY KEY,
        manifest_fingerprint TEXT NOT NULL,
        parser_fingerprint TEXT NOT NULL,
        chunker_fingerprint TEXT NOT NULL,
        embedding_config_fingerprint TEXT NOT NULL,
        updated_at TEXT
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS sources (
        source_id INTEGER PRIMARY KEY,
        vault_id TEXT NOT NULL REFERENCES vaults(vault_id) ON DELETE CASCADE,
        relative_path TEXT NOT NULL,
        folded_path TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
        mtime_ns INTEGER NOT NULL,
        parse_state TEXT NOT NULL CHECK (parse_state IN ('active', 'suppressed')),
        diagnostic TEXT,
        indexed_at TEXT NOT NULL,
        UNIQUE(vault_id, relative_path),
        UNIQUE(vault_id, folded_path)
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        source_id INTEGER NOT NULL REFERENCES sources(source_id) ON DELETE CASCADE,
        ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
        title TEXT NOT NULL,
        heading_json TEXT NOT NULL CHECK (json_valid(heading_json)),
        start_line INTEGER NOT NULL CHECK (start_line >= 1),
        end_line INTEGER NOT NULL CHECK (end_line >= start_line),
        body TEXT NOT NULL,
        embedding_text TEXT NOT NULL,
        token_count INTEGER NOT NULL CHECK (token_count >= 0),
        metadata_json TEXT NOT NULL CHECK (json_valid(metadata_json)),
        content_hash TEXT NOT NULL,
        embedding_state TEXT NOT NULL CHECK (
            embedding_state IN ('ready', 'pending', 'disabled')
        ),
        UNIQUE(source_id, ordinal),
        UNIQUE(chunk_id)
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS chunk_vectors (
        chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
        dimensions INTEGER NOT NULL CHECK (dimensions > 0),
        embedding_config_fingerprint TEXT NOT NULL,
        observed_fingerprint TEXT NOT NULL,
        vector BLOB NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS embedding_queue (
        chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
        category TEXT NOT NULL,
        message TEXT NOT NULL,
        attempted_at TEXT NOT NULL
    ) STRICT
    """,
    """
    CREATE TABLE IF NOT EXISTS index_state (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        schema_version INTEGER NOT NULL,
        manifest_fingerprint TEXT,
        parser_fingerprint TEXT,
        chunker_fingerprint TEXT,
        embedding_config_fingerprint TEXT,
        updated_at TEXT
    ) STRICT
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
        chunk_id UNINDEXED,
        path,
        title,
        heading,
        identifiers,
        aliases,
        tags,
        body
    )
    """,
    "CREATE INDEX IF NOT EXISTS sources_vault_state ON sources(vault_id, parse_state)",
    "CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source_id)",
    "CREATE INDEX IF NOT EXISTS vectors_fingerprint ON chunk_vectors(observed_fingerprint)",
)

REQUIRED_COLUMNS: dict[str, frozenset[str]] = {
    "vaults": frozenset(
        {
            "vault_id",
            "manifest_fingerprint",
            "parser_fingerprint",
            "chunker_fingerprint",
            "embedding_config_fingerprint",
            "updated_at",
        }
    )
}

REQUIRED_OBJECTS = frozenset(
    {
        "vaults",
        "sources",
        "chunks",
        "chunk_vectors",
        "embedding_queue",
        "index_state",
        "fts_chunks",
    }
)
