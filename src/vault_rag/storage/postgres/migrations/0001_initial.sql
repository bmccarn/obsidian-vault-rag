CREATE TABLE vault_rag.vaults (
    vault_id text PRIMARY KEY CHECK (length(vault_id) BETWEEN 1 AND 200),
    configured_ref text NOT NULL CHECK (length(configured_ref) BETWEEN 1 AND 500),
    active_revision_id uuid,
    fetched_sha text CHECK (fetched_sha IS NULL OR fetched_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'),
    attempted_sha text CHECK (attempted_sha IS NULL OR attempted_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'),
    reconciled_sha text CHECK (reconciled_sha IS NULL OR reconciled_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'),
    configured_sha text CHECK (
        configured_sha IS NULL OR configured_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'
    ),
    checkout_sha text CHECK (
        checkout_sha IS NULL OR checkout_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'
    ),
    configured_at timestamptz,
    fetched_at timestamptz,
    checkout_at timestamptz,
    attempted_at timestamptz,
    reconciled_at timestamptz,
    index_report jsonb,
    sync_degraded_at timestamptz,
    last_fetch_attempt_at timestamptz,
    last_fetch_success_at timestamptz,
    last_reconciliation_attempt_at timestamptz,
    last_reconciliation_success_at timestamptz,
    sync_degradation_category text CHECK (
        sync_degradation_category IS NULL OR length(sync_degradation_category) <= 100
    ),
    sync_degradation_message text CHECK (
        sync_degradation_message IS NULL OR length(sync_degradation_message) <= 1000
    ),
    semantic_degradation_category text CHECK (
        semantic_degradation_category IS NULL OR length(semantic_degradation_category) <= 100
    ),
    semantic_degradation_message text CHECK (
        semantic_degradation_message IS NULL OR length(semantic_degradation_message) <= 1000
    ),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE vault_rag.vault_revisions (
    revision_id uuid PRIMARY KEY,
    vault_id text NOT NULL REFERENCES vault_rag.vaults(vault_id) ON DELETE CASCADE,
    commit_sha text NOT NULL CHECK (commit_sha ~ '^[0-9a-f]{40}([0-9a-f]{24})?$'),
    state text NOT NULL CHECK (
        state IN ('building', 'active', 'active_degraded', 'superseded', 'failed')
    ),
    manifest_json jsonb,
    manifest_fingerprint text,
    parser_fingerprint text,
    chunker_fingerprint text,
    embedding_config_fingerprint text,
    observed_fingerprint text,
    lexical_complete boolean NOT NULL DEFAULT false,
    fully_reconciled boolean NOT NULL DEFAULT false,
    source_count integer NOT NULL DEFAULT 0 CHECK (source_count >= 0),
    chunk_count integer NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),
    ready_count integer NOT NULL DEFAULT 0 CHECK (ready_count >= 0),
    pending_count integer NOT NULL DEFAULT 0 CHECK (pending_count >= 0),
    vector_bytes bigint NOT NULL DEFAULT 0 CHECK (vector_bytes >= 0),
    diagnostics jsonb NOT NULL DEFAULT '[]'::jsonb,
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    promoted_at timestamptz,
    superseded_at timestamptz,
    CHECK (
        state IN ('building', 'failed')
        OR (
            manifest_fingerprint IS NOT NULL
            AND parser_fingerprint IS NOT NULL
            AND chunker_fingerprint IS NOT NULL
            AND embedding_config_fingerprint IS NOT NULL
        )
    ),
    UNIQUE (vault_id, revision_id)
);

ALTER TABLE vault_rag.vaults
    ADD CONSTRAINT vaults_active_revision_fk
    FOREIGN KEY (vault_id, active_revision_id)
    REFERENCES vault_rag.vault_revisions(vault_id, revision_id)
    DEFERRABLE INITIALLY DEFERRED;

CREATE TABLE vault_rag.source_blobs (
    content_hash text PRIMARY KEY CHECK (content_hash ~ '^sha256:[0-9a-f]{64}$'),
    byte_length bigint NOT NULL CHECK (byte_length >= 0),
    encoding text NOT NULL CHECK (encoding = 'utf-8'),
    content bytea NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (octet_length(content) = byte_length)
);

CREATE TABLE vault_rag.embeddings (
    content_hash text NOT NULL CHECK (content_hash ~ '^sha256:[0-9a-f]{64}$'),
    configuration_fingerprint text NOT NULL,
    observed_fingerprint text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions > 0),
    embedding vector NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (content_hash, configuration_fingerprint, observed_fingerprint),
    CHECK (vector_dims(embedding) = dimensions)
);

CREATE TABLE vault_rag.revision_sources (
    revision_id uuid NOT NULL REFERENCES vault_rag.vault_revisions(revision_id) ON DELETE CASCADE,
    relative_path text NOT NULL,
    folded_path text NOT NULL,
    source_kind text NOT NULL,
    content_hash text NOT NULL CHECK (content_hash ~ '^sha256:[0-9a-f]{64}$'),
    source_blob_hash text NOT NULL REFERENCES vault_rag.source_blobs(content_hash),
    size_bytes bigint NOT NULL CHECK (size_bytes >= 0),
    mtime_ns bigint NOT NULL,
    parse_state text NOT NULL CHECK (parse_state IN ('active', 'suppressed')),
    diagnostic text CHECK (diagnostic IS NULL OR length(diagnostic) <= 1000),
    indexed_at timestamptz NOT NULL,
    metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (revision_id, relative_path),
    UNIQUE (revision_id, folded_path),
    UNIQUE (revision_id, relative_path, source_blob_hash),
    CHECK (parse_state = 'suppressed' OR content_hash = source_blob_hash)
);

CREATE TABLE vault_rag.revision_chunks (
    revision_id uuid NOT NULL,
    chunk_id text NOT NULL,
    source_path text NOT NULL,
    source_blob_hash text NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal >= 0),
    title text NOT NULL,
    heading_json jsonb NOT NULL,
    start_line integer NOT NULL CHECK (start_line >= 1),
    end_line integer NOT NULL CHECK (end_line >= start_line),
    body text NOT NULL,
    embedding_text text NOT NULL,
    token_count integer NOT NULL CHECK (token_count >= 0),
    metadata_json jsonb NOT NULL,
    content_hash text NOT NULL CHECK (content_hash ~ '^sha256:[0-9a-f]{64}$'),
    embedding_state text NOT NULL CHECK (embedding_state IN ('ready', 'pending', 'disabled')),
    embedding_config_fingerprint text,
    observed_fingerprint text,
    dimensions integer CHECK (dimensions IS NULL OR dimensions > 0),
    search_document tsvector NOT NULL,
    PRIMARY KEY (revision_id, chunk_id),
    UNIQUE (revision_id, source_path, ordinal),
    FOREIGN KEY (revision_id, source_path, source_blob_hash)
        REFERENCES vault_rag.revision_sources(revision_id, relative_path, source_blob_hash)
        ON DELETE CASCADE,
    FOREIGN KEY (content_hash, embedding_config_fingerprint, observed_fingerprint)
        REFERENCES vault_rag.embeddings(
            content_hash,
            configuration_fingerprint,
            observed_fingerprint
        ),
    CHECK (
        (embedding_state = 'ready'
            AND embedding_config_fingerprint IS NOT NULL
            AND observed_fingerprint IS NOT NULL
            AND dimensions IS NOT NULL)
        OR
        (embedding_state <> 'ready'
            AND embedding_config_fingerprint IS NULL
            AND observed_fingerprint IS NULL
            AND dimensions IS NULL)
    )
);

CREATE TABLE vault_rag.embedding_queue (
    revision_id uuid NOT NULL,
    chunk_id text NOT NULL,
    category text NOT NULL CHECK (length(category) BETWEEN 1 AND 100),
    message text NOT NULL CHECK (length(message) BETWEEN 1 AND 1000),
    attempt_count integer NOT NULL DEFAULT 1 CHECK (attempt_count > 0),
    last_attempted_at timestamptz NOT NULL,
    next_eligible_at timestamptz NOT NULL,
    terminal boolean NOT NULL DEFAULT false,
    PRIMARY KEY (revision_id, chunk_id),
    FOREIGN KEY (revision_id, chunk_id)
        REFERENCES vault_rag.revision_chunks(revision_id, chunk_id) ON DELETE CASCADE
);

CREATE TABLE vault_rag.worker_leases (
    lease_name text PRIMARY KEY CHECK (length(lease_name) BETWEEN 1 AND 100),
    owner_id text NOT NULL CHECK (length(owner_id) BETWEEN 1 AND 200),
    generation bigint NOT NULL DEFAULT 1 CHECK (generation > 0),
    acquired_at timestamptz NOT NULL,
    heartbeat_at timestamptz NOT NULL,
    expires_at timestamptz NOT NULL,
    CHECK (expires_at > heartbeat_at)
);

CREATE TABLE vault_rag.sync_requests (
    request_id uuid PRIMARY KEY,
    vault_id text REFERENCES vault_rag.vaults(vault_id) ON DELETE CASCADE,
    state text NOT NULL CHECK (state IN ('pending', 'claimed', 'completed', 'failed')),
    claim_token uuid,
    outcome text CHECK (outcome IS NULL OR length(outcome) <= 100),
    created_at timestamptz NOT NULL DEFAULT now(),
    claimed_at timestamptz,
    completed_at timestamptz,
    CHECK (
        (state = 'pending' AND claimed_at IS NULL AND claim_token IS NULL
            AND completed_at IS NULL AND outcome IS NULL)
        OR (state = 'claimed' AND claimed_at IS NOT NULL AND claim_token IS NOT NULL
            AND completed_at IS NULL AND outcome IS NULL)
        OR (state IN ('completed', 'failed') AND claimed_at IS NOT NULL
            AND claim_token IS NULL AND completed_at IS NOT NULL AND outcome IS NOT NULL)
    )
);

CREATE INDEX vaults_active_revision_idx
    ON vault_rag.vaults(active_revision_id) WHERE active_revision_id IS NOT NULL;
CREATE INDEX vault_revisions_vault_state_started_idx
    ON vault_rag.vault_revisions(vault_id, state, started_at DESC);
CREATE INDEX revision_sources_revision_folded_idx
    ON vault_rag.revision_sources(revision_id, folded_path);
CREATE INDEX revision_chunks_revision_source_state_idx
    ON vault_rag.revision_chunks(revision_id, source_path, embedding_state);
CREATE INDEX revision_chunks_revision_fingerprint_idx
    ON vault_rag.revision_chunks(revision_id, observed_fingerprint, dimensions)
    WHERE embedding_state = 'ready';
CREATE INDEX revision_chunks_search_document_idx
    ON vault_rag.revision_chunks USING gin(search_document);
CREATE INDEX embeddings_identity_dimensions_idx
    ON vault_rag.embeddings(content_hash, observed_fingerprint, dimensions);
CREATE INDEX embedding_queue_eligibility_idx
    ON vault_rag.embedding_queue(next_eligible_at)
    WHERE terminal = false;
CREATE INDEX sync_requests_unclaimed_age_idx
    ON vault_rag.sync_requests(created_at)
    WHERE state = 'pending';
