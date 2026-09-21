CREATE INDEX revision_sources_source_blob_hash_idx
    ON vault_rag.revision_sources (source_blob_hash);

CREATE INDEX revision_chunks_embedding_identity_idx
    ON vault_rag.revision_chunks (
        content_hash,
        embedding_config_fingerprint,
        observed_fingerprint
    );
