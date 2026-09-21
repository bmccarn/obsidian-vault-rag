# Changelog

## 0.1.0 — Initial public release

- Local SQLite lexical, dense and hybrid retrieval with explicit vault profiles.
- Incremental indexing, source citations, filters and hash-verified reads.
- PostgreSQL 18/pgvector shared API and lease-coordinated workers.
- Stateless HTTP MCP tools for profile discovery, search, read and status.
- Synthetic CLI/Compose demos, Helm chart and private deployment guidance.

This repository starts with a clean public history. Commands and Python imports
remain `vault-rag` and `vault_rag`. Container images use immutable main-commit tags;
GitHub release versions do not imply an identically named container tag. See
[security limitations](SECURITY.md) before running the shared service.
