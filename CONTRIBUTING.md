# Contributing

Bug reports and focused pull requests are welcome. Use synthetic examples and
remove credentials, private note content, internal URLs and personal paths from
issues, logs and screenshots. Report vulnerabilities privately as described in
[SECURITY.md](SECURITY.md).

## Development

Install Python 3.12+ and uv, then from a clone:

```bash
uv sync --frozen
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest -q
uv build
npm ci --ignore-scripts
git ls-files -z '*.md' | xargs -0 node_modules/.bin/markdownlint-cli2 --
bash examples/lexical-demo.sh
```

Without a PostgreSQL test DSN, database-only tests skip; this is not full backend
coverage. CI runs PostgreSQL 18/pgvector contracts separately and exercises the
synthetic Compose demo and Helm rendering on GitHub-hosted Linux runners.

## Database safety

PostgreSQL tests destroy and recreate schemas. Use a disposable PostgreSQL 18
instance with pgvector, the exact database name `vault_rag_test`, and an explicit
loopback IP/port. Never use a live service database. Unset `PGDATABASE`, `PGHOST`,
`PGHOSTADDR`, `PGOPTIONS`, `PGPORT`, `PGSERVICE`, `PGSERVICEFILE` and `PGSYSCONFDIR`
to prevent libpq overrides. Supply `VAULT_RAG_TEST_POSTGRES_DSN` and set
`VAULT_RAG_REQUIRE_POSTGRES=1` when the database suite must run rather than skip.

## Review expectations

Keep changes scoped; explain user-visible behavior and test meaningful failure
cases. Maintain profile isolation, path containment, hash verification, egress
policy and bounded credential-free diagnostics. Update the public docs when
commands/configuration change. Do not add network calls to ordinary tests or
commit generated indexes, credentials, private operator config or agent transcripts.

PR CI uses GitHub-hosted runners with read-only repository permissions. Only
successful main pushes publish container images; contributors need no publishing
credentials. Do not add self-hosted runner labels or `pull_request_target` workflows.
