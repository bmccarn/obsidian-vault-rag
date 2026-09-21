# Obsidian Vault RAG

Search your Obsidian vault from a terminal or coding agent, with exact source citations.
This is a standalone tool, not an Obsidian plugin or an answer-generating chatbot.
The executable and Python package remain named `vault-rag`.

`vault-rag` is a local, vault-scoped retrieval CLI for Obsidian notes and selected
text files. It combines SQLite FTS5/BM25 lexical search with optional
OpenAI-compatible (including LiteLLM) embeddings, and returns source-grounded
citations instead of generated answers.

## Start here

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
The CLI works on macOS and Linux; CI exercises Linux with Python 3.12–3.14.

```bash
git clone https://github.com/bmccarn/obsidian-vault-rag.git
cd obsidian-vault-rag
uv sync --frozen
bash examples/lexical-demo.sh
```

The demo creates an isolated temporary vault, indexes one synthetic note, searches
for it, and reads its source. It requires **no API key or embedding service** and
sends no note/query text to a provider. First installation/tokenizer setup may
fetch public dependencies. It prints its temporary directory for inspection.
The demo's `local-only` policy prohibits remote embeddings; indexing and lexical
search complete without provider requests.

Choose your next step:

| Need | Guide |
| --- | --- |
| Search a local vault | [Configure one vault](#configure-one-vault) |
| Understand SQLite vs shared PostgreSQL | [Architecture](docs/architecture.md) |
| Connect an agent | [MCP guide](docs/mcp.md) |
| Try the shared service with synthetic data | [Compose demo](#run-the-synthetic-service-demo-with-docker-compose) |
| Deploy privately on Kubernetes | [Deployment guide](docs/postgresql-kubernetes-deployment.md) |
| Contribute or report a vulnerability | [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) |

## Read-only threat model

The CLI reads registered vaults and writes only its machine-local registry and
disposable SQLite index. It does not create, edit, move, or delete vault files;
use normal editor and Git workflows for vault changes. Every operation resolves
an explicit named profile. There is no implicit search across all registered
vaults, and profile filters are enforced before ranking and reading.

A remote embedding route receives text selected for indexing and query text.
Remote routes must use HTTPS; plaintext HTTP is accepted only for a loopback
endpoint classified as `local`. Choose the route and classify it honestly, use
`local-only` manifests where needed, and never place credentials in a manifest
or this repository.

## Included interfaces

This release provides:

- portable vault manifests and a machine-local registry;
- Markdown, `.txt`, `.log`, and `.json` discovery, parsing, and incremental
  SQLite indexing;
- explicit lexical-only, dense-only, and deterministic hybrid search with
  structured filters and hash-verified reads;
- `register`, `profile list`, `index`, `search`, `read`, `status`, `doctor`,
  and `evaluate` CLI commands;
- the private `vault-rag api`, `vault-rag worker`, and `vault-rag db migrate`
  service commands, one production Docker image, and a local Docker Compose
  deployment;
- a read-only MCP `2026-07-28` adapter at `/mcp` using stateless JSON
  Streamable HTTP, structured input/output schemas, resources, prompts, cache
  hints, and DNS-rebinding protection.

Filesystem watchers, client installers, Pi integrations, and public routing
remain outside this release. See the [MCP integration guide](docs/mcp.md) for
the complete agent workflow and private client configuration.

## Install

Install from this checkout or the public Git repository; no PyPI release is claimed.

For a local checkout, install the console script with:

```bash
uv tool install .
vault-rag version --json
```

Run without installing with:

```bash
uvx --from . vault-rag version --json
```

To install directly from the public repository:

```bash
uv tool install "git+https://github.com/bmccarn/obsidian-vault-rag.git"
uvx --from "git+https://github.com/bmccarn/obsidian-vault-rag.git" vault-rag version --json
```

The JSON version response is canonical, for example
`{"version":"0.1.0"}`.

## Configure one vault

Create a committed `.vault-rag.toml` at the vault root. This portable file
contains selection and egress policy, never local paths, endpoint URLs, or
secrets.

```toml
schema_version = 1
id = "example-vault"
egress_policy = "remote-allowed"

include = [
  "**/*.md",
  "attachments/**/*.txt",
  "attachments/**/*.log",
  "attachments/**/*.json",
]

exclude = [
  ".git/**",
  ".obsidian/**",
  "attachments/**/*.csv",
]

[metadata]
frontmatter_fields = ["type", "status", "tags"]
```

Create the machine-local registry at
`$XDG_CONFIG_HOME/vault-rag/config.toml` (default:
`~/.config/vault-rag/config.toml`). Its paths and embedding route are local to
this machine:

```toml
[vaults.example-vault]
path = "/absolute/path/to/your/vault"

[profiles.example]
vaults = ["example-vault"]

[embedding]
base_url = "http://127.0.0.1:4000/v1"
api_key_env = "LITELLM_API_KEY"
model_env = "VAULT_RAG_EMBEDDING_MODEL"
endpoint_class = "local"
batch_size = 64
max_batch_tokens = 300000
revision = "1"
dimensions = 1024 # example dimension setting; optional for other models
```

Keep the registry user-only (`chmod 600`); vault-rag creates its configuration
and data directories with user-only permissions. Export values rather than
writing them to TOML:

```bash
export LITELLM_API_KEY="…"
export VAULT_RAG_EMBEDDING_MODEL="text-embedding-3-large"
```

For existing configurations, registration canonicalizes the supplied root,
validates its manifest, and records it atomically:

```bash
vault-rag register /absolute/path/to/your/vault --json
```

See [configuration](docs/configuration.md) for all fields, validation, and
replacement rules.

## Core workflow

Every profile-sensitive command takes `--profile`; the environment fallback is
`VAULT_RAG_PROFILE`.

```bash
vault-rag profile list --json
vault-rag index --profile example --json
vault-rag search --profile example "deployment readiness" --mode hybrid --limit 5 --json
vault-rag search --profile example "deployment readiness" --mode lexical --path-prefix notes --frontmatter 'status="active"' --json
vault-rag search --profile example "deployment readiness" --mode dense --vault example-vault --source-kind markdown --json
vault-rag read --profile example notes/overview.md --start-line 1 --end-line 80 --json
vault-rag status --profile example --json
vault-rag doctor --profile example --json
vault-rag evaluate --profile example --file .vault-rag/eval.toml --mode hybrid --json
```

`read` infers the vault only for a single-vault profile. A multi-vault profile
must include `--vault <vault-id>` for reads. Search and read citations use the
stable form `vault://<vault-id>/<relative-path>#L<start>-L<end>`; percent
encoding protects unusual path characters.

`index` is incremental. Use `--rebuild` only when a compatibility change asks
for a controlled rebuild. `--rebuild` re-embeds the entire corpus, so do not run
it while the embedding endpoint is unavailable: the rebuilt index stays fully
searchable lexically, but every chunk is marked pending until a later `index`
succeeds. The database is disposable local state at
`$XDG_DATA_HOME/vault-rag/index.sqlite3` (default:
`~/.local/share/vault-rag/index.sqlite3`). Delete that database to discard an
index, then run `index --rebuild`; do not copy it between machines.

Evaluation is fail-closed: hybrid and dense runs fail acceptance when semantic
retrieval degrades, expected sources are vault-qualified for multi-vault
profiles, and the default gates require at least 25 cases, p95 latency at most
500 ms, configured quality, and zero invalid citations.

If an embedding attempt fails, affected chunks remain without dense vectors in
`pending`; they are retried on later reconciliation and gain vectors when
embedding succeeds. If embedding is disallowed by policy, affected chunks remain
without dense vectors in `disabled`, excluded from the pending count, and are
requeued if policy later permits embeddings. In both cases, lexical search
remains available. JSON search/status output marks the semantic degraded state
instead of pretending dense retrieval succeeded.

## Run the synthetic service demo with Docker Compose

Requires Docker Engine with Compose v2. This is a **test/demo stack**, not a
production deployment: PostgreSQL/pgvector, a local Git fixture server, two APIs,
two workers, and a verification client. It uses only committed synthetic notes
and generated fixture secrets. No GitHub token or provider key is needed.

```bash
bash deploy/docker/prepare-demo.sh
docker compose build
docker compose up -d postgres git-fixture migrate api-a api-b worker-a worker-b
docker compose run --rm smoke verify
curl --fail http://127.0.0.1:8080/health/ready
curl --fail --json '{"profile":"fixture","query":"deployment","mode":"lexical"}' \
  http://127.0.0.1:8080/v1/search
docker compose down
```

The setup script creates the exact secret filenames referenced by `compose.yaml`
in ignored `secrets/`, and refuses to overwrite existing files. Fixture files are
mode 0444 inside a user-only host directory so the non-root Linux containers can
read their bind mounts; never put real credentials in this fixture setup. Compose mounts
`deploy/docker/service.example.yaml` directly; editing `service.yaml` does not
change this demo. The fixture profile is `fixture`, not `example`. Only the first
API publishes loopback port 8080. The embedding URL is deliberately nonfunctional;
lexical results remain available and semantic degradation is reported explicitly.
The demo database uses a shared role and plaintext transport on its isolated
network; production needs least-privilege roles and verified TLS.

`docker compose down` retains demo data. Use `docker compose down --volumes` only
when intentionally discarding this disposable demo database and fixture volumes.
For production use the [private deployment guide](docs/postgresql-kubernetes-deployment.md),
not a fixture configuration adapted by adding real credentials.

CI publishes `ghcr.io/bmccarn/obsidian-vault-rag` only after its checks pass,
using an immutable SHA tag (`sha-<full-commit>`). Select a successful main build's
tag or digest; a `latest` or semantic-version image tag is not promised. Published
images target Linux amd64; ARM hosts need emulation or a native source build.

```bash
docker build -t obsidian-vault-rag:local .
# Substitute the full SHA of a successful published main build:
docker pull ghcr.io/bmccarn/obsidian-vault-rag:sha-<full-commit>
```

## Deploy the shared service with Helm

The Helm chart deploys the same production image as three PostgreSQL-backed
process roles:

- a pre-install/pre-upgrade migration Job with DDL-only credentials;
- a stateless API Deployment with read-only data access and no GitHub token;
- a lease-coordinated worker Deployment with Git and indexing access.

The API and worker default to two replicas. PostgreSQL holds all durable index
and synchronization state; checkout directories are bounded `emptyDir` volumes.
The chart creates no database, StatefulSet, or PVC.

Create separate existing Secrets for each process. Production database roles
should enforce the same API/worker/migrator privilege split:

```bash
namespace=default
kubectl -n "$namespace" create secret generic vault-rag-api-db \
  --from-literal=database_url="$API_DATABASE_URL"
kubectl -n "$namespace" create secret generic vault-rag-api-runtime \
  --from-literal=litellm_api_key="$LITELLM_API_KEY"
kubectl -n "$namespace" create secret generic vault-rag-worker-db \
  --from-literal=database_url="$WORKER_DATABASE_URL"
kubectl -n "$namespace" create secret generic vault-rag-worker-runtime \
  --from-literal=github_token="$GITHUB_TOKEN" \
  --from-literal=litellm_api_key="$LITELLM_API_KEY"
kubectl -n "$namespace" create secret generic vault-rag-migrator-db \
  --from-literal=database_url="$MIGRATOR_DATABASE_URL"
```

When the image is in a private registry, create a registry pull secret. The
token needs permission to pull the package (`read:packages` for private GHCR
packages):

```bash
kubectl -n "$namespace" create secret docker-registry vault-rag-registry \
  --docker-server=ghcr.io \
  --docker-username="$GITHUB_USERNAME" \
  --docker-password="$GHCR_TOKEN"
```

Author an operator-owned values file. This synthetic example omits optional
resource, autoscaling, monitoring, and NetworkPolicy overrides:

```yaml
# values.private.yaml
image:
  repository: ghcr.io/bmccarn/obsidian-vault-rag
  tag: "sha-<full-commit>"
  digest: "" # sha256:<64 lowercase hex>; takes precedence over tag
imagePullSecrets:
  - name: vault-rag-registry
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
api:
  databaseSecret:
    name: vault-rag-api-db
    key: database_url
  runtimeSecret:
    name: vault-rag-api-runtime
    litellmKey: litellm_api_key
worker:
  databaseSecret:
    name: vault-rag-worker-db
    key: database_url
  runtimeSecret:
    name: vault-rag-worker-runtime
    githubToken: github_token
    litellmKey: litellm_api_key
migrations:
  databaseSecret:
    name: vault-rag-migrator-db
    key: database_url
embeddingModel: text-embedding-3-large
serviceConfig:
  schemaVersion: 2
  syncInterval: 5m
  mcp:
    enabled: true
    allowedHosts:
      - vault-rag.example.test
    allowedOrigins:
      - https://vault-rag.example.test
  repositories:
    example-vault:
      url: https://github.com/your-org/your-vault.git
      ref: refs/heads/main
      credentialEnv: GITHUB_TOKEN
  profiles:
    example:
      vaults: [example-vault]
  embedding:
    baseUrl: https://litellm.example.com/v1
    apiKeyEnv: LITELLM_API_KEY
    modelEnv: VAULT_RAG_EMBEDDING_MODEL
    endpointClass: remote
    revision: baseline-2026-08
    dimensions: 1024
```

Validate and install it:

```bash
helm lint charts/vault-rag --values values.private.yaml
helm upgrade --install vault-rag charts/vault-rag \
  --namespace "$namespace" --create-namespace --values values.private.yaml
kubectl -n "$namespace" rollout status deployment/vault-rag-api
kubectl -n "$namespace" rollout status deployment/vault-rag-worker
helm status vault-rag --namespace "$namespace"
kubectl -n "$namespace" get deployment,job,service,hpa,pdb,networkpolicy \
  -l app.kubernetes.io/instance=vault-rag
```

The chart creates only a `ClusterIP` Service for the API. It creates no
Ingress, Gateway, HTTPRoute, LoadBalancer, DNS record, or public route. Configure
private LAN/tailnet routing in the operator's GitOps repository.

When enabled, the API serves MCP at exact path `/mcp`. It uses sessionless JSON
Streamable HTTP: configured clients do not launch a local Vault RAG process,
hold a sticky server session, or keep an idle SSE stream. Keep the MCP hostname
private and route neither `/metrics` nor `/v1/admin/sync` through the agent
hostname. See [MCP 2026-07-28 agent integration](docs/mcp.md).

Credential files are read at process startup. Restart only the roles affected
by a rotated runtime or database Secret:

```bash
kubectl rollout restart deployment/vault-rag-api --namespace "$namespace"
kubectl rollout restart deployment/vault-rag-worker --namespace "$namespace"
kubectl -n "$namespace" rollout status deployment/vault-rag-api
kubectl -n "$namespace" rollout status deployment/vault-rag-worker
```

`helm uninstall vault-rag --namespace "$namespace"` removes application
workloads but does not alter or delete the external PostgreSQL database.
Database backup, restore, role management, and destructive cleanup remain
operator-owned. See the
[PostgreSQL/Kubernetes deployment runbook](docs/postgresql-kubernetes-deployment.md)
for preflight, rollout, failure, and rollback gates.

## Verification boundaries

Repository CI proves the credential-free packaging, documentation, Docker/Compose
build and first-boot smoke, Helm, and repository-local synthetic service scenario
contracts. It uses only synthetic repositories and transports; it does not contact
an operator repository or embedding route.

Repository CI does not verify private routing, Kubernetes behavior,
private-vault credentials, live outages, two-machine access, or the 25-case p95 benchmark.
Those operator gates require operator-owned credentials, repositories, and network
resources. CI success does not certify a particular private deployment.

Private deployment inputs are not release artifacts.

## Exit codes

Every command uses one stable process exit code.

| Exit code | Meaning |
| --- | --- |
| `0` | The command completed. |
| `1` | Unexpected internal failure or a storage failure (`internal_error`, `storage_error`). |
| `2` | Configuration, input, parse, chunking, or rebuild-required error (`config_error`, `parse_error`, `chunking_error`, `rebuild_required`). |
| `3` | Path containment, profile isolation, or freshness error (`security_error`, `stale_source`). |
| `4` | `index` completed with chunks still awaiting embeddings, or `semantic_unavailable`. |
| `5` | `evaluate` completed and emitted the full JSON report, but at least one threshold failed. |

`index` returns exit code 4 on a **successful** run that left chunks pending;
the index is complete lexically and the pending chunks are retried on the next
reconciliation. Automation should treat `0` and `4` as success for `index` and
inspect `pending_chunks` in the JSON report rather than the exit code alone.

## Architecture and limitations

The local CLI uses SQLite. Shared API/worker deployments use PostgreSQL 18 with
pgvector and immutable indexed revisions. See [architecture](docs/architecture.md)
for data flow and [deployment](docs/postgresql-kubernetes-deployment.md) for setup.

There is no application authentication or per-user authorization. All clients
able to reach the service are trusted and may select its configured profiles.
Profiles scope retrieval; they do not isolate mutually untrusted users. Keep the
service on a private trusted network. Publishing this source does not make the
running endpoint safe to expose to the Internet. See [SECURITY.md](SECURITY.md).

## Development

GitHub Actions runs the locked suite on Python 3.12, 3.13, and 3.14. From a
checkout, run the same core checks used for this phase:

```bash
uv sync --frozen
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest --cov=vault_rag --cov-report=term-missing -q
uv build
uvx --from ./dist/vault_rag-0.1.0-py3-none-any.whl vault-rag version --json
```

PostgreSQL contract and scale tests destroy and recreate schemas. Run them only
against a dedicated PostgreSQL 18 database named `vault_rag_test`, using an
explicit loopback IP address and port. Unset `PGDATABASE`, `PGHOST`,
`PGHOSTADDR`, `PGOPTIONS`, `PGPORT`, `PGSERVICE`, `PGSERVICEFILE`, and
`PGSYSCONFDIR` so libpq cannot retarget the connection. CI additionally sets
`VAULT_RAG_REQUIRE_POSTGRES=1` so a missing `VAULT_RAG_TEST_POSTGRES_DSN` fails
rather than skips:

```bash
VAULT_RAG_TEST_POSTGRES_DSN=postgresql://vault_rag:password@127.0.0.1:5432/vault_rag_test \
VAULT_RAG_REQUIRE_POSTGRES=1 \
uv run pytest -q
```

For retrieval behavior, freshness, ranking, and evaluation details, see
[retrieval](docs/retrieval.md).

## License

License selection is pending. No open-source license is currently granted.
