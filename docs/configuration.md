# Configuration

`vault-rag` separates portable vault policy from machine-local registration and
embedding settings. TOML is parsed as a closed schema: unknown fields are
rejected, and no configuration file stores a credential value.

## Locations and permissions

The default registry is `$XDG_CONFIG_HOME/vault-rag/config.toml`, where
`XDG_CONFIG_HOME` defaults to `~/.config`. The default index is
`$XDG_DATA_HOME/vault-rag/index.sqlite3`, where `XDG_DATA_HOME` defaults to
`~/.local/share`.

Vault-rag creates the `vault-rag` configuration and data directories with mode
`0700`, and makes its SQLite database mode `0600`. The registry writer creates
its parent directory with mode `0700` and writes replacements atomically with
mode `0600`. Operators should likewise keep a hand-authored registry mode
`0600`; `doctor` reports registry permissions that are visible to group or
other users.

Pass `--config PATH` before a command to use a different registry. It is the
only registry-path override. Index state remains under the XDG data location.

## Vault manifest

Each registered root must contain `.vault-rag.toml`.

```toml
schema_version = 1
id = "example-vault"
egress_policy = "local-only"
include = ["**/*.md"]
exclude = ["drafts/**"]

[metadata]
frontmatter_fields = ["status", "owner"]
```

| Field | Required/default | Validation and effect |
| --- | --- | --- |
| `schema_version` | required | Must be integer `1`. |
| `id` | required | Lowercase identifier matching `^[a-z0-9][a-z0-9-]{0,62}$`; it must equal the registry key used by profiles. |
| `egress_policy` | required | `local-only` or `remote-allowed`. See [egress policy](#egress-policy). |
| `include` | required | A list of gitwildmatch patterns evaluated relative to the canonical vault root. Only selected supported text files can be indexed. |
| `exclude` | `[]` | A list of relative gitwildmatch patterns. Excludes win over includes. |
| `metadata.frontmatter_fields` | `[]` | A list of frontmatter keys preserved as selected metadata, searchable values, and permitted structured-filter keys. |

Vault paths are canonicalized before discovery. Default exclusions always block
`.git/**`, nested `.git/**`, `.obsidian/**`, and nested `.obsidian/**`, even if
an include pattern matches them. Keep generated runtime directories and
unsupported or sensitive attachments in `exclude`. Excluded paths are skipped before any containment,
extension, or decoding check, so an excluded symlink or
binary is never inspected. A selected file with an unsupported extension is a
bounded per-file indexing diagnostic, not a format conversion request, and it
does not stop other files from indexing.

The manifest is portable: do not put absolute paths, endpoint routes, model
names, API keys, or credentials in it.

## Machine registry

The registry has three top-level fields:

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
tokenizer = "cl100k_base"
revision = "1"
dimensions = 1024 # optional; omit for the model default
target_min_tokens = 500
target_max_tokens = 900
overlap_tokens = 80
max_input_tokens = 8191
```

### `vaults`

`vaults` defaults to an empty mapping. Each key is a vault ID and each value
has one required `path` field. The path is expanded and resolved strictly when
a profile is used. Its manifest must exist and its `id` must match the key.

### `profiles`

`profiles` defaults to an empty mapping. Each profile has a required non-empty
`vaults` list. Each entry must name a registered vault. A one-vault profile is
the normal isolated configuration; a multi-vault profile is an explicit,
deliberate allowlist.

No command searches all registered vaults. `--profile NAME` has precedence over
`VAULT_RAG_PROFILE`; if neither yields a name, profile-sensitive commands fail.
The `profile list` command reads only the registry and does not resolve the
embedding model.

### `embedding`

`embedding` is required.

| Field | Required/default | Validation and effect |
| --- | --- | --- |
| `base_url` | required | OpenAI-compatible embeddings API base. Remote endpoints require HTTPS. Plaintext HTTP is accepted only for `local` loopback hosts (`localhost`, IPv4 loopback, or `::1`). Vault-rag appends `/embeddings`. |
| `api_key_env` | optional | Name of the environment variable holding the API key. Its value is never read from TOML or printed. It must differ from `model_env` when present. |
| `model_env` | required | Name of the environment variable holding the embedding model identifier. The referenced value must be non-empty when resolving a profile. |
| `endpoint_class` | required | Exactly `local` or `remote`; this is the operator's classification of the effective route. |
| `batch_size` | `64` | Maximum inputs per embedding request; integer from `1` through `256`, inclusive. |
| `max_batch_tokens` | `300000` | Maximum tokens summed across all inputs in one embedding request. It must be at least `max_input_tokens` and at most `300000`. Both item and token limits apply. |
| `tokenizer` | `cl100k_base` | Tokenizer used for chunking and embedding-request token budgets. |
| `revision` | `1` | Bounded operator-controlled model revision. Increment it whenever a model alias changes backing model or semantics; a change forces re-embedding. |
| `dimensions` | omitted | Optional positive OpenAI-compatible output dimension request. It is sent to the endpoint and must match the returned width. A change forces re-embedding. |
| `target_min_tokens` | `500` | Lower target used when selecting chunk boundaries. |
| `target_max_tokens` | `900` | Upper target for normal chunk assembly. Must be less than `max_input_tokens`. |
| `overlap_tokens` | `80` | Maximum overlap used only for split sections. Must be less than `target_min_tokens`. |
| `max_input_tokens` | `8191` | Strict model input ceiling; a produced embedding text must remain below it. |

`target_min_tokens` must not exceed `target_max_tokens`. The model value, not
the `model_env` variable name, participates in embedding compatibility. The
operator revision, requested dimensions, model, tokenizer, chunking token
limits, endpoint class, normalization, parser, and chunker settings participate in
compatibility. Changing one makes existing vectors incompatible and requires
re-embedding. `status` and `doctor` report revision and requested dimensions
without exposing the route or credentials.

Set values only in the process environment, for example:

```bash
export LITELLM_API_KEY="…"
export VAULT_RAG_EMBEDDING_MODEL="your-embedding-model"
```

## Shared service configuration

`vault-rag serve` reads one strict YAML document instead of the machine-local
registry. The service owns all managed paths beneath its `--data-root`:

```text
/data/repos/<vault-id>          managed Git checkout
/data/index.sqlite3             shared SQLite index
/data/sync-state/<vault-id>.json  reconciliation state
```

Start from `deploy/docker/service.example.yaml`. Its synthetic URLs are safe to
commit but cannot synchronize a real repository. Copy it to the ignored
`deploy/docker/service.yaml`, then set real HTTPS repository and embedding
URLs:

```yaml
schemaVersion: 1
syncInterval: 5m
mcp:
  enabled: false
  allowedHosts: []
  allowedOrigins: []
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

The top-level fields are `schemaVersion`, a bounded `syncInterval` such as
`5m`, optional `mcp`, `repositories`, `profiles`, `embedding`, and optional
`storage`.
Repository keys are lowercase vault IDs; each repository requires an HTTPS URL
without embedded credentials and a full `refs/heads/...` or `refs/tags/...`
reference. `credentialEnv` names an environment variable, never a YAML
credential. Profiles are explicit non-empty allowlists of configured repository
keys. The `embedding` fields use the same validation and compatibility rules as
the [machine registry](#embedding), with camel-case YAML names.

`mcp.enabled` defaults to `false`. Enabling it requires at least one explicit
`allowedHosts` entry; `allowedOrigins` may remain empty for native clients that
send no Origin header. Disabled MCP must retain empty allowlists. The transport
is stateless JSON Streamable HTTP at exact path `/mcp`; see the
[MCP integration guide](mcp.md) for transport, security, schema, and client
contracts.

For a Docker or Kubernetes secret mount, leave the ordinary variable unset and
set `<NAME>_FILE` to the mounted file. The service reads one bounded value at
startup. Compose uses `GITHUB_TOKEN_FILE=/run/secrets/github_token` and
`LITELLM_API_KEY_FILE=/run/secrets/litellm_api_key`; it supplies
`VAULT_RAG_EMBEDDING_MODEL` directly as a non-secret model identifier. Do not
put tokens, local override YAML files, checkouts, state files, or SQLite
databases into version control or an image build context.

The production container runs:

```bash
vault-rag serve --service-config /config/service.yaml --data-root /data \
  --host 0.0.0.0 --port 8080
```

`/health/live` reports process liveness. `/health/ready` becomes successful
when at least one configured profile has a usable lexical source/index pair.
Cached healthy checkouts and indexes remain usable during a temporary GitHub or
LiteLLM failure, so a restart with unchanged repository SHA does not require
re-embedding before lexical requests can succeed.

## Helm values and Kubernetes secrets

The chart renders one schema-v2 service configuration for three process roles:
a migration Job, a stateless API Deployment, and a lease-coordinated worker
Deployment. PostgreSQL is the durable store; worker Git checkouts use bounded
ephemeral `emptyDir` storage.

Each role uses a distinct existing database Secret:

- `api.databaseSecret` supplies read-only served-state access;
- `worker.databaseSecret` supplies revision and synchronization writes;
- `migrations.databaseSecret` supplies application-schema DDL access.

The API runtime Secret contains only the LiteLLM key. The worker runtime Secret
contains the GitHub and LiteLLM keys. The chart rejects reused Secret names
across those trust boundaries. Helm supplies the selected keys through
`secretKeyRef` environment variables; it never copies values into the ConfigMap
or rendered manifest. The application also retains the bounded `<NAME>_FILE`
contract for operators that mount secrets as files and for Docker Compose.

The migration Job runs `vault-rag db migrate`. Application readiness and release
gates must use `vault-rag db check`; the check validates the complete packaged
migration history without changing the database.

An operator values file can begin:

```yaml
image:
  repository: ghcr.io/your-org/vault-rag
  tag: "0.1.0"
  digest: "" # sha256:<64 lowercase hex> overrides tag when set
imagePullSecrets:
  - name: vault-rag-registry
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  ownerRole: vault_rag_owner
  pool:
    minSize: 1
    maxSize: 4
    timeout: 5s
  connectTimeout: 5s
  statementTimeout: 5s
  lockTimeout: 5s
  idleTransactionTimeout: 30s
  allowInsecureTransport: true # required only for an approved non-TLS private DB route
  cleanup:
    enabled: false
    keepPromoted: 3
    minAge: 7d
    batchSize: 100

api:
  databaseSecret:
    name: vault-rag-api-db
    key: database_url
  runtimeSecret:
    name: vault-rag-api-runtime
    litellmKey: litellm_api_key
worker:
  terminationGracePeriodSeconds: 600
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

For non-loopback PostgreSQL DSNs, prefer `sslmode=require`, `verify-ca`, or
`verify-full`. `allowInsecureTransport: true` is an explicit exception for an
operator-approved private non-TLS route such as the current DB-core contract;
it is never inferred. Keep it `false` for TLS-enabled PostgreSQL.

Render and deploy with:

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

The API and worker default to two replicas. PostgreSQL session advisory locks
serialize reconciliation per vault while allowing different vaults to progress
concurrently. Losing a worker's lock session aborts that reconciliation; another
replica can acquire the released advisory lock. The API can optionally use the
chart HPA. The Service is `ClusterIP`, and the chart creates no public route or
database/PVC.

The worker has no serving endpoint, so the chart does not add a redundant
PID-only liveness probe. Container exit remains the worker liveness signal.
The default 600-second termination grace lets a `SIGTERM`-requested shutdown
finish its active reconciliation pass before Kubernetes sends `SIGKILL`.

The chart's default NetworkPolicy CIDRs are the non-routable documentation
range `192.0.2.0/32`. A production values file must replace the ingress,
database, and HTTPS CIDRs with the narrowest trusted client, PostgreSQL,
GitHub, and LiteLLM ranges. Worker and migration pods deny all ingress.

Credential files are read at startup. Restart only the affected Deployment
after rotating an API or worker Secret:

```bash
kubectl rollout restart deployment/vault-rag-api --namespace "$namespace"
kubectl rollout restart deployment/vault-rag-worker --namespace "$namespace"
kubectl -n "$namespace" rollout status deployment/vault-rag-api
kubectl -n "$namespace" rollout status deployment/vault-rag-worker
```

Uninstalling the chart does not alter or delete the external PostgreSQL
database. Follow the
[PostgreSQL/Kubernetes deployment runbook](postgresql-kubernetes-deployment.md)
for database roles, migrations, private routing, monitoring, backup/restore, and
rollback gates.

## Verification boundaries

Public CI uses only repository-local synthetic service and deployment contracts.
It never needs private repositories, credentials, or operator routing
configuration. The chart's ClusterIP-only boundary is intentional: configure
Envoy/Tailscale exposure in the private GitOps repository.

The operator is responsible for the external centralized 25-case evaluation and
two-machine routing verification. These deployment gates cannot be inferred
from a public package, chart render, or repository-local test.

Private deployment inputs are not release artifacts.

## Egress policy

A `remote-allowed` vault may use either endpoint class. A `local-only` vault
may use only a machine configuration classified as `local`. For a profile, the
most restrictive bound manifest wins: if any vault is `local-only` while the
endpoint is `remote`, semantic indexing and query embedding are disabled for
the entire profile. Lexical indexing and FTS search remain available, and
status/search report a semantic degraded reason.

The endpoint class describes the effective route as declared by the operator.
Vault-rag cannot infer whether a local proxy forwards data elsewhere. Remote
routes must use HTTPS. HTTP is allowed only for a loopback URL classified as
`local`; this prevents a bearer key and selected vault text from crossing a
plaintext network connection.

## Registration and replacement

Run:

```bash
vault-rag register /absolute/path/to/vault
```

Registration resolves the supplied root, validates its manifest, reads the
existing registry, and atomically records the canonical path under the manifest
ID. If that ID is already registered to a different canonical root, the command
fails unless `--replace` is supplied. Registration also removes stale registry
aliases that point to the same canonical root under a different ID. It does not
create profiles or modify the vault manifest.

## Evaluation configuration

`evaluate --file PATH` reads a separate operator-owned TOML document. It has
`schema_version = 1`, required `thresholds`, and one to 1,000 cases. Acceptance
requires at least 25 cases. Each case has a bounded identifier, `kind`
(`semantic` or `identifier`), a non-empty query of at most 10,000 characters,
and one to 20 expected source identities. Prefer:

```toml
expected_sources = [{ vault_id = "example-vault", path = "notes/expected.md" }]
```

Legacy `expected_paths = ["notes/expected.md"]` remains valid only when the
bound profile contains one vault. A multi-vault profile requires `vault_id` and
fails closed before searching if it is omitted. Absolute paths, backslashes,
`.` and `..` path segments are rejected.

Thresholds `recall_at_5` and `identifier_rank_1` are finite values in `[0, 1]`.
`minimum_cases` defaults to `25` and cannot be configured below 25 by a loaded
acceptance file. `p95_latency_ms` defaults to `500.0` and cannot be configured
above 500. `require_no_degradation` is always true. A report passes only when
all quality, case-count, latency, citation, and degradation gates pass. Unknown
fields are rejected.

Keep evaluation files operator-owned and out of public fixtures when they name
private notes or contain representative private queries.
