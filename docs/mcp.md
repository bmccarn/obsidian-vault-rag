# MCP 2026-07-28 agent integration

Vault RAG exposes its transport-neutral retrieval facade at the exact
Streamable HTTP route `/mcp`. The adapter uses the official Python MCP SDK 2.x
and the MCP `2026-07-28` protocol revision.

## Why the endpoint is stateless JSON over HTTP

The endpoint is configured with:

- `stateless_http=True`;
- `json_response=True`;
- a one-MiB request-body limit;
- explicit Host and Origin allowlists; and
- no application-owned client/session registry.

A modern client sends its protocol version, client information, and
capabilities with every request. Vault RAG returns one ordinary JSON response
and does not issue `Mcp-Session-Id`. It does not hold an idle SSE connection.
The optional Streamable HTTP GET receive channel is intentionally disabled
with HTTP 405 because this server sends no unsolicited messages. Only POST is
part of the supported MCP transport surface.
MCP `2026-07-28` also defines the POST-based `subscriptions/listen` stream.
Vault RAG explicitly does not advertise or serve it: tools, prompts, and
resources change only with a deployment, so an idle change stream would add
per-client resource use without delivering useful events. Discovery therefore
reports `listChanged: false` and `resources.subscribe: false`.
Requests can land on either API replica, so adding clients does not create one
resident process or sticky server session per machine. Memory is bounded by
normal concurrent in-flight HTTP requests and the existing PostgreSQL pool,
not by the number of configured clients.

The SDK can also negotiate with supported 2025-era clients. `stateless_http`
makes those HTTP exchanges sessionless too, but new integrations should use
normal MCP discovery rather than pinning an old protocol.

## Discovery surface

`server/discover` identifies the server, its instructions, supported protocol,
capabilities, and private cache hints. Agents then discover these tools:

| Tool | Purpose | Important contract |
| --- | --- | --- |
| `vault_profiles` | List caller-selectable profiles and included vaults. | Use when the profile is uncertain; never invent names. |
| `vault_search` | Hybrid, lexical, or dense search within one profile. | Returns degradation, commit identity, citations, and `recommended_read`. |
| `vault_read` | Read authoritative source text. | Requires the search hit's `expected_source_hash`; stale reads fail closed. |
| `vault_status` | Inspect readiness, revision identity, and degradation. | Diagnostic only; it does not trigger a sync. |

Every tool has full JSON Schema 2020-12 input and output schemas, structured
content, and annotations declaring it read-only, non-destructive, idempotent,
and closed-world. There is deliberately no MCP tool for sync, configuration,
Git URLs, credentials, arbitrary paths, database access, or mutation.

The server also publishes:

- `vault-rag://guide`: the complete agent workflow and error-recovery guide;
- `vault-rag://profiles`: a live JSON profile inventory; and
- `grounded_vault_research`: a reusable prompt for search/read/cite research.

List/discovery results carry one-hour private cache hints. Resource reads carry
a 30-second private hint. `private` means a client must not share cached
results across authorization contexts.

## Required agent workflow

1. Select the narrowest profile returned by `vault_profiles`.
2. Call `vault_search` in `hybrid` mode with a precise question or identifier.
   Start with five hits.
3. Inspect `degraded` and `vault_commits`. A lexical fallback may still be
   useful, but an agent must not silently imply semantic search succeeded.
4. Copy a relevant hit's complete `recommended_read` object into
   `vault_read`.
5. Keep `expected_source_hash`. If `stale_source` is returned, run the search
   again and use the new hash. Never retry by removing the hash.
6. Quote only returned source text and cite only the returned `citation`
   string. Never synthesize a path, line range, commit, or quote.
7. If the indexed sources do not support an answer, say so. Vault RAG is not a
   general web search or an answer generator.

The tool descriptions repeat these rules so a newly connected agent receives
the operational contract through discovery without needing this repository.

## Search arguments

- `profile` is the server-enforced retrieval scope (not a per-user authorization boundary).
- `query` accepts a question, title, system name, keyword set, or exact
  identifier. Preserve a known identifier verbatim.
- `limit` is 1-25. Start at five and widen only if recall is insufficient.
- `mode` is `hybrid`, `lexical`, or `dense`. `hybrid` is the default.
- `vault_ids` narrows inside the profile; it cannot expand access.
- `path_prefix` narrows to one vault-relative subtree.
- `source_kind` applies an exact `markdown`, `text`, `log`, or `json` filter.
- `frontmatter` uses exact typed equality. JSON boolean `true`, string
  `"true"`, number `1`, array `[1]`, and object `{"value":1}` are distinct.

Each hit returns text, metadata, ranking components, exact line citation,
source hash, snapshot commit/revision identity, and copy-safe read arguments.

## Read arguments

`profile`, `vault_id`, `path`, and `expected_source_hash` are required.
Search-provided `start_line` and `end_line` are preferred. A caller may instead
select an exact `heading`, but must not combine heading and line selection.
Both line endpoints are required together. A long result includes a bounded
`continuation`; continue from the returned position rather than guessing.

## Bounded errors

| Error | Agent action |
| --- | --- |
| `stale_source` | Search again and preserve the replacement hash. |
| `service_busy` | Retry with bounded backoff. |
| `storage_error` | Treat the service as temporarily unavailable and check status later. |
| `security_error` | Correct the profile/vault/path; the request crossed its boundary. |
| `invalid_request` | Correct arguments using the published schema. |
| `internal_error` | Report the returned correlation ID to the operator. |

Raw exception text, queries, source text, paths, credentials, and client IDs do
not enter MCP structured events. Metrics use only fixed tool and outcome
labels:

- `vault_rag_mcp_tool_calls_total{tool,outcome}`;
- `vault_rag_mcp_tool_call_duration_seconds{tool,outcome}`.

## Server configuration

MCP is off unless explicitly enabled in the strict service YAML:

```yaml
mcp:
  enabled: true
  allowedHosts:
    - vault-rag.example.test
    - vault-rag.tools.svc.cluster.local
    - vault-rag.tools.svc.cluster.local:*
  allowedOrigins:
    - https://vault-rag.example.test
```

`allowedHosts` is mandatory when enabled. A `:*` suffix means any port on that
exact host; it is not a hostname wildcard. `Origin` may be absent for native
agent clients. Browser-originated requests are accepted only from the explicit
origin list. Disabled MCP must have empty allowlists.

The security boundary remains the operator's private LAN/tailnet route and
Kubernetes NetworkPolicy. Do not publish `/mcp` through public DNS or a public
Gateway without adding an audited application authentication design first.

## Connect clients

Replace the example with the operator-owned private route. These commands store
only the endpoint URL; they do not launch a local Vault RAG process.

### mcporter and OpenClaw validation

```bash
mcporter config add vault-rag \
  --url https://vault-rag.example.test/mcp \
  --transport http \
  --scope home \
  --description "Private, source-grounded vault retrieval"
mcporter list vault-rag --schema
mcporter call vault-rag.vault_profiles --output json
mcporter call vault-rag.vault_search \
  --args '{"profile":"example","query":"deployment checklist","limit":5,"mode":"hybrid"}' \
  --output json
```

### Codex CLI

```bash
codex mcp add vault-rag --url https://vault-rag.example.test/mcp
codex mcp get vault-rag
```

### Claude Code

```bash
claude mcp add --transport http --scope user \
  vault-rag https://vault-rag.example.test/mcp
claude mcp get vault-rag
```

No bearer token is currently expected because the endpoint is private and the
network is the trust boundary. Do not add a fake Authorization header.

## Route only the intended paths

The application listens on port 8080 and exposes `/mcp`, health, metrics, and
the existing `/v1` HTTP facade. An operator-facing Gateway should route only:

- `/mcp` for agent clients;
- `/health/live` and `/health/ready` where private health checks need them; and
- selected read-only `/v1` routes only when a non-MCP client needs them.

Do not route `/metrics` outside the monitoring path or `/v1/admin/sync` through
the agent hostname. Prometheus and the worker control plane already have their
own cluster-internal access.

## Adding another vault

Adding a vault is a server operation, not an MCP client operation:

1. Add and validate `.vault-rag.toml` at the private Git repository root. Give
   it a stable lowercase ID and explicit include/exclude/frontmatter rules.
2. Grant the existing worker GitHub token read access to only that repository,
   or rotate to a new fine-grained token whose repository allowlist includes
   it. The API never receives the GitHub token.
3. Add the repository HTTPS URL, full `refs/heads/...` or `refs/tags/...` ref,
   and `credentialEnv: GITHUB_TOKEN` under `serviceConfig.repositories` in the
   operator GitOps values.
4. Add the vault ID to one or more explicit profiles. Prefer narrow profiles;
   do not silently add sensitive vaults to a broad existing profile.
5. Render Helm and Kustomize, review the ConfigMap and worker-only secret
   boundary, merge GitOps, and let Flux reconcile.
6. Confirm one worker acquires the vault lease, fetches/builds/promotes one
   immutable revision, and the API reports the expected commit.
7. Run a vault-specific retrieval fixture, exact-citation validation, stale-hash
   read test, and a client discovery/search/read test before treating it as
   available.

No client configuration changes are needed when a vault is added to a profile:
the shared endpoint and tools remain unchanged, and `vault_profiles` reflects
the new server configuration after rollout.

## Deliberately omitted protocol features

The 2026 protocol includes extensions such as long-running Tasks and MCP Apps.
Vault RAG tools are bounded, read-only, and normally subsecond, and there is no
interactive UI. Tasks, elicitation, sampling, roots, application UI, mutation,
and client-specific server sessions would add complexity or standing state
without helping this workload. The server uses the new discovery, structured
schemas/content, annotations, resources, prompts, cache hints, request metadata,
and stateless HTTP behavior that materially improve agent use.
