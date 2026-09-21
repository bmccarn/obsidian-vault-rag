# Private PostgreSQL/Kubernetes deployment

This guide is for operators with a private Kubernetes cluster and PostgreSQL 18
with pgvector. The chart installs application workloads, not a database or public
route. For a credential-free local trial use the [Compose demo](docker-demo.md).

## Prerequisites and trust boundary

- A tested PostgreSQL 18/pgvector database, verified TLS, and a backup/restore plan.
- A Kubernetes cluster, Helm 3, and a CNI that enforces NetworkPolicy.
- Read-only credentials for each configured source Git repository.
- An OpenAI-compatible embedding endpoint, only if semantic retrieval is desired.
- A private trusted client network. The service has no application authentication;
  every reachable client can select any configured profile.

Do not put real URLs, credentials, private manifests, or source text in this
repository. Store operator configuration and secrets separately. The chart does
not create Ingress, Gateway, DNS, or Internet routing. Do not publish `/metrics`
or `/v1/admin/sync` through an agent endpoint.

## Database preparation

Have your database administrator create a dedicated database and install the
`vector` extension. Use separate API, worker and migrator logins and a non-login
schema owner. The application schema is `vault_rag`; `storage.ownerRole` selects
its ownership role when configured. The migrator must be able to assume that
role. Never use a database administrator credential in a serving workload.

| Role | Required scope |
| --- | --- |
| API | Connect, schema USAGE, SELECT on served index/revision/configuration tables; bounded sync-request writes if the administrative sync endpoint is enabled |
| Worker | Connect, schema USAGE, application-table DML and required sequence access; no schema DDL |
| Migrator | Create/alter application schema objects and assume the configured owner; no cluster-wide administration |

Grant permissions and default privileges according to the versioned migrations
and enabled endpoints. Verify them with each role separately in staging: API
queries must work and worker/indexing operations must not be possible with API
credentials. Keep administration paths network-restricted. Review grants again
when upgrading migrations; do not copy permissive demo database credentials.

Use database URLs with `sslmode=verify-full` and a trusted CA, for example:

```text
postgresql://ROLE:PASSWORD@db.example.test:5432/vault_rag?sslmode=verify-full&sslrootcert=/etc/ssl/certs/ca-certificates.crt
```

Provision trust for private CAs in your image/environment before deployment.
Plaintext transport requires an explicit `allowInsecureTransport: true` opt-in;
that setting belongs to the isolated synthetic demo, not this production baseline.

## Configure the chart

The [Helm walkthrough](helm.md)
shows secret creation and an operator values file. Select a successful published
image's immutable SHA tag or digest. Render the chart before applying it:

```bash
helm lint charts/vault-rag --values values.private.yaml
helm template vault-rag charts/vault-rag --values values.private.yaml > rendered.yaml
```

Review the rendered API, worker and migration Job separately:

- API has its database/runtime secret, but no Git credential.
- Worker has its own database credential, read-only Git credential and embedding key.
- Migrator has only its migration database credential.
- `storage.backend: postgresql`, verified TLS, realistic bounded pools and timeouts.
- Worker `emptyDir` checkout space is large enough for the selected repositories.
- NetworkPolicy is enabled with your actual DNS, database, Git/provider egress and
  trusted-client/monitoring ingress selectors. Chart example CIDRs are placeholders;
  replace them and account for CNI/NAT behavior. The default policy is off, not safe
  isolation by itself.
- Resource requests, limits, disruption budgets, replica counts and optional HPA
  match the cluster. Monitoring CRDs must exist before enabling ServiceMonitor or
  PodMonitor.

Each source repository needs a `.vault-rag.toml`; its ID must match the configured
repository ID. Use full Git refs and explicit profiles. Start with a synthetic
repository and verify egress policy before adding private notes.

## Migrate and start

The chart runs migrations as a pre-install/pre-upgrade hook. The equivalent CLI
commands for a controlled operator environment are:

```bash
vault-rag db migrate --service-config /config/service.yaml
vault-rag db check --service-config /config/service.yaml
```

Database credentials come from the configured environment variable or its `_FILE`
variant, never from command-line arguments. Use the migrator credential only for
these commands. Do not run concurrent manual migrations alongside Helm hooks.

```bash
namespace=default
helm upgrade --install vault-rag charts/vault-rag \
  --namespace "$namespace" --create-namespace --values values.private.yaml
kubectl -n "$namespace" rollout status deployment/vault-rag-api
kubectl -n "$namespace" rollout status deployment/vault-rag-worker
kubectl -n "$namespace" get jobs,pods,service -l app.kubernetes.io/instance=vault-rag
kubectl -n "$namespace" port-forward service/vault-rag 8080:8080
```

From another terminal, check `/health/live`, `/health/ready`, and `/v1/status` over
loopback. Confirm the expected source commits, semantic state and profiles before
allowing client access. Readiness is not proof of embedding quality or fresh sync.

## Acceptance and operations

Test with synthetic data before adopting private sources:

1. Discovery, lexical/hybrid search, exact citations and hash-verified reads.
2. A Git update reconciles within the configured interval; unchanged commits do
   not trigger unnecessary indexing or embedding.
3. Multiple workers do not race promotion; stale-hash reads fail closed.
4. API/worker restarts preserve served PostgreSQL state.
5. Controlled Git/provider failures preserve usable results and report degradation.
6. A database backup restores into an isolated test instance and serves expected
   source hashes, commit identities and counts.

Do not run destructive PostgreSQL contract tests against a serving database.
They require the isolated `vault_rag_test` database described in [CONTRIBUTING.md](../CONTRIBUTING.md).

Automatic cleanup is disabled for the initial observation window. Keep
`storage.cleanup.enabled: false` until retention, recovery and the effect on old
revisions are understood. PostgreSQL is the durable state; worker checkouts can be
recreated. Backups contain private text and embeddings and need source-equivalent
protection.

After secret rotation, restart only affected roles and verify status. Pin the
previous image/chart for rollback, but confirm schema compatibility first: image
rollback does not undo migrations. Database restore is an operator-controlled
recovery action, not an automatic Helm rollback. `helm uninstall` does not delete
the external database. Test your actual networking, outage and recovery paths;
repository CI cannot certify them.
