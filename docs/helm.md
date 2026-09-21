# Install with Helm

Complete the database and network prerequisites in the
[deployment guide](postgresql-kubernetes-deployment.md) first. Run the commands
below from the repository root.

The Helm chart deploys the same production image as three PostgreSQL-backed
process roles:

- a pre-install/pre-upgrade migration Job running `vault-rag db migrate` with DDL-only credentials;
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
imagePullSecrets: [] # public GHCR image; add a pull secret for a private mirror
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
hostname. See [MCP 2026-07-28 agent integration](mcp.md).

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
[PostgreSQL/Kubernetes deployment runbook](postgresql-kubernetes-deployment.md)
for preflight, rollout, failure, and rollback gates.
