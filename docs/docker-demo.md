# Docker Compose demo

Run these commands from the repository root. To search your own local vault,
use the [CLI quickstart](../README.md#search-your-vault) instead.

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
For production use the [private deployment guide](postgresql-kubernetes-deployment.md),
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
