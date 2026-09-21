#!/usr/bin/env bash
# An isolated, credential-free CLI walkthrough. Run from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
demo_dir=$(mktemp -d "${TMPDIR:-/tmp}/vault-rag-demo.XXXXXX")
export XDG_CONFIG_HOME="$demo_dir/config"
export XDG_DATA_HOME="$demo_dir/data"
export VAULT_RAG_EMBEDDING_MODEL=disabled-demo
mkdir -p "$demo_dir/vault" "$XDG_CONFIG_HOME/vault-rag"
cat > "$demo_dir/vault/.vault-rag.toml" <<'TOML'
schema_version = 1
id = "demo"
egress_policy = "local-only"
include = ["**/*.md"]
exclude = [".git/**", ".obsidian/**"]
TOML
cat > "$demo_dir/vault/note.md" <<'NOTE'
# Deployment checklist

Back up the database before deploying a new version.
Verify health checks after the deployment.
NOTE
cat > "$XDG_CONFIG_HOME/vault-rag/config.toml" <<TOML
[vaults.demo]
path = "$demo_dir/vault"
[profiles.demo]
vaults = ["demo"]
[embedding]
base_url = "https://embedding.invalid/v1"
model_env = "VAULT_RAG_EMBEDDING_MODEL"
endpoint_class = "remote"
TOML
chmod 600 "$XDG_CONFIG_HOME/vault-rag/config.toml"
index_status=0
uv run --frozen vault-rag index --profile demo --json || index_status=$?
if [ "$index_status" -ne 0 ] && [ "$index_status" -ne 4 ]; then
  exit "$index_status"
fi
uv run --frozen vault-rag search --profile demo "deployment" --mode lexical --json
uv run --frozen vault-rag read --profile demo note.md --start-line 1 --end-line 4 --json
printf '\nDemo files retained at: %s\n' "$demo_dir"
