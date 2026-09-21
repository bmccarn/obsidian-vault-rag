#!/usr/bin/env bash
# Populate only the isolated Compose fixture's secret inputs, never real credentials.
set -euo pipefail
cd "$(dirname "$0")/../.."
umask 077
mkdir -p secrets
names=(postgres_password migrator_database_url api_a_database_url api_b_database_url worker_a_database_url worker_b_database_url smoke_database_url git_fixture_token api_litellm_api_key worker_litellm_api_key)
for name in "${names[@]}"; do
  if [ -e "secrets/$name" ]; then
    printf 'Refusing to overwrite secrets/%s; use a fresh demo checkout.\n' "$name" >&2
    exit 1
  fi
done
# noclobber also protects against a file appearing after the preflight.
set -o noclobber
demo_password=$(openssl rand -hex 32)
printf '%s\n' "$demo_password" > secrets/postgres_password
for name in migrator_database_url api_a_database_url api_b_database_url worker_a_database_url worker_b_database_url smoke_database_url; do
  printf 'postgresql://vault_rag:%s@postgres:5432/vault_rag\n' "$demo_password" > "secrets/$name"
done
printf 'fixture-token\n' > secrets/git_fixture_token
printf 'fixture-api-key\n' > secrets/api_litellm_api_key
printf 'fixture-worker-key\n' > secrets/worker_litellm_api_key
# Linux Compose bind-mounts these files without remapping ownership. The fixed
# container UID must be able to read them. These values are demo-only; the host
# directory remains user-only and each container receives only its named mounts.
for name in "${names[@]}"; do
  chmod 0444 "secrets/$name"
done
printf 'Created synthetic demo secret files. Next: docker compose build\n'
