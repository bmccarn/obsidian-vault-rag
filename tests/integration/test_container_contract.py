"""Deployment artifact contracts for split-process Docker Compose."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parents[2]


def _dockerfile() -> str:
    return (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")


def _compose() -> dict[str, Any]:
    with (PROJECT_ROOT / "compose.yaml").open(encoding="utf-8") as compose_file:
        document = yaml.safe_load(compose_file)
    assert isinstance(document, dict)
    return document


def _service(compose: dict[str, Any], name: str) -> dict[str, Any]:
    services = compose.get("services")
    assert isinstance(services, dict)
    service = services[name]
    assert isinstance(service, dict)
    return service


def test_final_image_defaults_to_the_api_as_fixed_numeric_non_root_user() -> None:
    dockerfile = _dockerfile()

    assert re.search(r"(?m)^USER\s+10001:10001\s*$", dockerfile)
    assert re.search(r'(?m)^ENTRYPOINT\s+\["vault-rag"\]\s*$', dockerfile)
    assert re.search(
        r'(?m)^CMD\s+\["api", "--service-config", "/config/service.yaml", '
        r'"--data-root", "/data", "--host", "0.0.0.0", "--port", "8080"\]\s*$',
        dockerfile,
    )
    assert "HEALTHCHECK" not in dockerfile
    assert "apt-get install -y --no-install-recommends git ca-certificates" in dockerfile
    assert "ghcr.io/astral-sh/uv:" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile


@pytest.mark.parametrize("path", ["/data", "/tmp/vault-rag"])
def test_final_image_owns_its_writable_paths(path: str) -> None:
    dockerfile = _dockerfile()

    assert path in dockerfile
    assert "10001:10001" in dockerfile


def test_compose_orders_database_migration_and_split_processes() -> None:
    compose = _compose()
    migrate = _service(compose, "migrate")

    assert migrate["depends_on"] == {"postgres": {"condition": "service_healthy"}}
    assert migrate["restart"] == "no"
    assert "vault-rag db migrate" in migrate["command"][0]
    assert "vault-rag db check" in migrate["command"][0]
    for name in ("api-a", "api-b", "worker-a", "worker-b"):
        assert _service(compose, name)["depends_on"]["migrate"] == {
            "condition": "service_completed_successfully"
        }


def test_compose_preinstalls_required_pgvector_extension() -> None:
    compose = _compose()
    postgres = _service(compose, "postgres")
    init_mount = (
        "./deploy/docker/postgres-init.sql:/docker-entrypoint-initdb.d/001-vault-rag.sql:ro"
    )

    assert init_mount in postgres["volumes"]
    init_sql = (PROJECT_ROOT / "deploy/docker/postgres-init.sql").read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector;" in init_sql


def test_compose_limits_networking_and_hardens_application_processes() -> None:
    compose = _compose()

    assert _service(compose, "api-a")["ports"] == ["127.0.0.1:8080:8080"]
    for name in (
        "postgres",
        "git-fixture-init",
        "git-fixture",
        "migrate",
        "api-b",
        "worker-a",
        "worker-b",
        "smoke",
    ):
        assert _service(compose, name).get("ports", []) == []

    for name in ("migrate", "api-a", "api-b", "worker-a", "worker-b", "smoke"):
        service = _service(compose, name)
        assert service["user"] == "10001:10001"
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
        assert any(mount.startswith("/tmp:") for mount in service["tmpfs"])


def test_compose_keeps_runtime_secret_references_separate_by_process() -> None:
    compose = _compose()

    for service_name, database_secret in (
        ("api-a", "api_a_database_url"),
        ("api-b", "api_b_database_url"),
        ("worker-a", "worker_a_database_url"),
        ("worker-b", "worker_b_database_url"),
        ("smoke", "smoke_database_url"),
    ):
        service = _service(compose, service_name)
        assert database_secret in service["secrets"]
        assert service["environment"]["VAULT_RAG_DATABASE_URL_FILE"] == (
            f"/run/secrets/{database_secret}"
        )
    assert "GIT_FIXTURE_TOKEN_FILE" in _service(compose, "worker-a")["environment"]


def test_compose_applies_api_readiness_without_probing_workers_or_migrator() -> None:
    compose = _compose()
    for name in ("api-a", "api-b"):
        healthcheck = _service(compose, name)["healthcheck"]
        assert healthcheck["test"][0:2] == ["CMD", "python"]
    for name in ("migrate", "worker-a", "worker-b"):
        assert "healthcheck" not in _service(compose, name)


def test_compose_is_a_pinned_postgresql_18_split_runtime_e2e() -> None:
    compose = _compose()
    services = compose["services"]

    assert set(services) == {
        "postgres",
        "git-fixture-init",
        "git-fixture",
        "migrate",
        "api-a",
        "api-b",
        "worker-a",
        "worker-b",
        "smoke",
    }
    assert services["postgres"]["image"] == (
        "pgvector/pgvector:pg18@sha256:"
        "691673308c99d2161ba298736f3147f1f22d79de2fb7ec93ae9b4afcab870b62"
    )
    assert services["git-fixture"]["image"].startswith("caddy:2.10.0@sha256:")
    fixture = services["git-fixture"]
    assert fixture["cap_drop"] == ["ALL"]
    assert fixture["cap_add"] == ["NET_BIND_SERVICE"]
    assert fixture["security_opt"] == ["no-new-privileges:true"]
    fixture_healthcheck = fixture["healthcheck"]["test"]
    assert fixture_healthcheck[0] == "CMD-SHELL"
    assert "/public-ca/root.crt" in fixture_healthcheck[1]
    assert "SSL_CERT_FILE=/public-ca/root.crt" in fixture_healthcheck[1]
    assert "wget -q --spider" in fixture_healthcheck[1]
    assert "https://git-fixture/fixture-vault.git/info/refs" in fixture_healthcheck[1]
    assert services["postgres"]["security_opt"] == ["no-new-privileges:true"]
    assert services["api-a"]["ports"] == ["127.0.0.1:8080:8080"]
    for name in (
        "postgres",
        "git-fixture-init",
        "git-fixture",
        "migrate",
        "api-b",
        "worker-a",
        "worker-b",
        "smoke",
    ):
        assert _service(compose, name).get("ports", []) == []
    for name in ("migrate", "api-a", "api-b", "worker-a", "worker-b", "smoke"):
        service = _service(compose, name)
        assert service["user"] == "10001:10001"
        assert service["read_only"] is True
        assert service["cap_drop"] == ["ALL"]
        assert service["security_opt"] == ["no-new-privileges:true"]
    migration_command = services["migrate"]["command"][0]
    assert "vault-rag db migrate --service-config /config/service.yaml" in migration_command
    assert "vault-rag db check --service-config /config/service.yaml" in migration_command
    assert services["smoke"]["entrypoint"] == ["python", "/smoke/compose-smoke.py"]
    assert services["smoke"]["command"] == ["verify"]
    assert (
        "./deploy/docker/fixture-vault:/fixture/source:ro"
        in services["git-fixture-init"]["volumes"]
    )
    assert "git-fixture-public-ca:/fixture/ca:ro" in services["worker-a"]["volumes"]


def test_container_artifacts_pin_bases_and_provide_secure_fixture_and_verifier() -> None:

    dockerfile = _dockerfile()
    fixture = (PROJECT_ROOT / "deploy/docker/compose-git-fixture.sh").read_text(encoding="utf-8")
    caddyfile = (PROJECT_ROOT / "deploy/docker/Caddyfile").read_text(encoding="utf-8")
    smoke = (PROJECT_ROOT / "deploy/docker/compose-smoke.py").read_text(encoding="utf-8")

    assert re.search(r"^FROM .+@sha256:[0-9a-f]{64}", dockerfile, re.MULTILINE)
    assert dockerfile.count("@sha256:") == 3
    assert "GIT_SSL_NO_VERIFY" not in fixture + caddyfile + smoke
    assert "tls internal" in caddyfile
    assert "update-server-info" in fixture
    assert 'if [ -f "$repository/HEAD" ]; then' in fixture
    assert "GIT_AUTHOR_DATE='2000-01-01T00:00:00Z'" in fixture
    assert "GIT_COMMITTER_DATE='2000-01-01T00:00:00Z'" in fixture
    assert 'rm -rf "$repository"' not in fixture
    for token in (
        "expected_source_hash",
        "/v1/search",
        "/v1/read",
        "/v1/status",
        "revision_count",
        "deadline",
    ):
        assert token in smoke
