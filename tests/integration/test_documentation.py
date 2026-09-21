"""Assertions that published documentation matches implemented behaviour."""

import json
import shlex
from pathlib import Path

import yaml

from vault_rag.errors import (  # type: ignore[import-untyped]
    ConfigError,
    SecurityError,
    SemanticUnavailableError,
    StorageError,
)

PROJECT_ROOT = Path(__file__).parents[2]
README = PROJECT_ROOT / "README.md"
RETRIEVAL_DOC = PROJECT_ROOT / "docs" / "retrieval.md"
CONFIGURATION_DOC = PROJECT_ROOT / "docs" / "configuration.md"
MCP_DOC = PROJECT_ROOT / "docs" / "mcp.md"
HELM_DOC = PROJECT_ROOT / "docs" / "helm.md"
CONTRIBUTING_DOC = PROJECT_ROOT / "CONTRIBUTING.md"
DOCKER_DOC = PROJECT_ROOT / "docs" / "docker-demo.md"

DOCUMENTED_EXIT_CODES = (
    0,
    StorageError.exit_code,
    ConfigError.exit_code,
    SecurityError.exit_code,
    SemanticUnavailableError.exit_code,
    5,
)


def test_readme_documents_every_stable_exit_code() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "## Exit codes" in readme
    for exit_code in sorted(set(DOCUMENTED_EXIT_CODES)):
        assert f"| `{exit_code}` |" in readme
    assert "exit code 4" in readme


def test_readme_warns_that_rebuild_reembeds_the_whole_corpus() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "re-embeds the entire corpus" in readme


def test_retrieval_doc_describes_phrase_queries_and_the_priority_tier() -> None:
    doc = RETRIEVAL_DOC.read_text(encoding="utf-8")

    assert "adds normalized alternatives" not in doc
    assert "deterministic boosts" not in doc
    assert "quoted FTS5 phrase" in doc
    assert "there is no score boost" in doc


def test_retrieval_doc_states_physical_line_and_diagnostic_behaviour() -> None:
    doc = RETRIEVAL_DOC.read_text(encoding="utf-8")

    assert "physical CommonMark lines" in doc
    assert "bounded per-file diagnostic" in doc
    assert "reported by the index run" in doc


def test_configuration_doc_states_exclusion_precedence() -> None:
    doc = CONFIGURATION_DOC.read_text(encoding="utf-8")

    assert "skipped before any containment" in doc


def test_mcp_doc_defines_stateless_transport_agent_contract_and_vault_onboarding() -> None:
    doc = MCP_DOC.read_text(encoding="utf-8")

    for term in (
        "2026-07-28",
        "stateless_http=True",
        "json_response=True",
        "Mcp-Session-Id",
        "expected_source_hash",
        "recommended_read",
        "vault-rag://guide",
        "grounded_vault_research",
        "allowedHosts",
        "mcporter config add",
        "codex mcp add",
        "claude mcp add",
        "Adding another vault",
    ):
        assert term in doc
    assert "Do not route `/metrics`" in doc
    assert "Do not publish `/mcp` through public DNS" in doc


def test_postgresql_docs_specify_nested_database_commands_and_cleanup_opt_in() -> None:
    configuration = CONFIGURATION_DOC.read_text(encoding="utf-8")
    deployment = (PROJECT_ROOT / "docs" / "postgresql-kubernetes-deployment.md").read_text(
        encoding="utf-8"
    )
    schema = json.loads(
        (PROJECT_ROOT / "charts" / "vault-rag" / "values.schema.json").read_text(encoding="utf-8")
    )
    storage_properties = schema["definitions"]["storage"]["properties"]

    assert "vault-rag db migrate" in configuration
    assert "vault-rag db check" in configuration
    assert "allowInsecureTransport: true" in configuration
    assert "cleanup:\n    enabled: false" in configuration
    assert "allowInsecureTransport" in storage_properties
    assert "cleanup" in storage_properties
    assert "`secretKeyRef` environment variables" in configuration
    assert "Secret keys are mounted as mode-`0400` files" not in configuration
    assert "vault-rag db migrate" in deployment
    assert "vault-rag db check" in deployment
    assert "Automatic cleanup is disabled for the initial observation window" in deployment


def test_helm_guide_exposes_only_nested_database_migration_command() -> None:
    readme = HELM_DOC.read_text(encoding="utf-8")
    assert "vault-rag db migrate" in readme
    assert "`vault-rag migrate`" not in readme


def test_helm_docs_explain_secret_rotation_rollout() -> None:
    for document in (HELM_DOC, CONFIGURATION_DOC):
        text = document.read_text(encoding="utf-8")

        assert "kubectl rollout restart deployment/vault-rag-api" in text
        assert "kubectl rollout restart deployment/vault-rag-worker" in text
        assert 'kubectl -n "$namespace" rollout status deployment/vault-rag-api' in text
        assert 'kubectl -n "$namespace" rollout status deployment/vault-rag-worker' in text


def test_helm_docs_and_schema_support_private_registry_credentials() -> None:
    values = yaml.safe_load(
        (PROJECT_ROOT / "charts" / "vault-rag" / "values.yaml").read_text(encoding="utf-8")
    )
    schema = json.loads(
        (PROJECT_ROOT / "charts" / "vault-rag" / "values.schema.json").read_text(encoding="utf-8")
    )
    workload_templates = [
        (PROJECT_ROOT / "charts" / "vault-rag" / "templates" / name).read_text(encoding="utf-8")
        for name in ("api-deployment.yaml", "worker-deployment.yaml", "migration-job.yaml")
    ]

    assert values["imagePullSecrets"] == []
    pull_secret_schema = schema["properties"]["imagePullSecrets"]
    assert pull_secret_schema["type"] == "array"
    assert pull_secret_schema["items"] == {"$ref": "#/definitions/imagePullSecret"}
    assert schema["definitions"]["imagePullSecret"]["required"] == ["name"]
    for template in workload_templates:
        assert ".Values.imagePullSecrets" in template
        assert "imagePullSecrets:" in template
    readme = HELM_DOC.read_text(encoding="utf-8")
    assert 'kubectl -n "$namespace" create secret docker-registry' in readme
    assert "read:packages" in readme
    assert "imagePullSecrets:" in readme


def test_ci_uses_pinned_tools_and_postgresql_18_on_the_required_python_versions() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "HELM_VERSION=3.16.3" in workflow
    assert "f5355c79190951eed23c5432a3b920e071f4c00a64f75e077de0dd4cb7b294ea" in workflow
    assert "sha256sum --check" in workflow
    assert "KUBECONFORM_VERSION=0.6.7" in workflow
    assert "uv export --frozen --no-dev --no-emit-project" in workflow
    assert "uv run pip-audit --strict --requirement" in workflow
    assert "npm ci --ignore-scripts" in workflow
    assert "node_modules/.bin/markdownlint-cli2 --" in workflow
    assert "npx" not in workflow
    assert "helm lint charts/vault-rag" in workflow
    assert "helm template vault-rag charts/vault-rag" in workflow
    assert "-strict -ignore-missing-schemas -summary" in workflow
    workflow = yaml.safe_load(workflow)
    postgres_image = (
        "pgvector/pgvector:pg18@sha256:"
        "691673308c99d2161ba298736f3147f1f22d79de2fb7ec93ae9b4afcab870b62"
    )
    for job_name in ("postgres-contract", "postgres-coverage"):
        job = workflow["jobs"][job_name]
        assert job["services"]["postgres"]["image"] == postgres_image
        assert job["strategy"]["matrix"]["python-version"] == ["3.12", "3.14"]
        assert job["services"]["postgres"]["env"]["POSTGRES_DB"] == "vault_rag_test"
        postgres_steps = [
            step for step in job["steps"] if "VAULT_RAG_TEST_POSTGRES_DSN" in step.get("env", {})
        ]
        assert postgres_steps
        assert all(step["env"].get("VAULT_RAG_REQUIRE_POSTGRES") == "1" for step in postgres_steps)
        assert all(
            step["env"]["VAULT_RAG_TEST_POSTGRES_DSN"].endswith("/vault_rag_test")
            for step in postgres_steps
        )
    coverage_step = next(
        step
        for step in workflow["jobs"]["postgres-coverage"]["steps"]
        if step.get("name") == "Run PostgreSQL subsystem coverage"
    )
    coverage_command = coverage_step["run"]
    assert shlex.split(coverage_command) == [
        "uv",
        "run",
        "--python",
        "${{ matrix.python-version }}",
        "pytest",
        "-q",
        "--cov=src/vault_rag/storage/postgres",
        "--cov=src/vault_rag/service",
        "--cov-report=term",
        "--cov-fail-under=90",
    ]
    assert workflow["jobs"]["sqlite-tests"]["strategy"]["matrix"]["python-version"] == [
        "3.12",
        "3.13",
        "3.14",
    ]
    readme = CONTRIBUTING_DOC.read_text(encoding="utf-8")
    assert "VAULT_RAG_TEST_POSTGRES_DSN" in readme
    assert "vault_rag_test" in readme
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lockfile = (PROJECT_ROOT / "uv.lock").read_text(encoding="utf-8")
    assert '"pip-audit>=2.9,<3"' in pyproject
    assert 'name = "pip-audit"' in lockfile
    package = json.loads((PROJECT_ROOT / "package.json").read_text(encoding="utf-8"))
    package_lock = json.loads((PROJECT_ROOT / "package-lock.json").read_text(encoding="utf-8"))
    assert package["devDependencies"]["markdownlint-cli2"] == "0.23.3"
    assert package_lock["lockfileVersion"] >= 3
    assert "node_modules/markdownlint-cli2" in package_lock["packages"]


def test_service_docs_distinguish_local_contracts_from_private_operations() -> None:
    readme = README.read_text(encoding="utf-8")
    configuration = CONFIGURATION_DOC.read_text(encoding="utf-8")
    retrieval = RETRIEVAL_DOC.read_text(encoding="utf-8")

    assert "synthetic notes" in readme
    assert "private routing" in configuration
    assert "operator-owned 25-case corpus" in retrieval
    assert "two-machine private routing" in retrieval


def test_ci_builds_smokes_and_publishes_the_production_image() -> None:
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    assert isinstance(workflow, dict)
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request_target" not in workflow_text
    assert all(job["runs-on"] == "ubuntu-latest" for job in workflow["jobs"].values())

    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    container = jobs["container"]
    publish_image = jobs["publish-image"]
    assert isinstance(container, dict)
    assert isinstance(publish_image, dict)
    assert container["name"] == "Build and smoke PostgreSQL 18 split runtime"
    assert publish_image["permissions"] == {"contents": "read", "packages": "write"}
    assert publish_image["needs"] == [
        "sqlite-tests",
        "postgres-contract",
        "postgres-coverage",
        "release-gates",
        "container",
    ]
    assert publish_image["if"] == "github.event_name == 'push' && github.ref == 'refs/heads/main'"
    for name, job in jobs.items():
        if name != "publish-image":
            assert isinstance(job, dict)
            permissions = job.get("permissions", {})
            assert isinstance(permissions, dict)
            assert permissions.get("packages") != "write"

    container_script = "\n".join(
        step["run"]
        for step in container["steps"]
        if isinstance(step, dict) and isinstance(step.get("run"), str)
    )
    for expected in (
        "docker compose build",
        "docker compose up --detach postgres git-fixture migrate api-a api-b worker-a worker-b",
        "docker compose run --rm smoke verify",
        "docker compose restart api-a",
        "docker compose restart worker-a",
        "docker compose down --volumes --remove-orphans",
    ):
        assert expected in container_script

    publish_script = "\n".join(
        step["run"]
        for step in publish_image["steps"]
        if isinstance(step, dict) and isinstance(step.get("run"), str)
    )
    for expected in (
        "ghcr.io/${GITHUB_REPOSITORY}",
        "sha-${GITHUB_SHA}",
        'docker push "${image}:${immutable_tag}"',
        "docker buildx imagetools inspect",
    ):
        assert expected in publish_script
    assert 'docker push "${image}:latest"' not in publish_script

    readme = DOCKER_DOC.read_text(encoding="utf-8")
    assert "ghcr.io/bmccarn/obsidian-vault-rag" in readme
    assert "immutable SHA tag" in readme
    deployment = (PROJECT_ROOT / "docs" / "postgresql-kubernetes-deployment.md").read_text(
        encoding="utf-8"
    )
    assert "repository CI cannot certify them" in deployment
