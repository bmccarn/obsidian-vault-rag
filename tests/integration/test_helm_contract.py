"""Rendered-manifest contract for the split PostgreSQL Helm release."""

from __future__ import annotations

import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parents[2]
CHART = PROJECT_ROOT / "charts" / "vault-rag"
HELM = shutil.which("helm")

FIXTURES = PROJECT_ROOT / "tests" / "fixtures" / "helm"

pytestmark = pytest.mark.skipif(HELM is None, reason="Helm binary is unavailable")


def _render(*extra_args: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [HELM or "helm", "template", "vault-rag", str(CHART), *extra_args],
        check=True,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    return [document for document in yaml.safe_load_all(completed.stdout) if document]


def _render_fixture(backend: str) -> list[dict[str, Any]]:
    return _render("--values", str(FIXTURES / f"{backend}-values.yaml"))


def _resources(documents: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [document for document in documents if document["kind"] == kind]


def _resource(documents: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    matches = _resources(documents, kind)
    assert len(matches) == 1
    return matches[0]


def _regular_config_map(documents: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [
        config_map
        for config_map in _resources(documents, "ConfigMap")
        if "helm.sh/hook" not in config_map["metadata"].get("annotations", {})
    ]
    assert len(matches) == 1
    return matches[0]


def _deployment(documents: list[dict[str, Any]], component: str) -> dict[str, Any]:
    matches = [
        deployment
        for deployment in _resources(documents, "Deployment")
        if deployment["spec"]["template"]["metadata"]["labels"]["app.kubernetes.io/component"]
        == component
    ]
    assert len(matches) == 1
    return matches[0]


def _container(workload: dict[str, Any]) -> dict[str, Any]:
    containers = workload["spec"]["template"]["spec"]["containers"]
    assert len(containers) == 1
    return containers[0]


def _environment(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in container["env"]}


@pytest.mark.parametrize(
    ("backend", "expected_resources", "component"),
    [
        (
            "sqlite",
            Counter(
                {
                    "ConfigMap": 1,
                    "Service": 1,
                    "StatefulSet": 1,
                    "ServiceMonitor": 1,
                    "PodMonitor": 1,
                    "NetworkPolicy": 1,
                }
            ),
            "serve",
        ),
        (
            "postgresql",
            Counter(
                {
                    "ConfigMap": 2,
                    "Service": 1,
                    "Deployment": 2,
                    "Job": 1,
                    "PodDisruptionBudget": 2,
                    "ServiceMonitor": 1,
                    "PodMonitor": 1,
                    "NetworkPolicy": 3,
                }
            ),
            "api",
        ),
    ],
)
def test_dual_mode_fixtures_render_exact_resource_and_selector_contracts(
    backend: str, expected_resources: Counter[str], component: str
) -> None:
    documents = _render_fixture(backend)
    assert Counter(document["kind"] for document in documents) == expected_resources

    service = _resource(documents, "Service")
    assert service["spec"]["selector"]["app.kubernetes.io/component"] == component
    assert service["metadata"]["labels"]["app.kubernetes.io/component"] == component

    for monitor_kind in ("ServiceMonitor", "PodMonitor"):
        monitor = _resource(documents, monitor_kind)
        assert (
            monitor["spec"]["selector"]["matchLabels"]["app.kubernetes.io/component"] == component
        )

    service_config = yaml.safe_load(_regular_config_map(documents)["data"]["service.yaml"])
    assert service_config["storage"]["backend"] == backend
    assert service_config["mcp"] == {
        "enabled": False,
        "allowedHosts": [],
        "allowedOrigins": [],
    }
    for repository in service_config["repositories"].values():
        parsed = urlsplit(repository["url"])
        assert parsed.username is None
        assert parsed.password is None
    expected_affinity = {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {
                        "matchExpressions": [
                            {
                                "key": "kubernetes.io/arch",
                                "operator": "In",
                                "values": ["arm64"],
                            }
                        ]
                    }
                ]
            }
        }
    }

    if backend == "sqlite":
        stateful_set = _resource(documents, "StatefulSet")
        assert stateful_set["spec"]["replicas"] == 1
        assert (
            stateful_set["spec"]["selector"]["matchLabels"]["app.kubernetes.io/component"]
            == "serve"
        )
        container = _container(stateful_set)
        assert container["args"] == [
            "serve",
            "--service-config",
            "/config/service.yaml",
            "--data-root",
            "/data",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
        ]
        environment = _environment(container)
        assert set(environment) == {
            "GITHUB_TOKEN_FILE",
            "LITELLM_API_KEY_FILE",
            "VAULT_RAG_EMBEDDING_MODEL",
        }
        assert "VAULT_RAG_DATABASE_URL" not in environment
        mounts = {mount["name"]: mount for mount in container["volumeMounts"]}
        assert mounts["data"]["mountPath"] == "/data"
        assert mounts["github-token"]["readOnly"] is True
        assert mounts["litellm-api-key"]["readOnly"] is True
        volumes = {
            volume["name"]: volume for volume in stateful_set["spec"]["template"]["spec"]["volumes"]
        }
        assert volumes["github-token"]["secret"]["secretName"] == "vault-rag-sqlite-runtime"
        assert volumes["litellm-api-key"]["secret"]["secretName"] == "vault-rag-sqlite-runtime"
        claim = stateful_set["spec"]["volumeClaimTemplates"][0]
        assert claim["metadata"]["name"] == "data"
        assert claim["spec"]["accessModes"] == ["ReadWriteOnce"]
        assert stateful_set["spec"]["template"]["spec"]["affinity"] == expected_affinity
        policy = _resource(documents, "NetworkPolicy")
        namespace_selector = policy["spec"]["ingress"][0]["from"][1]["namespaceSelector"]
        assert namespace_selector["matchExpressions"] == [
            {
                "key": "kubernetes.io/metadata.name",
                "operator": "In",
                "values": ["monitoring"],
            }
        ]
        assert not (
            {"Deployment", "Job", "PodDisruptionBudget", "HorizontalPodAutoscaler"}
            & {document["kind"] for document in documents}
        )
    else:
        api = _deployment(documents, "api")
        worker = _deployment(documents, "worker")
        migration = _resource(documents, "Job")
        assert _container(api)["livenessProbe"]["httpGet"]["path"] == "/health/live"
        assert _container(api)["readinessProbe"]["httpGet"]["path"] == "/health/ready"
        assert _container(worker)["readinessProbe"]["exec"]["command"][:3] == [
            "vault-rag",
            "db",
            "check",
        ]
        assert set(_environment(_container(api))) == {
            "VAULT_RAG_DATABASE_URL",
            "LITELLM_API_KEY",
            "VAULT_RAG_EMBEDDING_MODEL",
        }
        assert set(_environment(_container(worker))) == {
            "VAULT_RAG_DATABASE_URL",
            "GITHUB_TOKEN",
            "LITELLM_API_KEY",
            "VAULT_RAG_EMBEDDING_MODEL",
        }
        assert set(_environment(_container(migration))) == {"VAULT_RAG_DATABASE_URL"}
        assert api["spec"]["template"]["spec"]["affinity"] == expected_affinity
        assert worker["spec"]["template"]["spec"]["affinity"] == expected_affinity
        pdb_components = {
            pdb["spec"]["selector"]["matchLabels"]["app.kubernetes.io/component"]
            for pdb in _resources(documents, "PodDisruptionBudget")
        }
        assert pdb_components == {"api", "worker"}
        policy_components = {
            policy["spec"]["podSelector"]["matchLabels"]["app.kubernetes.io/component"]
            for policy in _resources(documents, "NetworkPolicy")
        }
        assert policy_components == {"api", "worker", "migration"}
        assert not (
            {"StatefulSet", "PersistentVolumeClaim"} & {document["kind"] for document in documents}
        )


@pytest.mark.parametrize(
    ("backend", "values_text", "field"),
    [
        ("sqlite", "storage:\n  replicas: 2\n", "replicas"),
        (
            "sqlite",
            "api:\n  databaseSecret:\n    name: forbidden-in-sqlite\n    key: database_url\n",
            "api",
        ),
        ("sqlite", "storage:\n  allowInsecureTransport: false\n", "allowInsecureTransport"),
        (
            "sqlite",
            "storage:\n  cleanup:\n    enabled: false\n"
            "    keepPromoted: 3\n    minAge: 7d\n    batchSize: 100\n",
            "cleanup",
        ),
        ("postgresql", "storage:\n  className: forbidden-in-postgresql\n", "className"),
        (
            "postgresql",
            "serviceConfig:\n  repositories:\n    example-vault:\n      url: https://user:secret@github.com/example/vault-rag-example.git\n",
            "url",
        ),
        ("postgresql", "api:\n  pdb:\n    minAvailable: null\n", "minAvailable"),
    ],
)
def test_schema_rejects_mode_violations_and_repository_userinfo(
    tmp_path: Path, backend: str, values_text: str, field: str
) -> None:
    invalid_values = tmp_path / "invalid-values.yaml"
    invalid_values.write_text(values_text, encoding="utf-8")
    completed = subprocess.run(
        [
            HELM or "helm",
            "template",
            "vault-rag",
            str(CHART),
            "--values",
            str(FIXTURES / f"{backend}-values.yaml"),
            "--values",
            str(invalid_values),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert completed.returncode != 0
    assert field in completed.stderr


def test_postgresql_fixture_is_a_private_hardened_split_release() -> None:
    documents = _render_fixture("postgresql")
    kinds = {document["kind"] for document in documents}

    assert {
        "ConfigMap",
        "Service",
        "Deployment",
        "Job",
        "PodDisruptionBudget",
        "NetworkPolicy",
    } <= kinds
    assert not ({"StatefulSet", "PersistentVolumeClaim", "Ingress", "Gateway", "HTTPRoute"} & kinds)

    api = _deployment(documents, "api")
    worker = _deployment(documents, "worker")
    migration = _resource(documents, "Job")
    assert api["spec"]["replicas"] == 2
    assert worker["spec"]["replicas"] == 2
    assert migration["metadata"]["annotations"]["helm.sh/hook"] == "pre-install,pre-upgrade"
    assert migration["metadata"]["annotations"]["helm.sh/hook-weight"] == "-5"
    migration_config = next(
        config_map
        for config_map in _resources(documents, "ConfigMap")
        if config_map["metadata"]["name"].endswith("-migration-config")
    )
    assert migration_config["metadata"]["annotations"]["helm.sh/hook"] == "pre-install,pre-upgrade"
    assert migration_config["metadata"]["annotations"]["helm.sh/hook-weight"] == "-10"
    assert (
        migration_config["metadata"]["annotations"]["helm.sh/hook-delete-policy"]
        == "before-hook-creation"
    )
    assert (
        migration_config["data"]["service.yaml"]
        == _regular_config_map(documents)["data"]["service.yaml"]
    )

    api_container = _container(api)
    worker_container = _container(worker)
    migration_container = migration["spec"]["template"]["spec"]["containers"][0]
    assert api_container["args"] == [
        "api",
        "--service-config",
        "/config/service.yaml",
        "--data-root",
        "/tmp/vault-rag",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ]
    assert worker_container["args"] == [
        "worker",
        "--service-config",
        "/config/service.yaml",
        "--data-root",
        "/work",
    ]
    assert migration_container["args"] == [
        "db",
        "migrate",
        "--service-config",
        "/config/service.yaml",
    ]
    migration_volumes = {
        volume["name"]: volume for volume in migration["spec"]["template"]["spec"]["volumes"]
    }
    assert migration_volumes["config"]["configMap"]["name"].endswith("-migration-config")
    assert api_container["ports"] == [{"name": "http", "containerPort": 8080, "protocol": "TCP"}]
    assert api_container["livenessProbe"]["httpGet"]["path"] == "/health/live"
    assert api_container["readinessProbe"]["httpGet"]["path"] == "/health/ready"
    assert "ports" not in worker_container
    assert worker_container["livenessProbe"]["exec"]["command"] == ["/bin/sh", "-c", "kill -0 1"]
    assert worker_container["readinessProbe"]["exec"]["command"] == [
        "vault-rag",
        "db",
        "check",
        "--service-config",
        "/config/service.yaml",
    ]

    api_environment = _environment(api_container)
    worker_environment = _environment(worker_container)
    migration_environment = _environment(migration_container)
    assert set(api_environment) == {
        "VAULT_RAG_DATABASE_URL",
        "LITELLM_API_KEY",
        "VAULT_RAG_EMBEDDING_MODEL",
    }
    assert set(worker_environment) == {
        "VAULT_RAG_DATABASE_URL",
        "GITHUB_TOKEN",
        "LITELLM_API_KEY",
        "VAULT_RAG_EMBEDDING_MODEL",
    }
    assert set(migration_environment) == {"VAULT_RAG_DATABASE_URL"}
    assert (
        api_environment["VAULT_RAG_DATABASE_URL"]["valueFrom"]["secretKeyRef"]["name"]
        == "vault-rag-api-db"
    )
    assert (
        worker_environment["VAULT_RAG_DATABASE_URL"]["valueFrom"]["secretKeyRef"]["name"]
        == "vault-rag-worker-db"
    )
    assert (
        api_environment["LITELLM_API_KEY"]["valueFrom"]["secretKeyRef"]["name"]
        == "vault-rag-api-runtime"
    )
    assert (
        worker_environment["GITHUB_TOKEN"]["valueFrom"]["secretKeyRef"]["name"]
        == "vault-rag-worker-runtime"
    )
    assert (
        worker_environment["LITELLM_API_KEY"]["valueFrom"]["secretKeyRef"]["name"]
        == "vault-rag-worker-runtime"
    )
    assert (
        migration_environment["VAULT_RAG_DATABASE_URL"]["valueFrom"]["secretKeyRef"]["name"]
        == "vault-rag-migrator-db"
    )

    worker_volumes = {
        volume["name"]: volume for volume in worker["spec"]["template"]["spec"]["volumes"]
    }
    assert worker_volumes["checkout"]["emptyDir"]["sizeLimit"] == "2Gi"
    assert "config" in worker_volumes
    assert worker["spec"]["template"]["spec"]["terminationGracePeriodSeconds"] == 600

    for workload in (api, worker, migration):
        pod_spec = workload["spec"]["template"]["spec"]
        container = (
            _container(workload) if workload["kind"] == "Deployment" else migration_container
        )
        assert pod_spec["securityContext"]["fsGroup"] == 10001
        assert pod_spec["automountServiceAccountToken"] is False
        assert pod_spec["securityContext"]["runAsNonRoot"] is True
        assert pod_spec["securityContext"]["seccompProfile"]["type"] == "RuntimeDefault"
        assert container["securityContext"]["readOnlyRootFilesystem"] is True
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
        assert container["securityContext"]["capabilities"]["drop"] == ["ALL"]
        assert any(mount["mountPath"] == "/tmp" for mount in container["volumeMounts"])

    service = _resource(documents, "Service")
    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["selector"]["app.kubernetes.io/component"] == "api"

    config_map = _regular_config_map(documents)
    service_config = yaml.safe_load(config_map["data"]["service.yaml"])
    assert service_config["schemaVersion"] == 2
    assert service_config["storage"]["backend"] == "postgresql"
    assert service_config["storage"]["databaseUrlEnv"] == "VAULT_RAG_DATABASE_URL"
    assert service_config["storage"]["ownerRole"] == "vault_rag_owner"
    assert service_config["storage"]["allowInsecureTransport"] is True
    assert service_config["storage"]["cleanup"] == {
        "enabled": False,
        "keepPromoted": 3,
        "minAge": "7d",
        "batchSize": 100,
    }

    pdbs = {
        pdb["spec"]["selector"]["matchLabels"]["app.kubernetes.io/component"]: pdb
        for pdb in _resources(documents, "PodDisruptionBudget")
    }
    assert set(pdbs) == {"api", "worker"}
    assert pdbs["api"]["spec"]["minAvailable"] == 1
    assert pdbs["worker"]["spec"]["minAvailable"] == 1
    network_policies = {
        policy["spec"]["podSelector"]["matchLabels"]["app.kubernetes.io/component"]: policy
        for policy in _resources(documents, "NetworkPolicy")
    }
    assert set(network_policies) == {"api", "worker", "migration"}
    api_policy = network_policies["api"]
    worker_policy = network_policies["worker"]
    migration_policy = network_policies["migration"]
    ingress = api_policy["spec"]["ingress"][0]
    assert any("ipBlock" in peer for peer in ingress["from"])
    assert any("namespaceSelector" in peer for peer in ingress["from"])
    assert ingress["ports"] == [{"protocol": "TCP", "port": 8080}]
    assert api_policy["spec"]["egress"][0]["to"] == [{"ipBlock": {"cidr": "192.0.2.0/32"}}]
    assert api_policy["spec"]["egress"][1]["to"] == [{"ipBlock": {"cidr": "192.0.2.0/32"}}]
    assert api_policy["spec"]["egress"][0]["ports"] == [{"protocol": "TCP", "port": 5432}]
    assert worker_policy["spec"]["policyTypes"] == ["Ingress", "Egress"]
    assert "ingress" not in worker_policy["spec"]
    assert worker_policy["spec"]["egress"][0]["ports"] == [{"protocol": "TCP", "port": 5432}]
    assert migration_policy["metadata"]["annotations"] == {
        "helm.sh/hook": "pre-install,pre-upgrade",
        "helm.sh/hook-weight": "-20",
        "helm.sh/hook-delete-policy": "before-hook-creation",
    }
    assert migration_policy["spec"]["policyTypes"] == ["Ingress", "Egress"]
    assert "ingress" not in migration_policy["spec"]
    assert migration_policy["spec"]["egress"] == [
        {
            "to": [{"ipBlock": {"cidr": "192.0.2.0/32"}}],
            "ports": [{"protocol": "TCP", "port": 5432}],
        },
        {
            "to": [
                {
                    "namespaceSelector": {
                        "matchLabels": {"kubernetes.io/metadata.name": "kube-system"}
                    }
                }
            ],
            "ports": [
                {"protocol": "UDP", "port": 53},
                {"protocol": "TCP", "port": 53},
            ],
        },
    ]


def test_digest_and_optional_operational_resources_render(tmp_path: Path) -> None:
    values = tmp_path / "values.yaml"
    values.write_text(
        """
image:
  repository: registry.example.test/platform/vault-rag
  tag: ignored-by-digest
  digest: sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
api:
  autoscaling:
    enabled: true
monitoring:
  serviceMonitor: true
  podMonitor: true
""".strip(),
        encoding="utf-8",
    )

    documents = _render(
        "--values",
        str(FIXTURES / "postgresql-values.yaml"),
        "--values",
        str(values),
    )
    api = _deployment(documents, "api")
    assert "replicas" not in api["spec"]
    worker = _deployment(documents, "worker")
    assert _container(api)["image"] == (
        "registry.example.test/platform/vault-rag@sha256:"
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )
    assert _container(worker)["image"] == _container(api)["image"]
    hpa = _resource(documents, "HorizontalPodAutoscaler")
    assert hpa["spec"]["scaleTargetRef"]["kind"] == "Deployment"
    assert hpa["spec"]["minReplicas"] == 2
    assert hpa["spec"]["maxReplicas"] >= hpa["spec"]["minReplicas"]
    assert (
        _resource(documents, "ServiceMonitor")["spec"]["selector"]["matchLabels"][
            "app.kubernetes.io/component"
        ]
        == "api"
    )
    assert (
        _resource(documents, "PodMonitor")["spec"]["selector"]["matchLabels"][
            "app.kubernetes.io/component"
        ]
        == "api"
    )


@pytest.mark.parametrize(
    ("field", "values_text"),
    [
        ("databaseSecret", 'api:\n  databaseSecret:\n    name: ""'),
        ("runtimeSecret", 'worker:\n  runtimeSecret:\n    githubToken: ""'),
        ("maxSize", "storage:\n  pool:\n    maxSize: 0"),
        ("schemaVersion", "serviceConfig:\n  schemaVersion: 1"),
        ("timeout", "storage:\n  pool:\n    timeout: 100ms"),
        ("statementTimeout", "storage:\n  statementTimeout: 301s"),
        ("allowInsecureTransport", "storage:\n  allowInsecureTransport: null"),
        ("keepPromoted", "storage:\n  cleanup:\n    keepPromoted: 1001"),
        ("minAge", "storage:\n  cleanup:\n    minAge: 366d"),
        ("batchSize", "storage:\n  cleanup:\n    batchSize: 0"),
        ("syncInterval", "serviceConfig:\n  syncInterval: 25h"),
    ],
)
def test_schema_rejects_incomplete_or_unsafe_postgresql_values(
    tmp_path: Path, field: str, values_text: str
) -> None:
    values = tmp_path / "invalid-values.yaml"
    values.write_text(values_text, encoding="utf-8")

    completed = subprocess.run(
        [
            HELM or "helm",
            "template",
            "vault-rag",
            str(CHART),
            "--values",
            str(FIXTURES / "postgresql-values.yaml"),
            "--values",
            str(values),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode != 0
    assert field in completed.stderr


def test_chart_renders_explicit_dns_rebinding_policy_for_stateless_mcp(tmp_path: Path) -> None:
    values = tmp_path / "mcp-values.yaml"
    values.write_text(
        """serviceConfig:
  mcp:
    enabled: true
    allowedHosts:
      - vault-rag.example.test
      - vault-rag.tools.svc.cluster.local:*
    allowedOrigins:
      - https://vault-rag.example.test
""",
        encoding="utf-8",
    )

    documents = _render(
        "--values",
        str(FIXTURES / "postgresql-values.yaml"),
        "--values",
        str(values),
    )
    config = yaml.safe_load(_regular_config_map(documents)["data"]["service.yaml"])
    assert config["mcp"] == {
        "enabled": True,
        "allowedHosts": [
            "vault-rag.example.test",
            "vault-rag.tools.svc.cluster.local:*",
        ],
        "allowedOrigins": ["https://vault-rag.example.test"],
    }

    invalid = tmp_path / "invalid-mcp-values.yaml"
    invalid.write_text(
        """serviceConfig:
  mcp:
    enabled: true
    allowedHosts: []
    allowedOrigins: []
""",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            HELM or "helm",
            "template",
            "vault-rag",
            str(CHART),
            "--values",
            str(FIXTURES / "postgresql-values.yaml"),
            "--values",
            str(invalid),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )
    assert completed.returncode != 0
    assert "allowedHosts" in completed.stderr


def test_template_rejects_reused_role_secrets(tmp_path: Path) -> None:
    values = tmp_path / "invalid-values.yaml"
    values.write_text(
        "worker:\n  databaseSecret:\n    name: vault-rag-api-db\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            HELM or "helm",
            "template",
            "vault-rag",
            str(CHART),
            "--values",
            str(FIXTURES / "postgresql-values.yaml"),
            "--values",
            str(values),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    assert completed.returncode != 0
    assert "must use distinct Secret names" in completed.stderr


def test_rendered_helm_and_ci_artifacts_exclude_runtime_content_sentinels() -> None:
    """Rendered deployment and CI artifacts must never carry runtime request content."""
    rendered = yaml.safe_dump_all(_render_fixture("postgresql"))
    ci = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    sentinels = (
        "task6-sentinel-query",
        "task6-sentinel-source",
        "task6-sentinel-path",
        "task6-sentinel-dsn",
        "task6-sentinel-credential",
        "task6-sentinel-vector",
        "task6-sentinel-fingerprint",
        "task6-sentinel-provider",
    )
    for sentinel in sentinels:
        assert sentinel not in rendered
        assert sentinel not in ci
