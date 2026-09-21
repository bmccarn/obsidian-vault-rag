from pathlib import Path

import pytest

from vault_rag.errors import ConfigError
from vault_rag.service.config import (
    ServiceConfig,
    load_service_config,
    materialize_secret_files,
)


def write_service_config(tmp_path: Path, **replacements: str) -> Path:
    """Write a valid service configuration, optionally replacing YAML fragments."""
    content = """
schemaVersion: 1
syncInterval: 5m
repositories:
  homelab-ops:
    url: https://github.com/example/homelab-ops.git
    ref: refs/heads/main
    credentialEnv: GITHUB_TOKEN
profiles:
  homelab:
    vaults: [homelab-ops]
embedding:
  baseUrl: https://litellm.example.test/v1
  apiKeyEnv: LITELLM_API_KEY
  modelEnv: VAULT_RAG_EMBEDDING_MODEL
  endpointClass: remote
  revision: baseline-2026-08
  dimensions: 1024
"""
    for old, new in replacements.items():
        content = content.replace(old, new)
    path = tmp_path / "service.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def test_service_config_loads_issue_shape(tmp_path: Path) -> None:
    config = load_service_config(write_service_config(tmp_path))

    assert config.sync_interval.total_seconds() == 300
    assert config.repositories["homelab-ops"].ref == "refs/heads/main"
    registry = config.registry(tmp_path / "data")
    assert registry.vaults["homelab-ops"].path == tmp_path / "data/repos/homelab-ops"
    assert registry.profiles["_reconcile_homelab-ops"].vaults == ("homelab-ops",)
    assert config.mcp.enabled is False


def test_mcp_transport_is_explicit_strict_and_requires_a_host_allowlist(tmp_path: Path) -> None:
    config = load_service_config(
        write_service_config(
            tmp_path,
            **{
                "syncInterval: 5m": """syncInterval: 5m
mcp:
  enabled: true
  allowedHosts:
    - vault-rag.example.test
    - vault-rag.tools.svc.cluster.local:*
  allowedOrigins:
    - https://vault-rag.example.test""",
            },
        )
    )

    assert config.mcp.enabled is True
    assert config.mcp.allowed_hosts == (
        "vault-rag.example.test",
        "vault-rag.tools.svc.cluster.local:*",
    )
    assert config.mcp.allowed_origins == ("https://vault-rag.example.test",)

    with pytest.raises(ConfigError, match="allowed_hosts"):
        load_service_config(
            write_service_config(
                tmp_path,
                **{"syncInterval: 5m": "syncInterval: 5m\nmcp:\n  enabled: true"},
            )
        )
    with pytest.raises(ConfigError):
        load_service_config(
            write_service_config(
                tmp_path,
                **{
                    "syncInterval: 5m": (
                        "syncInterval: 5m\nmcp:\n  enabled: false\n  allowedHosts: [localhost]"
                    )
                },
            )
        )


def test_repository_config_accepts_only_an_absolute_ca_file_path(tmp_path: Path) -> None:
    config = load_service_config(
        write_service_config(
            tmp_path,
            **{
                "credentialEnv: GITHUB_TOKEN": (
                    "credentialEnv: GITHUB_TOKEN\n    caCertPath: /fixture/ca/root.crt"
                )
            },
        )
    )

    assert config.repositories["homelab-ops"].ca_cert_path == Path("/fixture/ca/root.crt")
    with pytest.raises(ConfigError):
        load_service_config(
            write_service_config(
                tmp_path,
                **{"credentialEnv: GITHUB_TOKEN": "caCertPath: relative-ca.pem"},
            )
        )


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("https://github.com/example/homelab-ops.git", "https://token@github.com/example/repo.git"),
        ("https://github.com/example/homelab-ops.git", "http://github.com/example/repo.git"),
        ("refs/heads/main", "0123456789012345678901234567890123456789"),
        ("refs/heads/main", "refs/heads/main~1"),
        ("refs/heads/main", "refs/heads/feature..old"),
        ("refs/heads/main", "refs/heads/.hidden"),
        ("refs/heads/main", "refs/heads/feature\\branch"),
        ("refs/heads/main", "refs/heads/feature\u0001branch"),
    ],
)
def test_service_config_rejects_unsafe_repository_values(
    tmp_path: Path, old: str, new: str
) -> None:
    with pytest.raises(ConfigError):
        load_service_config(write_service_config(tmp_path, **{old: new}))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("vaults: [homelab-ops]", "vaults: [unknown]"),
        ("vaults: [homelab-ops]", "vaults: [homelab-ops, homelab-ops]"),
        ("  homelab:\n", "  _reconcile_homelab-ops:\n"),
        ("syncInterval: 5m", "syncInterval: 0s"),
        ("syncInterval: 5m", "syncInterval: 25h"),
        ("  dimensions: 1024", "  dimensions: 1024\n  unknown: value"),
    ],
)
def test_service_config_rejects_invalid_structure(tmp_path: Path, old: str, new: str) -> None:
    with pytest.raises(ConfigError):
        load_service_config(write_service_config(tmp_path, **{old: new}))


def test_service_config_rejects_files_larger_than_one_mebibyte(tmp_path: Path) -> None:
    path = tmp_path / "service.yaml"
    path.write_bytes(b"#" * (1024 * 1024 + 1))

    with pytest.raises(ConfigError, match="too large"):
        load_service_config(path)


@pytest.fixture
def config(tmp_path: Path) -> ServiceConfig:
    return load_service_config(write_service_config(tmp_path))


def test_materialize_secret_file_without_mutating_input(
    tmp_path: Path, config: ServiceConfig
) -> None:
    secret = tmp_path / "github-token"
    secret.write_text("synthetic-token-value\n", encoding="utf-8")
    environ = {"GITHUB_TOKEN_FILE": str(secret), "VAULT_RAG_EMBEDDING_MODEL": "model"}

    captured = materialize_secret_files(config, environ)

    assert captured["GITHUB_TOKEN"] == "synthetic-token-value"
    assert "GITHUB_TOKEN" not in environ
    assert captured["VAULT_RAG_EMBEDDING_MODEL"] == "model"
    with pytest.raises(TypeError):
        captured["GITHUB_TOKEN"] = "replacement"  # type: ignore[index]


def test_materialize_secret_files_prefers_direct_values_and_ignores_unreferenced_files(
    tmp_path: Path, config: ServiceConfig
) -> None:
    secret = tmp_path / "github-token"
    secret.write_text("from-file", encoding="utf-8")
    unrelated = tmp_path / "unrelated"
    unrelated.write_text("unrelated", encoding="utf-8")

    captured = materialize_secret_files(
        config,
        {
            "GITHUB_TOKEN": "direct-value",
            "GITHUB_TOKEN_FILE": str(secret),
            "VAULT_RAG_EMBEDDING_MODEL": "model",
            "UNRELATED_FILE": str(unrelated),
        },
    )

    assert captured == {
        "GITHUB_TOKEN": "direct-value",
        "VAULT_RAG_EMBEDDING_MODEL": "model",
    }


@pytest.mark.parametrize(
    "contents",
    ["", "x" * (64 * 1024 + 1)],
)
def test_materialize_secret_files_rejects_empty_or_oversized_files(
    tmp_path: Path, config: ServiceConfig, contents: str
) -> None:
    secret = tmp_path / "github-token"
    secret.write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError):
        materialize_secret_files(
            config,
            {"GITHUB_TOKEN_FILE": str(secret), "VAULT_RAG_EMBEDDING_MODEL": "model"},
        )


def test_materialize_secret_files_rejects_unreadable_file(
    tmp_path: Path, config: ServiceConfig
) -> None:
    with pytest.raises(ConfigError, match="could not read secret file"):
        materialize_secret_files(
            config,
            {
                "GITHUB_TOKEN_FILE": str(tmp_path / "missing"),
                "VAULT_RAG_EMBEDDING_MODEL": "model",
            },
        )


def test_postgresql_storage_configuration_materializes_database_secret(
    tmp_path: Path,
) -> None:
    path = write_service_config(
        tmp_path,
        **{
            "schemaVersion: 1": "schemaVersion: 2",
            "syncInterval: 5m": """syncInterval: 5m
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  ownerRole: vault_rag_owner
  pool:
    minSize: 2
    maxSize: 8
    timeout: 3s
  connectTimeout: 2s
  statementTimeout: 4s
  lockTimeout: 5s
  idleTransactionTimeout: 6s""",
        },
    )
    secret = tmp_path / "database-url"
    secret.write_text("postgresql://vault_rag:secret@postgres/vault_rag\n", encoding="utf-8")

    config = load_service_config(path)
    captured = materialize_secret_files(
        config,
        {
            "VAULT_RAG_DATABASE_URL_FILE": str(secret),
            "VAULT_RAG_EMBEDDING_MODEL": "model",
        },
    )

    assert config.storage.backend == "postgresql"
    assert config.storage.owner_role == "vault_rag_owner"
    assert config.storage.pool.min_size == 2
    assert config.storage.pool.max_size == 8
    assert config.storage.pool.timeout.total_seconds() == 3
    assert config.storage.connect_timeout.total_seconds() == 2
    assert config.storage.statement_timeout.total_seconds() == 4
    assert config.storage.lock_timeout.total_seconds() == 5
    assert config.storage.idle_transaction_timeout.total_seconds() == 6
    assert captured["VAULT_RAG_DATABASE_URL"].startswith("postgresql://")


def test_postgresql_cleanup_requires_explicit_enablement_and_validates_policy(
    tmp_path: Path,
) -> None:
    path = write_service_config(
        tmp_path,
        **{
            "schemaVersion: 1": "schemaVersion: 2",
            "syncInterval: 5m": """syncInterval: 5m
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  cleanup:
    enabled: true
    keepPromoted: 4
    minAge: 8d
    batchSize: 25
  allowInsecureTransport: true""",
        },
    )

    config = load_service_config(path)

    assert config.storage.cleanup.enabled is True
    assert config.storage.cleanup.keep_promoted == 4
    assert config.storage.cleanup.min_age.total_seconds() == 8 * 24 * 60 * 60
    assert config.storage.cleanup.batch_size == 25
    assert config.storage.allow_insecure_transport is True
    assert load_service_config(write_service_config(tmp_path)).storage.cleanup.enabled is False


def test_runtime_propagates_postgresql_pool_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vault_rag.service.runtime as runtime

    created: list[tuple[str, dict[str, object]]] = []

    class FakePool:
        def __init__(self, dsn: str, **kwargs: object) -> None:
            created.append((dsn, kwargs))

        def open(self, *, wait: bool = False) -> None:
            assert wait is True

        def close(self) -> None:
            raise AssertionError("successful pool construction must not close the pool")

    path = write_service_config(
        tmp_path,
        **{
            "schemaVersion: 1": "schemaVersion: 2",
            "syncInterval: 5m": """syncInterval: 5m
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  pool:
    minSize: 2
    maxSize: 8
    timeout: 3s
  connectTimeout: 2s
  statementTimeout: 4s
  lockTimeout: 5s
  idleTransactionTimeout: 6s""",
        },
    )
    config = load_service_config(path)
    monkeypatch.setattr(runtime, "PostgresPool", FakePool)

    runtime._postgres_pool(config, {"VAULT_RAG_DATABASE_URL": "postgresql://db/vault"})

    assert created == [
        (
            "postgresql://db/vault",
            {
                "min_size": 2,
                "max_size": 8,
                "timeout": 3.0,
                "connect_timeout": 2.0,
                "statement_timeout": 4.0,
                "lock_timeout": 5.0,
                "idle_transaction_timeout": 6.0,
                "allow_insecure_transport": False,
                "metrics": None,
            },
        )
    ]


def test_migration_uses_one_lazy_database_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import vault_rag.service.runtime as runtime

    path = write_service_config(
        tmp_path,
        **{
            "schemaVersion: 1": "schemaVersion: 2",
            "syncInterval: 5m": """syncInterval: 5m
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL""",
        },
    )
    observed: list[dict[str, int]] = []

    class FakePool:
        def close(self) -> None:
            pass

    class FakeMigrator:
        def __init__(self, pool: FakePool, *, owner_role: str | None = None) -> None:
            assert isinstance(pool, FakePool)
            assert owner_role is None

        def apply(self) -> tuple[int, ...]:
            return (1,)

        def check(self) -> tuple[int, ...]:
            return (1, 2)

    def build_pool(
        _config: object,
        _environ: object,
        **bounds: int,
    ) -> FakePool:
        observed.append(bounds)
        return FakePool()

    monkeypatch.setattr(runtime, "_postgres_pool", build_pool)
    monkeypatch.setattr(runtime, "PostgresMigrator", FakeMigrator)

    assert runtime.migrate_database(
        path,
        {"VAULT_RAG_DATABASE_URL": "postgresql://db/vault"},
    ) == (1,)
    assert runtime.check_database(
        path,
        {"VAULT_RAG_DATABASE_URL": "postgresql://db/vault"},
    ) == (1, 2)
    assert observed == [{"min_size": 0, "max_size": 1}, {"min_size": 0, "max_size": 1}]


def test_storage_defaults_to_sqlite(tmp_path: Path) -> None:
    config = load_service_config(write_service_config(tmp_path))

    assert config.storage.backend == "sqlite"
    assert config.storage.database_url_env is None
    assert config.storage.owner_role is None


@pytest.mark.parametrize(
    "storage",
    [
        "schemaVersion: 2\nstorage:\n  backend: postgresql",
        """schemaVersion: 2
storage:
  backend: sqlite
  databaseUrlEnv: VAULT_RAG_DATABASE_URL""",
        """schemaVersion: 2
storage:
  backend: postgresql
  databaseUrlEnv: invalid-name""",
        """schemaVersion: 2
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  ownerRole: Invalid-Role""",
        """schemaVersion: 2
storage:
  backend: sqlite
  ownerRole: vault_rag_owner""",
        """schemaVersion: 2
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  pool:
    minSize: 9
    maxSize: 2""",
        """schemaVersion: 2
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  pool:
    timeout: 0s""",
        """schemaVersion: 2
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  pool:
    timeout: 301s""",
        """schemaVersion: 2
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL
  statementTimeout: 301s""",
        """schemaVersion: 1
storage:
  backend: postgresql
  databaseUrlEnv: VAULT_RAG_DATABASE_URL""",
    ],
)
def test_storage_configuration_rejects_invalid_combinations(
    tmp_path: Path,
    storage: str,
) -> None:
    path = write_service_config(
        tmp_path,
        **{"syncInterval: 5m": f"syncInterval: 5m\n{storage}"},
    )

    with pytest.raises(ConfigError):
        load_service_config(path)
