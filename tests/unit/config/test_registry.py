from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from vault_rag.config.loader import (
    load_manifest,
    load_manifest_bytes,
    load_registry,
    resolve_profile,
)
from vault_rag.config.models import (
    EgressPolicy,
    EmbeddingConfig,
    RegistryConfig,
    VaultManifest,
)
from vault_rag.config.paths import config_home, data_home
from vault_rag.config.registry import register_vault
from vault_rag.errors import ConfigError


def registry_with(
    tmp_path: Path,
    *,
    vaults: dict[str, str],
    profile: tuple[str, ...] | None,
    endpoint_class: str = "local",
) -> RegistryConfig:
    """Create a registry and its vault-owned manifests for resolution tests."""
    registry_path = tmp_path / "config.toml"
    vault_sections: list[str] = []
    for vault_id, policy in vaults.items():
        root = tmp_path / vault_id
        root.mkdir()
        (root / ".vault-rag.toml").write_text(
            "\n".join(
                [
                    "schema_version = 1",
                    f'id = "{vault_id}"',
                    f'egress_policy = "{policy}"',
                    'include = ["**/*.md"]',
                ]
            )
        )
        vault_sections.extend([f"[vaults.{vault_id}]", f'path = "{root}"', ""])

    profile_section = []
    if profile is not None:
        profile_section = ["[profiles.combined]", f"vaults = {list(profile)!r}", ""]

    base_url = (
        "https://embedding.example.test/v1"
        if endpoint_class == "remote"
        else "http://127.0.0.1:4000/v1"
    )
    registry_path.write_text(
        "\n".join(
            [
                *vault_sections,
                *profile_section,
                "[embedding]",
                f'base_url = "{base_url}"',
                'api_key_env = "LITELLM_API_KEY"',
                'model_env = "EMBED_MODEL"',
                f'endpoint_class = "{endpoint_class}"',
            ]
        )
    )
    return load_registry(registry_path, environ={})


def test_multi_vault_profile_disables_remote_semantics_but_keeps_profile_usable(
    tmp_path: Path,
) -> None:
    registry = registry_with(
        tmp_path,
        vaults={"private": "local-only", "work": "remote-allowed"},
        profile=("private", "work"),
        endpoint_class="remote",
    )
    resolved = resolve_profile("combined", registry, environ={"EMBED_MODEL": "embed-v1"})
    assert resolved.effective_policy is EgressPolicy.LOCAL_ONLY
    assert resolved.semantic_enabled is False
    assert resolved.semantic_disabled_reason == "remote endpoint prohibited by local-only vault"


def test_no_implicit_all_profile(tmp_path: Path) -> None:
    registry = registry_with(tmp_path, vaults={"a": "remote-allowed"}, profile=None)
    with pytest.raises(ConfigError, match="profile is required"):
        resolve_profile("", registry, environ={"EMBED_MODEL": "embed-v1"})


def test_embedding_config_rejects_invalid_batch_and_chunk_bounds() -> None:
    base = {
        "base_url": "http://127.0.0.1:4000/v1",
        "model_env": "EMBED_MODEL",
        "endpoint_class": "local",
    }
    with pytest.raises(ValueError):
        EmbeddingConfig(**base, target_min_tokens=901, target_max_tokens=900)
    with pytest.raises(ValueError):
        EmbeddingConfig(**base, target_max_tokens=8191)
    with pytest.raises(ValueError):
        EmbeddingConfig(**base, overlap_tokens=500)
    with pytest.raises(ValueError):
        EmbeddingConfig(**base, batch_size=257)
    with pytest.raises(ValueError):
        EmbeddingConfig(**base, max_batch_tokens=8190)
    with pytest.raises(ValueError):
        EmbeddingConfig(**base, max_batch_tokens=300_001)
    with pytest.raises(ValueError):
        EmbeddingConfig(**base, dimensions=0)
    with pytest.raises(ValueError):
        EmbeddingConfig(**base, revision="")
    config = EmbeddingConfig(**base)
    assert config.tokenizer == "cl100k_base"
    assert config.max_batch_tokens == 300_000
    assert config.revision == "1"
    assert config.dimensions is None


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:4000/v1",
        "http://127.0.0.1:4000/v1",
        "http://[::1]:4000/v1",
        "https://local.example/v1",
    ],
)
def test_local_embedding_endpoint_accepts_loopback_http_or_https(base_url: str) -> None:
    config = EmbeddingConfig(
        base_url=base_url,
        model_env="EMBED_MODEL",
        endpoint_class="local",
    )
    assert config.base_url.scheme in {"http", "https"}


@pytest.mark.parametrize(
    ("base_url", "endpoint_class"),
    [
        ("http://api.example.test/v1?authorization=secret", "remote"),
        ("http://192.0.2.1:4000/v1", "local"),
        ("http://example.test/v1", "local"),
    ],
)
def test_embedding_endpoint_rejects_plaintext_non_loopback_routes(
    base_url: str, endpoint_class: str
) -> None:
    with pytest.raises(ValueError, match="plaintext embedding endpoints require local loopback"):
        EmbeddingConfig(
            base_url=base_url,
            model_env="EMBED_MODEL",
            endpoint_class=endpoint_class,
        )


def test_remote_embedding_endpoint_accepts_https() -> None:
    config = EmbeddingConfig(
        base_url="https://api.example.test/v1",
        model_env="EMBED_MODEL",
        endpoint_class="remote",
    )
    assert config.base_url.scheme == "https"


def test_manifest_id_must_be_lowercase_slug() -> None:
    with pytest.raises(ValueError):
        VaultManifest(
            schema_version=1,
            id="Not a valid ID",
            egress_policy="local-only",
            include=("**/*.md",),
        )


def test_model_is_resolved_from_named_environment(tmp_path: Path) -> None:
    registry = registry_with(tmp_path, vaults={"a": "remote-allowed"}, profile=("a",))
    resolved = resolve_profile(
        "combined",
        registry,
        environ={"EMBED_MODEL": "embed-v1", "LITELLM_API_KEY": "not-the-secret"},
    )
    assert resolved.embedding_model == "embed-v1"
    assert resolved.api_key_env == "LITELLM_API_KEY"
    assert "not-the-secret" not in repr(resolved)
    assert "not-the-secret" not in resolved.model_dump_json()


def test_register_requires_explicit_replacement_for_duplicate_id(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        (root / ".vault-rag.toml").write_text(
            "\n".join(
                [
                    "schema_version = 1",
                    'id = "shared"',
                    'egress_policy = "local-only"',
                    'include = ["**/*.md"]',
                ]
            )
        )
    config_path.write_text(
        "\n".join(
            [
                "[embedding]",
                'base_url = "http://127.0.0.1:4000/v1"',
                'model_env = "EMBED_MODEL"',
                'endpoint_class = "local"',
            ]
        )
    )

    register_vault(first, config_path)
    with pytest.raises(ConfigError, match="replace=True"):
        register_vault(second, config_path)
    registered = register_vault(second, config_path, replace=True)
    assert registered.vaults["shared"].path == second.resolve()


def test_reregister_changed_manifest_id_removes_stale_path_alias_atomically(
    tmp_path: Path,
) -> None:
    root = tmp_path / "vault"
    root.mkdir()
    manifest = root / ".vault-rag.toml"
    manifest.write_text(
        'schema_version = 1\nid = "vault-a"\negress_policy = "local-only"\ninclude = ["**/*.md"]\n',
        encoding="utf-8",
    )
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            [
                "[profiles.old]",
                'vaults = ["vault-a"]',
                "[profiles.new]",
                'vaults = ["vault-b"]',
                "[embedding]",
                'base_url = "http://127.0.0.1:4000/v1"',
                'model_env = "MODEL"',
                'endpoint_class = "local"',
            ]
        ),
        encoding="utf-8",
    )
    register_vault(root, config)
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("vault-a", "vault-b"),
        encoding="utf-8",
    )

    updated = register_vault(root, config)
    persisted = load_registry(config, environ={})

    assert set(updated.vaults) == {"vault-b"}
    assert set(persisted.vaults) == {"vault-b"}
    assert persisted.vaults["vault-b"].path == root.resolve()
    with pytest.raises(ConfigError, match="unregistered vault: vault-a"):
        resolve_profile("old", persisted, {"MODEL": "embed-v1"})
    assert resolve_profile("new", persisted, {"MODEL": "embed-v1"}).vaults[0].manifest.id == (
        "vault-b"
    )


@pytest.mark.parametrize(
    "base_url",
    [
        "http://embedding.test/v1",
        "http://user:password@[/v1",
    ],
)
def test_registry_validation_never_renders_raw_input_values(tmp_path: Path, base_url: str) -> None:
    secret = "sk-inline-supersecret-value"
    config = tmp_path / "config.toml"
    config.write_text(
        "\n".join(
            [
                "[embedding]",
                f'base_url = "{base_url}"',
                'model_env = "MODEL"',
                'endpoint_class = "remote"',
                f'api_key = "{secret}"',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError) as error:
        load_registry(config, environ={})

    rendered = str(error.value)
    assert secret not in rendered
    assert "user:password" not in rendered
    assert "input_value" not in rendered
    assert "embedding" in rendered


@pytest.mark.parametrize("resolver", [config_home, data_home])
def test_xdg_directory_permission_errors_are_typed_config_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    resolver: Callable[[Mapping[str, str]], Path],
) -> None:
    def denied(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("synthetic read-only home")

    monkeypatch.setattr(Path, "mkdir", denied)
    with pytest.raises(ConfigError, match="could not prepare"):
        resolver({"XDG_CONFIG_HOME": str(tmp_path), "XDG_DATA_HOME": str(tmp_path)})


def test_config_home_creates_user_only_directory(tmp_path: Path) -> None:
    home = config_home({"XDG_CONFIG_HOME": str(tmp_path)})
    assert home.stat().st_mode & 0o777 == 0o700


def test_manifest_bytes_are_bounded_and_never_echo_source_content(tmp_path: Path) -> None:
    """A parser regression must not retain or disclose untrusted manifest text."""
    secret = "manifest-secret-content-must-not-appear"
    oversized = b"x" * (1024 * 1024 + 1)
    with pytest.raises(ConfigError) as oversized_error:
        load_manifest_bytes(oversized)
    assert secret not in str(oversized_error.value)

    malformed = f'id = "{secret}" ='.encode()
    with pytest.raises(ConfigError) as malformed_error:
        load_manifest_bytes(malformed)
    assert secret not in str(malformed_error.value)

    manifest_path = tmp_path / ".vault-rag.toml"
    manifest_path.write_bytes(oversized)
    with pytest.raises(ConfigError, match="too large"):
        load_manifest(manifest_path)
