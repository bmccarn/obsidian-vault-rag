"""Load local registry configuration and resolve explicit vault profiles."""

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from vault_rag.errors import ConfigError

from .models import (
    EgressPolicy,
    RegistryConfig,
    ResolvedProfile,
    ResolvedVault,
    VaultManifest,
)

_MAX_VALIDATION_ERRORS = 20
_MAX_VALIDATION_MESSAGE = 200
_MAX_MANIFEST_BYTES = 1024 * 1024


def _validated_model[ModelT: BaseModel](model_type: type[ModelT], data: Any) -> ModelT:
    try:
        return model_type.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_url=False)
        rendered: list[str] = []
        for item in errors[:_MAX_VALIDATION_ERRORS]:
            location = ".".join(str(part) for part in item.get("loc", ())) or "configuration"
            message = " ".join(str(item.get("msg", "is invalid")).split())
            category = str(item.get("type", "validation_error"))
            rendered.append(f"{location}: {message[:_MAX_VALIDATION_MESSAGE]} [{category}]")
        if len(errors) > len(rendered):
            rendered.append(f"{len(errors) - len(rendered)} additional validation errors omitted")
        summary = "; ".join(rendered) or "configuration validation failed"
        raise ConfigError(f"invalid configuration: {summary}") from exc


def _load_toml(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as config_file:
            data = tomllib.load(config_file)
    except FileNotFoundError as exc:
        raise ConfigError(f"{description} does not exist: {path}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not load {description}: {path}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"invalid {description}: expected TOML table")
    return data


def load_manifest_bytes(data: bytes) -> VaultManifest:
    """Parse and validate one bounded vault manifest without exposing its source."""
    if len(data) > _MAX_MANIFEST_BYTES:
        raise ConfigError("vault manifest is too large")
    try:
        parsed = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError("could not load vault manifest") from exc
    if not isinstance(parsed, dict):
        raise ConfigError("invalid vault manifest: expected TOML table")
    return _validated_model(VaultManifest, parsed)


def load_manifest(path: Path) -> VaultManifest:
    """Read a bounded vault-owned manifest and delegate its parsing."""
    try:
        with path.open("rb") as manifest_file:
            data = manifest_file.read(_MAX_MANIFEST_BYTES + 1)
    except FileNotFoundError as exc:
        raise ConfigError(f"vault manifest does not exist: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"could not load vault manifest: {path}") from exc
    return load_manifest_bytes(data)


def load_registry(path: Path, environ: Mapping[str, str]) -> RegistryConfig:
    """Read and validate machine-local registry configuration.

    ``environ`` is accepted deliberately so callers keep all configuration
    inputs explicit; registry secrets are environment *names*, never values.
    """
    del environ
    return _validated_model(RegistryConfig, _load_toml(path, "registry"))


def resolve_profile(
    name: str,
    registry: RegistryConfig,
    environ: Mapping[str, str],
) -> ResolvedProfile:
    """Resolve one explicit profile and determine whether semantics are allowed."""
    profile_name = name or environ.get("VAULT_RAG_PROFILE", "")
    if not profile_name:
        raise ConfigError("profile is required")

    try:
        profile = registry.profiles[profile_name]
    except KeyError as exc:
        raise ConfigError(f"profile is not configured: {profile_name}") from exc

    resolved_vaults: list[ResolvedVault] = []
    for vault_id in profile.vaults:
        try:
            registration = registry.vaults[vault_id]
        except KeyError as exc:
            message = f"profile {profile_name} references unregistered vault: {vault_id}"
            raise ConfigError(message) from exc
        try:
            root = registration.path.expanduser().resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            message = f"could not resolve vault root for {vault_id}: {registration.path}"
            raise ConfigError(message) from exc
        if not root.is_dir():
            raise ConfigError(f"vault root is not a directory for {vault_id}: {root}")
        manifest = load_manifest(root / ".vault-rag.toml")
        if manifest.id != vault_id:
            raise ConfigError(
                f"registered vault ID {vault_id} does not match manifest ID {manifest.id}"
            )
        resolved_vaults.append(ResolvedVault(root=root, manifest=manifest))

    model = environ.get(registry.embedding.model_env)
    if not model:
        raise ConfigError(
            f"embedding model environment variable is required: {registry.embedding.model_env}"
        )

    effective_policy = (
        EgressPolicy.LOCAL_ONLY
        if any(vault.manifest.egress_policy is EgressPolicy.LOCAL_ONLY for vault in resolved_vaults)
        else EgressPolicy.REMOTE_ALLOWED
    )
    semantic_disabled_reason: str | None = None
    if (
        effective_policy is EgressPolicy.LOCAL_ONLY
        and registry.embedding.endpoint_class == "remote"
    ):
        semantic_disabled_reason = "remote endpoint prohibited by local-only vault"

    return ResolvedProfile(
        name=profile_name,
        vaults=tuple(resolved_vaults),
        embedding_model=model,
        api_key_env=registry.embedding.api_key_env,
        effective_policy=effective_policy,
        semantic_enabled=semantic_disabled_reason is None,
        semantic_disabled_reason=semantic_disabled_reason,
    )
