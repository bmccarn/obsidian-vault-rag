"""Safe persistence for the machine-local vault registry."""

import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomli_w

from vault_rag.errors import ConfigError

from .loader import load_manifest, load_registry
from .models import RegistryConfig, RegistryVault


@dataclass(frozen=True, slots=True)
class RegistrationOutcome:
    """Persisted identity returned by one atomic registration operation."""

    registry: RegistryConfig
    vault_id: str
    path: Path


def _registry_document(registry: RegistryConfig) -> dict[str, Any]:
    """Convert validated machine-local configuration to TOML-safe values."""
    return {
        "vaults": {
            vault_id: {"path": str(vault.path)} for vault_id, vault in registry.vaults.items()
        },
        "profiles": {
            profile_name: {"vaults": list(profile.vaults)}
            for profile_name, profile in registry.profiles.items()
        },
        "embedding": registry.embedding.model_dump(mode="json", exclude_none=True),
    }


def _write_registry(path: Path, registry: RegistryConfig) -> None:
    """Atomically persist a registry in a same-directory user-only file."""
    parent = path.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(parent, 0o700)
    contents = tomli_w.dumps(_registry_document(registry))
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(contents)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, 0o600)
        os.replace(temporary_path, path)
    except OSError as exc:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise ConfigError(f"could not persist registry: {path}") from exc


def _registered_path(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"could not resolve registered vault path: {path}") from exc


def register_vault_outcome(
    root: Path, config_path: Path, *, replace: bool = False
) -> RegistrationOutcome:
    """Register a vault and return the identity committed by the atomic write."""
    try:
        canonical_root = Path(root).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConfigError(f"could not resolve vault root: {root}") from exc
    if not canonical_root.is_dir():
        raise ConfigError(f"vault root is not a directory: {canonical_root}")

    manifest = load_manifest(canonical_root / ".vault-rag.toml")
    registry_path = Path(config_path).expanduser()
    registry = load_registry(registry_path, environ={})
    existing = registry.vaults.get(manifest.id)
    if existing is not None and _registered_path(existing.path) != canonical_root and not replace:
        raise ConfigError(
            f"vault ID {manifest.id} is already registered at {existing.path}; use replace=True"
        )

    stale_aliases = {
        vault_id
        for vault_id, registration in registry.vaults.items()
        if vault_id != manifest.id and _registered_path(registration.path) == canonical_root
    }
    updated_vaults = {
        vault_id: registration
        for vault_id, registration in registry.vaults.items()
        if vault_id not in stale_aliases
    }
    updated_vaults[manifest.id] = RegistryVault(path=canonical_root)
    updated = registry.model_copy(update={"vaults": updated_vaults})
    _write_registry(registry_path, updated)
    return RegistrationOutcome(updated, manifest.id, canonical_root)


def register_vault(root: Path, config_path: Path, *, replace: bool = False) -> RegistryConfig:
    """Register a manifest-validated vault root in a machine-local registry."""
    return register_vault_outcome(root, config_path, replace=replace).registry
