"""XDG paths for machine-local vault-rag state."""

import os
from collections.abc import Mapping
from pathlib import Path

from vault_rag.errors import ConfigError


def _configured_path(environ: Mapping[str, str], variable: str, fallback: str) -> Path:
    return Path(environ.get(variable, fallback)).expanduser() / "vault-rag"


def _user_only_directory(path: Path, description: str) -> Path:
    try:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)
    except OSError as exc:
        raise ConfigError(f"could not prepare {description}: {path}") from exc
    return path


def config_home_path(environ: Mapping[str, str]) -> Path:
    """Return the configured XDG configuration path without touching the filesystem."""
    return _configured_path(environ, "XDG_CONFIG_HOME", "~/.config")


def data_home_path(environ: Mapping[str, str]) -> Path:
    """Return the configured XDG data path without touching the filesystem."""
    return _configured_path(environ, "XDG_DATA_HOME", "~/.local/share")


def config_home(environ: Mapping[str, str]) -> Path:
    """Return and create the user-only XDG configuration directory."""
    return _user_only_directory(config_home_path(environ), "configuration directory")


def data_home(environ: Mapping[str, str]) -> Path:
    """Return and create the user-only XDG data directory."""
    return _user_only_directory(data_home_path(environ), "data directory")
