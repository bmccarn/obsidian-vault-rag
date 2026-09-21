"""Vault manifests, machine registry, and explicit profile resolution."""

from .loader import load_registry, resolve_profile
from .models import (
    ChunkingConfig,
    EgressPolicy,
    EmbeddingConfig,
    RegistryConfig,
    ResolvedProfile,
    ResolvedVault,
    VaultManifest,
)
from .registry import RegistrationOutcome, register_vault, register_vault_outcome

__all__ = [
    "ChunkingConfig",
    "EgressPolicy",
    "EmbeddingConfig",
    "RegistrationOutcome",
    "RegistryConfig",
    "ResolvedProfile",
    "ResolvedVault",
    "VaultManifest",
    "load_registry",
    "register_vault",
    "register_vault_outcome",
    "resolve_profile",
]
