"""Validated configuration models for vault registration and profiles."""

from enum import StrEnum
from ipaddress import ip_address
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class EgressPolicy(StrEnum):
    """Whether a vault permits calls to remote embedding endpoints."""

    LOCAL_ONLY = "local-only"
    REMOTE_ALLOWED = "remote-allowed"


class MetadataConfig(BaseModel):
    """Portable metadata extraction policy owned by a vault."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frontmatter_fields: tuple[str, ...] = ()


class VaultManifest(BaseModel):
    """Portable, vault-owned indexing policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,62}$")
    egress_policy: EgressPolicy
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    metadata: MetadataConfig = MetadataConfig()


class ChunkingConfig(BaseModel):
    """Validated token limits used while splitting sources into chunks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_min_tokens: int = 500
    target_max_tokens: int = 900
    overlap_tokens: int = 80
    max_input_tokens: int = 8191

    @model_validator(mode="after")
    def validate_token_limits(self) -> Self:
        """Require coherent chunking bounds."""
        if self.target_min_tokens > self.target_max_tokens:
            raise ValueError("target_min_tokens must not exceed target_max_tokens")
        if self.target_max_tokens >= self.max_input_tokens:
            raise ValueError("target_max_tokens must be less than max_input_tokens")
        if self.overlap_tokens >= self.target_min_tokens:
            raise ValueError("overlap_tokens must be less than target_min_tokens")
        return self


class EmbeddingConfig(ChunkingConfig):
    """Machine-local embedding endpoint configuration."""

    base_url: HttpUrl
    api_key_env: str | None = None
    model_env: str
    endpoint_class: Literal["local", "remote"]
    batch_size: int = 64
    max_batch_tokens: int = 300_000
    tokenizer: str = "cl100k_base"
    revision: str = Field(
        default="1",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    dimensions: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_embedding_config(self) -> Self:
        """Validate endpoint transport, batch bounds, and vector identity."""
        if not 1 <= self.batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")
        if (
            isinstance(self.max_batch_tokens, bool)
            or not self.max_input_tokens <= self.max_batch_tokens <= 300_000
        ):
            raise ValueError("max_batch_tokens must be between max_input_tokens and 300000")
        if isinstance(self.dimensions, bool):
            raise ValueError("dimensions must be a positive integer")
        if self.api_key_env is not None and self.api_key_env == self.model_env:
            raise ValueError("api_key_env and model_env must name different variables")
        if self.base_url.scheme == "http":
            host = (self.base_url.host or "").strip("[]").rstrip(".").casefold()
            loopback = host == "localhost"
            if not loopback:
                try:
                    loopback = ip_address(host).is_loopback
                except ValueError:
                    loopback = False
            if self.endpoint_class != "local" or not loopback:
                raise ValueError("plaintext embedding endpoints require local loopback")
        return self


class RegistryVault(BaseModel):
    """Machine-local registration for one vault root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path


class ProfileConfig(BaseModel):
    """Explicit allowlist of vault IDs for a profile."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vaults: tuple[str, ...] = Field(min_length=1)


class RegistryConfig(BaseModel):
    """Machine-local registry and its embedding endpoint configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vaults: dict[str, RegistryVault] = Field(default_factory=dict)
    profiles: dict[str, ProfileConfig] = Field(default_factory=dict)
    embedding: EmbeddingConfig


class ResolvedVault(BaseModel):
    """A registered vault with its canonical root and validated manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Path
    manifest: VaultManifest


class ResolvedProfile(BaseModel):
    """A named profile ready for indexing and retrieval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    vaults: tuple[ResolvedVault, ...]
    embedding_model: str
    api_key_env: str | None
    effective_policy: EgressPolicy
    semantic_enabled: bool
    semantic_disabled_reason: str | None = None
