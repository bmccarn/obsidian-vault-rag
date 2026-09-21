"""Strict YAML configuration for the shared Git-synced retrieval service."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic.alias_generators import to_camel

from vault_rag.config.models import EmbeddingConfig, ProfileConfig, RegistryConfig, RegistryVault
from vault_rag.errors import ConfigError

_MAX_CONFIG_BYTES = 1024 * 1024
_MAX_SECRET_BYTES = 64 * 1024
_MAX_VALIDATION_ERRORS = 20
_MAX_VALIDATION_MESSAGE = 200
_DURATION_PATTERN = re.compile(r"^([1-9][0-9]*)(s|m|h|d)$")
_VAULT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_POSTGRES_ROLE_PATTERN = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")
_ENVIRONMENT_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REF_PREFIX_PATTERN = re.compile(r"^refs/(?:heads|tags)/(.+)$")
_REF_FORBIDDEN_CHARACTERS = frozenset(" ~^:?*[\\")


class RepositoryConfig(BaseModel):
    """One Git repository managed as a local vault."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, alias_generator=to_camel
    )

    url: HttpUrl
    ref: str = Field(min_length=1, max_length=1024)
    credential_env: str | None = Field(default=None, min_length=1, max_length=128)
    ca_cert_path: Path | None = None

    @field_validator("ca_cert_path")
    @classmethod
    def validate_ca_cert_path(cls, value: Path | None) -> Path | None:
        """Accept a mounted CA file, never a relative host-dependent path."""
        if value is not None and (not value.is_absolute() or "\x00" in str(value)):
            raise ValueError("ca_cert_path must be an absolute path")
        return value

    @field_validator("credential_env")
    @classmethod
    def validate_credential_env(cls, value: str | None) -> str | None:
        """Require a portable environment variable name when configured."""
        if value is not None and _ENVIRONMENT_NAME_PATTERN.fullmatch(value) is None:
            raise ValueError("credential_env must be an environment variable name")
        return value

    @field_validator("ref")
    @classmethod
    def validate_ref(cls, value: str) -> str:
        """Accept only safe, fully-qualified branch and tag references."""
        matched = _REF_PREFIX_PATTERN.fullmatch(value)
        if matched is None:
            raise ValueError("ref must be a refs/heads/... or refs/tags/... reference")

        for component in matched.group(1).split("/"):
            if (
                not component
                or component.startswith(".")
                or component.endswith(".")
                or component.endswith(".lock")
                or component == "@"
                or ".." in component
                or "@{" in component
                or any(character in _REF_FORBIDDEN_CHARACTERS for character in component)
                or any(ord(character) < 32 or ord(character) == 127 for character in component)
            ):
                raise ValueError("ref contains an unsafe revision-expression component")
        return value

    @model_validator(mode="after")
    def validate_url(self) -> Self:
        """Require an HTTPS remote without embedded credentials."""
        if (
            self.url.scheme != "https"
            or self.url.username is not None
            or self.url.password is not None
        ):
            raise ValueError("repository url must be HTTPS without embedded credentials")
        return self


class ServiceProfileConfig(BaseModel):
    """A caller-selectable subset of managed repositories."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, alias_generator=to_camel
    )

    vaults: tuple[str, ...] = Field(min_length=1)


class ServiceEmbeddingConfig(EmbeddingConfig):
    """Embedding settings using the YAML service configuration spelling."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, alias_generator=to_camel
    )


class StorageBackend(StrEnum):
    """Supported service persistence backends."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class StoragePoolConfig(BaseModel):
    """Bounded PostgreSQL pool settings."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, alias_generator=to_camel
    )

    min_size: int = Field(default=1, ge=1, le=100)
    max_size: int = Field(default=10, ge=1, le=100)
    timeout: timedelta = timedelta(seconds=5)

    @field_validator("timeout", mode="before")
    @classmethod
    def parse_timeout(cls, value: Any) -> timedelta:
        return _parse_duration(value, name="pool timeout", maximum_seconds=300)

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        if self.min_size > self.max_size:
            raise ValueError("pool min_size must not exceed max_size")
        return self


class StorageCleanupConfig(BaseModel):
    """Explicit, bounded PostgreSQL revision cleanup settings."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, alias_generator=to_camel
    )

    enabled: bool = False
    keep_promoted: int = Field(default=3, ge=0, le=1_000)
    min_age: timedelta = timedelta(days=7)
    batch_size: int = Field(default=100, ge=1, le=10_000)

    @field_validator("min_age", mode="before")
    @classmethod
    def parse_min_age(cls, value: Any) -> timedelta:
        return _parse_duration(value, name="cleanup min_age", maximum_seconds=365 * 24 * 60 * 60)


class StorageConfig(BaseModel):
    """Validated storage selection without containing a database credential."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, alias_generator=to_camel
    )

    backend: StorageBackend = StorageBackend.SQLITE
    database_url_env: str | None = Field(default=None, min_length=1, max_length=128)
    owner_role: str | None = Field(default=None, min_length=1, max_length=63)
    pool: StoragePoolConfig = StoragePoolConfig()
    connect_timeout: timedelta = timedelta(seconds=5)
    statement_timeout: timedelta = timedelta(seconds=5)
    lock_timeout: timedelta = timedelta(seconds=5)
    idle_transaction_timeout: timedelta = timedelta(seconds=30)
    cleanup: StorageCleanupConfig = StorageCleanupConfig()
    allow_insecure_transport: bool = False

    @field_validator("database_url_env")
    @classmethod
    def validate_database_url_env(cls, value: str | None) -> str | None:
        if value is not None and _ENVIRONMENT_NAME_PATTERN.fullmatch(value) is None:
            raise ValueError("database_url_env must be an environment variable name")
        return value

    @field_validator("owner_role")
    @classmethod
    def validate_owner_role(cls, value: str | None) -> str | None:
        if value is not None and _POSTGRES_ROLE_PATTERN.fullmatch(value) is None:
            raise ValueError("owner_role must be a lowercase PostgreSQL identifier")
        return value

    @field_validator("connect_timeout", mode="before")
    @classmethod
    def parse_connect_timeout(cls, value: Any) -> timedelta:
        return _parse_duration(value, name="connect timeout", maximum_seconds=300)

    @field_validator("statement_timeout", mode="before")
    @classmethod
    def parse_statement_timeout(cls, value: Any) -> timedelta:
        return _parse_duration(value, name="statement timeout", maximum_seconds=300)

    @field_validator("lock_timeout", mode="before")
    @classmethod
    def parse_lock_timeout(cls, value: Any) -> timedelta:
        return _parse_duration(value, name="lock timeout", maximum_seconds=300)

    @field_validator("idle_transaction_timeout", mode="before")
    @classmethod
    def parse_idle_transaction_timeout(cls, value: Any) -> timedelta:
        return _parse_duration(value, name="idle transaction timeout", maximum_seconds=300)

    @model_validator(mode="after")
    def validate_backend(self) -> Self:
        if self.backend is StorageBackend.POSTGRESQL and self.database_url_env is None:
            raise ValueError("PostgreSQL storage requires database_url_env")
        if self.backend is StorageBackend.SQLITE and (
            self.database_url_env is not None
            or self.owner_role is not None
            or self.allow_insecure_transport
            or self.cleanup != StorageCleanupConfig()
        ):
            raise ValueError("SQLite storage must not configure PostgreSQL settings")
        return self


class MCPTransportConfig(BaseModel):
    """Stateless Streamable HTTP MCP transport and DNS-rebinding policy."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, alias_generator=to_camel
    )

    enabled: bool = False
    allowed_hosts: tuple[str, ...] = ()
    allowed_origins: tuple[str, ...] = ()

    @field_validator("allowed_hosts", "allowed_origins")
    @classmethod
    def validate_header_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Accept bounded literal values and the SDK's explicit wildcard-port form."""
        if len(value) > 100:
            raise ValueError("MCP transport allowlists may contain at most 100 values")
        for item in value:
            if (
                not item
                or len(item) > 500
                or item != item.strip()
                or any(ord(character) < 32 or ord(character) == 127 for character in item)
            ):
                raise ValueError("MCP transport allowlist values must be bounded printable text")
        if len(value) != len(set(value)):
            raise ValueError("MCP transport allowlist values must be unique")
        return value

    @model_validator(mode="after")
    def validate_enabled_transport(self) -> Self:
        """Never enable a network MCP endpoint without an explicit Host allowlist."""
        if self.enabled and not self.allowed_hosts:
            raise ValueError("enabled MCP transport requires allowed_hosts")
        if not self.enabled and (self.allowed_hosts or self.allowed_origins):
            raise ValueError("disabled MCP transport must not configure allowlists")
        return self


class ServiceConfig(BaseModel):
    """Validated service configuration and its derived local registry."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, populate_by_name=True, alias_generator=to_camel
    )

    schema_version: Literal[1, 2]
    sync_interval: timedelta
    repositories: dict[str, RepositoryConfig]
    profiles: dict[str, ServiceProfileConfig]
    embedding: ServiceEmbeddingConfig
    storage: StorageConfig = StorageConfig()
    mcp: MCPTransportConfig = MCPTransportConfig()

    @field_validator("sync_interval", mode="before")
    @classmethod
    def parse_sync_interval(cls, value: Any) -> timedelta:
        """Parse compact service intervals and enforce their supported range."""
        return _parse_duration(value, name="sync_interval", maximum_seconds=24 * 60 * 60)

    @field_validator("repositories")
    @classmethod
    def validate_repository_ids(
        cls, value: dict[str, RepositoryConfig]
    ) -> dict[str, RepositoryConfig]:
        """Keep managed paths stable by accepting only vault identifier keys."""
        for vault_id in value:
            if _VAULT_ID_PATTERN.fullmatch(vault_id) is None:
                raise ValueError("repository names must be lowercase vault identifiers")
        return value

    @model_validator(mode="after")
    def validate_schema_version(self) -> Self:
        if self.schema_version == 1 and self.storage.backend is StorageBackend.POSTGRESQL:
            raise ValueError("PostgreSQL storage requires schema_version 2")
        return self

    @model_validator(mode="after")
    def validate_profiles(self) -> Self:
        """Keep public profiles explicit and resolve every referenced vault."""
        repository_ids = set(self.repositories)
        for profile_name, profile in self.profiles.items():
            if profile_name.startswith("_reconcile_"):
                raise ValueError("public profile names must not begin with _reconcile_")
            if not profile_name:
                raise ValueError("profile names must not be empty")
            if len(profile.vaults) != len(set(profile.vaults)):
                raise ValueError(f"profile {profile_name!r} contains duplicate vaults")
            unknown_vaults = set(profile.vaults) - repository_ids
            if unknown_vaults:
                raise ValueError(f"profile {profile_name!r} references unknown vaults")
        return self

    def registry(self, data_root: Path) -> RegistryConfig:
        """Derive stable local repository paths and private reconciliation profiles."""
        vaults = {
            vault_id: RegistryVault(path=data_root / "repos" / vault_id)
            for vault_id in self.repositories
        }
        profiles = {
            profile_name: ProfileConfig(vaults=profile.vaults)
            for profile_name, profile in self.profiles.items()
        }
        profiles.update(
            {
                f"_reconcile_{vault_id}": ProfileConfig(vaults=(vault_id,))
                for vault_id in self.repositories
            }
        )
        return RegistryConfig(
            vaults=vaults,
            profiles=profiles,
            embedding=EmbeddingConfig.model_validate(self.embedding.model_dump()),
        )


def _parse_duration(value: Any, *, name: str, maximum_seconds: int) -> timedelta:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a duration such as 5s")
    matched = _DURATION_PATTERN.fullmatch(value)
    if matched is None:
        raise ValueError(f"{name} must match ^([1-9][0-9]*)(s|m|h|d)$")
    seconds = int(matched.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86_400}[matched.group(2)]
    if seconds > maximum_seconds:
        raise ValueError(f"{name} must not exceed {maximum_seconds}s")
    return timedelta(seconds=seconds)


def load_service_config(path: Path) -> ServiceConfig:
    """Read, size-limit, and validate a service YAML document."""
    try:
        with path.open("rb") as config_file:
            raw = config_file.read(_MAX_CONFIG_BYTES + 1)
    except FileNotFoundError as exc:
        raise ConfigError(f"service configuration does not exist: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"could not load service configuration: {path}") from exc
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ConfigError("service configuration is too large")

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError("could not parse service configuration") from exc
    if not isinstance(data, dict):
        raise ConfigError("invalid service configuration: expected YAML mapping")
    return _validated_service_config(data)


def materialize_secret_files(
    config: ServiceConfig, environ: Mapping[str, str]
) -> Mapping[str, str]:
    """Copy referenced environment values, resolving optional ``_FILE`` secrets."""
    captured: dict[str, str] = {}
    for environment_name in _referenced_environment_names(config):
        if environment_name in environ:
            captured[environment_name] = environ[environment_name]
            continue
        file_name = f"{environment_name}_FILE"
        if file_name in environ:
            captured[environment_name] = _read_secret_file(environ[file_name], environment_name)
    return MappingProxyType(captured)


def _validated_service_config(data: Any) -> ServiceConfig:
    try:
        return ServiceConfig.model_validate(data)
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
        raise ConfigError(f"invalid service configuration: {summary}") from exc


def _referenced_environment_names(config: ServiceConfig) -> tuple[str, ...]:
    names = [
        repository.credential_env
        for repository in config.repositories.values()
        if repository.credential_env is not None
    ]
    if config.embedding.api_key_env is not None:
        names.append(config.embedding.api_key_env)
    names.append(config.embedding.model_env)
    if config.storage.database_url_env is not None:
        names.append(config.storage.database_url_env)
    return tuple(dict.fromkeys(names))


def _read_secret_file(raw_path: str, environment_name: str) -> str:
    try:
        with Path(raw_path).open("rb") as secret_file:
            value = secret_file.read(_MAX_SECRET_BYTES + 1)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"could not read secret file for {environment_name}") from exc
    if len(value) > _MAX_SECRET_BYTES:
        raise ConfigError(f"secret file for {environment_name} is too large")
    try:
        secret = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"could not read secret file for {environment_name}") from exc
    if secret.endswith("\r\n"):
        secret = secret[:-2]
    elif secret.endswith(("\n", "\r")):
        secret = secret[:-1]
    if not secret:
        raise ConfigError(f"secret file for {environment_name} is empty")
    return secret
