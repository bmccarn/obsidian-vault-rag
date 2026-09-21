"""Application exceptions and stable CLI exit codes."""

from collections.abc import Mapping
from typing import ClassVar

from .domain import JsonValue


class VaultRagError(Exception):
    """Base exception for expected vault-rag failures."""

    code: ClassVar[str] = "vault_rag_error"
    exit_code: ClassVar[int] = 1

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        self.message = message
        self.details = dict(details) if details is not None else {}
        super().__init__(message)


class ConfigError(VaultRagError):
    code = "config_error"
    exit_code = 2


class SecurityError(VaultRagError):
    code = "security_error"
    exit_code = 3


class ParseError(VaultRagError):
    code = "parse_error"
    exit_code = 2


class ChunkingError(VaultRagError):
    code = "chunking_error"
    exit_code = 2


class StorageError(VaultRagError):
    code = "storage_error"
    exit_code = 1


class SemanticUnavailableError(VaultRagError):
    code = "semantic_unavailable"
    exit_code = 4


class RebuildRequiredError(VaultRagError):
    code = "rebuild_required"
    exit_code = 2


class StaleSourceError(VaultRagError):
    code = "stale_source"
    exit_code = 3


class ServiceBusyError(VaultRagError):
    code = "service_busy"
    exit_code = 1


class RepositorySyncError(VaultRagError):
    """A bounded failure while synchronizing a managed Git repository."""

    code = "repository_sync_error"
