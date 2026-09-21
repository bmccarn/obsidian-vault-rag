"""Durable, bounded per-vault synchronization state."""

from __future__ import annotations

import json
import os
import re
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vault_rag.indexing import IndexReport

_MAX_STATE_BYTES = 1024 * 1024
_MAX_DIAGNOSTICS = 100
_MAX_MESSAGE_LENGTH = 500
_MAX_REF_LENGTH = 1024
_MAX_VAULT_ID_LENGTH = 63
_SHA_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_VAULT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class IndexDiagnosticState(BaseModel):
    """A bounded, JSON-safe diagnostic retained from one index run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vault_id: str = Field(min_length=1, max_length=_MAX_VAULT_ID_LENGTH)
    path: str = Field(min_length=1, max_length=_MAX_REF_LENGTH)
    category: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=_MAX_MESSAGE_LENGTH)


class IndexReportState(BaseModel):
    """The aggregate and bounded portion of an immutable :class:`IndexReport`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_sources: int = Field(ge=0)
    added_sources: int = Field(ge=0)
    changed_sources: int = Field(ge=0)
    unchanged_sources: int = Field(ge=0)
    deleted_sources: int = Field(ge=0)
    ready_chunks: int = Field(ge=0)
    pending_chunks: int = Field(ge=0)
    parse_failures: int = Field(ge=0)
    embedding_failures: int = Field(ge=0)
    embedding_requests: int = Field(ge=0)
    blocking_failures: int = Field(default=0, ge=0)
    diagnostics: tuple[IndexDiagnosticState, ...] = Field(max_length=_MAX_DIAGNOSTICS)
    elapsed_ms: int = Field(ge=0)

    @classmethod
    def from_report(cls, report: IndexReport) -> Self:
        """Copy report aggregates while retaining only bounded diagnostic data."""
        return cls(
            total_sources=report.total_sources,
            added_sources=report.added_sources,
            changed_sources=report.changed_sources,
            unchanged_sources=report.unchanged_sources,
            deleted_sources=report.deleted_sources,
            ready_chunks=report.ready_chunks,
            pending_chunks=report.pending_chunks,
            parse_failures=report.parse_failures,
            embedding_failures=report.embedding_failures,
            embedding_requests=report.embedding_requests,
            blocking_failures=report.blocking_failures,
            diagnostics=tuple(
                IndexDiagnosticState(
                    vault_id=diagnostic.vault_id,
                    path=diagnostic.path,
                    category=diagnostic.category,
                    message=diagnostic.message[:_MAX_MESSAGE_LENGTH],
                )
                for diagnostic in report.diagnostics[:_MAX_DIAGNOSTICS]
            ),
            elapsed_ms=report.elapsed_ms,
        )


class VaultSyncState(BaseModel):
    """Validated progress and outcome state for one managed vault."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    vault_id: str = Field(min_length=1, max_length=_MAX_VAULT_ID_LENGTH)
    configured_ref: str = Field(min_length=1, max_length=_MAX_REF_LENGTH)
    configured_sha: str | None = None
    fetched_sha: str | None = None
    checkout_sha: str | None = None
    attempted_sha: str | None = None
    reconciled_sha: str | None = None
    configured_at: datetime | None = None
    fetched_at: datetime | None = None
    checkout_at: datetime | None = None
    attempted_at: datetime | None = None
    reconciled_at: datetime | None = None
    report: IndexReportState | None = None
    sync_degraded_reason: str | None = Field(default=None, max_length=_MAX_MESSAGE_LENGTH)
    sync_degraded_at: datetime | None = None

    @field_validator("vault_id")
    @classmethod
    def validate_vault_id(cls, value: str) -> str:
        """Keep the state filename within the configured state directory."""
        if _VAULT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("vault_id must be a safe vault identifier")
        return value

    @field_validator(
        "configured_sha", "fetched_sha", "checkout_sha", "attempted_sha", "reconciled_sha"
    )
    @classmethod
    def validate_sha(cls, value: str | None) -> str | None:
        """Accept only full lowercase SHA-1 or SHA-256 object identifiers."""
        if value is not None and _SHA_PATTERN.fullmatch(value) is None:
            raise ValueError("SHA must be 40 or 64 lowercase hexadecimal characters")
        return value

    @field_validator(
        "configured_at",
        "fetched_at",
        "checkout_at",
        "attempted_at",
        "reconciled_at",
        "sync_degraded_at",
    )
    @classmethod
    def normalize_utc_datetime(cls, value: datetime | None) -> datetime | None:
        """Reject naive time values and normalize aware values to UTC."""
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must include a UTC offset")
        return value.astimezone(UTC)


class StateRead(BaseModel):
    """A synchronization-state snapshot with an optional safe read warning."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: VaultSyncState
    warning: str | None = Field(default=None, max_length=_MAX_MESSAGE_LENGTH)


@runtime_checkable
class SyncStateStorePort(Protocol):
    """Persistence boundary shared by local files and PostgreSQL."""

    @property
    def checkout_independent(self) -> bool: ...

    def read(self, vault_id: str, configured_ref: str) -> StateRead: ...

    def write(self, state: VaultSyncState) -> None: ...


class SyncStateStore:
    """Read and atomically replace per-vault state without persisting secrets."""

    checkout_independent = False

    def __init__(self, state_directory: Path) -> None:
        self._state_directory = state_directory

    def read(self, vault_id: str, configured_ref: str) -> StateRead:
        """Return persisted state or a safe initial state when it cannot be trusted."""
        initial = VaultSyncState(vault_id=vault_id, configured_ref=configured_ref)
        path = self._path_for(vault_id)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return StateRead(state=initial)
        except OSError:
            return self._invalid(initial)

        if len(raw) > _MAX_STATE_BYTES:
            return self._invalid(initial)

        try:
            decoded: Any = json.loads(raw)
            stored = VaultSyncState.model_validate(decoded)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return self._invalid(initial)

        if stored.vault_id != vault_id or stored.configured_ref != configured_ref:
            return self._invalid(initial)
        return StateRead(state=stored)

    def write(self, state: VaultSyncState) -> None:
        """Durably replace a vault state file using a same-directory temporary file."""
        self._ensure_directory()
        destination = self._path_for(state.vault_id)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=self._state_directory, prefix=f".{state.vault_id}.json.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            payload = json.dumps(
                state.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            self._fsync_directory()
        except Exception:
            if descriptor != -1:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                temporary.unlink()
            raise

    def _path_for(self, vault_id: str) -> Path:
        validated = VaultSyncState(vault_id=vault_id, configured_ref="refs/heads/state")
        return self._state_directory / f"{validated.vault_id}.json"

    def _ensure_directory(self) -> None:
        self._state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._state_directory, 0o700)

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        try:
            descriptor = os.open(self._state_directory, flags)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    def _invalid(initial: VaultSyncState) -> StateRead:
        return StateRead(
            state=initial.model_copy(update={"sync_degraded_reason": "state_invalid"}),
            warning="stored synchronization state is invalid",
        )
