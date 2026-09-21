"""Shared domain types for vault-rag."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TypeAlias
from urllib.parse import quote


class SourceKind(StrEnum):
    MARKDOWN = "markdown"
    TEXT = "text"
    LOG = "log"
    JSON = "json"


class EmbeddingState(StrEnum):
    READY = "ready"
    PENDING = "pending"
    DISABLED = "disabled"


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]  # noqa: RUF036, UP040


@dataclass(frozen=True, slots=True)
class DegradedState:
    semantic_search: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class LineRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1:
            raise ValueError("line range start must be at least 1")
        if self.end < 1:
            raise ValueError("line range end must be at least 1")
        if self.end < self.start:
            raise ValueError("line range end must not precede start")


@dataclass(frozen=True, slots=True)
class SourceRef:
    vault_id: str
    path: str
    heading: tuple[str, ...]
    lines: LineRange
    source_hash: str

    def __post_init__(self) -> None:
        if not self.vault_id:
            raise ValueError("vault id must not be empty")
        if not self.path or self.path.startswith("/") or "\\" in self.path:
            raise ValueError("source path must be a non-empty POSIX-relative path")
        path = PurePosixPath(self.path)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("source path must be a non-empty POSIX-relative path")

    @property
    def citation(self) -> str:
        quoted = quote(self.path, safe="/")
        return f"vault://{self.vault_id}/{quoted}#L{self.lines.start}-L{self.lines.end}"
