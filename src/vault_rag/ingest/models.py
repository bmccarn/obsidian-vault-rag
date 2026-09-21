"""Immutable contracts for discovered and structurally parsed sources."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from vault_rag.domain import JsonValue, LineRange, SourceKind


@dataclass(frozen=True, slots=True)
class DiscoveredSource:
    vault_id: str
    root: Path
    relative_path: str
    folded_path: str
    kind: SourceKind
    text: str
    content_hash: str
    size_bytes: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class SourceSection:
    heading: tuple[str, ...]
    lines: LineRange
    text: str
    wikilinks: tuple[str, ...]
    aliases: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedSource:
    source: DiscoveredSource
    title: str
    frontmatter: Mapping[str, JsonValue]
    sections: tuple[SourceSection, ...]


@dataclass(frozen=True, slots=True, order=True)
class DiscoveryFailure:
    """One manifest-selected file that could not become a source."""

    relative_path: str
    category: str
    message: str


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Sources that reconcile and the bounded failures that were skipped."""

    sources: tuple[DiscoveredSource, ...]
    failures: tuple[DiscoveryFailure, ...]
