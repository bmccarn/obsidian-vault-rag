"""Versioned exact-identifier recognition and safe FTS query construction."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath

IDENTIFIER_SCHEMA_VERSION = 1
_SUPPORTED_PATH_SUFFIXES = (".md", ".txt", ".log", ".json")
_JIRA = re.compile(r"\b[A-Z][A-Z0-9]+-[0-9]+\b")
_PR = re.compile(r"\b([a-z0-9][a-z0-9-]+)\s+#([0-9]+)\b", re.IGNORECASE)
_QUALIFIED_GIT = re.compile(r"\b(?:sha|commit):([0-9a-fA-F]{7,40})\b", re.IGNORECASE)
_UNQUALIFIED_GIT = re.compile(r"\b[0-9a-fA-F]{7,40}\b")
_PATH = re.compile(
    r"(?<![.\w/%+@~-])"
    r"("
    r"[\w%+@~-][\w.%+@~-]*(?:/[\w%+@~-][\w.%+@~-]*)*/"
    r"[\w%+@~-][\w.%+@~-]*?"
    r"(?:\.md|\.txt|\.log|\.json)"
    r")"
    r"(?![\w%+@~/-]|\.[\w])",
    re.IGNORECASE | re.UNICODE,
)
_LEXICAL_TOKEN = re.compile(r"[\w]+(?:[-./:][\w]+)*", re.UNICODE)


@dataclass(frozen=True, slots=True)
class IdentifierMatch:
    """One complete, normalized identifier recognized in query text."""

    kind: str
    value: str

    def __post_init__(self) -> None:
        if self.kind not in {"jira", "pr", "git", "path"}:
            raise ValueError("unsupported identifier kind")
        if not self.value:
            raise ValueError("identifier value must not be empty")


def _valid_path(value: str) -> bool:
    normalized = unicodedata.normalize("NFC", value)
    path = PurePosixPath(normalized)
    return (
        "/" in normalized
        and not normalized.startswith("/")
        and normalized.casefold().endswith(_SUPPORTED_PATH_SUFFIXES)
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def recognize_identifiers(query: str) -> tuple[IdentifierMatch, ...]:
    """Recognize complete identifiers in source order using schema version 1."""
    if not isinstance(query, str):
        raise TypeError("query must be text")
    normalized_query = unicodedata.normalize("NFC", query)
    matches: list[tuple[int, int, IdentifierMatch]] = []
    qualified_spans: list[tuple[int, int]] = []

    for match in _JIRA.finditer(normalized_query):
        matches.append((match.start(), match.end(), IdentifierMatch("jira", match.group(0))))
    for match in _PR.finditer(normalized_query):
        value = f"{match.group(1).casefold()}#{match.group(2)}"
        matches.append((match.start(), match.end(), IdentifierMatch("pr", value)))
    for match in _QUALIFIED_GIT.finditer(normalized_query):
        qualified_spans.append(match.span())
        matches.append(
            (match.start(), match.end(), IdentifierMatch("git", match.group(1).casefold()))
        )
    for match in _UNQUALIFIED_GIT.finditer(normalized_query):
        if any(start <= match.start() and match.end() <= end for start, end in qualified_spans):
            continue
        value = match.group(0)
        if any(character.casefold() in "abcdef" for character in value):
            matches.append((match.start(), match.end(), IdentifierMatch("git", value.casefold())))
    for match in _PATH.finditer(normalized_query):
        value = unicodedata.normalize("NFC", match.group(1))
        if _valid_path(value):
            matches.append((match.start(), match.end(), IdentifierMatch("path", value)))

    ordered: list[IdentifierMatch] = []
    seen: set[IdentifierMatch] = set()
    for _start, _end, identifier in sorted(matches, key=lambda item: (item[0], item[1])):
        if identifier not in seen:
            seen.add(identifier)
            ordered.append(identifier)
    return tuple(ordered)


def build_fts_query(query: str) -> str:
    """Compile arbitrary text to a disjunction of quoted FTS5 tokens."""
    if not isinstance(query, str):
        raise TypeError("query must be text")
    tokens = (match.group(0).replace('"', '""') for match in _LEXICAL_TOKEN.finditer(query))
    return " OR ".join(f'"{token}"' for token in tokens)
