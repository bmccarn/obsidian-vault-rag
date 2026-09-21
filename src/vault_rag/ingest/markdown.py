"""Structural Markdown parsing with exact source-line provenance."""

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import cast

from markdown_it import MarkdownIt  # pyright: ignore[reportMissingImports]
from markdown_it.token import Token  # pyright: ignore[reportMissingImports]
from yaml import safe_load
from yaml.error import MarkedYAMLError, YAMLError

from vault_rag.domain import JsonValue, LineRange
from vault_rag.errors import ParseError

from .lines import split_source_lines
from .models import DiscoveredSource, ParsedSource, SourceSection

_WIKILINK = re.compile(r"!?\[\[([^\]\n]+)\]\]")
_TAG = re.compile(r"(?<![\w#])#([A-Za-z0-9][A-Za-z0-9_/-]*)")


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_value(item) for item in cast(list[object], value)]
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("frontmatter keys must be strings")
            result[key] = _json_value(item)
        return result
    raise ValueError(f"frontmatter value has unsupported type: {type(value).__name__}")


def _parse_frontmatter(
    source: DiscoveredSource,
    lines: Sequence[str],
) -> tuple[Mapping[str, JsonValue], int]:
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}, 0

    closing_index = next(
        (index for index, line in enumerate(lines[1:], start=1) if line.rstrip("\r\n") == "---"),
        None,
    )
    if closing_index is None:
        raise ParseError(
            f"unclosed frontmatter in {source.relative_path} at line 1",
            details={"path": source.relative_path, "line": 1},
        )

    yaml_text = "".join(lines[1:closing_index])
    try:
        loaded: object = safe_load(yaml_text)
    except MarkedYAMLError as exc:
        marked_line = 2 if exc.problem_mark is None else exc.problem_mark.line + 2
        line = min(max(marked_line, 2), max(2, closing_index))
        raise ParseError(
            f"malformed frontmatter in {source.relative_path} at line {line}",
            details={"path": source.relative_path, "line": line},
        ) from exc
    except YAMLError as exc:
        raise ParseError(
            f"malformed frontmatter in {source.relative_path} at line 1",
            details={"path": source.relative_path, "line": 1},
        ) from exc

    if loaded is None:
        return {}, closing_index + 1
    try:
        converted = _json_value(loaded)
    except ValueError as exc:
        raise ParseError(
            f"invalid frontmatter in {source.relative_path} at line 2",
            details={"path": source.relative_path, "line": 2},
        ) from exc
    if not isinstance(converted, dict):
        raise ParseError(
            f"frontmatter must be a mapping in {source.relative_path} at line 2",
            details={"path": source.relative_path, "line": 2},
        )
    return converted, closing_index + 1


def _plain_heading(inline: Token) -> str:
    if inline.children is None:
        return inline.content.strip()
    visible_types = {"text", "code_inline", "html_inline"}
    text = "".join(child.content for child in inline.children if child.type in visible_types)
    return text.strip()


def _headings(tokens: Sequence[Token]) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        level = int(token.tag[1:])
        inline = tokens[index + 1]
        headings.append((token.map[0] + 1, level, _plain_heading(inline)))
    return headings


def _append_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value and value not in target:
            target.append(value)


def _metadata_from_tokens(
    tokens: Sequence[Token],
    line_range: LineRange,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    wikilinks: list[str] = []
    aliases: list[str] = []
    tags: list[str] = []
    for token in tokens:
        if token.type != "inline" or token.map is None or token.children is None:
            continue
        token_start = token.map[0] + 1
        if not line_range.start <= token_start <= line_range.end:
            continue
        for child in token.children:
            if child.type != "text":
                continue
            for match in _WIKILINK.finditer(child.content):
                target, separator, alias = match.group(1).partition("|")
                _append_unique(wikilinks, (target.strip(),))
                if separator:
                    _append_unique(aliases, (alias.strip(),))
            _append_unique(tags, (match.group(1) for match in _TAG.finditer(child.content)))
    return tuple(wikilinks), tuple(aliases), tuple(tags)


def _merge_metadata(
    inherited: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    local: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    merged: list[tuple[str, ...]] = []
    for inherited_values, local_values in zip(inherited, local, strict=True):
        values: list[str] = []
        _append_unique(values, inherited_values)
        _append_unique(values, local_values)
        merged.append(tuple(values))
    return merged[0], merged[1], merged[2]


def parse_markdown(source: DiscoveredSource) -> ParsedSource:
    """Parse frontmatter and real Markdown headings while retaining exact lines."""
    source_lines = split_source_lines(source.text)
    frontmatter, content_start_index = _parse_frontmatter(source, source_lines)

    parser_text = "".join(
        "\n" if index < content_start_index else line for index, line in enumerate(source_lines)
    )
    tokens = MarkdownIt("commonmark").parse(parser_text)
    headings = _headings(tokens)
    total_lines = len(source_lines)

    section_specs: list[tuple[tuple[str, ...], LineRange]] = []
    first_heading_line = headings[0][0] if headings else total_lines + 1
    preamble_start = content_start_index + 1
    preamble_end = first_heading_line - 1
    while preamble_start <= preamble_end and not source_lines[preamble_start - 1].strip():
        preamble_start += 1
    if preamble_start <= preamble_end:
        section_specs.append(((), LineRange(preamble_start, preamble_end)))

    breadcrumb_stack: list[tuple[int, str]] = []
    for index, (line, level, heading_text) in enumerate(headings):
        while breadcrumb_stack and breadcrumb_stack[-1][0] >= level:
            breadcrumb_stack.pop()
        breadcrumb_stack.append((level, heading_text))
        end = headings[index + 1][0] - 1 if index + 1 < len(headings) else total_lines
        section_specs.append((tuple(item[1] for item in breadcrumb_stack), LineRange(line, end)))

    sections: list[SourceSection] = []
    metadata_by_heading: dict[
        tuple[str, ...], tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
    ] = {}
    for heading_path, line_range in section_specs:
        local_metadata = _metadata_from_tokens(tokens, line_range)
        inherited_metadata: tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]] = (
            (),
            (),
            (),
        )
        if len(heading_path) > 1:
            inherited_metadata = metadata_by_heading.get(heading_path[:-1], inherited_metadata)
        metadata = _merge_metadata(inherited_metadata, local_metadata)
        if heading_path:
            metadata_by_heading[heading_path] = metadata
        sections.append(
            SourceSection(
                heading=heading_path,
                lines=line_range,
                text="".join(source_lines[line_range.start - 1 : line_range.end]),
                wikilinks=metadata[0],
                aliases=metadata[1],
                tags=metadata[2],
            )
        )

    title = next(
        (heading for _line, level, heading in headings if level == 1),
        PurePosixPath(source.relative_path).stem,
    )
    return ParsedSource(
        source=source,
        title=title,
        frontmatter=frontmatter,
        sections=tuple(sections),
    )
