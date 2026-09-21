"""Deterministic, heading-aware source chunking."""

import hashlib
import json
import re
import unicodedata
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import tiktoken  # pyright: ignore[reportMissingImports]

from vault_rag.config.models import ChunkingConfig
from vault_rag.domain import JsonValue, LineRange
from vault_rag.errors import ChunkingError

from .lines import split_source_lines
from .models import ParsedSource, SourceSection

CHUNKER_SCHEMA_VERSION = 3
_FENCE_OPEN = re.compile(r"^[ \t]*([`~]{3,})")


class TokenCounter(Protocol):
    """Count tokens and return token-bounded suffixes without breaking text."""

    def count(self, text: str) -> int: ...

    def take_tail(self, text: str, tokens: int) -> str: ...


class TiktokenCounter:
    """A ``tiktoken``-backed counter for the configured embedding tokenizer."""

    def __init__(self, tokenizer: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(tokenizer)

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def take_tail(self, text: str, tokens: int) -> str:
        if tokens <= 0:
            return ""
        encoded = self._encoding.encode(text)
        if len(encoded) <= tokens:
            return text
        raw = self._encoding.decode_bytes(encoded[-tokens:])
        # A tokenizer token may start in the middle of a UTF-8 sequence.  Drop
        # only the incomplete leading bytes so the returned tail is valid text.
        for index in range(len(raw)):
            try:
                return raw[index:].decode("utf-8")
            except UnicodeDecodeError:
                continue
        return ""


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    id: str
    vault_id: str
    relative_path: str
    ordinal: int
    title: str
    heading: tuple[str, ...]
    lines: LineRange
    text: str
    embedding_text: str
    token_count: int
    metadata: Mapping[str, JsonValue]
    content_hash: str


@dataclass(frozen=True, slots=True)
class _Block:
    text: str
    start_offset: int
    lines: LineRange
    indivisible: bool = False


@dataclass(frozen=True, slots=True)
class _ChunkText:
    text: str
    start_offset: int


def _is_fence_open(line: str) -> tuple[str, int] | None:
    match = _FENCE_OPEN.match(line)
    if match is None:
        return None
    marker = match.group(1)
    return marker[0], len(marker)


def _is_fence_close(line: str, marker: str, minimum: int) -> bool:
    stripped = line.lstrip(" \t").rstrip("\r\n")
    marker_count = 0
    while marker_count < len(stripped) and stripped[marker_count] == marker:
        marker_count += 1
    return marker_count >= minimum and not stripped[marker_count:].strip()


def _structural_blocks(section: SourceSection) -> tuple[_Block, ...]:
    """Return blank-line blocks, keeping each fenced block indivisible."""
    lines = split_source_lines(section.text)
    offsets: list[int] = []
    offset = 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)

    blocks: list[_Block] = []
    index = 0
    ordinary_start = 0
    while index < len(lines):
        fence = _is_fence_open(lines[index])
        if fence is not None:
            if ordinary_start < index:
                start = offsets[ordinary_start]
                text = "".join(lines[ordinary_start:index])
                blocks.append(
                    _Block(
                        text=text,
                        start_offset=start,
                        lines=LineRange(
                            section.lines.start + ordinary_start, section.lines.start + index - 1
                        ),
                    )
                )
            fence_start = index
            marker, minimum = fence
            index += 1
            while index < len(lines):
                if _is_fence_close(lines[index], marker, minimum):
                    index += 1
                    break
                index += 1
            start = offsets[fence_start]
            text = "".join(lines[fence_start:index])
            blocks.append(
                _Block(
                    text=text,
                    start_offset=start,
                    lines=LineRange(
                        section.lines.start + fence_start,
                        section.lines.start + index - 1,
                    ),
                    indivisible=True,
                )
            )
            ordinary_start = index
            continue

        if not lines[index].strip():
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1
            start = offsets[ordinary_start]
            text = "".join(lines[ordinary_start:index])
            if text:
                blocks.append(
                    _Block(
                        text=text,
                        start_offset=start,
                        lines=LineRange(
                            section.lines.start + ordinary_start, section.lines.start + index - 1
                        ),
                    )
                )
            ordinary_start = index
            continue
        index += 1

    if ordinary_start < len(lines):
        start = offsets[ordinary_start]
        blocks.append(
            _Block(
                text="".join(lines[ordinary_start:]),
                start_offset=start,
                lines=LineRange(section.lines.start + ordinary_start, section.lines.end),
            )
        )
    return tuple(blocks)


def _metadata_value(value: JsonValue) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _prefix(source: ParsedSource, heading: tuple[str, ...]) -> str:
    lines = [
        f"Vault: {source.source.vault_id}",
        f"Title: {source.title}",
        f"Heading: {' > '.join(heading)}",
    ]
    lines.extend(f"{key}: {_metadata_value(value)}" for key, value in source.frontmatter.items())
    return "\n".join(lines) + "\n\n"


def _embedding_text(prefix: str, text: str) -> str:
    return prefix + text


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _chunk_id(source: ParsedSource, heading: tuple[str, ...], ordinal: int) -> str:
    identity: dict[str, object] = {
        "breadcrumb": list(heading),
        "chunker_schema_version": CHUNKER_SCHEMA_VERSION,
        "relative_path": unicodedata.normalize("NFC", source.source.relative_path),
        "structural_ordinal": ordinal,
        "vault_id": source.source.vault_id,
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return f"sha256:{digest}"


def _content_hash(embedding_text: str) -> str:
    canonical = unicodedata.normalize("NFC", embedding_text).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _line_starts(text: str) -> list[int]:
    starts = [0]
    offset = 0
    for line in split_source_lines(text):
        offset += len(line)
        starts.append(offset)
    return starts


def _lines_for_text(section: SourceSection, text: str, start_offset: int) -> LineRange:
    starts = _line_starts(section.text)
    start_line = section.lines.start + bisect_right(starts, start_offset) - 1
    last_character = start_offset + len(text) - 1
    end_line = section.lines.start + bisect_right(starts, last_character) - 1
    return LineRange(start_line, end_line)


def _bounded_tail(counter: TokenCounter, text: str, tokens: int) -> str:
    """Return an exact source suffix no larger than the requested token count."""
    if tokens <= 0:
        return ""
    tail = counter.take_tail(text, tokens)
    if tail and text.endswith(tail) and counter.count(tail) <= tokens:
        return tail
    return ""


def _largest_bounded_prefix(
    prefix: str,
    text: str,
    token_limit: int,
    counter: TokenCounter,
) -> str:
    low = 1
    high = len(text)
    best = 0
    while low <= high:
        middle = (low + high) // 2
        if counter.count(_embedding_text(prefix, text[:middle])) <= token_limit:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return text[:best]


def _split_ordinary_block(
    block: _Block,
    section: SourceSection,
    prefix: str,
    config: ChunkingConfig,
    counter: TokenCounter,
) -> tuple[_Block, ...]:
    if counter.count(_embedding_text(prefix, block.text)) < config.max_input_tokens:
        return (block,)

    prefix_tokens = counter.count(prefix)
    token_limit = (
        config.target_max_tokens
        if prefix_tokens < config.target_max_tokens
        else config.max_input_tokens - 1
    )
    pieces: list[_Block] = []
    current = ""
    current_offset = block.start_offset
    consumed = 0

    def append_piece(text: str, start_offset: int) -> None:
        pieces.append(
            _Block(
                text=text,
                start_offset=start_offset,
                lines=_lines_for_text(section, text, start_offset),
            )
        )

    for source_line in split_source_lines(block.text):
        if current and counter.count(_embedding_text(prefix, current + source_line)) <= token_limit:
            current += source_line
            consumed += len(source_line)
            continue
        if current:
            append_piece(current, current_offset)
            current = ""
            current_offset = block.start_offset + consumed
        remaining = source_line
        while remaining and counter.count(_embedding_text(prefix, remaining)) > token_limit:
            bounded = _largest_bounded_prefix(prefix, remaining, token_limit, counter)
            if not bounded:
                raise ChunkingError("metadata prefix leaves no room for source text")
            append_piece(bounded, current_offset)
            current_offset += len(bounded)
            consumed += len(bounded)
            remaining = remaining[len(bounded) :]
        current = remaining
        consumed += len(remaining)
    if current:
        append_piece(current, current_offset)
    return tuple(pieces)


def _split_section(
    source: ParsedSource,
    section: SourceSection,
    config: ChunkingConfig,
    counter: TokenCounter,
) -> tuple[_ChunkText, ...]:
    prefix = _prefix(source, section.heading)
    prefix_tokens = counter.count(prefix)
    if prefix_tokens >= config.max_input_tokens:
        raise ChunkingError(
            f"metadata prefix exceeds input limit in {source.source.relative_path} lines "
            f"{section.lines.start}-{section.lines.end}",
            details={
                "path": source.source.relative_path,
                "start_line": section.lines.start,
                "end_line": section.lines.end,
            },
        )

    blocks: list[_Block] = []
    for block in _structural_blocks(section):
        if block.indivisible:
            if counter.count(_embedding_text(prefix, block.text)) >= config.max_input_tokens:
                raise ChunkingError(
                    f"indivisible block exceeds input limit in {source.source.relative_path} lines "
                    f"{block.lines.start}-{block.lines.end}",
                    details={
                        "path": source.source.relative_path,
                        "start_line": block.lines.start,
                        "end_line": block.lines.end,
                    },
                )
            blocks.append(block)
        else:
            blocks.extend(_split_ordinary_block(block, section, prefix, config, counter))

    chunks: list[_ChunkText] = []
    current: _ChunkText | None = None
    for block in blocks:
        if current is None:
            current = _ChunkText(block.text, block.start_offset)
            continue

        combined = current.text + block.text
        if counter.count(_embedding_text(prefix, combined)) <= config.target_max_tokens:
            current = _ChunkText(combined, current.start_offset)
            continue

        chunks.append(current)
        block_tokens = counter.count(_embedding_text(prefix, block.text))
        tail = ""
        if block_tokens <= config.target_max_tokens:
            for requested in range(config.overlap_tokens, 0, -1):
                candidate = _bounded_tail(counter, current.text, requested)
                candidate_tokens = counter.count(_embedding_text(prefix, candidate + block.text))
                if candidate and candidate_tokens <= config.target_max_tokens:
                    tail = candidate
                    break
        current = _ChunkText(
            text=tail + block.text,
            start_offset=current.start_offset + len(current.text) - len(tail),
        )

    if current is not None:
        chunks.append(current)
    return tuple(chunks)


def chunk_source(
    source: ParsedSource,
    config: ChunkingConfig,
    counter: TokenCounter,
) -> tuple[ChunkRecord, ...]:
    """Chunk parsed sections without crossing headings or model input limits."""
    records: list[ChunkRecord] = []
    for section in source.sections:
        prefix = _prefix(source, section.heading)
        for chunk_text in _split_section(source, section, config, counter):
            embedding_text = _embedding_text(prefix, chunk_text.text)
            token_count = counter.count(embedding_text)
            if token_count >= config.max_input_tokens:
                raise ChunkingError(
                    f"chunk exceeds input limit in {source.source.relative_path} lines "
                    f"{section.lines.start}-{section.lines.end}",
                    details={
                        "path": source.source.relative_path,
                        "start_line": section.lines.start,
                        "end_line": section.lines.end,
                    },
                )
            ordinal = len(records)
            records.append(
                ChunkRecord(
                    id=_chunk_id(source, section.heading, ordinal),
                    vault_id=source.source.vault_id,
                    relative_path=source.source.relative_path,
                    ordinal=ordinal,
                    title=source.title,
                    heading=section.heading,
                    lines=_lines_for_text(section, chunk_text.text, chunk_text.start_offset),
                    text=chunk_text.text,
                    embedding_text=embedding_text,
                    token_count=token_count,
                    metadata=dict(source.frontmatter),
                    content_hash=_content_hash(embedding_text),
                )
            )
    return tuple(records)
