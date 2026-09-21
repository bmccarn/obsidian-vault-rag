import hashlib
import re
from dataclasses import replace
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.config.models import ChunkingConfig  # type: ignore[import-untyped]
from vault_rag.domain import LineRange, SourceKind  # type: ignore[import-untyped]
from vault_rag.errors import ChunkingError  # type: ignore[import-untyped]
from vault_rag.ingest.chunker import (  # type: ignore[import-untyped]
    ChunkRecord,
    TiktokenCounter,
    chunk_source,
)
from vault_rag.ingest.markdown import parse_markdown  # type: ignore[import-untyped]
from vault_rag.ingest.models import DiscoveredSource, ParsedSource  # type: ignore[import-untyped]


class WhitespaceCounter:
    """Deterministic test counter that retains a source-text token tail."""

    def count(self, text: str) -> int:
        return len(re.findall(r"\S+", text))

    def take_tail(self, text: str, tokens: int) -> str:
        if tokens <= 0:
            return ""
        matches = list(re.finditer(r"\S+", text))
        if len(matches) <= tokens:
            return text
        return text[matches[-tokens].start() :]


def config() -> ChunkingConfig:
    return ChunkingConfig(
        target_min_tokens=10,
        target_max_tokens=40,
        overlap_tokens=3,
        max_input_tokens=100,
    )


def tiny_config(max_tokens: int) -> ChunkingConfig:
    return ChunkingConfig(
        target_min_tokens=4,
        target_max_tokens=max_tokens,
        overlap_tokens=3,
        max_input_tokens=100,
    )


def discovered_source(
    text: str, *, path: str = "example.md", vault_id: str = "example-vault"
) -> DiscoveredSource:
    raw = text.encode("utf-8")
    return DiscoveredSource(
        vault_id=vault_id,
        root=Path("/vault"),
        relative_path=path,
        folded_path=path.casefold(),
        kind=SourceKind.MARKDOWN,
        text=text,
        content_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        size_bytes=len(raw),
        mtime_ns=0,
    )


@pytest.fixture
def parsed_source() -> ParsedSource:
    text = (
        Path(__file__).parents[2] / "fixtures" / "vault-a" / "projects" / "example.md"
    ).read_text(encoding="utf-8")
    parsed = parse_markdown(discovered_source(text))
    return replace(parsed, title="Example", frontmatter={"status": "active"})


def long_paragraph_source() -> ParsedSource:
    paragraph = "one two three four five six seven eight nine ten\n\n"
    return parse_markdown(discovered_source(f"# Heading\n{paragraph}{paragraph}{paragraph}"))


def overlap_size(left: ChunkRecord, right: ChunkRecord) -> int:
    left_words = re.findall(r"\S+", left.text)
    right_words = re.findall(r"\S+", right.text)
    maximum = min(len(left_words), len(right_words))
    for size in range(maximum, 0, -1):
        if left_words[-size:] == right_words[:size]:
            return size
    return 0


def test_small_heading_section_is_one_chunk(parsed_source: ParsedSource) -> None:
    state_only = replace(parsed_source, sections=(parsed_source.sections[-1],))

    chunks = chunk_source(state_only, tiny_config(max_tokens=40), WhitespaceCounter())

    assert len(chunks) == 1
    assert chunks[0].heading == ("Project", "State")
    assert chunks[0].lines == LineRange(17, 24)
    assert chunks[0].embedding_text.startswith(
        "Vault: example-vault\nTitle: Example\nHeading: Project > State\nstatus: active\n\n"
    )


def test_oversized_section_splits_on_blocks_with_bounded_overlap() -> None:
    chunks = chunk_source(long_paragraph_source(), tiny_config(max_tokens=20), WhitespaceCounter())

    assert len(chunks) == 3
    assert all(chunk.token_count <= 20 for chunk in chunks)
    assert overlap_size(chunks[0], chunks[1]) <= 3
    assert chunks[0].lines.end >= chunks[1].lines.start


def test_chunk_id_is_stable_and_path_sensitive(parsed_source: ParsedSource) -> None:
    source = replace(parsed_source, sections=(parsed_source.sections[-1],))
    first = chunk_source(source, config(), WhitespaceCounter())[0]
    again = chunk_source(source, config(), WhitespaceCounter())[0]
    moved = chunk_source(
        replace(source, source=replace(source.source, relative_path="moved.md")),
        config(),
        WhitespaceCounter(),
    )[0]

    assert first.id == again.id
    assert first.id != moved.id


def test_sections_are_never_merged_across_headings(parsed_source: ParsedSource) -> None:
    chunks = chunk_source(parsed_source, config(), WhitespaceCounter())

    assert [chunk.heading for chunk in chunks] == [(), ("Project",), ("Project", "State")]
    assert [chunk.lines for chunk in chunks] == [
        LineRange(7, 8),
        LineRange(9, 16),
        LineRange(17, 24),
    ]


def test_code_fence_remains_whole() -> None:
    source = parse_markdown(
        discovered_source(
            "# Code\nBefore fence.\n```python\nfirst = 1\n\nsecond = 2\n```\nAfter fence.\n"
        )
    )

    chunks = chunk_source(source, tiny_config(max_tokens=11), WhitespaceCounter())

    fenced = next(chunk for chunk in chunks if "```python" in chunk.text)
    assert "```python\nfirst = 1\n\nsecond = 2\n```" in fenced.text


def test_over_limit_unfenced_block_splits_on_source_lines() -> None:
    rows = "".join(f"row {index} alpha beta\n" for index in range(12))
    source = parse_markdown(discovered_source(f"# Table\n{rows}"))
    limited = ChunkingConfig(
        target_min_tokens=1,
        target_max_tokens=10,
        overlap_tokens=0,
        max_input_tokens=13,
    )

    chunks = chunk_source(source, limited, WhitespaceCounter())

    assert len(chunks) > 1
    assert all(chunk.token_count <= limited.target_max_tokens for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == source.sections[0].text
    assert chunks[0].lines.start == 1
    assert chunks[-1].lines.end == 13


def test_over_limit_indivisible_block_has_source_lines() -> None:
    source = parse_markdown(discovered_source("# Code\n```text\none two three four five\n```\n"))
    limited = ChunkingConfig(
        target_min_tokens=1,
        target_max_tokens=5,
        overlap_tokens=0,
        max_input_tokens=13,
    )

    with pytest.raises(ChunkingError, match=r"example\.md.*lines 2-4") as error:
        chunk_source(source, limited, WhitespaceCounter())

    assert error.value.details == {"path": "example.md", "start_line": 2, "end_line": 4}


def test_overlap_is_only_used_for_a_split_section() -> None:
    source = parse_markdown(discovered_source("# One\none two\n\n# Two\nthree four\n"))

    chunks = chunk_source(source, tiny_config(max_tokens=20), WhitespaceCounter())

    assert len(chunks) == 2
    assert overlap_size(chunks[0], chunks[1]) == 0


def test_frontmatter_order_is_preserved_in_embedding_text(parsed_source: ParsedSource) -> None:
    source = replace(
        parsed_source,
        sections=(parsed_source.sections[-1],),
        frontmatter={"type": "project", "status": "active"},
    )

    chunk = chunk_source(source, config(), WhitespaceCounter())[0]

    assert "type: project\nstatus: active\n\n" in chunk.embedding_text


def test_tiktoken_tail_is_valid_utf8_and_token_bounded() -> None:
    counter = TiktokenCounter()
    tail = counter.take_tail("alpha 😃 café", 2)

    assert tail.encode("utf-8").decode("utf-8") == tail
    assert counter.count(tail) <= 2


def test_subchunk_line_ranges_cover_their_source_text() -> None:
    source = long_paragraph_source()
    section = source.sections[0]

    chunks = chunk_source(source, tiny_config(max_tokens=20), WhitespaceCounter())
    source_lines = source.source.text.splitlines(keepends=True)
    for chunk in chunks:
        covered = "".join(source_lines[chunk.lines.start - 1 : chunk.lines.end])
        assert chunk.text in covered
        assert chunk.lines.start >= section.lines.start
        assert chunk.lines.end <= section.lines.end


def test_chunk_line_ranges_ignore_non_commonmark_separators() -> None:
    text = "# Heading\n\nalpha\u2028more alpha\n\nbeta line\n"
    parsed = parse_markdown(discovered_source(text))

    chunks = chunk_source(parsed, tiny_config(max_tokens=6), WhitespaceCounter())

    assert [chunk.text for chunk in chunks] == [
        "# Heading\n\n",
        "alpha\u2028more alpha\n\n",
        "beta line\n",
    ]
    assert [chunk.lines for chunk in chunks] == [
        LineRange(1, 2),
        LineRange(3, 4),
        LineRange(5, 5),
    ]
