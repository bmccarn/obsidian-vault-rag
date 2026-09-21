import hashlib
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.domain import LineRange, SourceKind  # type: ignore[import-untyped]
from vault_rag.errors import ParseError  # type: ignore[import-untyped]
from vault_rag.ingest.lines import split_source_lines  # type: ignore[import-untyped]
from vault_rag.ingest.markdown import parse_markdown  # type: ignore[import-untyped]
from vault_rag.ingest.models import DiscoveredSource  # type: ignore[import-untyped]

FIXTURE_PATH = Path(__file__).parents[2] / "fixtures" / "vault-a" / "projects" / "example.md"


def discovered_markdown(path: Path, text: str) -> DiscoveredSource:
    raw = text.encode("utf-8")
    return DiscoveredSource(
        vault_id="vault-a",
        root=path.parent,
        relative_path=path.name,
        folded_path=path.name.casefold(),
        kind=SourceKind.MARKDOWN,
        text=text,
        content_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        size_bytes=len(raw),
        mtime_ns=0,
    )


@pytest.fixture
def markdown_source() -> DiscoveredSource:
    text = FIXTURE_PATH.read_text(encoding="utf-8")
    return discovered_markdown(FIXTURE_PATH, text)


def test_markdown_sections_preserve_breadcrumbs_and_lines(
    markdown_source: DiscoveredSource,
) -> None:
    parsed = parse_markdown(markdown_source)
    assert parsed.title == "Project"
    assert parsed.frontmatter["status"] == "active"
    assert [(section.heading, section.lines) for section in parsed.sections] == [
        ((), LineRange(7, 8)),
        (("Project",), LineRange(9, 16)),
        (("Project", "State"), LineRange(17, 24)),
    ]
    state = parsed.sections[-1]
    assert state.wikilinks == ("target", "embed")
    assert state.aliases == ("Alias",)
    assert state.tags == ("operational-tag",)


def test_markdown_sections_preserve_exact_source_text(
    markdown_source: DiscoveredSource,
) -> None:
    parsed = parse_markdown(markdown_source)

    assert parsed.sections[0].text == "Preamble text.\n\n"
    assert parsed.sections[1].text.startswith("# Project\n")
    assert parsed.sections[-1].text.endswith("Final state line.\n")


def test_markdown_ignores_fenced_and_inline_code_metadata(
    markdown_source: DiscoveredSource,
) -> None:
    text = markdown_source.text.replace(
        "Final state line.",
        "Final `[[inline-hidden]] #inline-hidden` state with [[visible]].",
    )
    source = discovered_markdown(FIXTURE_PATH, text)

    parsed = parse_markdown(source)
    state = parsed.sections[-1]

    assert [section.heading for section in parsed.sections] == [
        (),
        ("Project",),
        ("Project", "State"),
    ]
    assert state.wikilinks == ("target", "embed", "visible")
    assert state.aliases == ("Alias",)
    assert state.tags == ("operational-tag",)
    assert "not-a-link" not in state.wikilinks


@pytest.mark.parametrize(
    "text, expected_heading_line",
    [
        ("---\rowner: core\n---\n# H\n\nbody\n", 4),
        ("---\nowner: core\r---\r# H\r\rbody\r", 4),
        ("---\r\nowner: core\r---\n# H\r\n\r\nbody\r\n", 4),
    ],
)
def test_frontmatter_blanking_preserves_mixed_physical_line_count(
    tmp_path: Path,
    text: str,
    expected_heading_line: int,
) -> None:
    source = discovered_markdown(tmp_path / "mixed.md", text)

    parsed = parse_markdown(source)
    heading = parsed.sections[-1]

    assert heading.heading == ("H",)
    assert heading.lines.start == expected_heading_line
    assert heading.lines.end == len(split_source_lines(text))
    assert heading.text == text[text.index("# H") :]
    assert parsed.frontmatter["owner"] == "core"


def test_markdown_frontmatter_dates_become_canonical_json_strings(tmp_path: Path) -> None:
    path = tmp_path / "dated.md"
    source = discovered_markdown(
        path,
        "---\nupdated: 2026-08-06\nverified: 2026-08-06T14:30:00Z\n---\n# Heading\n",
    )

    parsed = parse_markdown(source)

    assert parsed.frontmatter["updated"] == "2026-08-06"
    assert parsed.frontmatter["verified"] == "2026-08-06T14:30:00+00:00"


def test_markdown_malformed_frontmatter_has_bounded_path_and_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.md"
    source = discovered_markdown(path, "---\nstatus: [active\n---\n# Heading\n")

    with pytest.raises(ParseError, match=r"broken\.md.*line 2") as error:
        parse_markdown(source)

    assert error.value.details == {"path": "broken.md", "line": 2}


def test_markdown_unclosed_frontmatter_reports_opening_line(tmp_path: Path) -> None:
    path = tmp_path / "unclosed.md"
    source = discovered_markdown(path, "---\nstatus: active\n# Heading\n")

    with pytest.raises(ParseError, match="unclosed frontmatter") as error:
        parse_markdown(source)

    assert error.value.details == {"path": "unclosed.md", "line": 1}


def test_markdown_uses_filename_stem_without_h1(tmp_path: Path) -> None:
    path = tmp_path / "fallback-name.md"
    source = discovered_markdown(path, "## State\nBody\n")

    assert parse_markdown(source).title == "fallback-name"


def test_markdown_section_lines_ignore_non_commonmark_separators(tmp_path: Path) -> None:
    text = "# Drift Note\n\nalpha\u2028more alpha\n\n## Target Section\n\ntarget body line\n"
    source = discovered_markdown(tmp_path / "drift.md", text)

    parsed = parse_markdown(source)

    physical_lines = text.split("\n")[:-1]
    assert len(physical_lines) == 7
    assert physical_lines[4] == "## Target Section"
    assert [(section.heading, section.lines) for section in parsed.sections] == [
        (("Drift Note",), LineRange(1, 4)),
        (("Drift Note", "Target Section"), LineRange(5, 7)),
    ]
    assert parsed.sections[-1].text == "## Target Section\n\ntarget body line\n"
