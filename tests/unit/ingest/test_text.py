import hashlib
from pathlib import Path

from vault_rag.domain import LineRange, SourceKind  # type: ignore[import-untyped]
from vault_rag.ingest.models import DiscoveredSource  # type: ignore[import-untyped]
from vault_rag.ingest.text import parse_text  # type: ignore[import-untyped]


def discovered_log(path: Path, text: str) -> DiscoveredSource:
    raw = text.encode("utf-8")
    return DiscoveredSource(
        vault_id="vault-a",
        root=path.parent,
        relative_path=path.name,
        folded_path=path.name.casefold(),
        kind=SourceKind.LOG,
        text=text,
        content_hash=f"sha256:{hashlib.sha256(raw).hexdigest()}",
        size_bytes=len(raw),
        mtime_ns=0,
    )


def test_log_file_is_one_line_mapped_section(tmp_path: Path) -> None:
    text = "first event\nsecond event\nthird event\n"
    source = discovered_log(tmp_path / "events.log", text)

    parsed = parse_text(source)

    assert parsed.title == "events"
    assert parsed.frontmatter == {}
    assert len(parsed.sections) == 1
    assert parsed.sections[0].heading == ()
    assert parsed.sections[0].lines == LineRange(1, 3)
    assert parsed.sections[0].text == text
    assert parsed.sections[0].wikilinks == ()
    assert parsed.sections[0].aliases == ()
    assert parsed.sections[0].tags == ()


def test_empty_text_file_has_no_sections(tmp_path: Path) -> None:
    source = discovered_log(tmp_path / "empty.log", "")

    assert parse_text(source).sections == ()


def test_text_line_count_ignores_non_commonmark_separators(tmp_path: Path) -> None:
    text = "first\u2028event\nsecond event\n"
    source = discovered_log(tmp_path / "events.log", text)

    parsed = parse_text(source)

    assert parsed.sections[0].lines == LineRange(1, 2)
    assert parsed.sections[0].text == text
