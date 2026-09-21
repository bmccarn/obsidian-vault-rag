import pytest  # pyright: ignore[reportMissingImports]

from vault_rag.ingest.lines import (  # type: ignore[import-untyped]
    count_source_lines,
    split_source_lines,
)

NON_COMMONMARK_SEPARATORS = (
    "\x0b",
    "\x0c",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("alpha", ["alpha"]),
        ("alpha\n", ["alpha\n"]),
        ("alpha\nbeta", ["alpha\n", "beta"]),
        ("alpha\n\nbeta", ["alpha\n", "\n", "beta"]),
        ("alpha\r\nbeta\r\n", ["alpha\r\n", "beta\r\n"]),
        ("alpha\rbeta", ["alpha\r", "beta"]),
        ("alpha\r\rbeta", ["alpha\r", "\r", "beta"]),
        ("\n", ["\n"]),
        ("\r\n", ["\r\n"]),
    ],
)
def test_split_source_lines_preserves_commonmark_terminators(
    text: str, expected: list[str]
) -> None:
    assert split_source_lines(text) == expected
    assert "".join(split_source_lines(text)) == text
    assert count_source_lines(text) == len(expected)


@pytest.mark.parametrize("separator", NON_COMMONMARK_SEPARATORS)
def test_split_source_lines_ignores_non_commonmark_separators(separator: str) -> None:
    text = f"alpha{separator}beta\n"

    assert split_source_lines(text) == [text]
    assert count_source_lines(text) == 1
    # Documents the divergence this helper exists to remove: markdown-it
    # normalizes only ``\r\n?|\n``, while str.splitlines breaks here.
    assert len(text.splitlines()) == 2
