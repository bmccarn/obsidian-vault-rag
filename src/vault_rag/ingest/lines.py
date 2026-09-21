"""CommonMark line boundaries shared by parsing, chunking, and reads."""

import re

# CommonMark treats a newline, a carriage return not followed by a newline, and
# a carriage return plus newline as line endings, and nothing else.  markdown-it
# normalizes exactly ``\r\n?|\n`` before assigning ``token.map`` line numbers,
# so heading provenance is counted in these units.  ``str.splitlines`` also
# breaks on VT, FF, FS, GS, RS, NEL, LS, and PS, which would shift every
# downstream line citation relative to the file an operator opens.
_LINE_BREAK = re.compile(r"\r\n|\r|\n")


def split_source_lines(text: str) -> list[str]:
    """Split ``text`` into physical CommonMark lines, keeping line endings.

    ``"".join(split_source_lines(text)) == text`` holds for every input, an
    empty string yields no lines, and a trailing line terminator does not
    create an extra empty line.
    """
    lines: list[str] = []
    start = 0
    for match in _LINE_BREAK.finditer(text):
        lines.append(text[start : match.end()])
        start = match.end()
    if start < len(text):
        lines.append(text[start:])
    return lines


def count_source_lines(text: str) -> int:
    """Count physical CommonMark lines in ``text``."""
    return len(split_source_lines(text))
