"""Line-oriented parsing for approved plain-text source kinds."""

from pathlib import PurePosixPath

from vault_rag.domain import LineRange

from .lines import count_source_lines
from .models import DiscoveredSource, ParsedSource, SourceSection


def parse_text(source: DiscoveredSource) -> ParsedSource:
    """Represent a text source as one exact, line-mapped pre-chunk section."""
    line_count = count_source_lines(source.text)
    sections: tuple[SourceSection, ...] = ()
    if line_count:
        sections = (
            SourceSection(
                heading=(),
                lines=LineRange(1, line_count),
                text=source.text,
                wikilinks=(),
                aliases=(),
                tags=(),
            ),
        )
    return ParsedSource(
        source=source,
        title=PurePosixPath(source.relative_path).stem,
        frontmatter={},
        sections=sections,
    )
