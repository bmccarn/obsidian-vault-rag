"""Vault source discovery and structural parsing."""

from .discovery import discover_sources
from .lines import count_source_lines, split_source_lines
from .markdown import parse_markdown
from .models import (
    DiscoveredSource,
    DiscoveryFailure,
    DiscoveryResult,
    ParsedSource,
    SourceSection,
)
from .text import parse_text

__all__ = [
    "DiscoveredSource",
    "DiscoveryFailure",
    "DiscoveryResult",
    "ParsedSource",
    "SourceSection",
    "count_source_lines",
    "discover_sources",
    "parse_markdown",
    "parse_text",
    "split_source_lines",
]
