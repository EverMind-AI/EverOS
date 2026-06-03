"""Markdown file reader.

Loads a markdown document and splits it into:

    1. ``frontmatter`` — parsed YAML (empty dict if absent)
    2. ``body`` — raw text after the closing ``---`` delimiter
    3. ``entries`` — marker-delimited spans inside ``body``

The reader is purely parsing; it does not validate frontmatter shape,
entry content, or cross-references. Higher layers add business-aware
checks. The :class:`ParsedMarkdown` data type lives in :mod:`.parsed`.

``parse`` is sync (pure in-memory string processing). ``read`` is async
and uses :class:`anyio.Path` so file I/O does not block the event loop.
"""

from __future__ import annotations

from pathlib import Path

import anyio

from .entries import split_entries
from .frontmatter import parse_frontmatter
from .parsed import ParsedMarkdown


class MarkdownReader:
    """Parse markdown files / strings into :class:`ParsedMarkdown`."""

    @staticmethod
    def parse(text: str) -> ParsedMarkdown:
        """Parse already-loaded text (no IO)."""
        meta, body = parse_frontmatter(text)
        entries = split_entries(body)
        return ParsedMarkdown(frontmatter=meta, body=body, entries=entries)

    @staticmethod
    async def read(path: Path) -> ParsedMarkdown:
        """Read the file at ``path`` and parse its content."""
        text = await anyio.Path(path).read_text(encoding="utf-8")
        return MarkdownReader.parse(text)
