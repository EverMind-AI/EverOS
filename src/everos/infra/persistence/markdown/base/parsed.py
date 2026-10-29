"""Parsed-markdown data type.

The output shape of :class:`MarkdownReader` is held here, separate
from the reader implementation: callers that only consume parse
results don't need to import the reader machinery, and downstream
modules (writer, business readers) can produce :class:`ParsedMarkdown`
without going through ``MarkdownReader.read`` if they already hold
the pieces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .entries import Entry


@dataclass(frozen=True)
class ParsedMarkdown:
    """A markdown document after parsing.

    Attributes:
        frontmatter: Parsed YAML mapping (empty dict if no frontmatter block).
        body: Document text after the frontmatter block; not entry-stripped.
        entries: Marker-delimited entries discovered inside ``body``.
    """

    frontmatter: dict[str, Any]
    body: str
    entries: list[Entry] = field(default_factory=list)
