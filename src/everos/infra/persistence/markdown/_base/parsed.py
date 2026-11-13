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

from .entries import Entry, UnreadableEntry


@dataclass(frozen=True)
class ParsedMarkdown:
    """A markdown document after parsing.

    Attributes:
        frontmatter: Parsed YAML mapping (empty dict if no frontmatter block).
        body: Document text after the frontmatter block; not entry-stripped.
        entries: Marker-delimited entries discovered inside ``body``.
        unreadable: Open markers in ``body`` left without a close marker
            (a hand edit); the scan skipped each and went on.
        body_line: Line of the document the body starts on (1-based).
    """

    frontmatter: dict[str, Any]
    body: str
    entries: list[Entry] = field(default_factory=list)
    unreadable: list[UnreadableEntry] = field(default_factory=list)
    body_line: int = 1

    def line_of(self, start: int) -> int:
        """The document line (1-based) of the body offset ``start``.

        What a warning about an entry names besides its id: the line of its
        open marker is ``line_of(entry.start)``.
        """
        return self.body_line + self.body.count("\n", 0, start)
