"""Markdown file reader.

Loads a markdown document and splits it into:

    1. ``frontmatter`` — parsed YAML (empty dict if absent)
    2. ``body`` — raw text after the closing ``---`` delimiter
    3. ``entries`` — marker-delimited spans inside ``body``

The reader is purely parsing; it does not validate frontmatter shape,
entry content, or cross-references. Higher layers add business-aware
checks. The :class:`ParsedMarkdown` data type lives in :mod:`.parsed`.

One broken entry costs that entry only. An open marker without its close
marker is skipped and the entries after it are still read
(:func:`~.entries.scan_entries`); :meth:`MarkdownReader.read` reports each
such entry with :func:`warn_unreadable_entry`, the one structured warning
(``markdown_entry_unreadable``: path, entry id, line, reason) every reader
that skips an entry -- the cascade, the stores -- logs.

``parse`` is sync (pure in-memory string processing). ``read`` and
``read_text`` are async and run the file I/O in a worker thread so it does
not block the event loop. Every read of a markdown file under the memory
root goes through ``read_text``: on Windows an ``open`` that races another
writer's ``os.replace`` of the same file can be refused for a moment, and
``read_text`` waits that out (see :func:`_read_text_with_retry`).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import anyio

from everos.core.observability.logging import get_logger

from .entries import scan_entries
from .frontmatter import parse_frontmatter
from .parsed import ParsedMarkdown

logger = get_logger(__name__)

# The same budget as the writer's ``_replace_with_retry``: the two sides of
# one race, so they wait equally long.
_READ_ATTEMPTS = 8
_READ_FIRST_BACKOFF_S = 0.02  # doubles each time: ~2.5 s of patience in total


def _read_text_with_retry(path: Path) -> str:
    """``Path.read_text`` that outlasts a Windows sharing violation.

    On Windows, opening a file while another thread or process is replacing
    it with ``os.replace`` can fail with ``PermissionError`` (errno 13) for a
    moment. The writer already retries its own ``os.replace`` for the
    mirror-image case (``writer._replace_with_retry``); a read that races it
    needs the same patience, or a forget fails half way and a cascade
    projection fails permanently.
    Measured on the GitHub Windows runner: 16 refused reads in 1.26 million
    while one thread kept replacing the file, none with this retry.

    On POSIX the first attempt succeeds. A POSIX ``PermissionError`` (a file
    without read permission) is permanent and only costs the backoff before it
    surfaces. Only ``PermissionError`` is retried; ``FileNotFoundError`` and
    everything else propagate at once, unchanged.
    """
    delay = _READ_FIRST_BACKOFF_S
    for attempt in range(1, _READ_ATTEMPTS):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            logger.debug(
                "markdown_read_retried",
                path=str(path),
                attempt=attempt,
                backoff_seconds=delay,
            )
            time.sleep(delay)
            delay *= 2
    return path.read_text(encoding="utf-8")  # the last attempt lets the error out


def warn_unreadable_entry(
    path: os.PathLike[str] | str,
    entry_id: str,
    *,
    reason: str,
    line: int | None = None,
    kind: str | None = None,
) -> None:
    """Log one skipped entry as ``markdown_entry_unreadable``.

    Args:
        path: The file, relative to the memory root where the caller knows it.
        entry_id: The id its open marker names.
        reason: Why it could not be read, on one line, cut to 200 characters.
        line: The line of its open marker, when the caller has the file.
        kind: The memory kind the caller reads it as (``episode``, ...).
    """
    logger.warning(
        "markdown_entry_unreadable",
        path=str(path),
        entry_id=entry_id,
        line=line,
        reason=" ".join(reason.split())[:200],
        kind=kind,
    )


class MarkdownReader:
    """Parse markdown files / strings into :class:`ParsedMarkdown`."""

    @staticmethod
    def parse(text: str) -> ParsedMarkdown:
        """Parse already-loaded text (no IO)."""
        meta, body = parse_frontmatter(text)
        entries, unreadable = scan_entries(body)
        return ParsedMarkdown(
            frontmatter=meta,
            body=body,
            entries=entries,
            unreadable=unreadable,
            body_line=text.count("\n", 0, len(text) - len(body)) + 1,
        )

    @staticmethod
    async def read_text(path: os.PathLike[str] | str) -> str:
        """Read the file at ``path`` as UTF-8 text, outlasting a Windows
        sharing violation (see :func:`_read_text_with_retry`)."""
        return await anyio.to_thread.run_sync(_read_text_with_retry, Path(path))

    @staticmethod
    async def read(path: Path) -> ParsedMarkdown:
        """Read the file at ``path`` and parse its content.

        Each entry left without its close marker is logged
        (:func:`warn_unreadable_entry`) and left out; the rest are read.
        """
        parsed = MarkdownReader.parse(await MarkdownReader.read_text(path))
        for skipped in parsed.unreadable:
            warn_unreadable_entry(
                path,
                skipped.id,
                reason=skipped.reason,
                line=parsed.line_of(skipped.start),
            )
        return parsed
