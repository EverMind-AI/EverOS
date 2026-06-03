"""Markdown entries — id format, marker spans, and audit-form parsing.

Three closely-related entry concepts live together here so a reader
sees the whole entry surface in one file:

1. :class:`EntryId` — the ``<prefix>_<YYYYMMDD>_<NNNN>`` structured id
   stamped into each daily-log entry's open / close markers. Carries
   the prefix declared by the frontmatter schema, the date bucket, and
   the in-file zero-padded sequence.

2. :class:`Entry` — a marker-delimited span inside a markdown body::

       <!-- entry:abc123 -->
       ...content...
       <!-- /entry:abc123 -->

   :func:`split_entries` and :func:`find_entry` locate these spans
   without interpreting the inner content. Higher layers (writers,
   cascade) parse it per record type.

3. :class:`StructuredEntry` — :class:`Entry` extended with the parsed
   audit-form body fields (header / inline / sections). Built either
   from a raw body string via :func:`parse_structured_entry` or from
   an existing :class:`Entry` via :meth:`Entry.as_structured`.

Audit-form layout::

    ## <header>                ← optional H2 (usually entry id, for grep)

    **key**: value             ← inline fields, one per line
    **key2**: value2

    ### Section Title          ← section fields: H3 + free-form text
    body content...

    ### Another Section
    more content...

The audit chassis is intentionally **type-agnostic** — every field
round-trips as a string. Inline values are stringified on render
(lists become ``[a, b, c]``, scalars use ``str()``); on parse
everything is the raw text after the colon. Section titles are kept
verbatim. This keeps parsing tolerant of stray fields, wrapped
strings, and manually-typed timestamps; the strong-typed model lives
in business writers + the SQLite/LanceDB indexes.

Cross-user uniqueness is handled at the database layer via a composite
``<user_id>_<entry_id>`` field; it is *not* encoded into the
:class:`EntryId` string itself.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

# ── EntryId — structured id for marker stamping ─────────────────────────

_DATE_FMT = "%Y%m%d"
_SEQ_DIGITS = 8
"""Minimum zero-padding for the in-file seq.

8 digits keeps lexicographic order == numeric order up to 10**8
entries per file (per user, per day). ``format()`` is "at least 8" —
larger seqs emit more digits without truncation. ``parse`` is
permissive: shorter (legacy 4-digit) and longer seq strings both
parse cleanly; format normalises to >= 8 digits on round-trip.
"""


@dataclass(frozen=True, slots=True)
class EntryId:
    """Parsed components of an entry id (``<prefix>_<YYYYMMDD>_<NNNN>``)."""

    prefix: str
    date: _dt.date
    seq: int

    def format(self) -> str:
        """Render as ``<prefix>_<YYYYMMDD>_<NNNN>``."""
        return (
            f"{self.prefix}_{self.date.strftime(_DATE_FMT)}_{self.seq:0{_SEQ_DIGITS}d}"
        )

    def __str__(self) -> str:  # noqa: D401
        return self.format()

    @classmethod
    def parse(cls, s: str) -> Self:
        """Parse ``<prefix>_<YYYYMMDD>_<NNNN>``.

        Uses ``rsplit("_", 2)`` so a multi-segment prefix (rare, but
        possible) is preserved as-is.
        """
        parts = s.rsplit("_", 2)
        if len(parts) != 3:
            raise ValueError(f"invalid entry id format: {s!r}")
        prefix, date_str, seq_str = parts
        if not prefix:
            raise ValueError(f"empty prefix in entry id: {s!r}")
        try:
            d = _dt.datetime.strptime(date_str, _DATE_FMT).date()
        except ValueError as exc:
            raise ValueError(f"invalid date in entry id: {s!r}") from exc
        try:
            seq = int(seq_str)
        except ValueError as exc:
            raise ValueError(f"invalid seq in entry id: {s!r}") from exc
        if seq < 0:
            raise ValueError(f"negative seq in entry id: {s!r}")
        return cls(prefix=prefix, date=d, seq=seq)

    @classmethod
    def next_for(cls, prefix: str, date: _dt.date, current_count: int) -> Self:
        """Build the id for the next entry given the file's current count.

        ``current_count`` is the value of ``frontmatter.entry_count``
        *before* this append. The new id gets ``seq = current_count + 1``.
        """
        if current_count < 0:
            raise ValueError(f"current_count must be >= 0, got {current_count}")
        return cls(prefix=prefix, date=date, seq=current_count + 1)


# ── Entry — marker-delimited span inside a body ─────────────────────────

# Filename / URL-safe id alphabet for the marker.
_ID_PATTERN = r"[A-Za-z0-9_-]+"
_OPEN_RE = re.compile(rf"<!-- entry:({_ID_PATTERN}) -->")


@dataclass(frozen=True)
class Entry:
    """One marker-delimited entry within a markdown body.

    Attributes:
        id: Value between ``entry:`` and ``-->`` in the open marker.
        body: Content between the open and close markers, with one leading
            and one trailing newline removed (typical formatter output).
        start: Offset of the opening ``<!-- entry:id -->`` in the source body.
        end: Offset just past the closing ``<!-- /entry:id -->`` in the source.
    """

    id: str
    body: str
    start: int
    end: int

    def as_structured(self) -> StructuredEntry:
        """Parse my body as audit-form and return a :class:`StructuredEntry`.

        The id / body / start / end fields are preserved; the parsed
        ``header`` / ``inline`` / ``sections`` are added on top.
        """
        return parse_structured_entry(self.body, _origin=self)


def split_entries(body: str) -> list[Entry]:
    """Scan ``body`` and return every entry in order.

    Unmatched / unterminated open markers stop the scan at the first
    such marker — partial entries are not returned. Callers needing
    strict validation should layer a dedicated check on top.
    """
    entries: list[Entry] = []
    pos = 0
    while True:
        open_match = _OPEN_RE.search(body, pos)
        if open_match is None:
            break
        entry_id = open_match.group(1)
        close_match = _close_re_for(entry_id).search(body, open_match.end())
        if close_match is None:
            # Unterminated entry — abort further scanning.
            break
        entries.append(
            Entry(
                id=entry_id,
                body=_strip_one_newline(body[open_match.end() : close_match.start()]),
                start=open_match.start(),
                end=close_match.end(),
            )
        )
        pos = close_match.end()
    return entries


def find_entry(body: str, entry_id: str) -> Entry | None:
    """Find the first entry with ``entry_id``, or ``None``."""
    open_re = re.compile(rf"<!-- entry:{re.escape(entry_id)} -->")
    open_match = open_re.search(body)
    if open_match is None:
        return None
    close_match = _close_re_for(entry_id).search(body, open_match.end())
    if close_match is None:
        return None
    return Entry(
        id=entry_id,
        body=_strip_one_newline(body[open_match.end() : close_match.start()]),
        start=open_match.start(),
        end=close_match.end(),
    )


def _close_re_for(entry_id: str) -> re.Pattern[str]:
    """Build the close-marker regex for a specific id."""
    return re.compile(rf"<!-- /entry:{re.escape(entry_id)} -->")


def _strip_one_newline(text: str) -> str:
    """Strip one leading and one trailing newline (typical formatter padding)."""
    if text.startswith("\r\n"):
        text = text[2:]
    elif text.startswith("\n"):
        text = text[1:]
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    return text


# ── StructuredEntry — Entry + parsed audit-form fields ──────────────────

# H2 line: ``## <header>``.
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
# Inline field: ``**key**: value``. Anchored to line start so a stray
# ``**emphasis**`` mid-paragraph isn't mistaken for a field.
_INLINE_RE = re.compile(
    r"^\*\*(?P<key>[^*\n]+?)\*\*:\s*(?P<value>.*?)\s*$",
    re.MULTILINE,
)
# H3 line: ``### Title``.
_H3_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class StructuredEntry(Entry):
    """:class:`Entry` whose body has been parsed as audit-form data.

    Inherits ``id`` / ``body`` / ``start`` / ``end`` from :class:`Entry`
    (zeroed when built from a raw body string with no marker context)
    and adds three parsed views of the body: the optional H2 header,
    the inline ``**key**: value`` map, and the ``### Title`` sections.

    Audit-form values are strings only; type coercion is the caller's
    job (a strong-typed model lives in the writer / index).
    """

    header: str | None = None
    inline: dict[str, str] = field(default_factory=dict)
    sections: dict[str, str] = field(default_factory=dict)


def render_structured_entry(
    *,
    header: str | None = None,
    inline: Mapping[str, object] | None = None,
    sections: Mapping[str, str] | None = None,
) -> str:
    """Render an audit-form entry body.

    Args:
        header: Optional H2 line at the top (typically the entry id —
            redundant with the marker but useful for plain-text grep).
        inline: ``{key: value}`` rendered as ``**key**: value``. Values
            are stringified: ``list``/``tuple`` become ``[a, b, c]``;
            ``None`` becomes the empty string; everything else uses
            ``str()``.
        sections: ``{title: body}`` rendered as ``### Title`` plus the
            body text. Title is verbatim; body's trailing whitespace is
            stripped.

    Returns:
        The rendered string, no trailing newline (the caller — typically
        :meth:`MarkdownWriter.append_entry` — handles markers + newlines).
    """
    inline = inline or {}
    sections = sections or {}
    lines: list[str] = []

    if header:
        lines.append(f"## {header}")
        lines.append("")

    for key, value in inline.items():
        lines.append(f"**{key}**: {_render_value(value)}")

    for title, body in sections.items():
        lines.append("")
        lines.append(f"### {title}")
        lines.append(body.rstrip())

    return "\n".join(lines)


def parse_structured_entry(
    body: str, *, _origin: Entry | None = None
) -> StructuredEntry:
    """Parse an audit-form entry body. Strings only — no type coercion.

    Tolerant of:

    - missing H2 (``header`` will be ``None``)
    - inline fields appearing before, between or after sections
      (only matches before the first H3 are taken as the inline block)
    - extra whitespace and stray lines (silently kept inside the
      enclosing section's body)

    When called via :meth:`Entry.as_structured`, the ``_origin`` Entry
    contributes its ``id`` / ``start`` / ``end``; otherwise those fall
    back to ``""`` / ``0`` / ``len(body)``.

    Returns:
        :class:`StructuredEntry` with everything as strings.
    """
    text = body.strip("\n")

    # Split on H3 lines.
    parts = _H3_RE.split(text)
    head = parts[0]
    sections_dict: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        sections_dict[title] = content.strip("\n").rstrip()

    header: str | None = None
    h2 = _H2_RE.search(head)
    if h2:
        header = h2.group(1).strip()

    inline_dict: dict[str, str] = {
        m.group("key").strip(): m.group("value").strip()
        for m in _INLINE_RE.finditer(head)
    }

    if _origin is not None:
        return StructuredEntry(
            id=_origin.id,
            body=_origin.body,
            start=_origin.start,
            end=_origin.end,
            header=header,
            inline=inline_dict,
            sections=sections_dict,
        )
    return StructuredEntry(
        id="",
        body=body,
        start=0,
        end=len(body),
        header=header,
        inline=inline_dict,
        sections=sections_dict,
    )


def _render_value(value: object) -> str:
    """Stringify an inline value the audit-friendly way."""
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)
