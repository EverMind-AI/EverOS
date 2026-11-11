"""The group profile in markdown: its daily logs, their entries, the pending batch.

Paths (docs/group-memory.md, sections 6.1 and 7.5)::

    groups/<group_id>/profile/group_profile-<YYYY-MM-DD>.md   one entry per item
    groups/<group_id>/profile/pending.md                      the pending batch

The daily logs are append-only (D7). A new version of an item is a new entry;
the entry it replaces is listed in *its own* file's ``deprecated_entries``
against the new entry's id, and a forget lists an entry against
:data:`~everos.memory.models.FORGOTTEN_MARK` instead, so a reader can tell
"replaced by this entry" from "forgotten". Entry ids are
``gp_<YYYYMMDD>_<seq>``, the date being the file's (the day the entry was
written), so an id alone locates its file. The cascade projects the logs into
the ``group_profile_item`` table. ``pending.md`` is state, not memory: it is
frontmatter only (``type: group_profile_pending``, ``group_id``,
``track: group``, ``pending_memcell_ids``) and matches no cascade kind's glob,
so it is never indexed.

Entry format -- :func:`render_group_profile_entry` writes it and
:func:`group_profile_item_from_entry` reads it back, for the store and for the
cascade alike; nothing else may assume more than this::

    <!-- entry:gp_20260302_00000002 -->
    ## gp_20260302_00000002

    **kind**: decision
    **topic_id**: tp_4c1e
    **decided_by**: [u_chen]
    **agreed_by**: [u_betty]
    **supersedes**: gp_20260301_00000001
    **subject_ids**: [u_chen, u_betty]
    **sender_ids**: [u_chen, u_betty]
    **origin**: member
    **sources**: [mc_7f3a/m-102, mc_7f3a/m-103]
    **source_start**: 2026-03-02T09:02:00+00:00
    **source_end**: 2026-03-02T09:05:00+00:00
    **observed_at**: 2026-03-02T09:08:00+00:00

    ### Statement
    Depart on April 30 instead of May 1.

    ### Evidence
    - {"memcell_id": "mc_7f3a", "message_id": "m-102", "speaker_id": "u_chen",
      "at": 1772442120000, "quote": "Let's leave on April 30 instead."}
    - {"memcell_id": "mc_7f3a", "message_id": "m-103", "speaker_id": "u_betty",
      "at": 1772442300000, "quote": "Fine by me."}
    <!-- /entry:gp_20260302_00000002 -->

(each evidence line is one line in the file; wrapped here.)

Inline fields, in this order, each written only when it is set (non-empty,
non-``None``): ``kind`` (always), ``topic_id``, ``topic_status``,
``decided_by``, ``agreed_by``, ``assignees``, ``applies_to``, ``held_by``,
``about``, ``resolved`` (``true``, only when true), ``resolved_by``,
``member_id``, ``role``, ``confidence``, ``retired`` (``true``, only when
true), ``supersedes``, ``subject_ids``, ``sender_ids``, ``origin``,
``sources``, then ``source_start`` / ``source_end`` / ``observed_at``
(always; ISO-8601 in UTC, held as epoch ms by the domain). Lists render as
``[a, b]``, so an id holding ``,`` or ``]`` cannot round-trip -- the same
limit as every inline list of the other kinds.

``sources`` is informational (``<memcell_id>/<message_id>`` per source, for a
human reader and for grep) and is never read back. The authoritative copy of
the sources is the ``Evidence`` section: one line per source, ``- `` followed
by the source as one JSON object (``json.dumps(..., ensure_ascii=False)``, keys
``memcell_id`` / ``message_id`` / ``speaker_id`` / ``at`` (epoch ms, an
integer) / ``quote``). JSON keeps any quote on one line and verbatim --
quotes, colons, newlines and non-Latin text included. ``Statement``,
``Name`` and ``Summary`` are written when non-empty (``Name`` and ``Summary``
belong to topics). Like every entry section, their text must not hold a line
that starts with ``### `` (it would open a section of its own).
"""

from __future__ import annotations

__all__ = [
    "GroupProfileDailyFrontmatter",
    "GroupProfilePendingFrontmatter",
    "group_profile_item_from_entry",
    "render_group_profile_entry",
]

import datetime as _dt
import json
from typing import ClassVar, Literal

from pydantic import Field

from everos.core.utils.datetime import (
    ensure_utc,
    from_iso_format,
    from_timestamp_ms,
    to_iso_format,
    to_timestamp_ms,
)
from everos.infra.persistence.markdown._base.entries import Entry, StructuredEntry
from everos.infra.persistence.markdown._base.frontmatter import (
    DailyLogPathMixin,
    GroupScopedFrontmatter,
)
from everos.memory.models import GroupProfileItem, ProfileSource

_DIR_NAME = "profile"


class GroupProfileDailyFrontmatter(DailyLogPathMixin, GroupScopedFrontmatter):
    """Frontmatter for ``groups/<g>/profile/group_profile-<YYYY-MM-DD>.md``.

    One entry per item version; ``deprecated_entries`` maps an entry of this
    file to the entry that replaced it, or to ``forgotten``.
    """

    ENTRY_ID_PREFIX: ClassVar[str] = "gp"
    DIR_NAME: ClassVar[str] = _DIR_NAME
    FILE_PREFIX: ClassVar[str] = "group_profile"

    type: Literal["group_profile_daily"] = "group_profile_daily"
    file_type: Literal["group_profile_daily"] = "group_profile_daily"
    date: _dt.date
    entry_count: int = 0
    last_appended_at: _dt.datetime | None = None
    deprecated_entries: dict[str, str] = Field(default_factory=dict)


class GroupProfilePendingFrontmatter(GroupScopedFrontmatter):
    """``groups/<g>/profile/pending.md``: the segments the next update reads.

    Frontmatter only, rewritten whole. Its name is outside the daily logs'
    ``group_profile-*.md`` glob on purpose: the file is never indexed.
    ``failures`` and ``retry_at`` (epoch ms) record the batch's failed
    updates and when the next may try; both are absent from a file written
    before they existed, and read as none.
    """

    DIR_NAME: ClassVar[str] = _DIR_NAME
    FILENAME: ClassVar[str] = "pending.md"

    type: Literal["group_profile_pending"] = "group_profile_pending"
    pending_memcell_ids: list[str] = Field(default_factory=list)
    failures: int = 0
    retry_at: int = 0


_STATEMENT = "Statement"
_NAME = "Name"
_SUMMARY = "Summary"
_EVIDENCE = "Evidence"
_EVIDENCE_BULLET = "- "
_TRUE = "true"


def render_group_profile_entry(
    item: GroupProfileItem,
) -> tuple[dict[str, object], dict[str, str]]:
    """``(inline, sections)`` of one item, in the format the module documents.

    The read-side fields (``item_id`` / ``status`` / ``superseded_by``) and
    ``owner_id`` (the file's ``group_id``) are not written.
    """
    inline: dict[str, object] = {"kind": item.kind}
    _put(inline, "topic_id", item.topic_id)
    _put(inline, "topic_status", item.topic_status)
    _put(inline, "decided_by", item.decided_by)
    _put(inline, "agreed_by", item.agreed_by)
    _put(inline, "assignees", item.assignees)
    _put(inline, "applies_to", item.applies_to)
    _put(inline, "held_by", item.held_by)
    _put(inline, "about", item.about)
    if item.resolved:
        inline["resolved"] = _TRUE
    _put(inline, "resolved_by", item.resolved_by)
    _put(inline, "member_id", item.member_id)
    _put(inline, "role", item.role)
    _put(inline, "confidence", item.confidence)
    if item.retired:
        inline["retired"] = _TRUE
    _put(inline, "supersedes", item.supersedes)
    _put(inline, "subject_ids", item.subject_ids)
    _put(inline, "sender_ids", item.sender_ids)
    _put(inline, "origin", item.origin)
    _put(
        inline,
        "sources",
        [f"{source.memcell_id}/{source.message_id}" for source in item.sources],
    )
    inline["source_start"] = _iso(item.source_start)
    inline["source_end"] = _iso(item.source_end)
    inline["observed_at"] = _iso(item.observed_at)

    sections: dict[str, str] = {}
    if item.statement:
        sections[_STATEMENT] = item.statement
    if item.name:
        sections[_NAME] = item.name
    if item.summary:
        sections[_SUMMARY] = item.summary
    if item.sources:
        sections[_EVIDENCE] = "\n".join(
            _EVIDENCE_BULLET + json.dumps(source.model_dump(), ensure_ascii=False)
            for source in item.sources
        )
    return inline, sections


def group_profile_item_from_entry(entry: Entry, *, owner_id: str) -> GroupProfileItem:
    """The item an entry written by :func:`render_group_profile_entry` holds.

    ``owner_id`` is the group the file belongs to. The read-side fields are
    left unset: what a mark makes of the item is the reader's call.

    Raises:
        ValueError: The entry is not a readable item -- a time is missing, an
            evidence line is not a source, or a field fails the model (a
            ``pydantic.ValidationError`` is a ``ValueError``).
    """
    structured = entry if isinstance(entry, StructuredEntry) else entry.as_structured()
    inline, sections = structured.inline, structured.sections
    return GroupProfileItem(
        owner_id=owner_id,
        kind=inline.get("kind", ""),  # type: ignore[arg-type]
        statement=sections.get(_STATEMENT, ""),
        topic_id=inline.get("topic_id") or None,
        name=sections.get(_NAME, ""),
        summary=sections.get(_SUMMARY, ""),
        topic_status=inline.get("topic_status") or None,  # type: ignore[arg-type]
        decided_by=_list(inline.get("decided_by")),
        agreed_by=_list(inline.get("agreed_by")),
        assignees=_list(inline.get("assignees")),
        applies_to=_list(inline.get("applies_to")),
        held_by=_list(inline.get("held_by")),
        about=inline.get("about") or None,
        resolved=_flag(inline.get("resolved")),
        resolved_by=_list(inline.get("resolved_by")),
        member_id=inline.get("member_id") or None,
        role=inline.get("role") or None,  # type: ignore[arg-type]
        confidence=inline.get("confidence") or None,  # type: ignore[arg-type]
        retired=_flag(inline.get("retired")),
        supersedes=inline.get("supersedes") or None,
        sources=_sources(sections.get(_EVIDENCE, ""), entry_id=structured.id),
        subject_ids=_list(inline.get("subject_ids")),
        sender_ids=_list(inline.get("sender_ids")),
        origin=inline.get("origin") or None,  # type: ignore[arg-type]
        source_start=_ms(inline, "source_start", entry_id=structured.id),
        source_end=_ms(inline, "source_end", entry_id=structured.id),
        observed_at=_ms(inline, "observed_at", entry_id=structured.id),
    )


def _put(inline: dict[str, object], key: str, value: object) -> None:
    """Set ``key`` unless ``value`` is unset (``None``, ``""`` or ``[]``)."""
    if value is None or value == "" or value == []:
        return
    inline[key] = value


def _iso(timestamp_ms: int) -> str:
    rendered = to_iso_format(ensure_utc(from_timestamp_ms(timestamp_ms)))
    assert rendered is not None  # a concrete instant always renders
    return rendered


def _ms(inline: dict[str, str], key: str, *, entry_id: str) -> int:
    raw = inline.get(key)
    if not raw:
        raise ValueError(f"group profile entry {entry_id!r} has no {key!r}")
    return to_timestamp_ms(from_iso_format(raw))


def _list(raw: str | None) -> list[str]:
    """The inverse of the ``[a, b]`` rendering; absent reads as empty."""
    text = (raw or "").strip()
    if not (text.startswith("[") and text.endswith("]")):
        return []
    return [part.strip() for part in text[1:-1].split(",") if part.strip()]


def _flag(raw: str | None) -> bool:
    return (raw or "").strip().lower() == _TRUE


def _sources(text: str, *, entry_id: str) -> list[ProfileSource]:
    """The ``Evidence`` section's sources, in order; blank lines are skipped.

    Lines are split at line feeds only, never with :meth:`str.splitlines`:
    JSON escapes a line feed inside a quote but leaves U+2028, U+2029 and
    U+0085 raw, and ``splitlines`` would cut a quote's line in two at them.
    """
    sources: list[ProfileSource] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if not stripped.startswith(_EVIDENCE_BULLET.strip()):
            raise ValueError(
                f"group profile entry {entry_id!r}: evidence line is not a source"
            )
        payload = stripped[1:].strip()
        sources.append(ProfileSource.model_validate(json.loads(payload)))
    return sources
