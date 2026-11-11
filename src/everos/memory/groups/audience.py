"""Who knew a group record: its audience (docs/group-memory.md, section 8.6).

Computed at read time from the roster and the record's span; nothing is
stored. A record's span ``[start, end]`` runs from its first source to its
last (a segment's first and last message, an item's evidence):

* ``present`` -- members, current or former, humans and assistants, one of
  whose membership stays covers the whole span (adjacent stays merged, as in
  the read windows);
* ``joined_after`` -- current members who were not present and whose current
  stay began after the span's end: the people who have not heard of it.

Membership here is the roster's, not visibility: a group whose history is
visible from its creation still has members who were not there. The caller
attaches an audience only to records the asker may see.
"""

from __future__ import annotations

__all__ = ["RecordAudience", "record_audience"]

from dataclasses import dataclass

from everos.memory.models import Roster


@dataclass(frozen=True)
class RecordAudience:
    """``present`` and ``joined_after`` member ids, each in roster order."""

    present: tuple[str, ...]
    joined_after: tuple[str, ...]


def record_audience(roster: Roster, *, start: int, end: int) -> RecordAudience:
    """The audience of a record whose sources span ``[start, end]`` (epoch ms)."""
    members = list(dict.fromkeys(i.member_id for i in roster.intervals))
    present = [
        m
        for m in members
        if any(
            lo <= start and (hi is None or end < hi) for lo, hi in roster.windows_of(m)
        )
    ]
    joined_after = [
        interval.member_id
        for interval in roster.current_members()
        if interval.member_id not in present and interval.joined_at > end
    ]
    return RecordAudience(present=tuple(present), joined_after=tuple(joined_after))
