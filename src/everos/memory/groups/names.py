"""Who a text names: roster members matched by display name and alias.

docs/group-memory.md, section 7.2: a group record's ``subject_ids`` are the
roster members whose ``display_name`` or an alias occurs in its title or body;
section 8.2 finds the members a query names the same way. The rule is
deliberately simple -- cheap, testable, explainable -- and it is the first
thing to replace if the attribution sample of section 13.5 shows misses:

* whole-word and case-insensitive for scripts that separate words with spaces,
  so ``Al`` is not found inside ``Alice``;
* a plain substring for scripts written without word spacing (Chinese,
  Japanese, Thai, Lao, Khmer, Myanmar), where a text has no word boundaries
  to anchor on.

Pronouns and unlisted nicknames are not found; aliases in the roster are the
remedy.
"""

from __future__ import annotations

__all__ = ["NamedMember", "named_members", "names_of", "roster_names"]

import re
from collections.abc import Iterable
from dataclasses import dataclass

from everos.memory.models import MemberKind, Roster

_UNSPACED_RANGES: tuple[tuple[int, int], ...] = (
    (0x0E00, 0x0EFF),  # Thai, Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF66, 0xFF9F),  # Halfwidth Katakana
    (0x20000, 0x3134F),  # CJK Extensions B-G
)
"""Code points of the scripts written without spaces between words."""


@dataclass(frozen=True)
class NamedMember:
    """A member and every name a text may call them by."""

    member_id: str
    names: tuple[str, ...]


def names_of(
    roster: Roster, member_id: str, *, until: int | None = None
) -> tuple[str, ...]:
    """Every display name and alias ``member_id`` held, oldest interval first.

    ``until`` (epoch ms) keeps only the intervals that had begun by then, so a
    record is never attributed to someone who joined after it.
    """
    names: list[str] = []
    for interval in sorted(roster.intervals_of(member_id), key=lambda i: i.joined_at):
        if until is not None and interval.joined_at > until:
            continue
        for name in (interval.display_name, *interval.aliases):
            if name and name not in names:
                names.append(name)
    return tuple(names)


def roster_names(
    roster: Roster, *, until: int | None = None, kind: MemberKind | None = None
) -> list[NamedMember]:
    """Every member who had joined by ``until`` -- former members included --
    with their names; only members of ``kind`` (by their latest interval begun
    by then) when it is given."""
    members: list[NamedMember] = []
    for member_id in dict.fromkeys(i.member_id for i in roster.intervals):
        begun = [
            i
            for i in roster.intervals_of(member_id)
            if until is None or i.joined_at <= until
        ]
        if not begun:
            continue
        latest = max(begun, key=lambda i: i.joined_at)
        if kind is not None and latest.kind != kind:
            continue
        names = names_of(roster, member_id, until=until)
        if names:
            members.append(NamedMember(member_id=member_id, names=names))
    return members


def named_members(text: str, members: Iterable[NamedMember]) -> list[str]:
    """The ids of the members ``text`` names, in the order the text names them."""
    folded = text.casefold()
    hits: list[tuple[int, str]] = []
    for member in members:
        positions = [p for p in (_find(folded, n) for n in member.names) if p >= 0]
        if positions:
            hits.append((min(positions), member.member_id))
    return [member_id for _, member_id in sorted(hits)]


def _find(folded_text: str, name: str) -> int:
    """Where ``name`` first occurs in an already case-folded text; ``-1`` if not."""
    needle = name.strip().casefold()
    if not needle:
        return -1
    if any(_unspaced(ch) for ch in needle):
        return folded_text.find(needle)
    found = re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", folded_text)
    return found.start() if found else -1


def _unspaced(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _UNSPACED_RANGES)
