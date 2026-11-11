"""Member-in-group profiles: the source check.

docs/group-memory.md, section 7.4. A member's profile in a group is updated
by the one profile writer (everalgo ``ProfileExtractor.aupdate``, its lineage
stamped by :mod:`.profile_items`) over the member's new segments in the
group. What this module adds is the source check: only the segments holding a
message by the member are read, so no item can rest on others' words alone
(:func:`wrote_in`, :func:`own_segments`).
"""

from __future__ import annotations

__all__ = ["own_segments", "wrote_in"]

from collections.abc import Sequence

from everos.memory.models import MemCell


def wrote_in(cell: MemCell, member_id: str) -> bool:
    """Whether ``member_id`` wrote a message of their own in the segment."""
    return any(
        getattr(item, "sender_id", None) == member_id
        and getattr(item, "role", None) == "user"
        for item in cell.items
    )


def own_segments(memcells: Sequence[MemCell], member_id: str) -> list[MemCell]:
    """The segments in which ``member_id`` wrote a message of their own."""
    return [cell for cell in memcells if wrote_in(cell, member_id)]
