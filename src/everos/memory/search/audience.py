"""Who knew what, on the records a group search returns (section 8.6).

``audience.present`` and ``audience.joined_after`` are computed at read time
from the roster of the record's group (the read scope already holds it) and
the record's span; nothing is stored. Only records the asker may see are ever
given one -- the callers attach it after the predicate chose them.
"""

from __future__ import annotations

__all__ = ["audience_of", "with_audience"]

from collections.abc import Sequence

from everos.memory.access import ReadScope
from everos.memory.groups import record_audience

from .dto import AudienceItem, GroupProfileEntry, SearchEpisodeItem, SearchImageItem


def audience_of(
    scope: ReadScope, group_id: str, *, start: int, end: int
) -> AudienceItem | None:
    """The audience of a record of ``group_id`` spanning ``[start, end]``."""
    view = scope.group(group_id)
    if view is None:
        return None
    found = record_audience(view.roster, start=start, end=end)
    return AudienceItem(
        present=list(found.present), joined_after=list(found.joined_after)
    )


def with_audience[T: (SearchEpisodeItem, SearchImageItem, GroupProfileEntry)](
    scope: ReadScope, items: Sequence[T]
) -> list[T]:
    """``items`` with their audience set; records of no group pass unchanged."""
    out: list[T] = []
    for item in items:
        group_id = item.group_id
        span = _span(item)
        if group_id is None or span is None:
            out.append(item)
            continue
        audience = audience_of(scope, group_id, start=span[0], end=span[1])
        out.append(item.model_copy(update={"audience": audience}))
    return out


def _span(
    item: SearchEpisodeItem | SearchImageItem | GroupProfileEntry,
) -> tuple[int, int] | None:
    if isinstance(item, SearchImageItem):
        stamp = int(item.timestamp.timestamp() * 1000)
        return stamp, stamp
    if item.source is None:
        return None
    return item.source.from_, item.source.to
