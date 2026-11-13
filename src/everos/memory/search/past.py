"""Asking about the past: each returned record's version state.

A search with ``include_past`` reads through the access policy's predicate of
the past (:meth:`~everos.memory.access.AccessPolicy.past_predicate`), so its
results can hold a superseded version or a forgotten memory beside the
current ones. :func:`with_versions` says which is which on every returned
episode and picture memory -- ``status`` always; on a superseded one also the
version that replaced it (``superseded_by``) and that version's own time
(``valid_until``): the old version said what it says from its ``timestamp``
until then. A replacement the asker may not see is not named.

The record's state is the index row's mark (``Stored.mark``). The versions
that replaced the returned ones are read in one index read per kind, through
the same predicate of the past, so a version outside the asker's windows --
or retired by a live forget -- never reaches the answer, not even as an id.
No model call; a search without ``include_past`` never comes here, and its
records carry none of these fields.
"""

from __future__ import annotations

__all__ = ["with_versions"]

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Protocol

from everos.core.utils.datetime import from_timestamp_ms, to_display_tz
from everos.memory.models import FORGOTTEN_MARK, Episode, ImageMemory
from everos.memory.ports import Filter, Stored

from .dto import SearchEpisodeItem, SearchImageItem
from .forgetting import state_of

type Record = Stored[Episode] | Stored[ImageMemory]
type Later = Mapping[tuple[str, str], Stored[Any]]
"""A replacing version by ``(owner id, entry id)`` of the version it replaced."""


class _EntryReader(Protocol):
    async def find_entries(
        self, where: Filter, entries: Mapping[str, Collection[str]]
    ) -> Sequence[Stored[Any]]: ...


async def with_versions(
    owners: Filter,
    records: Mapping[str, Record],
    episodes: Sequence[SearchEpisodeItem],
    images: Sequence[SearchImageItem],
    *,
    episode_reader: _EntryReader,
    image_reader: _EntryReader | None,
) -> tuple[list[SearchEpisodeItem], list[SearchImageItem]]:
    """``episodes`` and ``images`` with their version state.

    Args:
        owners: The predicate of the past the search read through.
        records: The stored record behind every candidate, by result id.
        episodes / images: The returned items.
        episode_reader / image_reader: Where a replacing version is read;
            without a picture reader a picture's replacement is not named.
    """
    later_episodes = await _later(owners, records, episodes, episode_reader)
    later_images = (
        await _later(owners, records, images, image_reader)
        if image_reader is not None
        else {}
    )
    return (
        [_versioned(e, records, later_episodes) for e in episodes],
        [_versioned(i, records, later_images) for i in images],
    )


def _versioned[T: (SearchEpisodeItem, SearchImageItem)](
    item: T, records: Mapping[str, Record], later: Later
) -> T:
    record = records.get(item.id)
    if record is None:
        return item
    update: dict[str, object] = {"status": state_of(record)}
    successor = later.get((record.scope.owner_id, record.mark or ""))
    if successor is not None:
        update["superseded_by"] = successor.ref.id
        update["valid_until"] = to_display_tz(
            from_timestamp_ms(successor.item.timestamp)
        )
    return item.model_copy(update=update)


async def _later(
    owners: Filter,
    records: Mapping[str, Record],
    items: Sequence[SearchEpisodeItem] | Sequence[SearchImageItem],
    reader: _EntryReader,
) -> dict[tuple[str, str], Stored[Any]]:
    """The visible versions that replaced ``items``, in one index read."""
    wanted: dict[str, set[str]] = {}
    for item in items:
        record = records.get(item.id)
        if record is None or record.mark is None or record.mark == FORGOTTEN_MARK:
            continue
        wanted.setdefault(record.scope.owner_id, set()).add(record.mark)
    if not wanted:
        return {}
    found = await reader.find_entries(owners, wanted)
    return {(r.scope.owner_id, r.ref.entry_id): r for r in found}
