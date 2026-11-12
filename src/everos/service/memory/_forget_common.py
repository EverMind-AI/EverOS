"""What every forget of one user's memory and its undo read from the log.

A user's forgets mark: the marks stay in markdown, and the log says which
forget wrote which. Two forgets can name the same entry -- a conversation and
then one of its episodes by id, or two time ranges that overlap -- and an
entry stays forgotten while any forget naming it stands. So an undo asks
:func:`named_by_others` before it clears a mark, and the profile's source
rule asks :func:`forgotten_segments` which segments are forgotten already.
"""

from __future__ import annotations

__all__ = [
    "Named",
    "entries_named",
    "forgotten_segments",
    "named_by_others",
    "removed_by_others",
    "removed_item_ids",
    "segments_of",
]

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from everos.memory.extract.pipeline import profile_items
from everos.memory.ports import SessionKey

if TYPE_CHECKING:
    from everos.memory.deps import Stores
    from everos.memory.ports import ForgetEntry, Scope


@dataclass(frozen=True)
class Named:
    """The entries live forgets other than one name, by kind (entry ids)."""

    episodes: set[str] = field(default_factory=set)
    image_memories: set[str] = field(default_factory=set)
    scenes: set[str] = field(default_factory=set)


def entries_named(entry: ForgetEntry) -> Named:
    """The entries one forget of a user marked forgotten, by kind.

    A forget by records, owner or condition keeps its exact sets in
    ``undo``; a conversation's forget names its episodes in ``episode_ids``
    and its picture memories in ``entry_ids``; a picture's forget its
    picture memories (its episodes are superseded, not forgotten).
    """
    undo = entry.undo or {}
    if undo:
        return Named(
            episodes={str(row[1]) for row in undo.get("episodes") or []},
            image_memories={str(row[1]) for row in undo.get("image_memories") or []},
            scenes={str(row[1]) for row in undo.get("scenes") or []},
        )
    if entry.target == "session":
        return Named(
            episodes={old for old, _ in entry.episode_ids},
            image_memories=set(entry.entry_ids),
        )
    if entry.target == "image":
        return Named(image_memories=set(entry.entry_ids))
    return Named()


async def named_by_others(stores: Stores, scope: Scope, *, except_id: str) -> Named:
    """What the user's live forgets other than ``except_id`` keep forgotten."""
    named = Named()
    for entry in await stores.forget_log.live_entries(
        app_id=scope.app_id,
        project_id=scope.project_id,
        owner_id=scope.owner_id,
        owner_type=scope.owner_type,
    ):
        if entry.forget_id == except_id:
            continue
        found = entries_named(entry)
        named.episodes.update(found.episodes)
        named.image_memories.update(found.image_memories)
        named.scenes.update(found.scenes)
    return named


async def segments_of(
    stores: Stores, scope: Scope, sessions: Collection[str]
) -> set[str]:
    """The memcell ids of the segments of ``sessions`` in ``scope``'s space."""
    found: set[str] = set()
    for session_id in sessions:
        cells = await stores.memcell.cells_in_session(
            SessionKey(
                app_id=scope.app_id,
                project_id=scope.project_id,
                session_id=session_id,
            )
        )
        found.update(cells)
    return found


async def forgotten_segments(
    stores: Stores,
    scope: Scope,
    *,
    except_id: str | None = None,
    except_session: str | None = None,
) -> set[str]:
    """The segments the user's live forgets have forgotten, but one.

    A conversation's forget covers every segment of the conversation; a forget
    by records, owner or condition the segments of the episodes it marked
    (``undo["segments"]``). The profile's source rule takes an item out once
    every segment it rests on is among them.
    """
    sessions: list[str] = []
    found: set[str] = set()
    for entry in await stores.forget_log.live_entries(
        app_id=scope.app_id,
        project_id=scope.project_id,
        owner_id=scope.owner_id,
        owner_type=scope.owner_type,
    ):
        if entry.forget_id == except_id:
            continue
        if entry.target == "session":
            if entry.target_id != except_session:
                sessions.append(entry.target_id)
        elif entry.undo:
            found.update(str(s) for s in entry.undo.get("segments") or [])
    return found | await segments_of(stores, scope, dict.fromkeys(sessions))


def removed_item_ids(removed: Sequence[tuple[str, dict[str, object]]]) -> list[str]:
    """The item ids of what a forget took out of a profile, once each."""
    return list(
        dict.fromkeys(
            str(item["item_id"])
            for _bucket, item in removed
            if isinstance(item.get("item_id"), str)
        )
    )


async def removed_by_others(
    stores: Stores, scope: Scope, segments: Collection[str], *, except_id: str
) -> list[tuple[str, dict[str, object]]]:
    """What the user's other live forgets took out of the profile by the
    segment rule, resting on one of ``segments``.

    Undoing a forget remembers its segments again; an item another forget took
    out because every segment it rests on was forgotten -- these among them --
    is no longer wholly forgotten, so it comes back with this undo. An item a
    forget named by id, or took with a picture, stays out.
    """
    wanted = set(segments)
    found: list[tuple[str, dict[str, object]]] = []
    for other in await stores.forget_log.live_entries(
        app_id=scope.app_id,
        project_id=scope.project_id,
        owner_id=scope.owner_id,
        owner_type=scope.owner_type,
    ):
        if other.forget_id == except_id:
            continue
        if other.target == "session":
            candidates = list(other.removed_profile_items)
        elif other.undo:
            by_segments = {str(i) for i in other.undo.get("segment_items") or []}
            candidates = [
                (bucket, item)
                for bucket, item in other.removed_profile_items
                if item.get("item_id") in by_segments
            ]
        else:
            continue
        found += [
            (bucket, item)
            for bucket, item in candidates
            if profile_items.rests_on(item, wanted)
        ]
    return found
