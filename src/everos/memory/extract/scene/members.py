"""Which of a scene's episodes are still alive, and in which version.

A scene names its episodes by entry id when a version is written. An episode
can change after that without the scene knowing: a picture forget writes a
new version without the picture and marks the old one superseded by it; a
Reflection merge marks its sources merged into the merged episode; a session
forget marks it ``forgotten``; a hard delete removes it. So a member is read
through its marks: a supersession leads to the successor, ``forgotten`` or a
missing entry ends it. That answers both what a scene update reads (the
current text of each member, never a version a forget replaced) and when a
forget reaches the scene (no member alive -- docs/scenes.md, "forgetting
follows the sources").
"""

from __future__ import annotations

__all__ = ["alive_members", "current_versions"]

from collections.abc import Collection, Sequence

from everos.memory.models import FORGOTTEN_MARK, Episode
from everos.memory.ports import EpisodeStore, Scope, Stored

_CHAIN_CAP = 8
"""Supersession links followed per member; a longer chain reads as gone."""


async def current_versions(
    episodes: EpisodeStore, scope: Scope, entry_ids: Collection[str]
) -> dict[str, Stored[Episode] | None]:
    """Each id's current version, or ``None`` when it was forgotten or removed.

    Reads markdown (the files the ids name, and their marks), never the
    index: a forget changes the truth first and must not wait for the
    cascade.
    """
    out: dict[str, Stored[Episode] | None] = {}
    pending = {entry_id: entry_id for entry_id in dict.fromkeys(entry_ids)}
    for _ in range(_CHAIN_CAP):
        if not pending:
            break
        found = {
            stored.ref.entry_id: stored
            for stored in await episodes.read_many(scope, set(pending.values()))
        }
        marks: dict[str, dict[str, str]] = {}
        for md_path in {stored.ref.md_path for stored in found.values()}:
            marks[md_path] = await episodes.deprecations(scope, md_path)
        following: dict[str, str] = {}
        for start, entry_id in pending.items():
            stored = found.get(entry_id)
            if stored is None:
                out[start] = None
                continue
            mark = marks[stored.ref.md_path].get(entry_id)
            if not mark:
                out[start] = stored
            elif mark == FORGOTTEN_MARK:
                out[start] = None
            else:
                following[start] = str(mark)
        pending = following
    out.update(dict.fromkeys(pending))
    return out


async def alive_members(
    episodes: EpisodeStore, scope: Scope, member_ids: Sequence[str]
) -> list[Stored[Episode]]:
    """The current versions of ``member_ids`` that are alive, oldest first.

    Two members that lead to one version (two sources of one merge) count
    once.
    """
    current = await current_versions(episodes, scope, member_ids)
    unique = {s.ref.entry_id: s for s in current.values() if s is not None}
    return sorted(unique.values(), key=lambda s: s.item.timestamp)
