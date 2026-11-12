"""Forgetting reaches a user's scenes through their sources (docs/scenes.md).

A scene is written from its episodes, so what happens to them decides what
happens to it, with no model call:

- **A forget** (of a conversation) marks a scene's current version forgotten
  once none of its episodes is alive (:func:`forget_orphaned_scenes`). While
  one is, the scene stays; its next update is written from the live episodes
  only (the strategy never carries a forgotten episode's narrative forward).
- **An undo** clears that mark on a scene's newest version as soon as one of
  its episodes is alive again (:func:`restore_revived_scenes`).
- **A deletion** removes every version that names a removed episode
  (:func:`remove_scenes_naming`): what a deleted episode said must not stay
  in a narrative.

Each step holds the owner's scene lock, so a scene update in flight finishes
before a forget reads the scenes, and one that starts later sees the marks.
"""

from __future__ import annotations

__all__ = [
    "forget_orphaned_scenes",
    "newest_versions",
    "remove_scenes_naming",
    "restore_revived_scenes",
]

from collections.abc import Collection

from everos.memory._partition_locks import scenes_lock
from everos.memory.models import FORGOTTEN_MARK, Scene
from everos.memory.ports import (
    NO_SOURCE_CHECK,
    EpisodeStore,
    MemoryRef,
    SceneStore,
    Scope,
    Stored,
)

from .members import alive_members


async def newest_versions(
    scenes: SceneStore, scope: Scope
) -> list[tuple[Stored[Scene], str | None]]:
    """Each scene's newest version with its mark (``None`` when current)."""
    newest: dict[str, Stored[Scene]] = {}
    for version in await scenes.versions(scope):
        held = newest.get(version.item.scene_id)
        if held is None or version.item.version > held.item.version:
            newest[version.item.scene_id] = version
    marks: dict[str, dict[str, str]] = {}
    for md_path in {v.ref.md_path for v in newest.values()}:
        marks[md_path] = await scenes.deprecations(scope, md_path)
    return [
        (v, marks[v.ref.md_path].get(v.ref.entry_id) or None) for v in newest.values()
    ]


async def forget_orphaned_scenes(
    episodes: EpisodeStore, scenes: SceneStore, scope: Scope
) -> list[Stored[Scene]]:
    """Mark forgotten each current scene version none of whose episodes is alive."""
    async with scenes_lock(scope):
        orphaned = [
            version
            for version, mark in await newest_versions(scenes, scope)
            if mark is None
            and not await alive_members(episodes, scope, version.item.member_ids)
        ]
        if orphaned:
            await scenes.deprecate(
                scope,
                entries=[(v.ref.md_path, v.ref.entry_id) for v in orphaned],
                replaced_by=FORGOTTEN_MARK,
                only_if=NO_SOURCE_CHECK,
            )
    return orphaned


async def restore_revived_scenes(
    episodes: EpisodeStore,
    scenes: SceneStore,
    scope: Scope,
    *,
    keep: Collection[str] = (),
) -> list[Stored[Scene]]:
    """Clear the forget mark of each scene's newest version with a live episode.

    Only the newest version of a scene: an older one was replaced, and
    bringing it back would make two versions current. A version in ``keep``
    (one a live forget names itself, rather than through its episodes) stays
    forgotten.
    """
    kept = set(keep)
    async with scenes_lock(scope):
        revived = [
            version
            for version, mark in await newest_versions(scenes, scope)
            if mark == FORGOTTEN_MARK
            and version.ref.entry_id not in kept
            and await alive_members(episodes, scope, version.item.member_ids)
        ]
        if revived:
            await scenes.clear_deprecations(scope, [v.ref.entry_id for v in revived])
    return revived


async def remove_scenes_naming(
    scenes: SceneStore, scope: Scope, episode_ids: Collection[str]
) -> list[MemoryRef]:
    """Remove every scene version that names one of ``episode_ids``."""
    if not episode_ids:
        return []
    async with scenes_lock(scope):
        return await scenes.remove_naming(scope, episode_ids)
