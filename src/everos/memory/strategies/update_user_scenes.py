"""update_user_scenes strategy -- a user's scenes, built from their episodes.

docs/scenes.md. A scene is the layer above the episodes: a user's episodes
about one matter -- close in vector and in time -- with one narrative written
over them, ending on where things stand. It links to its episodes and never
changes them; scenes and episodes compete in the same search pool (scene
versions are rows of the episode table, ``parent_type = scene``).

- **Membership.** Each new episode of the one-to-one pipeline is folded into
  the user's scene clusters (kind ``user_scene``), the placement group scenes
  use (:mod:`._scene_clusters`).
- **Narrative.** A cluster of ``[scenes] min_members`` live episodes or more
  is a scene. Each time episodes join it, one ``chat.scene_update`` call
  writes the narrative again over the current one and the joining episodes
  (:func:`~everos.memory.extract.scene.write_narrative`). The narrative and
  what the call reads are capped, so its price does not grow with the scene.
  An episode that joins no scene costs no call.
- **Versions.** Every change is a new version under ``users/<u>/scenes/``;
  the version it replaces is marked superseded, so only the newest competes.
- **Held versions.** A version a person wrote through ``POST /memory/update``
  is held unless they released it: while the newest version is held, an
  episode joining the scene joins its cluster and no narrative is written (no
  model call). Released (an update with ``hold = false``), the next join
  writes over the person's narrative with every episode that joined since.
- **Forgetting.** Members are read through their marks
  (:mod:`~everos.memory.extract.scene.members`): a forgotten episode is no
  member, a rewritten one counts as its new version. A version whose members
  were forgotten since is not carried forward -- the next version is written
  from the live members only -- and a forgotten version is never superseded,
  so its mark stays.

Off by default (``[scenes] enabled = false``) until its cost is accepted.
Runs on the pipeline's episodes only (a Reflection merge is not a new
episode), and does nothing without an embedding provider.
"""

from __future__ import annotations

__all__ = ["UPDATE_USER_SCENES_SPEC", "update_user_scenes"]

from collections.abc import Sequence

import numpy as np
from everalgo.types import Episode as AlgoEpisode

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import scenes_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.pipeline.group_scene import scene_version
from everos.memory.extract.scene import (
    SceneCaps,
    alive_members,
    current_versions,
    write_narrative,
)
from everos.memory.models import FORGOTTEN_MARK, Scene
from everos.memory.ports import (
    NO_SOURCE_CHECK,
    Event,
    Immediate,
    Scope,
    Stored,
    StrategySpec,
)

from ._scene_clusters import place_in_scenes

logger = get_logger(__name__)

_KIND = "user_scene"


def _applies(event: EpisodeExtracted, deps: MemoryDeps) -> bool:
    return (
        event.source == "pipeline"
        and deps.embedding is not None
        and load_settings().scenes.enabled
    )


async def update_user_scenes(event: EpisodeExtracted, deps: MemoryDeps) -> list[Event]:
    """Place the new episode in the user's scenes and rewrite the one it joins."""
    if deps.embedding is None:
        return []
    settings = load_settings()
    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.owner_id,
        owner_type="user",
    )
    # Before the lock: a model call under it would hold every other writer.
    vector = np.asarray(await deps.embedding.embed(event.episode_text), np.float32)
    async with scenes_lock(scope):
        if not await deps.stores.episode.exists(scope, [event.episode_entry_id]):
            return []
        cluster = await place_in_scenes(
            deps,
            scope,
            _KIND,
            entry_id=event.episode_entry_id,
            text=event.episode_text,
            timestamp_ms=event.episode_timestamp_ms,
            vector=vector,
            threshold=settings.clustering.threshold,
            time_window_days=settings.clustering.time_window_days,
        )
        if len(cluster.members) < settings.scenes.min_members:
            return []
        assert cluster.id is not None  # minted at placement or read back
        await _write_version(deps, scope, cluster.id, cluster.members, event)
    return []


async def _write_version(
    deps: MemoryDeps,
    scope: Scope,
    scene_id: str,
    member_ids: Sequence[str],
    event: EpisodeExtracted,
) -> None:
    """Write the scene's next version over its live members, when one is due.

    Nothing is written when the newest version already holds the episode (a
    replay of an event whose version was written), or when fewer than
    ``min_members`` members are alive.
    """
    settings = load_settings().scenes
    previous, mark = await _newest(deps, scope, scene_id)
    if previous is not None and event.episode_entry_id in previous.item.member_ids:
        return
    if previous is not None and mark is None and previous.item.held:
        logger.info(
            "user_scene_held",
            owner_id=scope.owner_id,
            scene_id=scene_id,
            version=previous.item.version,
            entry_id=previous.ref.entry_id,
        )
        return
    members = await alive_members(deps.stores.episode, scope, member_ids)
    if len(members) < settings.min_members:
        return
    carried, known = await _carried(deps, scope, previous, mark)
    joining = [m for m in members if m.ref.entry_id not in known] or members[-1:]
    text = await write_narrative(
        deps.llm,
        previous=carried or "",
        episodes=[m.item.episode for m in joining],
        caps=SceneCaps(
            narrative_max_words=settings.narrative_max_words,
            max_new_episodes=settings.max_new_episodes,
            episode_max_chars=settings.episode_max_chars,
        ),
        language=event.language,
        custom_instructions=deps.instructions.load("reflect"),
    )
    scene = scene_version(
        scope,
        scene_id=scene_id,
        version=previous.item.version + 1 if previous is not None else 1,
        members=members,
        narrative=AlgoEpisode(
            owner_id=None,
            episode=text.narrative,
            subject=text.title,
            summary=text.summary,
            timestamp=members[-1].item.timestamp,
        ),
        language=event.language,
    )
    # A forgotten version keeps its mark: superseding it would overwrite the
    # forget with this version's id, and an undo could not find it again.
    supersedes = previous if previous is not None and mark is None else None
    ref = await deps.stores.scene.append(
        scope, scene, supersedes=supersedes, only_if=NO_SOURCE_CHECK
    )
    logger.info(
        "user_scene_written",
        owner_id=scope.owner_id,
        scene_id=scene_id,
        version=scene.version,
        members=len(members),
        joining=len(joining),
        read=text.episodes,
        mode="update" if carried is not None else "init",
        narrative_chars=len(text.narrative),
        clipped=text.clipped,
        entry_id=ref.entry_id if ref is not None else None,
    )


async def _newest(
    deps: MemoryDeps, scope: Scope, scene_id: str
) -> tuple[Stored[Scene] | None, str | None]:
    """The scene's newest version and its mark (``None`` when it has none)."""
    newest = await deps.stores.scene.latest(scope, scene_id)
    if newest is None:
        return None, None
    marks = await deps.stores.scene.deprecations(scope, newest.ref.md_path)
    return newest, marks.get(newest.ref.entry_id) or None


async def _carried(
    deps: MemoryDeps,
    scope: Scope,
    previous: Stored[Scene] | None,
    mark: str | None,
) -> tuple[str | None, set[str]]:
    """The narrative the next version builds on, and the members it covers.

    ``(None, set())`` starts over: a forgotten version, or one written from an
    episode that was forgotten or removed since, may say what that episode
    said, so the next version is written from the live members alone. The
    covered members are named by their current versions, so an episode
    rewritten since (a picture forget) is not read as joining.
    """
    if previous is None or mark == FORGOTTEN_MARK:
        return None, set()
    current = await current_versions(
        deps.stores.episode, scope, previous.item.member_ids
    )
    if any(stored is None for stored in current.values()):
        return None, set()
    covered = {stored.ref.entry_id for stored in current.values() if stored}
    return previous.item.narrative, covered


UPDATE_USER_SCENES_SPEC = StrategySpec(
    name="update_user_scenes",
    fn=update_user_scenes,
    trigger=Immediate(on=[EpisodeExtracted]),
    applies_to=_applies,
    max_retries=2,
)
