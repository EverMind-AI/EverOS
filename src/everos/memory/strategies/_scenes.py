"""An owner's scenes, built from its episodes: the core both scene strategies run.

docs/scenes.md; a group's, docs/group-memory.md, section 7.7. A scene is the
layer above the episodes: an owner's episodes about one matter -- close in
vector and in time -- with one narrative written over them, ending on where
things stand. It links to its episodes and never changes them; scenes and
episodes compete in the same search pool (scene versions are rows of the
episode table, ``parent_type = scene``). A user's scenes and a group's are
built the same way; the strategies ``update_user_scenes`` and
``update_group_scenes`` only name the owner and the switch.

- **Membership.** Each new episode is folded into the owner's scene clusters
  with everalgo's ``cluster_by_geometry`` (``[clustering]`` threshold and
  time window), or starts a cluster of its own. The clusters are stored by
  kind, ``user_scene`` or ``group_scene``.
- **Narrative.** A cluster of ``[scenes] min_members`` live episodes or more
  is a scene. Each time episodes join it, one ``chat.scene_update`` call
  writes the narrative again over the current one and the joining episodes
  (:class:`~everos.memory.extract.scene.SceneNarrator`). The narrative and
  what the call reads are capped, so its price does not grow with the scene.
  An episode that joins no scene costs no call.
- **Versions.** Every change is a new version under the owner's ``scenes/``;
  the version it replaces is marked superseded, so only the newest competes.
  A version spans its members, so a group read shows it only to an asker who
  can see every one of them.
- **Held versions.** A version a person wrote through ``POST /memory/update``
  is held unless they released it: while the newest version is held, an
  episode joining the scene joins its cluster and no narrative is written (no
  model call). Released (an update with ``hold = false``), the next join
  writes over the person's narrative with every episode that joined since.
- **Forgetting.** Members are read through their marks
  (:mod:`~everos.memory.extract.scene.members`): a forgotten episode is no
  member, a rewritten one counts as its new version. A version whose members
  were forgotten or removed since is not carried forward -- the next version
  is written from the live members only -- and a forgotten version is never
  superseded, so its mark stays.

Does nothing without an embedding provider (the episodes cannot be placed).
"""

from __future__ import annotations

__all__ = ["place_in_scenes", "update_scenes"]

from collections.abc import Sequence

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.clustering import cluster_by_geometry

from everos.config import Settings
from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import scenes_lock
from everos.memory.deps import MemoryDeps
from everos.memory.extract.scene import (
    SceneNarrator,
    alive_members,
    current_versions,
    scene_cluster_kind,
)
from everos.memory.mappers import cluster as cluster_mapper
from everos.memory.models import FORGOTTEN_MARK, ClusterKind, Scene, mint_cluster_id
from everos.memory.ports import NO_SOURCE_CHECK, Scope, Stored

logger = get_logger(__name__)


async def update_scenes(
    deps: MemoryDeps,
    scope: Scope,
    *,
    entry_id: str,
    text: str,
    timestamp_ms: int,
    language: str | None,
    settings: Settings,
) -> None:
    """Place the owner's new episode in its scenes and rewrite the one it joins."""
    if deps.embedding is None:
        return
    # Before the lock: a model call under it would hold every other writer.
    vector = np.asarray(await deps.embedding.embed(text), np.float32)
    async with scenes_lock(scope):
        if not await deps.stores.episode.exists(scope, [entry_id]):
            return
        cluster = await place_in_scenes(
            deps,
            scope,
            scene_cluster_kind(scope),
            entry_id=entry_id,
            text=text,
            timestamp_ms=timestamp_ms,
            vector=vector,
            threshold=settings.clustering.threshold,
            time_window_days=settings.clustering.time_window_days,
        )
        if len(cluster.members) < settings.scenes.min_members:
            return
        assert cluster.id is not None  # minted at placement or read back
        await _write_version(
            deps,
            scope,
            cluster.id,
            cluster.members,
            entry_id=entry_id,
            language=language,
            narrator=SceneNarrator(deps.llm, settings.scenes, deps.instructions),
        )


async def place_in_scenes(
    deps: MemoryDeps,
    scope: Scope,
    kind: ClusterKind,
    *,
    entry_id: str,
    text: str,
    timestamp_ms: int,
    vector: np.ndarray,
    threshold: float,
    time_window_days: float,
) -> AlgoCluster:
    """The scene cluster that holds the episode; placed and persisted if new.

    A replay (or a retry after the narrative call failed) finds the episode
    placed already and gets that cluster back unchanged, so a member is never
    folded in twice. The caller holds the owner's scene lock.
    """
    existing = [
        cluster_mapper.to_algo(c)
        for c in await deps.stores.cluster.list_for_owner(scope, kind)
    ]
    placed = next((c for c in existing if entry_id in c.members), None)
    if placed is not None:
        return placed
    single = AlgoCluster(
        id=mint_cluster_id(),
        centroid=vector,
        count=1,
        last_ts=timestamp_ms,
        preview=[text],
        members=[entry_id],
    )
    cluster = (
        cluster_by_geometry(
            single, existing, threshold=threshold, time_window_days=time_window_days
        )
        or single
    )
    await deps.stores.cluster.upsert(
        scope, kind, cluster_mapper.from_algo(cluster), member_type="episode"
    )
    return cluster


async def _write_version(
    deps: MemoryDeps,
    scope: Scope,
    scene_id: str,
    member_ids: Sequence[str],
    *,
    entry_id: str,
    language: str | None,
    narrator: SceneNarrator,
) -> None:
    """Write the scene's next version over its live members, when one is due.

    Nothing is written when the newest version already holds the episode (a
    replay of an event whose version was written), when the newest version
    is held, or when fewer than ``min_members`` members are alive.
    """
    previous, mark = await _newest(deps, scope, scene_id)
    if previous is not None and entry_id in previous.item.member_ids:
        return
    if previous is not None and mark is None and previous.item.held:
        logger.info(
            f"{scope.owner_type}_scene_held",
            owner_id=scope.owner_id,
            scene_id=scene_id,
            version=previous.item.version,
            entry_id=previous.ref.entry_id,
        )
        return
    members = await alive_members(deps.stores.episode, scope, member_ids)
    if len(members) < narrator.min_members:
        return
    carried, known = await _carried(deps, scope, previous, mark)
    joining = [m for m in members if m.ref.entry_id not in known] or members[-1:]
    scene, text = await narrator.version(
        scope,
        scene_id=scene_id,
        version=previous.item.version + 1 if previous is not None else 1,
        members=members,
        language=language,
        previous=carried or "",
        joining=joining,
    )
    # A forgotten version keeps its mark: superseding it would overwrite the
    # forget with this version's id, and an undo could not find it again.
    supersedes = previous if previous is not None and mark is None else None
    ref = await deps.stores.scene.append(
        scope, scene, supersedes=supersedes, only_if=NO_SOURCE_CHECK
    )
    logger.info(
        f"{scope.owner_type}_scene_written",
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
    rewritten since (a picture forget, a person's update) is not read as
    joining.
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
