"""update_group_scenes strategy — a group's scenes, built from its episodes.

docs/group-memory.md, section 7.7. A scene is a layer above the episodes: the
group's episodes about one thing -- close in vector and in time -- with one
narrative written over them. It links to its episodes and never changes them;
episodes and scenes compete in the same recall.

Built from operators that already exist, with no new prompt text (D15):

- **Membership.** Each new group episode is folded into the group's scene
  clusters with everalgo's ``cluster_by_geometry`` (``[clustering]``
  threshold and time window), the geometry users' episode clusters use. The
  clusters live in the cluster store as kind ``group_scene``.
- **Narrative.** A cluster that holds two episodes or more is a scene; its
  narrative is everalgo's ``EpisodeReflector`` -- INIT over the members when
  the scene is new, UPDATE of the current narrative with the new episode
  after that.
- **Versions.** Every change is a new version entry under
  ``groups/<g>/scenes/``; the version it replaces is marked superseded, so
  only the newest competes. A version spans its members (``source_start`` of
  the first to the end of the last), so a read shows it only to an asker who
  can see every one of them; the episodes still return on their own.

An opt-in: the strategy runs only with ``[groups] scenes = true`` (default
``false``). The design builds group scenes on 2.0's per-owner scenes, which
do not exist yet; until they do, a group gets no record type users do not
have, and ingestion makes no extra reflector calls. Without an embedding
provider it does nothing (the episodes cannot be placed).
"""

from __future__ import annotations

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.clustering import cluster_by_geometry

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import group_scenes_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import GroupEpisodeExtracted
from everos.memory.extract.pipeline.group_scene import narrate_scene, scene_version
from everos.memory.mappers import cluster as cluster_mapper
from everos.memory.models import mint_cluster_id
from everos.memory.ports import (
    NO_SOURCE_CHECK,
    Event,
    Immediate,
    Scope,
    StrategySpec,
)

logger = get_logger(__name__)

_KIND = "group_scene"


def _applies(_event: GroupEpisodeExtracted, deps: MemoryDeps) -> bool:
    return deps.embedding is not None and load_settings().groups.scenes


async def update_group_scenes(
    event: GroupEpisodeExtracted, deps: MemoryDeps
) -> list[Event]:
    """Place the new episode in the group's scenes and rewrite the one it joins."""
    if deps.embedding is None:
        return []
    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.group_id,
        owner_type="group",
    )
    # Before the lock: a model call under it would hold every other writer.
    vector = np.asarray(await deps.embedding.embed(event.episode_text), np.float32)
    async with group_scenes_lock(event.app_id, event.project_id, event.group_id):
        if not await deps.stores.episode.exists(scope, [event.episode_entry_id]):
            return []
        existing = [
            cluster_mapper.to_algo(c)
            for c in await deps.stores.cluster.list_for_owner(scope, _KIND)
        ]
        # A replay (or a retry after the narrative call failed) finds the
        # episode placed already; it still writes the version if it is due.
        placed = next(
            (c for c in existing if event.episode_entry_id in c.members), None
        )
        cluster = placed or await _place(deps, scope, existing, vector, event)
        if len(cluster.members) >= 2:
            assert cluster.id is not None  # minted above or read back
            await _write_version(deps, scope, cluster.id, cluster.members, event)
    return []


async def _place(
    deps: MemoryDeps,
    scope: Scope,
    existing: list[AlgoCluster],
    vector: np.ndarray,
    event: GroupEpisodeExtracted,
) -> AlgoCluster:
    """Fold the episode into the group's scene clusters (or start one); persist."""
    settings = load_settings().clustering
    single = AlgoCluster(
        id=mint_cluster_id(),
        centroid=vector,
        count=1,
        last_ts=event.episode_timestamp_ms,
        preview=[event.episode_text],
        members=[event.episode_entry_id],
    )
    cluster = (
        cluster_by_geometry(
            single,
            existing,
            threshold=settings.threshold,
            time_window_days=settings.time_window_days,
        )
        or single
    )
    await deps.stores.cluster.upsert(
        scope, _KIND, cluster_mapper.from_algo(cluster), member_type="episode"
    )
    return cluster


async def _write_version(
    deps: MemoryDeps,
    scope: Scope,
    scene_id: str,
    member_ids: list[str],
    event: GroupEpisodeExtracted,
) -> None:
    """Write the scene's next version over its current members.

    Nothing is written when the newest version already holds the episode (a
    replay of an event whose version was written).
    """
    previous = await deps.stores.scene.latest(scope, scene_id)
    if previous is not None and event.episode_entry_id in previous.item.member_ids:
        return
    members = sorted(
        await deps.stores.episode.read_many(scope, member_ids),
        key=lambda s: s.item.timestamp,
    )
    if len(members) < 2:
        return  # a member is gone from markdown; the next episode retries
    previous_scene = previous.item if previous is not None else None
    known = set(previous_scene.member_ids) if previous_scene is not None else set()
    new = [m for m in members if m.ref.entry_id not in known] or members[-1:]
    narrative = await narrate_scene(
        deps.llm,
        owner_id=scope.owner_id,
        members=members if previous_scene is None else new,
        previous=previous_scene,
        custom_instructions=deps.instructions.load("reflect"),
        language=event.language,
    )
    scene = scene_version(
        scope,
        scene_id=scene_id,
        version=previous.item.version + 1 if previous is not None else 1,
        members=members,
        narrative=narrative,
        language=event.language,
    )
    ref = await deps.stores.scene.append(
        scope, scene, supersedes=previous, only_if=NO_SOURCE_CHECK
    )
    logger.info(
        "group_scene_written",
        group_id=scope.owner_id,
        scene_id=scene_id,
        version=scene.version,
        members=len(members),
        entry_id=ref.entry_id if ref is not None else None,
    )


UPDATE_GROUP_SCENES_SPEC = StrategySpec(
    name="update_group_scenes",
    fn=update_group_scenes,
    trigger=Immediate(on=[GroupEpisodeExtracted]),
    applies_to=_applies,
    max_retries=2,
)
