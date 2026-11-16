"""trigger_profile_clustering strategy — group user episodes by topic.

Listens to pipeline :class:`EpisodeExtracted` events (one per memcell
episode), embeds the ``episode_text`` once, and merges a size-1
:class:`everalgo.clustering.Cluster` into the user-memory cluster set of
every copy's owner, each under that owner's lock: the trigger is per
episode, clusters are per owner.

Uses :func:`cluster_by_geometry` (embedding-only cosine + time-window).
"""

from __future__ import annotations

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.clustering import cluster_by_geometry

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import user_cluster_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeCopy, EpisodeExtracted
from everos.memory.mappers import cluster as cluster_mapper
from everos.memory.models import mint_cluster_id
from everos.memory.ports import ClusterStore, Event, Immediate, Scope, StrategySpec

logger = get_logger(__name__)


async def trigger_profile_clustering(
    event: EpisodeExtracted, deps: MemoryDeps
) -> list[Event]:
    """Fold the episode into the clusters of every copy's owner."""
    # Body-guard: without an embedding provider (Tier 1) the episode
    # cannot be vectorised, so the strategy silently no-ops — no work, no
    # owner lock, no OME retry pressure. Tier upgrades require a server
    # restart; this guard is not a hot-reload mechanism.
    #
    # Log level is intentionally ``debug`` here (and at the three sibling
    # body-guards in ``trigger_skill_clustering``, ``extract_agent_skill``,
    # and ``reflect_episodes``) rather than the ``info`` used by
    # ``cascade.registry`` and the reflection orchestrator's
    # ``cascade_registry_knowledge_gated_off`` /
    # ``reflection_orchestrator_disabled`` events. Rationale (PR #361
    # round-3 review #8, accepted intentional asymmetry): the
    # registry/orchestrator gate events fire ONCE per process at startup
    # and are useful lifecycle signals worth ``info``. The per-dispatch
    # body-guards fire on EVERY memorize/reflection dispatch under Tier 1
    # — a busy chat session emits hundreds. At ``info`` they would flood
    # structured-log consumers even when the process-wide default level
    # is ``WARNING`` (which suppresses stdout but not the sink). ``debug``
    # keeps this per-event signal cheap and opt-in via ``--verbose`` for
    # diagnosing why a strategy isn't running.
    if deps.embedding is None:
        logger.debug(
            "strategy_gated_off_embedding_unavailable",
            strategy_name="trigger_profile_clustering",
            memcell_id=event.memcell_id,
        )
        return []

    # 1. Embed the episode_text once, before any lock: a model call under
    #    one would hold every other writer of that owner's clusters (and a
    #    deletion clearing one) for its whole duration.
    vector_list = await deps.embedding.embed(event.episode_text)
    vector = np.asarray(vector_list, dtype=np.float32)
    # Owners one after another in this task, not gathered: only the run's
    # own task hands its execution slot back while it waits for a lock.
    for copy in event.copies:
        await _cluster_for_owner(event, copy, vector, deps)
    return []


async def _cluster_for_owner(
    event: EpisodeExtracted, copy: EpisodeCopy, vector: np.ndarray, deps: MemoryDeps
) -> None:
    """Fold one owner's copy of the episode into that owner's clusters."""
    scope = event.owner_scope(copy)
    # Serialise per (app, project, owner): the fold reads the owner's full
    # cluster set, picks a merge target by geometry, then upserts, so
    # concurrent runs on one owner would race that cycle. Clusters are
    # scoped to a space, so another space's run neither waits nor merges.
    async with user_cluster_lock(event.app_id, event.project_id, copy.owner_id):
        # The episode may have been deleted since it was extracted. A
        # single-id deletion removes it from md first and then clears it from
        # its cluster under this same lock, so checking here keeps a run
        # queued behind that deletion from adding it back.
        if not await _sources_alive(event, copy, scope, deps):
            logger.info(
                "profile_cluster_skipped_source_gone",
                episode_entry_id=copy.entry_id,
                owner_id=copy.owner_id,
            )
            return

        # 2. Load this user's existing user-memory clusters (scoped to space).
        existing = [
            cluster_mapper.to_algo(c)
            for c in await deps.stores.cluster.list_for_owner(scope, "user_memory")
        ]

        # OME delivers at least once: a replay of this event (a crash after
        # the upsert, or a retry after another owner's fold failed) must not
        # fold the episode in a second time -- the cluster's count and
        # centroid would drift. An episode already among the owner's members
        # is left where it is.
        existing_ids = {c.id for c in existing}
        already = next((c for c in existing if copy.entry_id in c.members), None)
        if already is not None:
            to_save = already
            logger.info(
                "profile_cluster_replay_skipped",
                episode_entry_id=copy.entry_id,
                cluster_id=already.id,
            )
        else:
            to_save = await _fold_in(
                event,
                copy.entry_id,
                vector,
                existing=existing,
                clusters=deps.stores.cluster,
                scope=scope,
            )
        assert to_save.id is not None  # both branches above set id

    logger.info(
        "profile_cluster_updated",
        memcell_id=event.memcell_id,
        cluster_id=to_save.id,
        owner_id=copy.owner_id,
        merged=to_save.id in existing_ids,
        replayed=already is not None,
        cluster_count=to_save.count,
    )


async def _sources_alive(
    event: EpisodeExtracted, copy: EpisodeCopy, scope: Scope, deps: MemoryDeps
) -> bool:
    """The owner's copy is still in md and its memcell still exists."""
    if not await deps.stores.episode.exists(scope, [copy.entry_id]):
        return False
    return bool(await deps.stores.memcell.find_by_ids([event.memcell_id]))


async def _fold_in(
    event: EpisodeExtracted,
    entry_id: str,
    vector: np.ndarray,
    *,
    existing: list[AlgoCluster],
    clusters: ClusterStore,
    scope: Scope,
) -> AlgoCluster:
    """Merge the episode into the owner's clusters (or start one) and persist."""
    # 3. Build a size-1 cluster for the new episode.
    new_cluster = AlgoCluster(
        id=mint_cluster_id(),
        centroid=vector,
        count=1,
        last_ts=event.episode_timestamp_ms,
        preview=[event.episode_text],
        members=[entry_id],
    )
    # 4. Geometry-merge it into an existing cluster (or keep as-is).
    # ``cluster_by_geometry`` is a pure synchronous CPU function (cosine +
    # time-window math, no I/O) returning ``Cluster | None`` directly, so
    # it must not be awaited (``await None`` raises when there is no
    # existing cluster to merge into).
    settings = load_settings()
    merged = cluster_by_geometry(
        new_cluster,
        existing,
        threshold=settings.clustering.threshold,
        time_window_days=settings.clustering.time_window_days,
    )
    to_save = merged if merged is not None else new_cluster

    # 5. Persist the (possibly-merged) cluster.
    await clusters.upsert(
        scope, "user_memory", cluster_mapper.from_algo(to_save), member_type="episode"
    )
    return to_save


TRIGGER_PROFILE_CLUSTERING_SPEC = StrategySpec(
    name="trigger_profile_clustering",
    fn=trigger_profile_clustering,
    trigger=Immediate(on=[EpisodeExtracted]),
    applies_to=lambda e, _deps: e.source == "pipeline",
    max_retries=2,
)
