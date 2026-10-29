"""trigger_profile_clustering strategy — group user episodes by topic.

Listens to :class:`EpisodeExtracted` (emitted per-episode after the user
pipeline writes its md), embeds the ``episode_text``, and merges the
resulting size-1 :class:`everalgo.clustering.Cluster` into the user's
existing user-memory cluster set.

Uses :func:`cluster_by_geometry` (embedding-only cosine + time-window).
"""

from __future__ import annotations

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.clustering import cluster_by_geometry

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import get_partition_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted, ProfileClusterUpdated
from everos.memory.mappers import cluster as cluster_mapper
from everos.memory.models import mint_cluster_id
from everos.memory.ports import ClusterStore, Event, Immediate, Scope, StrategySpec

logger = get_logger(__name__)


async def trigger_profile_clustering(
    event: EpisodeExtracted, deps: MemoryDeps
) -> list[Event]:
    """Fold the episode into its owner's clusters and announce the cluster."""
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
            owner_id=event.owner_id,
        )
        return []

    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.owner_id,
        owner_type="user",
    )
    # Serialise on owner_id: the strategy reads the user's full cluster
    # set, picks merge target by geometry, then upserts — concurrent runs
    # on the same owner_id would race the read → decide → write cycle.
    # Different users run fully in parallel.
    # Lock per (app, project, owner): clusters are scoped to a space, so a
    # different space's run must not serialise on (or merge into) this one.
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("trigger_profile_clustering", partition):
        # 1. Embed the episode_text into a vector.
        vector_list = await deps.embedding.embed(event.episode_text)
        vector = np.asarray(vector_list, dtype=np.float32)

        # 2. Load this user's existing user-memory clusters (scoped to space).
        existing = [
            cluster_mapper.to_algo(c)
            for c in await deps.stores.cluster.list_for_owner(scope, "user_memory")
        ]

        # OME delivers at least once: a replay of this event (a crash after
        # the upsert, a retry after the emit failed) must not fold the episode
        # in a second time -- the cluster's count and centroid would drift
        # and extract_user_profile's gate reads that count. An episode already
        # among the owner's members is left where it is; the downstream event
        # is still returned so a run lost with the crash is not lost for good.
        existing_ids = {c.id for c in existing}
        already = next(
            (c for c in existing if event.episode_entry_id in c.members), None
        )
        if already is not None:
            to_save = already
            logger.info(
                "profile_cluster_replay_skipped",
                episode_entry_id=event.episode_entry_id,
                cluster_id=already.id,
            )
        else:
            to_save = await _fold_in(
                event, vector, existing, deps.stores.cluster, scope
            )
        assert to_save.id is not None  # both branches above set id

    logger.info(
        "profile_cluster_updated",
        memcell_id=event.memcell_id,
        cluster_id=to_save.id,
        owner_id=event.owner_id,
        merged=to_save.id in existing_ids,
        replayed=already is not None,
        cluster_count=to_save.count,
    )
    # 6. Announce the cluster → downstream extract_user_profile.
    return [
        ProfileClusterUpdated(
            memcell_id=event.memcell_id,
            cluster_id=to_save.id,
            owner_id=event.owner_id,
            app_id=event.app_id,
            project_id=event.project_id,
            language=event.language,
        )
    ]


async def _fold_in(
    event: EpisodeExtracted,
    vector: np.ndarray,
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
        members=[event.episode_entry_id],
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
    emits=frozenset({ProfileClusterUpdated}),
    applies_to=lambda e, _deps: e.source == "pipeline",
    max_retries=2,
)
