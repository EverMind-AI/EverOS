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

from everos.component.embedding import get_embedding_capability
from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.sqlite import cluster_repo, mint_cluster_id
from everos.memory._partition_locks import get_partition_lock
from everos.memory.events import EpisodeExtracted, ProfileClusterUpdated

logger = get_logger(__name__)


@offline_strategy(
    name="trigger_profile_clustering",
    trigger=Immediate(on=[EpisodeExtracted]),
    emits=[ProfileClusterUpdated],
    applies_to=lambda e: e.source == "pipeline",
    max_retries=2,
)
async def trigger_profile_clustering(
    event: EpisodeExtracted, ctx: StrategyContext
) -> None:
    # Body-guard: capability is checked here for defensive degradation.
    # When embedding is unavailable we cannot vectorise the episode, so
    # the strategy silently no-ops — no work, no owner lock, no OME
    # retry pressure. Tier upgrades require a server restart; this
    # guard is not a hot-reload mechanism, it just keeps dispatch clean
    # when capability was absent at engine start.
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
    if not get_embedding_capability().available:
        logger.debug(
            "strategy_gated_off_embedding_unavailable",
            strategy_name="trigger_profile_clustering",
            owner_id=event.owner_id,
        )
        return

    # Serialise on owner_id: the strategy reads the user's full cluster
    # set, picks merge target by geometry, then upserts — concurrent runs
    # on the same owner_id would race the read → decide → write cycle.
    # Different users run fully in parallel.
    # Lock per (app, project, owner): clusters are scoped to a space, so a
    # different space's run must not serialise on (or merge into) this one.
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("trigger_profile_clustering", partition):
        # 1. Embed the episode_text into a vector.
        # ``.require()`` is defensive: the body-guard above already
        # returned when the capability was missing, so this cannot raise
        # in the guarded path. Routing through the capability keeps a
        # single shared provider (one client, one semaphore) per process.
        embedder = get_embedding_capability().require()
        vector_list = await embedder.embed(event.episode_text)
        vector = np.asarray(vector_list, dtype=np.float32)

        # 2. Load this user's existing user-memory clusters (scoped to space).
        existing = await cluster_repo.list_for_owner(
            event.owner_id,
            "user_memory",
            app_id=event.app_id,
            project_id=event.project_id,
        )

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

        # 5. Persist the (possibly-merged) cluster back to SQLite.
        await cluster_repo.upsert_with_members(
            to_save,
            owner_id=event.owner_id,
            owner_type="user",
            kind="user_memory",
            member_type="episode",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 6. Emit ProfileClusterUpdated → downstream extract_user_profile.
        assert to_save.id is not None  # both branches above set id
        await ctx.emit(
            ProfileClusterUpdated(
                memcell_id=event.memcell_id,
                cluster_id=to_save.id,
                owner_id=event.owner_id,
                app_id=event.app_id,
                project_id=event.project_id,
            )
        )
    logger.info(
        "profile_cluster_updated",
        memcell_id=event.memcell_id,
        cluster_id=to_save.id,
        owner_id=event.owner_id,
        merged=merged is not None,
        cluster_count=to_save.count,
    )
