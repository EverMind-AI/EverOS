"""trigger_profile_clustering strategy — group user memcells by episode topic.

Listens to :class:`EpisodeExtracted` (emitted per-episode after the user
pipeline writes its md), embeds the ``episode_text``, and merges the
resulting size-1 :class:`everalgo.clustering.Cluster` into the user's
existing user-memory cluster set.

Profile-track parity with opensource: uses :func:`cluster_by_geometry`
(rather than the LLM-refined variant) — opensource routes
``has_case=False`` (user-memory) memcells through the embedding-only
path. The members on the merged cluster are ``memcell_id`` rather than
``episode_entry_id`` because the downstream profile-extraction step
needs to feed full memcells (chat messages) back into
:class:`everalgo.user_memory.ProfileExtractor`, not the per-sender
episode summaries.
"""

from __future__ import annotations

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.clustering import cluster_by_geometry

from everos.component.embedding import get_embedder
from everos.core.observability.logging import get_logger
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.sqlite import cluster_repo, mint_cluster_id
from everos.memory.events import EpisodeExtracted, ProfileClusterUpdated
from everos.memory.strategies._partition_locks import get_partition_lock

logger = get_logger(__name__)


@offline_strategy(
    name="trigger_profile_clustering",
    trigger=Immediate(on=[EpisodeExtracted]),
    emits=[ProfileClusterUpdated],
    max_retries=2,
)
async def trigger_profile_clustering(
    event: EpisodeExtracted, ctx: StrategyContext
) -> None:
    # Serialise on owner_id: the strategy reads the user's full cluster
    # set, picks merge target by geometry, then upserts — concurrent runs
    # on the same owner_id would race the read → decide → write cycle.
    # Different users run fully in parallel.
    # Lock per (app, project, owner): clusters are scoped to a space, so a
    # different space's run must not serialise on (or merge into) this one.
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("trigger_profile_clustering", partition):
        # 1. Embed the episode_text into a vector.
        vector_list = await get_embedder().embed(event.episode_text)
        vector = np.asarray(vector_list, dtype=np.float32)

        # 2. Load this user's existing user-memory clusters (scoped to space).
        existing = await cluster_repo.list_for_owner(
            event.owner_id,
            "user_memory",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 3. Build a size-1 cluster for the fresh memcell (id minted upfront).
        new_cluster = AlgoCluster(
            id=mint_cluster_id(),
            centroid=vector,
            count=1,
            last_ts=event.episode_timestamp_ms,
            preview=[event.episode_text],
            members=[event.memcell_id],
        )

        # 4. Geometry-merge it into an existing cluster (or keep as-is).
        # ``cluster_by_geometry`` is a pure synchronous CPU function (cosine +
        # time-window math, no I/O) returning ``Cluster | None`` directly, so
        # it must not be awaited (``await None`` raises when there is no
        # existing cluster to merge into).
        merged = cluster_by_geometry(new_cluster, existing)
        to_save = merged if merged is not None else new_cluster

        # 5. Persist the (possibly-merged) cluster back to SQLite.
        await cluster_repo.upsert_with_members(
            to_save,
            owner_id=event.owner_id,
            owner_type="user",
            kind="user_memory",
            member_type="memcell",
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
