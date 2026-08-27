"""trigger_decision_clustering strategy — group user decisions by topic.

Listens to :class:`DecisionExtracted` (emitted per written decision after
``extract_decision`` appends its daily-log entry), embeds the
``decision_text``, and merges the resulting size-1
:class:`everalgo.clustering.Cluster` into the owner's existing decision
cluster set.

Uses :func:`cluster_by_geometry` (embedding-only cosine + time-window).
Sqlite ``kind`` / ``member_type`` are both ``"decision"`` — not the
user-memory episode track.
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
from everos.memory.events import DecisionClusterUpdated, DecisionExtracted

logger = get_logger(__name__)


@offline_strategy(
    name="trigger_decision_clustering",
    trigger=Immediate(on=[DecisionExtracted]),
    emits=[DecisionClusterUpdated],
    applies_to=lambda e: e.source == "pipeline",
    max_retries=2,
)
async def trigger_decision_clustering(
    event: DecisionExtracted, ctx: StrategyContext
) -> None:
    # Body-guard: capability is checked here for defensive degradation.
    # When embedding is unavailable we cannot vectorise the decision, so
    # the strategy silently no-ops — no work, no owner lock, no OME
    # retry pressure. Same debug-level rationale as
    # ``trigger_profile_clustering``: per-dispatch body-guards fire on
    # every memorize under Tier 1 and must not flood structured logs.
    if not get_embedding_capability().available:
        logger.debug(
            "strategy_gated_off_embedding_unavailable",
            strategy_name="trigger_decision_clustering",
            owner_id=event.owner_id,
        )
        return

    # Serialise on owner_id: the strategy reads the owner's full cluster
    # set, picks merge target by geometry, then upserts — concurrent runs
    # on the same owner_id would race the read → decide → write cycle.
    # Different users run fully in parallel.
    # Lock per (app, project, owner): clusters are scoped to a space, so a
    # different space's run must not serialise on (or merge into) this one.
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("trigger_decision_clustering", partition):
        # 1. Embed the decision_text into a vector.
        # ``.require()`` is defensive: the body-guard above already
        # returned when the capability was missing, so this cannot raise
        # in the guarded path. Routing through the capability keeps a
        # single shared provider (one client, one semaphore) per process.
        embedder = get_embedding_capability().require()
        vector_list = await embedder.embed(event.decision_text)
        vector = np.asarray(vector_list, dtype=np.float32)

        # 2. Load this owner's existing decision clusters (scoped to space).
        existing = await cluster_repo.list_for_owner(
            event.owner_id,
            "decision",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 3. Build a size-1 cluster for the new decision.
        new_cluster = AlgoCluster(
            id=mint_cluster_id(),
            centroid=vector,
            count=1,
            last_ts=event.decision_timestamp_ms,
            preview=[event.decision_text],
            members=[event.decision_entry_id],
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
            kind="decision",
            member_type="decision",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 6. Emit DecisionClusterUpdated with a row snapshot so a later
        # principle extractor can consume the triggering decision without
        # racing cascade / polling LanceDB.
        assert to_save.id is not None  # both branches above set id
        await ctx.emit(
            DecisionClusterUpdated(
                memcell_id=event.memcell_id,
                decision_entry_id=event.decision_entry_id,
                cluster_id=to_save.id,
                owner_id=event.owner_id,
                app_id=event.app_id,
                project_id=event.project_id,
                title=event.title,
                decision_text=event.decision_text,
                reason=event.reason,
                impact=event.impact,
                tags=list(event.tags),
                decision_timestamp_ms=event.decision_timestamp_ms,
            )
        )
    logger.info(
        "decision_cluster_updated",
        memcell_id=event.memcell_id,
        cluster_id=to_save.id,
        owner_id=event.owner_id,
        merged=merged is not None,
        cluster_count=to_save.count,
    )
