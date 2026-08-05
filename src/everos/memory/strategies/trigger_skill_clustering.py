"""trigger_skill_clustering strategy — group fresh AgentCases into clusters.

Listens to :class:`AgentCaseExtracted`, embeds the case's ``task_intent``,
and merges the resulting size-1 :class:`everalgo.clustering.Cluster` into
the agent's existing cluster set. Cluster identity is the algo's
"caller-supplied id"; the strategy mints one upfront via
:func:`mint_cluster_id` so the value flows unchanged whether the algo
merges into an existing cluster (id transparently swapped to the existing
cluster's id by ``_merge``) or returns ``None`` (the brand-new id is used
as-is when persisting).

Skill-track parity with opensource: uses :func:`cluster_by_llm` (rather
than the geometry-only variant) — opensource routes ``has_case=True``
memcells through the LLM-refined path. The low-quality short-circuit
(``quality_score < 0.2``) mirrors :class:`AgentSkillExtractor`'s own
``skip_quality_threshold`` so we avoid both an embedding call and an LLM
ranking call for cases that won't drive a skill anyway.
"""

from __future__ import annotations

import numpy as np
from everalgo.clustering import Cluster as AlgoCluster
from everalgo.clustering import cluster_by_llm

from everos.component.embedding import get_embedding_capability
from everos.component.llm import get_llm_client
from everos.core.observability.logging import get_logger
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.sqlite import cluster_repo, mint_cluster_id
from everos.memory._partition_locks import get_partition_lock
from everos.memory.events import AgentCaseExtracted, SkillClusterUpdated

logger = get_logger(__name__)

_SKIP_QUALITY_THRESHOLD = 0.2
"""Lower bound mirroring :attr:`AgentSkillExtractor.skip_quality_threshold`;
cases below this never produce a skill, so we don't bother clustering them."""


@offline_strategy(
    name="trigger_skill_clustering",
    trigger=Immediate(on=[AgentCaseExtracted]),
    emits=[SkillClusterUpdated],
    max_retries=2,
)
async def trigger_skill_clustering(
    event: AgentCaseExtracted, ctx: StrategyContext
) -> None:
    # Body-guard: capability is checked here for defensive degradation.
    # When embedding is unavailable we cannot vectorise the case, so
    # the strategy silently no-ops — no work, no agent lock, no OME
    # retry pressure. Tier upgrades require a server restart; this
    # guard is not a hot-reload mechanism.
    #
    # ``debug`` level (not ``info``) is intentional; see the body-guard
    # in :func:`everos.memory.strategies.trigger_profile_clustering` for
    # the shared rationale (PR #361 round-3 review #8).
    if not get_embedding_capability().available:
        logger.debug(
            "strategy_gated_off_embedding_unavailable",
            strategy_name="trigger_skill_clustering",
            agent_id=event.agent_id,
        )
        return

    # Serialise on agent_id: the strategy reads the agent's full cluster
    # set, lets the LLM decide merge vs. mint, then upserts — concurrent
    # runs on the same agent_id would race the read → decide → write
    # cycle. Different agents run fully in parallel.
    # Lock per (app, project, agent): clusters are scoped to a space.
    partition = f"{event.app_id}:{event.project_id}:{event.agent_id}"
    async with get_partition_lock("trigger_skill_clustering", partition):
        # 1. Drop low-quality cases — they won't yield a skill anyway.
        if event.quality_score < _SKIP_QUALITY_THRESHOLD:
            logger.info(
                "skill_clustering_skipped_low_quality",
                case_entry_id=event.case_entry_id,
                quality_score=event.quality_score,
                threshold=_SKIP_QUALITY_THRESHOLD,
            )
            return

        # 2. Embed the case's task_intent into a vector.
        # ``.require()`` is defensive: the body-guard above already
        # returned when the capability was missing, so this cannot raise
        # in the guarded path. Routing through the capability keeps a
        # single shared provider (one client, one semaphore) per process.
        embedder = get_embedding_capability().require()
        vector_list = await embedder.embed(event.task_intent)
        vector = np.asarray(vector_list, dtype=np.float32)

        # 3. Load this agent's existing skill clusters (scoped to space).
        existing = await cluster_repo.list_for_owner(
            event.agent_id,
            "agent_case",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 4. Build a size-1 cluster for the fresh case (id minted upfront).
        new_cluster = AlgoCluster(
            id=mint_cluster_id(),
            centroid=vector,
            count=1,
            last_ts=event.case_timestamp_ms,
            preview=[event.task_intent],
            members=[event.case_entry_id],
        )

        # 5. Ask the LLM to merge it into an existing cluster (or keep as-is).
        merged = await cluster_by_llm(new_cluster, existing, llm=get_llm_client())
        to_save = merged if merged is not None else new_cluster

        # 6. Persist the (possibly-merged) cluster back to SQLite.
        await cluster_repo.upsert_with_members(
            to_save,
            owner_id=event.agent_id,
            owner_type="agent",
            kind="agent_case",
            member_type="case",
            app_id=event.app_id,
            project_id=event.project_id,
        )

        # 7. Emit SkillClusterUpdated → downstream extract_agent_skill.
        assert to_save.id is not None  # both branches above set id
        await ctx.emit(
            SkillClusterUpdated(
                case_entry_id=event.case_entry_id,
                cluster_id=to_save.id,
                agent_id=event.agent_id,
                app_id=event.app_id,
                project_id=event.project_id,
                task_intent=event.task_intent,
                approach=event.approach,
                key_insight=event.key_insight,
                quality_score=event.quality_score,
                case_timestamp_ms=event.case_timestamp_ms,
                case_vector=vector_list,
            )
        )
    logger.info(
        "skill_cluster_updated",
        case_entry_id=event.case_entry_id,
        cluster_id=to_save.id,
        agent_id=event.agent_id,
        merged=merged is not None,
        cluster_count=to_save.count,
    )
