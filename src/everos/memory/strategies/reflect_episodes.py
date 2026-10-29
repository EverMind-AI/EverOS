"""reflect_episodes Cron strategy — nightly Reflection consolidation.

Triggered by a cron schedule (default: ``0 2 * * 1``). Enumerates all
distinct owner scopes from the cluster store and runs the
:class:`ReflectionOrchestrator` for each, returning one
``EpisodeExtracted(source="reflection")`` per merged cluster. The merge is
finished by :mod:`.finish_reflection` once the merged episode's atomic
facts exist. Configuration lives in ``[reflection]`` of
``config/default.toml``.

The strategy is a thin entry point; all business logic lives in
:mod:`everos.memory.reflection.orchestrator`.
"""

from __future__ import annotations

import asyncio

from everos.core.observability.logging import get_logger
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.ports import Cron, CronTick, Event, StrategySpec
from everos.memory.reflection import ReflectionOrchestrator

logger = get_logger(__name__)


async def reflect_episodes(event: CronTick, deps: MemoryDeps) -> list[Event]:
    """Merge the eligible clusters of every owner scope.

    Args:
        event: Cron tick event (unused; triggers the scheduled run).
        deps: Stores, indexes and providers the merge works through.

    Returns:
        One ``EpisodeExtracted`` per merged cluster.
    """
    # Body-guard: Reflection re-embeds merged narratives (in its finishing
    # step), so it cannot run without an embedder — silently no-op instead
    # of raising deep inside the orchestrator. Tier upgrades require a
    # server restart; this guard is not a hot-reload mechanism.
    #
    # ``debug`` level (not ``info``) is intentional; see the body-guard
    # in :func:`everos.memory.strategies.trigger_profile_clustering` for
    # the shared rationale (PR #361 round-3 review #8).
    if deps.embedding is None:
        logger.debug(
            "strategy_gated_off_embedding_unavailable",
            strategy_name="reflect_episodes",
        )
        return []

    # Deferred: avoid pulling LLM libs at module import time.
    from everalgo.user_memory import EpisodeReflector

    orchestrator = ReflectionOrchestrator(
        cluster_store=deps.stores.cluster,
        episode_store=deps.stores.episode,
        episode_index=deps.indexes.episode,
        report_store=deps.stores.reflection_report,
        reflector=EpisodeReflector(llm=deps.llm),
        custom_instructions=deps.instructions.load("reflect"),
    )
    owners = await deps.stores.cluster.list_distinct_owners()
    merged = await asyncio.gather(*(orchestrator.run(scope=scope) for scope in owners))
    return [event for per_owner in merged for event in per_owner]


REFLECT_EPISODES_SPEC = StrategySpec(
    name="reflect_episodes",
    fn=reflect_episodes,
    trigger=Cron(expr="0 2 * * 1"),
    emits=frozenset({EpisodeExtracted}),
    enabled=False,
    max_retries=1,
)
