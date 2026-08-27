"""reflect_decisions Cron strategy — weekly Decision consolidation.

Triggered by a cron schedule (default: ``0 2 * * 1``). Enumerates all
distinct owner scopes from the cluster table and runs the
:class:`DecisionReflectionOrchestrator` for each. Disabled by default
(same cadence as ``reflect_episodes``).

The strategy is a thin entry point: it constructs the orchestrator with
production singletons and iterates over owners. All business logic
lives in :mod:`everos.memory.reflection.decision_orchestrator`.
"""

from __future__ import annotations

import asyncio

from everos.component.embedding import get_embedding_capability
from everos.component.llm import get_llm_client
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.events import CronTick
from everos.infra.ome.triggers import Cron
from everos.infra.persistence.lancedb import decision_repo
from everos.infra.persistence.markdown import DecisionWriter
from everos.infra.persistence.sqlite import (
    cluster_repo,
    reflection_report_repo,
)
from everos.memory.events import DecisionExtracted
from everos.memory.reflection import DecisionReflectionOrchestrator

logger = get_logger(__name__)

_writer: DecisionWriter | None = None


def _get_writer() -> DecisionWriter:
    """Return the lazily-initialised DecisionWriter singleton."""
    global _writer
    if _writer is None:
        _writer = DecisionWriter(root=MemoryRoot.resolve())
    return _writer


@offline_strategy(
    name="reflect_decisions",
    trigger=Cron(expr="0 2 * * 1"),
    emits=[DecisionExtracted],
    enabled=False,
    max_retries=1,
)
async def reflect_decisions(event: CronTick, ctx: StrategyContext) -> None:
    """Run Decision Reflection for all owner scopes.

    Args:
        event: Cron tick event (unused; triggers the scheduled run).
        ctx: OME strategy context for emit and logging.
    """
    # Body-guard: capability is checked here for defensive degradation.
    # Reflection re-embeds merged decision text, so it cannot run without
    # an embedder — silently no-op instead of raising deep inside the
    # orchestrator. Tier upgrades require a server restart; this guard
    # is not a hot-reload mechanism.
    if not get_embedding_capability().available:
        logger.debug(
            "strategy_gated_off_embedding_unavailable",
            strategy_name="reflect_decisions",
        )
        return

    from everalgo.user_memory import DecisionReflector

    orchestrator = DecisionReflectionOrchestrator(
        cluster_repo=cluster_repo,
        decision_store=decision_repo,
        decision_writer=_get_writer(),
        report_repo=reflection_report_repo,
        reflector=DecisionReflector(llm=get_llm_client()),
        embedder=get_embedding_capability().require(),
    )

    owners = await cluster_repo.list_distinct_owners()
    await asyncio.gather(
        *(
            orchestrator.run(
                ctx=ctx,
                owner_id=owner_id,
                owner_type=owner_type,
                kind="decision",
                app_id=app_id,
                project_id=project_id,
            )
            for owner_id, owner_type, app_id, project_id in owners
        )
    )
