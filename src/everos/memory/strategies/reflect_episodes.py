"""reflect_episodes Cron strategy — nightly Reflection consolidation.

Triggered by a cron schedule (default: ``0 2 * * 1``). Enumerates all
distinct owner scopes from the cluster table and runs the
:class:`ReflectionOrchestrator` for each. Configuration lives in
``[reflection]`` of ``config/default.toml``.

The strategy is a thin entry point: it constructs the orchestrator with
production singletons and iterates over owners. All business logic
lives in :mod:`everos.memory.reflection.orchestrator`.
"""

from __future__ import annotations

import asyncio

from everos.component.embedding import get_embedder
from everos.component.llm import get_llm_client
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.events import CronTick
from everos.infra.ome.triggers import Cron
from everos.infra.persistence.lancedb import (
    atomic_fact_repo,
    episode_repo,
)
from everos.infra.persistence.markdown import EpisodeWriter
from everos.infra.persistence.sqlite import (
    cluster_repo,
    reflection_report_repo,
)
from everos.memory.events import EpisodeExtracted
from everos.memory.reflection import ReflectionOrchestrator

logger = get_logger(__name__)

_episode_writer: EpisodeWriter | None = None


def _get_episode_writer() -> EpisodeWriter:
    """Return the lazily-initialised EpisodeWriter singleton."""
    global _episode_writer
    if _episode_writer is None:
        _episode_writer = EpisodeWriter(root=MemoryRoot.default())
    return _episode_writer


@offline_strategy(
    name="reflect_episodes",
    trigger=Cron(expr="0 2 * * 1"),
    emits=[EpisodeExtracted],
    enabled=False,
    max_retries=1,
)
async def reflect_episodes(event: CronTick, ctx: StrategyContext) -> None:
    """Run Reflection for all owner scopes.

    Args:
        event: Cron tick event (unused; triggers the scheduled run).
        ctx: OME strategy context for emit and logging.
    """
    # Deferred: avoid pulling LLM libs at module import time.
    from everalgo.user_memory import EpisodeReflector

    orchestrator = ReflectionOrchestrator(
        cluster_repo=cluster_repo,
        episode_store=episode_repo,
        atomic_fact_store=atomic_fact_repo,
        episode_writer=_get_episode_writer(),
        report_repo=reflection_report_repo,
        reflector=EpisodeReflector(llm=get_llm_client()),
        embedder=get_embedder(),
    )

    owners = await cluster_repo.list_distinct_owners()
    await asyncio.gather(
        *(
            orchestrator.run(
                ctx=ctx,
                owner_id=owner_id,
                owner_type=owner_type,
                app_id=app_id,
                project_id=project_id,
            )
            for owner_id, owner_type, app_id, project_id in owners
        )
    )
