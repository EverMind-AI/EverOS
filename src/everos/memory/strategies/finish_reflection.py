"""finish_reflection strategy — retire what a Reflection merge replaced.

Second half of Reflection (see :mod:`everos.memory.reflection.orchestrator`).
``reflect_episodes`` writes a merged episode and publishes it as
``EpisodeExtracted(source="reflection")``; ``extract_atomic_facts``
extracts its facts and publishes ``AtomicFactsExtracted`` carrying the
same :class:`~everos.memory.events.ReflectionContext`. This strategy fires
on that event — so the originals are deprecated only after the merged
episode's facts exist — and hands the context to
:class:`~everos.memory.reflection.ReflectionFinisher`.

Pipeline episodes also produce ``AtomicFactsExtracted``; ``applies_to``
keeps them from ever dispatching a run here.
"""

from __future__ import annotations

from everos.core.observability.logging import get_logger
from everos.memory.deps import MemoryDeps
from everos.memory.events import AtomicFactsExtracted
from everos.memory.ports import Event, Immediate, Scope, StrategySpec
from everos.memory.reflection import ReflectionFinisher

logger = get_logger(__name__)


def _finishes_a_merge(event: AtomicFactsExtracted, deps: MemoryDeps) -> bool:
    return event.source == "reflection" and event.reflection is not None


async def finish_reflection(
    event: AtomicFactsExtracted, deps: MemoryDeps
) -> list[Event]:
    """Deprecate the merged cluster's replaced members and record the merge."""
    context = event.reflection
    assert context is not None  # applies_to admits only reflection events
    # Body-guard: the cluster centroid is re-embedded from the merged text.
    # Reflection only merges with an embedder, so this trips only when the
    # provider was removed between the merge and this run (a restart).
    if deps.embedding is None:
        logger.debug(
            "strategy_gated_off_embedding_unavailable",
            strategy_name="finish_reflection",
            cluster_id=context.cluster_id,
        )
        return []

    finisher = ReflectionFinisher(
        cluster_store=deps.stores.cluster,
        episode_store=deps.stores.episode,
        atomic_fact_store=deps.stores.atomic_fact,
        atomic_fact_index=deps.indexes.atomic_fact,
        report_store=deps.stores.reflection_report,
        embedder=deps.embedding,
    )
    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.owner_id,
        owner_type="user",
    )
    await finisher.finish(
        scope=scope, merged_entry_id=event.episode_entry_id, context=context
    )
    return []


FINISH_REFLECTION_SPEC = StrategySpec(
    name="finish_reflection",
    fn=finish_reflection,
    trigger=Immediate(on=[AtomicFactsExtracted]),
    applies_to=_finishes_a_merge,
    max_retries=1,
)
