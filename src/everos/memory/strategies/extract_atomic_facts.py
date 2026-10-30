"""extract_atomic_facts strategy — derive AtomicFacts from an Episode.

Triggered per :class:`EpisodeExtracted` event (one per episode per
sender). Uses :meth:`AtomicFactExtractor.aextract_from_text` to extract
facts from the episode narrative. Each event carries a single
``owner_id``; all facts are written under that owner in one batched
append.

Always returns one :class:`AtomicFactsExtracted`, even for an episode that
yields no facts: Reflection finishes a merge on that event, and it carries
the event's Reflection context through unchanged.
"""

from __future__ import annotations

from everalgo.user_memory import AtomicFactExtractor

from everos.core.observability.logging import get_logger
from everos.memory.deps import MemoryDeps
from everos.memory.events import AtomicFactsExtracted, EpisodeExtracted
from everos.memory.models import AtomicFact
from everos.memory.ports import Event, Immediate, Scope, StrategySpec

logger = get_logger(__name__)


async def extract_atomic_facts(
    event: EpisodeExtracted, deps: MemoryDeps
) -> list[Event]:
    """Extract atomic facts from an episode and persist them under its owner."""
    # 1. Run LLM extractor on episode text.
    extractor = AtomicFactExtractor(llm=deps.llm)
    algo_facts = await extractor.aextract_from_text(
        event.episode_text,
        timestamp=event.episode_timestamp_ms,
        output_language=event.language,
    )

    # 2. Build domain AtomicFacts (single owner from event).
    facts = [
        AtomicFact.from_algo(
            af,
            owner_id=event.owner_id,
            session_id=event.session_id,
            parent_id=event.episode_entry_id,
        )
        for af in algo_facts
    ]

    # 3. Write all facts in one batched append.
    refs = (
        await deps.stores.atomic_fact.append_many(_scope(event), facts) if facts else []
    )

    logger.info(
        "atomic_facts_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        count=len(facts),
        owner_id=event.owner_id,
    )
    return [
        AtomicFactsExtracted(
            episode_entry_id=event.episode_entry_id,
            owner_id=event.owner_id,
            app_id=event.app_id,
            project_id=event.project_id,
            source=event.source,
            fact_entry_ids=[ref.entry_id for ref in refs],
            reflection=event.reflection,
        )
    ]


def _scope(event: EpisodeExtracted) -> Scope:
    return Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.owner_id,
        owner_type="user",
    )


EXTRACT_ATOMIC_FACTS_SPEC = StrategySpec(
    name="extract_atomic_facts",
    fn=extract_atomic_facts,
    trigger=Immediate(on=[EpisodeExtracted]),
    emits=frozenset({AtomicFactsExtracted}),
    max_retries=2,
)
