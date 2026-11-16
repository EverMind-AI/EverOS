"""extract_atomic_facts strategy — derive AtomicFacts from an Episode.

Triggered once per :class:`EpisodeExtracted`, that is once per memcell
episode. :meth:`AtomicFactExtractor.aextract_from_text` runs once over the
episode text, in the episode's language, and the facts are written under
every copy's owner, parented to that owner's copy of the episode: the copies
are one extraction fanned out per sender, so per-owner calls would pay N
times for the same answer.

Each owner's write keeps its own source check, so a copy deleted while the
model ran gets no facts while the others do. Returns one
:class:`AtomicFactsExtracted` per copy, even for a copy whose episode yielded
no facts or whose write was skipped: Reflection finishes a merge on that
event (its merged episode is a one-copy event), and the event carries the
Reflection context through unchanged.

A failing write fails the run; OME's retry extracts again and appends to
every owner again, the owners already written included: like every append,
facts are written at least once, not exactly once.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from everalgo.types import AtomicFact as AlgoAtomicFact
from everalgo.user_memory import AtomicFactExtractor

from everos.core.observability.logging import get_logger
from everos.memory.deps import MemoryDeps
from everos.memory.events import AtomicFactsExtracted, EpisodeCopy, EpisodeExtracted
from everos.memory.models import AtomicFact
from everos.memory.ports import Event, Immediate, StrategySpec

logger = get_logger(__name__)


async def extract_atomic_facts(
    event: EpisodeExtracted, deps: MemoryDeps
) -> list[Event]:
    """Extract the episode's facts once and write them under every copy's owner."""
    extractor = AtomicFactExtractor(llm=deps.llms.atomic_fact)
    algo_facts = await extractor.aextract_from_text(
        event.episode_text,
        timestamp=event.episode_timestamp_ms,
        output_language=event.language,
        tz=deps.tz,
    )
    done = await asyncio.gather(
        *(_write_for_copy(event, copy, algo_facts, deps) for copy in event.copies)
    )
    logger.info(
        "atomic_facts_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        count=len(algo_facts),
        owners=len(event.copies),
    )
    return list(done)


async def _write_for_copy(
    event: EpisodeExtracted,
    copy: EpisodeCopy,
    algo_facts: Sequence[AlgoAtomicFact],
    deps: MemoryDeps,
) -> AtomicFactsExtracted:
    """Write the facts under one copy's owner, if its sources still exist.

    The sources are the owner's copy of the episode in md (a single-id
    deletion removes it there) and, for a pipeline episode, its memcell (a
    session deletion removes that first, before it reaches this owner's
    files). A Reflection event carries the merged episode's id as
    ``memcell_id``.
    """
    scope = event.owner_scope(copy)
    facts = [
        AtomicFact.from_algo(
            af,
            owner_id=copy.owner_id,
            session_id=event.session_id,
            parent_id=copy.entry_id,
        )
        for af in algo_facts
    ]

    async def sources_alive() -> bool:
        if not await deps.stores.episode.exists(scope, [copy.entry_id]):
            return False
        if event.source == "reflection":
            return True
        return bool(await deps.stores.memcell.find_by_ids([event.memcell_id]))

    refs = await deps.stores.atomic_fact.append_many(
        scope, facts, only_if=sources_alive
    )
    return AtomicFactsExtracted(
        episode_entry_id=copy.entry_id,
        owner_id=copy.owner_id,
        app_id=event.app_id,
        project_id=event.project_id,
        source=event.source,
        fact_entry_ids=[ref.entry_id for ref in refs],
        reflection=event.reflection,
    )


EXTRACT_ATOMIC_FACTS_SPEC = StrategySpec(
    name="extract_atomic_facts",
    fn=extract_atomic_facts,
    trigger=Immediate(on=[EpisodeExtracted]),
    emits=frozenset({AtomicFactsExtracted}),
    max_retries=2,
)
