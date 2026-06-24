"""extract_atomic_facts strategy — derive AtomicFacts from an Episode.

Triggered per :class:`EpisodeExtracted` event (one per episode per
sender). Uses :meth:`AtomicFactExtractor.aextract_from_text` to extract
facts from the episode narrative. Each event carries a single
``owner_id``; all facts are written under that owner in one batched
:meth:`append_entries` call.
"""

from __future__ import annotations

from everalgo.user_memory import AtomicFactExtractor

from everos.component.llm import get_llm_client
from everos.component.utils.datetime import from_timestamp, to_iso_format
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.triggers import Immediate
from everos.infra.persistence.markdown import AtomicFactWriter
from everos.memory.events import EpisodeExtracted
from everos.memory.models import AtomicFact

logger = get_logger(__name__)

_writer: AtomicFactWriter | None = None


def _get_writer() -> AtomicFactWriter:
    """Return the lazily-initialised AtomicFactWriter singleton."""
    global _writer
    if _writer is None:
        _writer = AtomicFactWriter(root=MemoryRoot.default())
    return _writer


@offline_strategy(
    name="extract_atomic_facts",
    trigger=Immediate(on=[EpisodeExtracted]),
    emits=[],
    max_retries=2,
)
async def extract_atomic_facts(event: EpisodeExtracted, ctx: StrategyContext) -> None:
    """Extract atomic facts from an episode and persist as markdown entries."""
    # 1. Run LLM extractor on episode text.
    extractor = AtomicFactExtractor(llm=get_llm_client())
    algo_facts = await extractor.aextract_from_text(
        event.episode_text, timestamp=event.episode_timestamp_ms
    )
    if not algo_facts:
        logger.info(
            "atomic_facts_extracted",
            memcell_id=event.memcell_id,
            session_id=event.session_id,
            count=0,
            owner_id=event.owner_id,
        )
        return

    # 2. Build domain AtomicFacts (single owner from event).
    facts: list[AtomicFact] = [
        AtomicFact.from_algo(
            af,
            owner_id=event.owner_id,
            session_id=event.session_id,
            parent_id=event.episode_entry_id,
        )
        for af in algo_facts
    ]

    # 3. Write all facts in one batched append.
    writer = _get_writer()
    items = [_atomic_fact_to_entry_body(f) for f in facts]
    await writer.append_entries(
        event.owner_id, items, app_id=event.app_id, project_id=event.project_id
    )

    logger.info(
        "atomic_facts_extracted",
        memcell_id=event.memcell_id,
        session_id=event.session_id,
        count=len(facts),
        owner_id=event.owner_id,
    )


def _atomic_fact_to_entry_body(
    fact: AtomicFact,
) -> tuple[dict[str, object], dict[str, str]]:
    """Split a domain AtomicFact into ``(inline, sections)`` for md rendering.

    Mirrors ``_episode_to_entry_body`` in the user_memory pipeline. Lives in
    the memory layer (strategy module) rather than the writer (infra)
    because it depends on :class:`everos.memory.AtomicFact` — infra is
    not allowed to import memory per the layered architecture contract.
    """
    inline: dict[str, object] = {
        "owner_id": fact.owner_id,
        "timestamp": to_iso_format(from_timestamp(fact.timestamp)),
        "parent_type": "episode",
        "parent_id": fact.parent_id,
    }
    if fact.session_id is not None:
        inline["session_id"] = fact.session_id
    sections = {"Fact": fact.fact}
    return inline, sections
