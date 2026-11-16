"""summarize_episode strategy — write an episode's model preview offline.

The pipeline writes every copy of an episode with
:func:`~everos.memory.episode_summary.placeholder_summary` as its summary,
so a memcell costs one model call online. This strategy writes the real
preview afterwards:

1. **Skip what has landed.** The placeholder is recomputed from the event's
   ``episode_text``; when no copy still shows it (an edit, an earlier
   delivery), no model call is made.
2. **One preview for every copy.** ``EpisodeSummarizer.asummarize`` reads
   the episode text -- what the user reads -- and writes in the episode's
   language (``event.language``), with the ``summary`` task's model.
3. **Compare-and-set.** A copy is replaced only while its summary still
   equals the placeholder, so a summary edited meanwhile stays.

Pipeline episodes only: Reflection's merged episode carries the reflector's
own summary. A failure propagates for OME's retries; there is no fallback
cut, so once they are spent the placeholder stays.
"""

from __future__ import annotations

import asyncio

from everalgo.user_memory import EpisodeSummarizer

from everos.core.observability.logging import get_logger
from everos.memory.deps import MemoryDeps
from everos.memory.episode_summary import placeholder_summary
from everos.memory.events import EpisodeExtracted
from everos.memory.ports import Event, Immediate, StrategySpec

logger = get_logger(__name__)


async def summarize_episode(event: EpisodeExtracted, deps: MemoryDeps) -> list[Event]:
    """Replace the placeholder summary of every copy with the model's preview."""
    placeholder = placeholder_summary(event.episode_text)
    if not await _any_copy_shows(event, placeholder, deps):
        logger.info("episode_summary_skipped_landed", memcell_id=event.memcell_id)
        return []
    summary = await _write_preview(event, deps)
    replaced = await _replace_placeholders(event, placeholder, summary, deps)
    logger.info(
        "episode_summary_written",
        memcell_id=event.memcell_id,
        copies=replaced,
        language=event.language,
    )
    return []


async def _any_copy_shows(
    event: EpisodeExtracted, placeholder: str, deps: MemoryDeps
) -> bool:
    """Whether a copy is still in md with ``placeholder`` as its summary."""
    stored = await asyncio.gather(
        *(
            deps.stores.episode.read_many(event.owner_scope(copy), [copy.entry_id])
            for copy in event.copies
        )
    )
    return any(s.item.summary == placeholder for found in stored for s in found)


async def _write_preview(event: EpisodeExtracted, deps: MemoryDeps) -> str:
    """The model's preview of the episode text, in the episode's language."""
    summarizer = EpisodeSummarizer(llm=deps.llms.summary)
    summary = await summarizer.asummarize(
        event.episode_text, output_language=event.language
    )
    return summary.strip()


async def _replace_placeholders(
    event: EpisodeExtracted, placeholder: str, summary: str, deps: MemoryDeps
) -> int:
    """Swap ``summary`` in on every copy still showing ``placeholder``; how many."""
    replaced = await asyncio.gather(
        *(
            deps.stores.episode.replace_summary(
                event.owner_scope(copy),
                copy.entry_id,
                expected=placeholder,
                summary=summary,
            )
            for copy in event.copies
        )
    )
    return sum(replaced)


SUMMARIZE_EPISODE_SPEC = StrategySpec(
    name="summarize_episode",
    fn=summarize_episode,
    trigger=Immediate(on=[EpisodeExtracted]),
    applies_to=lambda event, _deps: event.source == "pipeline",
    max_retries=2,
)
