"""split_episode_by_owner strategy — one event per owner's copy of an episode.

:class:`EpisodeExtracted` is per memcell: one extraction, copied to every
user sender. Work that reads and rewrites one owner's state (the profile)
needs one run per owner instead, so each owner is serialised on its own lock
and retried on its own. This strategy is that split and nothing else: no
model call, no storage access.

Pipeline episodes only: Reflection's merged episodes are not profiled. A
run replayed after a crash re-emits every copy; the profile accepts that
rare repeat like any other at-least-once delivery.
"""

from __future__ import annotations

from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted, EpisodeExtractedForOwner
from everos.memory.ports import Event, Immediate, StrategySpec


async def split_episode_by_owner(
    event: EpisodeExtracted, deps: MemoryDeps
) -> list[Event]:
    """One :class:`EpisodeExtractedForOwner` per copy, in the event's order."""
    return [
        EpisodeExtractedForOwner(
            memcell_id=event.memcell_id,
            episode_entry_id=copy.entry_id,
            owner_id=copy.owner_id,
            owner_name=copy.owner_name,
            app_id=event.app_id,
            project_id=event.project_id,
            language=event.language,
        )
        for copy in event.copies
    ]


SPLIT_EPISODE_BY_OWNER_SPEC = StrategySpec(
    name="split_episode_by_owner",
    fn=split_episode_by_owner,
    trigger=Immediate(on=[EpisodeExtracted]),
    emits=frozenset({EpisodeExtractedForOwner}),
    applies_to=lambda event, _deps: event.source == "pipeline",
    max_retries=2,
)
