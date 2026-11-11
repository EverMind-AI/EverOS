"""extract_user_profile strategy — fold one new episode into its owner's profile.

Fires on every pipeline :class:`EpisodeExtracted` and hands that one episode's
narrative to everalgo's ``ProfileExtractor.aextract_from_episodes``: INIT when
the owner has no ``user.md`` yet, UPDATE on the stored profile otherwise.

Invariants a change must keep:

- **One owner per event.** A memcell with several user senders yields the
  same narrative once per sender, each copy in its own owner's files with its
  own ``EpisodeExtracted``; each copy updates only that owner's profile.
- **The narrative names the owner by the label it was written with.** The
  pipeline puts that label on the event (``owner_name``: the owner's
  ``sender_name``, else their id); an event without one falls back to
  ``owner_id``.
- **Sources are read back, not trusted from the event.** The episode comes
  from md (the truth), and its memcell must still exist; either one gone (the
  user deleted it before this run) skips the run.
- **Reflection's merged episodes are not profiled**: their source episodes
  already were.
- **At least once.** OME re-enqueues a run that was RUNNING when the process
  died, so a crash after the write processes its episode a second time; that
  rare repeat is accepted rather than tracked.
- **Serialised per owner**: ``user.md`` is a read -> LLM merge -> overwrite
  sequence. Different owners run in parallel.
"""

from __future__ import annotations

import asyncio

from everalgo.types import Profile as AlgoProfile
from everalgo.user_memory import ProfileExtractor
from everalgo.utils import InvalidInputError as AlgoInvalidInputError

from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import get_partition_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.mappers import episode as episode_mapper
from everos.memory.mappers import profile as profile_mapper
from everos.memory.models import Episode, Profile
from everos.memory.ports import Event, Immediate, Scope, StrategySpec

logger = get_logger(__name__)


async def extract_user_profile(
    event: EpisodeExtracted, deps: MemoryDeps
) -> list[Event]:
    """Fold the event's episode into its owner's profile document."""
    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.owner_id,
        owner_type="user",
    )
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("extract_user_profile", partition):
        episode = await _read_episode(event, scope, deps)
        if episode is None:
            return []
        existing = await deps.stores.profile.read(scope)
        language = _profile_language(existing, event)
        new_profile = await _extract(
            event, deps, episode=episode, existing=existing, language=language
        )
        if new_profile is None:
            return []
        await deps.stores.profile.write(
            scope, profile_mapper.from_algo(new_profile, primary_language=language)
        )
    logger.info(
        "user_profile_extracted",
        owner_id=event.owner_id,
        episode_entry_id=event.episode_entry_id,
        mode="UPDATE" if existing is not None else "INIT",
    )
    return []


async def _read_episode(
    event: EpisodeExtracted, scope: Scope, deps: MemoryDeps
) -> Episode | None:
    """The event's episode from md; ``None`` when it or its memcell is gone."""
    episodes, memcells = await asyncio.gather(
        deps.stores.episode.read_many(scope, [event.episode_entry_id]),
        deps.stores.memcell.find_by_ids([event.memcell_id]),
    )
    if not episodes or not memcells:
        logger.info(
            "profile_skipped_source_gone",
            owner_id=event.owner_id,
            episode_entry_id=event.episode_entry_id,
            memcell_id=event.memcell_id,
        )
        return None
    return episodes[0].item


async def _extract(
    event: EpisodeExtracted,
    deps: MemoryDeps,
    *,
    episode: Episode,
    existing: Profile | None,
    language: str | None,
) -> AlgoProfile | None:
    """INIT or UPDATE the owner's profile from the episode; ``None`` if refused."""
    try:
        return await ProfileExtractor(llm=deps.profile_llm).aextract_from_episodes(
            [episode_mapper.to_algo(episode)],
            owner_id=event.owner_id,
            owner_name=event.owner_name or event.owner_id,
            old_profile=profile_mapper.to_algo(existing) if existing else None,
            custom_instructions=deps.instructions.load("profile_extract"),
            output_language=language,
        )
    except AlgoInvalidInputError as exc:
        # The narrative does not name the owner (or is otherwise unusable):
        # a retry would read the same text, so the episode is skipped.
        logger.warning(
            "profile_episode_skipped",
            owner_id=event.owner_id,
            episode_entry_id=event.episode_entry_id,
            reason=str(exc),
        )
        return None


def _profile_language(existing: Profile | None, event: EpisodeExtracted) -> str | None:
    """The language to write the profile in, frozen once known.

    A stored profile's ``primary_language`` wins and is never changed
    automatically; changing it is a manual edit. Until one is stored, the
    triggering episode's language is used and then frozen.
    """
    if existing is not None and existing.primary_language:
        return existing.primary_language
    return event.language


EXTRACT_USER_PROFILE_SPEC = StrategySpec(
    name="extract_user_profile",
    fn=extract_user_profile,
    trigger=Immediate(on=[EpisodeExtracted]),
    applies_to=lambda event, _deps: event.source == "pipeline",
    max_retries=2,
)
