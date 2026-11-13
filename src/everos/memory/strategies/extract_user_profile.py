"""extract_user_profile strategy — fold new episodes into their owner's profile.

Fires on every pipeline :class:`EpisodeExtracted` and hands the event's episode,
with up to ``max_episodes_per_call - 1`` of the owner's newer episodes still
waiting, to everalgo's ``ProfileExtractor.aextract_from_episodes`` in one call:
INIT when the owner has no ``user.md`` yet, UPDATE on the stored profile
otherwise. The runs of the episodes folded along find them marked and end
without a call, so a backlog drains in a fraction of the calls.

Invariants a change must keep:

- **One owner per event.** A memcell with several user senders yields the
  same narrative once per sender, each copy in its own owner's files with its
  own ``EpisodeExtracted``; each copy updates only that owner's profile.
- **The narrative names the owner by the label it was written with.** The
  pipeline puts that label on the event (``owner_name``: the owner's
  ``sender_name``, else their id); an event without one falls back to
  ``owner_id``.
- **Every episode is folded by its own run unless another run folded it.**
  The waiting episodes are the owner's pipeline episodes the index holds
  newer than ``Profile.timestamp`` (the newest observation folded so far;
  everalgo never moves it back), oldest first, so the profile's timestamp
  advances over them in order and never re-selects a folded one. An
  episode the index does not hold yet, or one older than the profile (a late
  import), is left to its own run; everalgo folds such an older episode
  without letting it rewrite what a later one established.
- **Sources are read back, not trusted from the index or the event.** Every
  episode comes from md (the truth), and its memcell must still exist; one
  gone (the user deleted it before this run) is left out, and a run whose own
  episode is gone ends there.
- **Reflection's merged episodes are not profiled**: their source episodes
  already were. Their runs do not apply, and the index read takes
  ``parent_type="memcell"`` only.
- **At least once.** OME re-enqueues a run that was RUNNING when the process
  died, and :class:`~everos.memory.profile_batcher.ProfileBatcher` keeps the
  marks of folded-along episodes in memory, so a crash can fold an episode a
  second time; that rare repeat is accepted rather than tracked.
- **Serialised per owner**: ``user.md`` is a read -> LLM merge -> overwrite
  sequence, and the marks are read and written under the same lock.
  Different owners run in parallel.
- **everalgo's lists ride with the profile.** ``do_not_record`` and
  ``pending`` go back into every call and the returned ones replace them, an
  empty ``pending`` included; the extractor runs with its default item
  checks and never adds a new ``implicit_traits`` item.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection

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
    """Fold the event's episode, and the owner's waiting ones, into the profile."""
    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.owner_id,
        owner_type="user",
    )
    partition = f"{event.app_id}:{event.project_id}:{event.owner_id}"
    async with get_partition_lock("extract_user_profile", partition):
        if deps.profile_batcher.take(scope, event.episode_entry_id):
            logger.debug(
                "profile_episode_already_folded",
                owner_id=event.owner_id,
                episode_entry_id=event.episode_entry_id,
            )
            return []
        await _fold(event, scope, deps)
    return []


async def _fold(event: EpisodeExtracted, scope: Scope, deps: MemoryDeps) -> None:
    """Fold the event's episode and the owner's waiting ones into ``user.md``."""
    own = await _read_episodes(scope, [event.episode_entry_id], deps)
    if not own:
        logger.info(
            "profile_skipped_source_gone",
            owner_id=event.owner_id,
            episode_entry_id=event.episode_entry_id,
            memcell_id=event.memcell_id,
        )
        return
    existing = await deps.stores.profile.read(scope)
    waiting = await _waiting_episodes(scope, existing, event, deps)
    batch = [*own, *waiting]
    language = _profile_language(existing, event)
    new_profile = await _extract(
        event, deps, episodes=batch, existing=existing, language=language
    )
    if new_profile is not None:
        await deps.stores.profile.write(
            scope, profile_mapper.from_algo(new_profile, primary_language=language)
        )
    deps.profile_batcher.mark(scope, (entry_id for entry_id, _ in waiting))
    logger.info(
        "user_profile_extracted",
        owner_id=event.owner_id,
        episode_entry_id=event.episode_entry_id,
        episodes=len(batch),
        mode="UPDATE" if existing is not None else "INIT",
        written=new_profile is not None,
    )


async def _waiting_episodes(
    scope: Scope, existing: Profile | None, event: EpisodeExtracted, deps: MemoryDeps
) -> list[tuple[str, Episode]]:
    """The owner's oldest unfolded episodes newer than the profile, up to the cap.

    Excludes the event's own episode and those another run already folded.
    """
    room = deps.profile_batcher.max_episodes - 1
    if room == 0:
        return []
    after_ts = existing.timestamp if existing is not None else 0
    rows = await deps.indexes.episode.list_after(scope, after_ts, parent_type="memcell")
    candidates = [
        row.ref.entry_id
        for row in rows
        if row.ref.entry_id != event.episode_entry_id
        and not deps.profile_batcher.is_marked(scope, row.ref.entry_id)
    ][:room]
    return await _read_episodes(scope, candidates, deps)


async def _read_episodes(
    scope: Scope, entry_ids: Collection[str], deps: MemoryDeps
) -> list[tuple[str, Episode]]:
    """``(entry_id, episode)`` of ``entry_ids`` still in md with their memcell."""
    if not entry_ids:
        return []
    stored = await deps.stores.episode.read_many(scope, entry_ids)
    alive = await asyncio.gather(
        *(_memcell_exists(s.item.parent_id, deps) for s in stored)
    )
    return [(s.ref.entry_id, s.item) for s, ok in zip(stored, alive, strict=True) if ok]


async def _memcell_exists(memcell_id: str, deps: MemoryDeps) -> bool:
    return bool(await deps.stores.memcell.find_by_ids([memcell_id]))


async def _extract(
    event: EpisodeExtracted,
    deps: MemoryDeps,
    *,
    episodes: list[tuple[str, Episode]],
    existing: Profile | None,
    language: str | None,
) -> AlgoProfile | None:
    """INIT or UPDATE the owner's profile from ``episodes``; ``None`` if refused."""
    try:
        return await ProfileExtractor(llm=deps.llms.profile).aextract_from_episodes(
            [episode_mapper.to_algo(episode) for _, episode in episodes],
            owner_id=event.owner_id,
            owner_name=event.owner_name or event.owner_id,
            old_profile=profile_mapper.to_algo(existing) if existing else None,
            custom_instructions=deps.instructions.load("profile_extract"),
            output_language=language,
            tz=deps.tz,
        )
    except AlgoInvalidInputError as exc:
        # No narrative names the owner (or they are otherwise unusable): a
        # retry would read the same text, so the episodes are skipped.
        logger.warning(
            "profile_episode_skipped",
            owner_id=event.owner_id,
            episode_entry_id=event.episode_entry_id,
            episodes=len(episodes),
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
