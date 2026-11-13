"""extract_user_profile strategy — the owner's profile, one new segment at a time.

The 2.0 design's profile update: every new segment of a one-to-one
conversation updates the profile of each user who spoke in it, and the update
reads that segment only -- with the pictures the owner posted in it -- against
the stored profile. What older segments said is already in the profile, so
reading them again only re-adds it. The update is the one a member's profile
in a group runs (:mod:`._profiles`); a one-to-one space's cadence is every
segment, at once.

- **Trigger.** :class:`EpisodeExtracted` of a user's own episode (one per
  one-to-one segment), with or without an embedding provider. A group's
  episode -- a session of three or more people is a group's -- never
  triggers it; each of two people talking without a ``group_id`` has their
  own episode, so their own update.
- **Input.** The event's segment, read from the SQLite ledger (the event
  carries its id, so nothing waits for the index) without the messages a live
  forget names, and the pictures the owner posted in it
  (:func:`~everos.memory.extract.pipeline.pictures.profile_pictures`; none
  when ``[extraction] vision`` is off). With it come the segments a message
  forget queued (``pending_memcell_ids``): the items it took out because they
  rested on a segment that lost a message are written again from what the
  owner's segments still say. A queued segment is read only while it has a
  live episode and a message of the owner's. A segment the profile already
  rests on (a replayed event) is not read again, and neither is a segment
  whose episode the owner forgot -- by its conversation, by id, by a time
  range or with everything -- nor a picture they forgot: a forget leaves the
  bytes in place for its undo, so the episode's mark and the forget log, read
  under the profile lock a forget holds while it records, are what tell. A
  conversation sent again (or continued) after its forget is new material:
  its new episodes are not forgotten, and the profile reads them.
- **Operator.** everalgo's ``ProfileExtractor.aupdate``: one update call with
  the two-prompt design's add / modify / deprecate / none, then a
  consolidation call only for a part of the profile that is over its cap and
  has grown since it was last consolidated.
- **Items.** Every item carries its id, its source segments and their span,
  and the pictures it was read off
  (:mod:`~everos.memory.extract.pipeline.profile_items`), so forgetting a
  picture reaches the items resting on it alone. As on a member profile, a
  fact given a new value is a new item linked to the old one (``supersedes``,
  ``since``), and every item the update took out moves to ``user.md``'s
  history with what replaced it: nothing is lost in place. Forgetting a
  conversation takes out the items every segment of which is forgotten; the
  forget does that, under the same lock, not this strategy. A forget that
  takes an item out also drops the summary (it may repeat the item); every
  update writes the summary again from the items, so the next one restores
  it.
- **Held items.** An item a person changed through ``POST /memory/update`` is
  held unless they released it: the update reads it, and whatever the reply
  did to it is undone (:func:`~everos.memory.extract.pipeline.profile_items.keep_held`);
  a new value the model wrote for it stays as an item of its own.
- **Language.** A stored ``primary_language`` wins and is never changed
  automatically; until one is stored, the segment's language (resolved in
  code by the pipeline) is used and then frozen.
"""

from __future__ import annotations

from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import profile_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.pipeline.member_profile import wrote_in
from everos.memory.extract.pipeline.profile_items import already_read
from everos.memory.groups import readable_segments
from everos.memory.models import FORGOTTEN_MARK, MemCell, Profile
from everos.memory.ports import Event, Immediate, Scope, StrategySpec

from ._profiles import update_profile

logger = get_logger(__name__)


def _profile_applies(event: Event, deps: MemoryDeps) -> bool:
    """A user's segment episode; not a group's."""
    del deps
    return isinstance(event, EpisodeExtracted) and event.owner_type == "user"


def _profile_language(existing: Profile | None, event: EpisodeExtracted) -> str | None:
    """The language to write the profile in, frozen once known.

    A stored profile's ``primary_language`` wins and is never changed
    automatically; changing it is a manual edit. Until one is stored, the
    triggering episode's language is used and then frozen.
    """
    if existing is not None and existing.primary_language:
        return existing.primary_language
    return event.language


async def _forgotten_segment(
    deps: MemoryDeps, scope: Scope, event: EpisodeExtracted
) -> bool:
    """Whether the owner forgot the segment's episode and no undo reversed it.

    Read off the episode's own mark in markdown: every forget of the owner
    marks the episodes it covers (a conversation's, by id, by a time range,
    everything), and only those.
    """
    found = await deps.stores.episode.read_many(scope, [event.episode_entry_id])
    if not found:
        return False
    marks = await deps.stores.episode.deprecations(scope, found[0].ref.md_path)
    return marks.get(event.episode_entry_id) == FORGOTTEN_MARK


async def _segments(
    deps: MemoryDeps, scope: Scope, existing: Profile | None, memcell_id: str
) -> list[tuple[str, MemCell]]:
    """The segments one update reads, oldest first, each without the messages
    a live forget names: the event's, and the ones a message forget queued.

    Empty when the event's segment is gone (its session deleted since); the
    queue then waits for the next update. A queued segment whose episodes
    are all forgotten, or that keeps no message of the owner's, is not read.
    """
    queued = [
        m for m in (existing.pending_memcell_ids if existing else []) if m != memcell_id
    ]
    cells = await readable_segments(
        deps.stores.message_forget,
        await deps.stores.memcell.cells_by_id([*queued, memcell_id]),
    )
    if memcell_id not in cells:
        return []
    alive = (
        {
            e.item.parent_id
            for e in await deps.stores.episode.read_for_segments(scope, queued)
        }
        if queued
        else set()
    )
    found = [
        (m, cells[m])
        for m in queued
        if m in cells and m in alive and wrote_in(cells[m], scope.owner_id)
    ]
    found.append((memcell_id, cells[memcell_id]))
    return sorted(found, key=lambda pair: pair[1].timestamp)


async def extract_user_profile(
    event: EpisodeExtracted, deps: MemoryDeps
) -> list[Event]:
    """Update the owner's profile with the segment the event announces."""
    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.owner_id,
        owner_type="user",
    )
    # Serialise on the owner: user.md is a single per-user file and the
    # body is a read → LLM update → overwrite sequence; a forget and its undo
    # hold the same lock. Different users run fully in parallel.
    async with profile_lock(scope):
        if await _forgotten_segment(deps, scope, event):
            logger.info(
                "profile_segment_forgotten",
                owner_id=event.owner_id,
                memcell_id=event.memcell_id,
            )
            return []
        existing = await deps.stores.profile.read(scope)
        if already_read(existing, event.memcell_id):
            logger.info(
                "profile_segment_already_read",
                owner_id=event.owner_id,
                memcell_id=event.memcell_id,
            )
            return []
        segments = await _segments(deps, scope, existing, event.memcell_id)
        if not segments:  # the session was deleted since
            return []
        updated = await update_profile(
            deps,
            scope,
            person_id=event.owner_id,
            old=existing,
            history=existing.history if existing else [],
            segments=segments,
            shown=[cell for _, cell in segments],
            language=_profile_language(existing, event),
        )
        if updated.held:
            logger.info(
                "profile_held_items_kept",
                owner_id=event.owner_id,
                item_ids=updated.held,
            )
        await deps.stores.profile.write(scope, updated.profile)
    logger.info(
        "user_profile_extracted",
        owner_id=event.owner_id,
        memcell_id=event.memcell_id,
        pictures=updated.pictures,
        mode="UPDATE" if existing is not None else "FIRST",
        retired=updated.retired,
        consolidated=updated.consolidated,
    )
    return []


EXTRACT_USER_PROFILE_SPEC = StrategySpec(
    name="extract_user_profile",
    fn=extract_user_profile,
    trigger=Immediate(on=[EpisodeExtracted]),
    applies_to=_profile_applies,
    max_retries=2,
)
