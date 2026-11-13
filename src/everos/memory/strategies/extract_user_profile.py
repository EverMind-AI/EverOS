"""extract_user_profile strategy — the owner's profile, its new segments read together.

The 2.0 design's profile update: the new segments of a one-to-one
conversation update the profile of each user who spoke in them, and the update
reads those segments only -- with the pictures the owner posted in them --
against the stored profile. What older segments said is already in the
profile, so reading them again only re-adds it. The update is the one a
member's profile in a group runs (:mod:`._profiles`), and so is the cadence:

- **Cadence.** ``[profile] update_every`` (5): a user's segment is queued on
  their profile (``pending_memcell_ids``), and the profile is updated once
  that many are queued -- all of them in one update -- and at every flush of
  one of the user's conversations (:class:`SessionFlushed`), for every sender
  of its segments with something queued. A user's first segment, while they
  have no profile, is read at once: a personal search attaches the user's
  profile, and a profile of a queue only would be an empty one. ``1`` reads
  every segment at once, as before. Episodes are searchable at once; only the
  profile lags. An update that fails leaves every segment it was to read
  queued (the event's included), so a later update -- the engine's retry,
  the next segment past the cadence, a flush -- reads them: a failed window
  is never skipped.
- **Trigger.** :class:`EpisodeExtracted` of a user's own episode (one per
  one-to-one segment), with or without an embedding provider, and
  :class:`SessionFlushed`. A group's episode -- a session of three or more
  people is a group's -- never triggers it; each of two people talking
  without a ``group_id`` has their own episode, so their own update.
- **Input.** The queued segments, read from the SQLite ledger (an event
  carries its segment's id, so nothing waits for the index), each without the
  messages a live forget names, oldest first, and the pictures the owner
  posted in them (:func:`~everos.memory.extract.pipeline.pictures.profile_pictures`;
  none when ``[extraction] vision`` is off). The queue also holds the segments
  a message forget queued: the items it took out because they rested on a
  segment that lost a message are written again from what the owner's
  segments still say. A queued segment is read only while it has a live
  episode and a message of the owner's. A segment the profile already rests
  on (a replayed event) is not queued again, and neither is a segment whose
  episode the owner forgot -- by its conversation, by id, by a time range or
  with everything -- nor is a picture they forgot read: a forget leaves the
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
  automatically; until one is stored, the triggering segment's language
  (resolved in code by the pipeline) -- at a flush, the language of the
  segments read -- is used and then frozen.
"""

from __future__ import annotations

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import profile_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted, SessionFlushed
from everos.memory.extract.conversation_language import conversation_language
from everos.memory.extract.pipeline.member_profile import wrote_in
from everos.memory.extract.pipeline.profile_items import already_read
from everos.memory.groups import readable_segments
from everos.memory.models import FORGOTTEN_MARK, MemCell, Profile
from everos.memory.ports import Event, Immediate, Scope, SessionKey, StrategySpec

from ._profiles import update_profile

logger = get_logger(__name__)


def _profile_applies(event: Event, deps: MemoryDeps) -> bool:
    """A user's segment episode, or a flush of a session bound to no group."""
    del deps
    if isinstance(event, SessionFlushed):
        return True
    return isinstance(event, EpisodeExtracted) and event.owner_type == "user"


def _profile_language(existing: Profile | None, language: str | None) -> str | None:
    """The language to write the profile in, frozen once known.

    A stored profile's ``primary_language`` wins and is never changed
    automatically; changing it is a manual edit. Until one is stored, the
    segments' language is used and then frozen.
    """
    if existing is not None and existing.primary_language:
        return existing.primary_language
    return language


def _has_items(profile: Profile | None) -> bool:
    """Whether ``profile`` says anything yet (a queue alone says nothing)."""
    return profile is not None and bool(
        profile.explicit_info or profile.implicit_traits or profile.summary
    )


def _scope(event: EpisodeExtracted | SessionFlushed, owner_id: str) -> Scope:
    return Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=owner_id,
        owner_type="user",
    )


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
    deps: MemoryDeps, scope: Scope, queued: list[str], memcell_id: str | None
) -> list[tuple[str, MemCell]]:
    """The segments one update reads, oldest first, each without the messages
    a live forget names: the queued ones, and the event's (``memcell_id``;
    ``None`` at a flush).

    Empty when the event's segment is gone (its session deleted since); the
    queue then waits for the next update. A queued segment whose episodes
    are all forgotten, or that keeps no message of the owner's, is not read.
    """
    queued = [m for m in dict.fromkeys(queued) if m != memcell_id]
    wanted = [*queued, memcell_id] if memcell_id is not None else queued
    cells = await readable_segments(
        deps.stores.message_forget, await deps.stores.memcell.cells_by_id(wanted)
    )
    if memcell_id is not None and memcell_id not in cells:
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
    if memcell_id is not None:
        found.append((memcell_id, cells[memcell_id]))
    return sorted(found, key=lambda pair: pair[1].timestamp)


async def extract_user_profile(
    event: EpisodeExtracted | SessionFlushed, deps: MemoryDeps
) -> list[Event]:
    """Queue the event's segment on its owner's profile and read the queue
    when it is due; at a flush, read what each sender's profile queued."""
    if isinstance(event, SessionFlushed):
        session = SessionKey(event.app_id, event.project_id, event.session_id)
        for person in await deps.stores.memcell.senders_in_session(session):
            await _flush_pending(deps, _scope(event, person))
        return []
    every = load_settings().profile.update_every
    scope = _scope(event, event.owner_id)
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
        queued = list(existing.pending_memcell_ids) if existing is not None else []
        due = [*queued, event.memcell_id] if event.memcell_id not in queued else queued
        if _has_items(existing) and len(due) < every:
            if event.memcell_id not in queued:
                assert existing is not None
                await deps.stores.profile.write(
                    scope, existing.model_copy(update={"pending_memcell_ids": due})
                )
            return []
        await _read(
            deps,
            scope,
            existing,
            queued,
            memcell_id=event.memcell_id,
            language=event.language,
        )
    return []


async def _flush_pending(deps: MemoryDeps, scope: Scope) -> None:
    """Read what the person's profile has queued, if anything."""
    async with profile_lock(scope):
        existing = await deps.stores.profile.read(scope)
        if existing is None or not existing.pending_memcell_ids:
            return
        await _read(
            deps,
            scope,
            existing,
            list(existing.pending_memcell_ids),
            memcell_id=None,
            language=None,
        )


async def _read(
    deps: MemoryDeps,
    scope: Scope,
    existing: Profile | None,
    queued: list[str],
    *,
    memcell_id: str | None,
    language: str | None,
) -> None:
    """Read the queue (and the event's segment) into the profile and write
    it; the caller holds the lock.

    An update that fails writes the queue with the event's segment in it
    before the error goes on: those segments stay to be read by a later
    update, never skipped.
    """
    segments = await _segments(deps, scope, queued, memcell_id)
    if not segments:
        return
    if language is None:
        found = conversation_language([cell for _, cell in reversed(segments)])
        language = None if found is None else str(found)
    try:
        updated = await update_profile(
            deps,
            scope,
            person_id=scope.owner_id,
            old=existing.model_copy(update={"pending_memcell_ids": []})
            if _has_items(existing)
            else None,
            history=existing.history if existing else [],
            segments=segments,
            shown=[cell for _, cell in segments],
            language=_profile_language(existing, language),
        )
    except Exception:
        await _keep_queued(deps, scope, existing, [m for m, _ in segments])
        raise
    if updated.held:
        logger.info(
            "profile_held_items_kept",
            owner_id=scope.owner_id,
            item_ids=updated.held,
        )
    await deps.stores.profile.write(scope, updated.profile)
    logger.info(
        "user_profile_extracted",
        owner_id=scope.owner_id,
        segments=len(segments),
        pictures=updated.pictures,
        mode="UPDATE" if _has_items(existing) else "FIRST",
        retired=updated.retired,
        consolidated=updated.consolidated,
    )


async def _keep_queued(
    deps: MemoryDeps, scope: Scope, existing: Profile | None, memcell_ids: list[str]
) -> None:
    """Write ``memcell_ids`` into the profile's queue (a profile of a queue
    alone when there is none yet), so a later update reads them."""
    queued = list(existing.pending_memcell_ids) if existing is not None else []
    pending = [*queued, *(m for m in memcell_ids if m not in queued)]
    if pending == queued:
        return
    base = existing or Profile(owner_id=scope.owner_id, summary="", timestamp=0)
    await deps.stores.profile.write(
        scope, base.model_copy(update={"pending_memcell_ids": pending})
    )
    logger.warning(
        "profile_update_failed_segments_kept",
        owner_id=scope.owner_id,
        pending=len(pending),
    )


EXTRACT_USER_PROFILE_SPEC = StrategySpec(
    name="extract_user_profile",
    fn=extract_user_profile,
    trigger=Immediate(on=[EpisodeExtracted, SessionFlushed]),
    applies_to=_profile_applies,
    max_retries=2,
)
