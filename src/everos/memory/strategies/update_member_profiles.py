"""update_member_profiles strategy — the member-in-group profiles of a group.

docs/group-memory.md, section 7.4. ``groups/<g>/members/<m>.md`` holds what the
group's conversations say about one member: preferences, constraints,
commitments, expertise. It is a document of its own (D6): the member's
personal ``users/<m>/user.md`` is never read or written here, and no other
group's segments are.

- **Operator.** everalgo's ``ProfileExtractor.aupdate`` -- the same two
  prompts as the one-to-one profile (update, then a consolidation only for a
  part over its cap that has grown) -- called with ``sender_id = member_id``
  over the member's new segments in this group, speakers labelled by their
  roster names, and the pictures the member posted in them (D17: a picture
  says something about its poster only). Its prompt scopes extraction to the
  target speaker and keeps assistant suggestions out of the person's traits.
- **Input.** Only this group's segments in which the member spoke:
  :class:`GroupEpisodeExtracted` names the human speakers of each segment,
  and :func:`~everos.memory.extract.pipeline.member_profile.wrote_in`
  drops any segment without a message of the member's own (the source check).
  Each segment is read without its forgotten messages
  (:func:`~everos.memory.groups.readable_segments`, section 9.4), so a
  forget that queued a segment again has its rewrite read the rest only.
- **Throttle.** A segment is queued on the member's profile
  (``pending_memcell_ids``); the profile is updated after
  ``[groups] member_profile_every`` of them, and on :class:`GroupFlushed`
  for every member with something pending. A replayed event queues its
  segment once.
- **Lock.** One per ``(group, member)``: two updates of one member never lose
  each other's items, and the members of one segment update in parallel.
- **Fan-out.** A flush updates every member with pending segments at once,
  so the storage reads of one run are bounded: each member's pending
  segments come back in one query (``cells_by_id``), and at most
  ``[sqlite] pool_size`` members read storage at a time (their model calls
  stay parallel). Without the bound one flush asked for a connection per
  pending segment of every member at once and saturated the pool.
- **Sources.** Every new or rewritten item records the segments its
  operation named and their span, and the pictures it was read off; items the
  update retired move to the history with what replaced them
  (:mod:`~everos.memory.extract.pipeline.profile_items`).
- **Versions.** A fact the update's ``modify`` gave a new value becomes a new
  value of it: the new item names the old one (``supersedes``) and the end of
  this update's span (``since``), and the old one's history entry names the
  new one (``superseded_by``). The history keeps the values a current item
  still shows as replaced when older entries drop off.
- **Held items.** An item a person changed through ``POST /memory/update`` is
  held unless they released it, as on ``user.md``: whatever the update's reply
  did to it is undone, and a new value the model wrote for it stays as an
  item of its own (:func:`~everos.memory.extract.pipeline.profile_items.keep_held`).
- **Language.** A stored ``primary_language`` wins, as for ``user.md``. Until
  one is stored, an update that a segment's event runs is told that event's
  language -- the conversation's, resolved in code by the group pipeline --
  and an update on :class:`GroupFlushed`, which brings no language, resolves
  it the same way from the segments it reads. Never left to the model, unless
  no segment has a letter in it.
"""

from __future__ import annotations

import asyncio

from everalgo.user_memory import ProfileExtractor

from everos.config import SqliteSettings, load_settings
from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import member_profile_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import GroupEpisodeExtracted, GroupFlushed
from everos.memory.extract.conversation_language import conversation_language
from everos.memory.extract.pipeline.group_render import label_cell
from everos.memory.extract.pipeline.member_profile import wrote_in
from everos.memory.extract.pipeline.pictures import profile_pictures
from everos.memory.extract.pipeline.profile_items import (
    bounded_history,
    keep_held,
    stamp_update,
)
from everos.memory.groups import readable_segments
from everos.memory.mappers import profile as profile_mapper
from everos.memory.models import MemberProfile, MemCell, Profile, Roster
from everos.memory.ports import Event, Immediate, Scope, StrategySpec

logger = get_logger(__name__)


async def update_member_profiles(
    event: GroupEpisodeExtracted | GroupFlushed, deps: MemoryDeps
) -> list[Event]:
    """Queue a new segment on its speakers' profiles, or bring them up to date."""
    scope = Scope(
        app_id=event.app_id,
        project_id=event.project_id,
        owner_id=event.group_id,
        owner_type="group",
    )
    settings = load_settings()
    every = settings.groups.member_profile_every
    reads = asyncio.Semaphore(read_fanout(settings.sqlite))
    if isinstance(event, GroupEpisodeExtracted):
        await asyncio.gather(
            *(
                _queue(deps, scope, member, event, every=every, reads=reads)
                for member in dict.fromkeys(event.speaker_ids)
            )
        )
    else:
        members = await deps.stores.member_profile.members(scope)
        await asyncio.gather(
            *(_update_pending(deps, scope, m, reads=reads) for m in members)
        )
    return []


def read_fanout(sqlite: SqliteSettings) -> int:
    """How many members of one run may read storage at once.

    The pool's steady size: the overflow above it stays free for every other
    reader of ``system.db`` (the cascade, the queue, requests), so one large
    group's flush cannot exhaust the pool on its own.
    """
    return max(1, sqlite.pool_size)


async def _queue(
    deps: MemoryDeps,
    scope: Scope,
    member_id: str,
    event: GroupEpisodeExtracted,
    *,
    every: int,
    reads: asyncio.Semaphore,
) -> None:
    async with member_profile_lock(
        scope.app_id, scope.project_id, scope.owner_id, member_id
    ):
        stored = await deps.stores.member_profile.read(scope, member_id)
        profile = stored or MemberProfile(
            group_id=scope.owner_id,
            member_id=member_id,
            profile=Profile(owner_id=member_id, summary="", timestamp=0),
        )
        if event.memcell_id in profile.pending_memcell_ids:
            return
        profile = profile.model_copy(
            update={
                "pending_memcell_ids": [*profile.pending_memcell_ids, event.memcell_id]
            }
        )
        if len(profile.pending_memcell_ids) >= every:
            profile = await _update(
                deps, scope, profile, language=event.language, reads=reads
            )
        await deps.stores.member_profile.write(scope, profile)


async def _update_pending(
    deps: MemoryDeps, scope: Scope, member_id: str, *, reads: asyncio.Semaphore
) -> None:
    async with member_profile_lock(
        scope.app_id, scope.project_id, scope.owner_id, member_id
    ):
        stored = await deps.stores.member_profile.read(scope, member_id)
        if stored is None or not stored.pending_memcell_ids:
            return
        await deps.stores.member_profile.write(
            scope, await _update(deps, scope, stored, language=None, reads=reads)
        )


async def _update(
    deps: MemoryDeps,
    scope: Scope,
    profile: MemberProfile,
    *,
    language: str | None,
    reads: asyncio.Semaphore,
) -> MemberProfile:
    """Read the pending segments into the profile; the caller holds the lock."""
    member_id = profile.member_id
    pending = profile.pending_memcell_ids
    async with reads:
        # Forgotten messages are not read (section 9.4): the payload keeps
        # them for an undo, the profile never sees them.
        cells = await readable_segments(
            deps.stores.message_forget, await deps.stores.memcell.cells_by_id(pending)
        )
        # A segment that is gone (its session deleted) or holds no message of
        # the member's own is not read: no item may rest on others' words alone.
        segments = sorted(
            (
                (memcell_id, cells[memcell_id])
                for memcell_id in pending
                if memcell_id in cells and wrote_in(cells[memcell_id], member_id)
            ),
            key=lambda pair: pair[1].timestamp,
        )
        record = await deps.stores.groups.get(scope) if segments else None
    done = profile.model_copy(update={"pending_memcell_ids": []})
    if not segments:
        logger.info(
            "member_profile_nothing_to_read",
            group_id=scope.owner_id,
            member_id=member_id,
            pending=len(profile.pending_memcell_ids),
        )
        return done
    roster = record.roster if record is not None else Roster()
    existing = profile.profile
    old = existing if existing.explicit_info or existing.implicit_traits else None
    output_language = existing.primary_language or language or _language_of(segments)
    async with reads:
        cells, pictures = await profile_pictures(
            [label_cell(cell, roster) for _, cell in segments],
            member_id,
            images=deps.stores.images,
            limits=deps.picture_limits,
            forgotten=await deps.stores.forget_log.live_targets(
                target="image",
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
            ),
        )
    update = await ProfileExtractor(llm=deps.llm).aupdate(
        cells,
        sender_id=member_id,
        old_profile=profile_mapper.to_algo(old) if old is not None else None,
        segment_ids=[memcell_id for memcell_id, _ in segments],
        pictures=pictures,
        custom_instructions=deps.instructions.load("profile_extract"),
        output_language=output_language,
    )
    stamped, replaced = stamp_update(
        old,
        update,
        cells=dict(segments),
        replaced_at=segments[-1][1].timestamp,
        primary_language=output_language,
    )
    stamped, replaced, held = keep_held(old, stamped, replaced)
    if held:
        logger.info(
            "member_profile_held_items_kept",
            group_id=scope.owner_id,
            member_id=member_id,
            item_ids=held,
        )
    logger.info(
        "member_profile_updated",
        group_id=scope.owner_id,
        member_id=member_id,
        segments=len(segments),
        pictures=len(pictures),
        mode="UPDATE" if old is not None else "FIRST",
        replaced=len(replaced),
        superseded=sum(1 for entry in replaced if "superseded_by" in entry),
        consolidated=len(update.consolidated),
    )
    return done.model_copy(
        update={
            "profile": stamped,
            "history": bounded_history(replaced, profile.history, stamped),
        }
    )


def _language_of(segments: list[tuple[str, MemCell]]) -> str | None:
    """The segments' conversation language, newest first, as a cell's is read."""
    found = conversation_language([cell for _, cell in reversed(segments)])
    return None if found is None else str(found)


UPDATE_MEMBER_PROFILES_SPEC = StrategySpec(
    name="update_member_profiles",
    fn=update_member_profiles,
    trigger=Immediate(on=[GroupEpisodeExtracted, GroupFlushed]),
    max_retries=2,
)
