"""update_member_profiles strategy — the member-in-group profiles of a group.

docs/group-memory.md, section 7.4. ``groups/<g>/members/<m>.md`` holds what the
group's conversations say about one member: preferences, constraints,
commitments, expertise. It is a document of its own (D6): the member's
personal ``users/<m>/user.md`` is never read or written here, and no other
group's segments are.

- **Operator.** everalgo's ``ProfileExtractor``, called with
  ``sender_id = member_id`` over the member's new segments in this group,
  speakers labelled by their roster names. Its prompt scopes extraction to
  the target speaker and keeps assistant suggestions out of the person's
  traits.
- **Input.** Only this group's segments in which the member spoke:
  :class:`GroupEpisodeExtracted` names the human speakers of each segment,
  and :func:`~everos.memory.extract.pipeline.member_profile.wrote_in`
  drops any segment without a message of the member's own (the source check).
- **Throttle.** A segment is queued on the member's profile
  (``pending_memcell_ids``); the profile is updated after
  ``[groups] member_profile_every`` of them, and on :class:`GroupFlushed`
  for every member with something pending. A replayed event queues its
  segment once.
- **Lock.** One per ``(group, member)``: two updates of one member never lose
  each other's items, and the members of one segment update in parallel.
- **Sources.** Every new or rewritten item records the segments it was read
  from and their span; items the update dropped move to the history.
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

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.memory._partition_locks import member_profile_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import GroupEpisodeExtracted, GroupFlushed
from everos.memory.extract.conversation_language import conversation_language
from everos.memory.extract.pipeline.group_render import label_cell
from everos.memory.extract.pipeline.member_profile import (
    HISTORY_LIMIT,
    with_provenance,
    wrote_in,
)
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
    every = load_settings().groups.member_profile_every
    if isinstance(event, GroupEpisodeExtracted):
        await asyncio.gather(
            *(
                _queue(deps, scope, member, event, every=every)
                for member in dict.fromkeys(event.speaker_ids)
            )
        )
    else:
        members = await deps.stores.member_profile.members(scope)
        await asyncio.gather(*(_update_pending(deps, scope, m) for m in members))
    return []


async def _queue(
    deps: MemoryDeps,
    scope: Scope,
    member_id: str,
    event: GroupEpisodeExtracted,
    *,
    every: int,
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
            profile = await _update(deps, scope, profile, language=event.language)
        await deps.stores.member_profile.write(scope, profile)


async def _update_pending(deps: MemoryDeps, scope: Scope, member_id: str) -> None:
    async with member_profile_lock(
        scope.app_id, scope.project_id, scope.owner_id, member_id
    ):
        stored = await deps.stores.member_profile.read(scope, member_id)
        if stored is None or not stored.pending_memcell_ids:
            return
        await deps.stores.member_profile.write(
            scope, await _update(deps, scope, stored, language=None)
        )


async def _update(
    deps: MemoryDeps,
    scope: Scope,
    profile: MemberProfile,
    *,
    language: str | None,
) -> MemberProfile:
    """Read the pending segments into the profile; the caller holds the lock."""
    member_id = profile.member_id
    found = await asyncio.gather(
        *(deps.stores.memcell.find_by_ids([m]) for m in profile.pending_memcell_ids)
    )
    # A segment that is gone (its session deleted) or holds no message of the
    # member's own is not read: no item may rest on others' words alone.
    segments = sorted(
        (
            (memcell_id, cells[0])
            for memcell_id, cells in zip(
                profile.pending_memcell_ids, found, strict=True
            )
            if cells and wrote_in(cells[0], member_id)
        ),
        key=lambda pair: pair[1].timestamp,
    )
    done = profile.model_copy(update={"pending_memcell_ids": []})
    if not segments:
        logger.info(
            "member_profile_nothing_to_read",
            group_id=scope.owner_id,
            member_id=member_id,
            pending=len(profile.pending_memcell_ids),
        )
        return done
    record = await deps.stores.groups.get(scope)
    roster = record.roster if record is not None else Roster()
    existing = profile.profile
    old = existing if existing.explicit_info or existing.implicit_traits else None
    output_language = existing.primary_language or language or _language_of(segments)
    extracted = await ProfileExtractor(llm=deps.llm).aextract(
        [label_cell(cell, roster) for _, cell in segments],
        sender_id=member_id,
        old_profile=profile_mapper.to_algo(old) if old is not None else None,
        custom_instructions=deps.instructions.load("profile_extract"),
        output_language=output_language,
    )
    new = profile_mapper.from_algo(extracted, primary_language=output_language)
    span = (_first_message(segments[0][1]), segments[-1][1].timestamp)
    stamped, replaced = with_provenance(
        old,
        new,
        sources=[memcell_id for memcell_id, _ in segments],
        span=span,
        replaced_at=span[1],
    )
    logger.info(
        "member_profile_updated",
        group_id=scope.owner_id,
        member_id=member_id,
        segments=len(segments),
        mode="UPDATE" if old is not None else "INIT",
        replaced=len(replaced),
    )
    return done.model_copy(
        update={
            "profile": stamped,
            "history": [*replaced, *profile.history][:HISTORY_LIMIT],
        }
    )


def _language_of(segments: list[tuple[str, MemCell]]) -> str | None:
    """The segments' conversation language, newest first, as a cell's is read."""
    found = conversation_language([cell for _, cell in reversed(segments)])
    return None if found is None else str(found)


def _first_message(cell: MemCell) -> int:
    times = [t for t in (getattr(i, "timestamp", 0) for i in cell.items) if t]
    return min(times) if times else cell.timestamp


UPDATE_MEMBER_PROFILES_SPEC = StrategySpec(
    name="update_member_profiles",
    fn=update_member_profiles,
    trigger=Immediate(on=[GroupEpisodeExtracted, GroupFlushed]),
    max_retries=2,
)
