"""Member-in-group profiles read in batches (docs/group-memory.md, section 7.4).

``[groups] member_profile_mode = "batched"``, the default. The group is read like
one more person: its new segments are queued on the group
(``GroupProfileStore.member_pending_batch``), not on each speaker, and the
batch is read on the group profile's cadence -- after ``[groups]
group_profile_batch`` new segments, after ``group_profile_idle_seconds``
without one, on a flush with ``profile_on_flush``, and when a catch-up asks
(:class:`~everos.memory.events.GroupProfilesDue`).

- **One call per batch.** everalgo's ``ProfileExtractor.aupdate_many`` reads
  the batch's segments once, speakers labelled by their roster names, for
  every member who wrote in them: each one's stored profile under the label
  their refs carry, and the pictures they posted (D17: a picture is evidence
  about its poster only). The reply is grouped per member; a member it
  leaves out has nothing new, and their document is not written.
- **What the call reads.** ``[profile] input = "full"`` (the default) shows
  the whole batch and every picture its members posted, as before cues.
  With ``input = "cues"`` it reads what the batch's episode calls marked
  about each member
  (:func:`~everos.memory.extract.pipeline.profile_input.select_members_input`):
  a member with no cue in the batch is not read (no operation, no token), the
  conversation is rendered once from the union of the members' marked
  messages -- each after the nearest earlier message by someone else, cut to
  200 characters -- and a marked picture is attached as its picture memory's
  text (``[profile] picture_pixels`` attaches the picture); a batch that
  marked nothing for anyone makes no call. Every segment a member's update
  read with a cue record is marked read once the batch is written, so a
  batch read again reads it no more.
- **Each member as before.** A named member's operations apply to their own
  document as a one-person update's do (:func:`~._profiles.finish_update`):
  item ids, sources and spans, ``supersedes`` / ``since`` and the history,
  held items; a consolidation per member and part, only over its cap and
  grown. A member is read from the batch's segments holding a message of
  theirs, less any their profile already rests on (a batch read again after
  a crash), and less any whose only message of theirs was a forgotten
  picture: those stay in the conversation, and nothing of theirs may rest
  on them.
- **Attribution.** everalgo leaves out an operation that names another
  member's item, rests only on segments without a line of the member's or
  only on someone else's pictures, or quotes only lines others wrote; a
  quote found nowhere, or none at all, keeps it. Every outcome is counted
  (``everos_member_profile_attribution_total`` by ``outcome``).
- **Languages.** A stored ``primary_language`` wins, as on ``user.md``; for a
  member with none, the batch's, resolved in code from its segments newest
  first. Members whose languages differ are read in one call per language.
- **Locks.** The batch's (:func:`member_profiles_lock`), then the profile
  lock of every member who wrote in the batch, in sorted order, held across
  the call.
- **Forget queues.** A forget that needs a segment read again for a member
  queues it on that member's document, as in ``per_member`` mode; after the
  group's batch is read, and at a flush with ``profile_on_flush`` or a
  catch-up, those queues are read as there, one member per call
  (``update_member_profiles._update_pending``). An idle tick with nothing
  queued reads no member's document: the scheduler fires one on every scan
  while a group stays idle.
- **A call that yields nothing usable** -- or input everalgo refuses, which
  would otherwise fail every later batch of the group -- keeps the batch
  queued with the group profile's back-off: no automatic try before
  ``group_profile_idle_seconds`` has passed, then twice that; the third
  failure in a row drops the batch (its segments stay searchable). Each
  failure is counted (``everos_member_profile_batch_failures_total``) and
  fails its run (:class:`~everos.memory.ports.DeferredFailure`). A catch-up
  reads a batch that waits out a back-off at once.
"""

from __future__ import annotations

__all__ = ["update_batched"]

from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass

from everalgo.types import ChatMessage
from everalgo.user_memory import ProfileExtractor
from everalgo.user_memory.profile import ProfileTarget, ProfileUpdates
from everalgo.utils import InvalidInputError

from everos.config import Settings
from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Counter as MetricCounter
from everos.core.observability.tracing import memory_span
from everos.core.utils.datetime import get_utc_now, to_timestamp_ms
from everos.memory._partition_locks import member_profiles_lock, profile_locks
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.conversation_language import conversation_language
from everos.memory.extract.llm_retry import (
    UNUSABLE_CALL,
    failure_reason,
    retry_llm_output,
)
from everos.memory.extract.pipeline.group_render import label_cell
from everos.memory.extract.pipeline.member_profile import wrote_in
from everos.memory.extract.pipeline.pictures import members_pictures
from everos.memory.extract.pipeline.profile_input import select_members_input
from everos.memory.extract.pipeline.profile_items import already_read
from everos.memory.groups import readable_segments
from everos.memory.mappers import profile as profile_mapper
from everos.memory.models import MemCell, Profile, Roster
from everos.memory.ports import DeferredFailure, PendingBatch, Scope

from ._profiles import ProfileUpdated, finish_update

logger = get_logger(__name__)

ATTEMPTS = 3
"""Reads of one batch whose call yields nothing usable before it is dropped
(the group profile's number)."""

_attribution = MetricCounter(
    name="member_profile_attribution_total",
    description=(
        "Operations of batched member-profile updates by how they fared against "
        "what they rest on: kept (own, unmatched, unquoted) or left out "
        "(other_speaker, other_segment, other_picture, other_ref)."
    ),
    labelnames=("outcome",),
    namespace="everos",
)

_failures = MetricCounter(
    name="member_profile_batch_failures_total",
    description=(
        "Batched member-profile updates whose model call yielded nothing usable, "
        "by reason and by what became of the batch (backed_off, dropped)."
    ),
    labelnames=("reason", "outcome"),
    namespace="everos",
)

_OUTCOMES = (
    "own",
    "unmatched",
    "unquoted",
    "other_speaker",
    "other_segment",
    "other_picture",
    "other_ref",
)

type _Targets = dict[str, list[tuple[str, MemCell]]]


async def update_batched(
    deps: MemoryDeps,
    scope: Scope,
    *,
    settings: Settings,
    new_segment: EpisodeExtracted | None,
    force: bool,
    own_queues: Callable[[], Awaitable[None]],
    queues_always: bool,
) -> None:
    """Queue ``new_segment`` on the group, and read the batch when it is due.

    Without a new segment (an idle tick, a flush with ``profile_on_flush``, a
    catch-up) the batch is read whatever its size; ``force`` (a catch-up)
    reads one that waits out a back-off too. After a batch is read --
    always, with ``queues_always`` (a flush, a catch-up) -- ``own_queues``
    reads what forgets queued on the members' own documents. An idle tick
    with nothing queued reads nothing: the scheduler fires one on every scan
    while a group stays idle.

    Raises:
        DeferredFailure: The batch's call yielded nothing usable; the batch
            was kept with a back-off, or dropped.
    """
    groups = settings.groups
    failure: str | None = None
    read = False
    async with member_profiles_lock(scope.app_id, scope.project_id, scope.owner_id):
        batch = await deps.stores.group_profile.member_pending_batch(scope)
        pending = list(batch.memcell_ids)
        waiting = batch.retry_at > _now_ms() and not force
        if new_segment is not None:
            if new_segment.memcell_id in pending or await _read_by_every_speaker(
                deps, scope, new_segment
            ):
                return
            pending = [*pending, new_segment.memcell_id]
            if len(pending) < groups.group_profile_batch or waiting:
                await deps.stores.group_profile.set_member_pending(
                    scope, pending, failures=batch.failures, retry_at=batch.retry_at
                )
                return
        if pending and not waiting:
            reason = await _read_batch(deps, scope, pending)
            if reason is None:
                await deps.stores.group_profile.set_member_pending(scope, [])
                read = True
            else:
                failure = await _back_off(
                    deps,
                    scope,
                    batch,
                    pending,
                    reason=reason,
                    wait_seconds=groups.group_profile_idle_seconds,
                )
        if failure is None and (read or queues_always):
            await own_queues()
    if failure is not None:
        raise DeferredFailure(failure)


async def _read_by_every_speaker(
    deps: MemoryDeps, scope: Scope, event: EpisodeExtracted
) -> bool:
    """Whether every speaker's profile already rests on the segment, or read
    it (its cue record says so): a replay."""
    if not event.speaker_ids:
        return False
    for member_id in dict.fromkeys(event.speaker_ids):
        if not already_read(
            await deps.stores.profile.read(scope, member_id), event.memcell_id
        ) and not await _cue_read(deps, member_id, event.memcell_id):
            return False
    logger.info(
        "member_profile_segment_already_read",
        group_id=scope.owner_id,
        memcell_id=event.memcell_id,
    )
    return True


async def _read_batch(
    deps: MemoryDeps, scope: Scope, pending: Sequence[str]
) -> str | None:
    """Read the batch into every speaker's profile; the caller holds its lock.

    Returns:
        ``None`` once the batch is read (or there is nothing to read); the
        failure's reason when a call yielded nothing usable, in which case no
        profile was written.
    """
    # Forgotten messages are not read (section 9.4); a segment that is gone
    # (its session deleted) or left with no message is not shown at all.
    cells = await readable_segments(
        deps.stores.message_forget, await deps.stores.memcell.cells_by_id(pending)
    )
    segments = sorted(
        (
            (memcell_id, cells[memcell_id])
            for memcell_id in pending
            if memcell_id in cells and _has_messages(cells[memcell_id])
        ),
        key=lambda pair: pair[1].timestamp,
    )
    speakers = sorted(
        {
            item.sender_id
            for _, cell in segments
            for item in cell.items
            if isinstance(item, ChatMessage) and item.role == "user" and item.sender_id
        }
    )
    if not speakers:
        logger.info(
            "member_profiles_nothing_to_read",
            group_id=scope.owner_id,
            pending=len(pending),
        )
        return None
    record = await deps.stores.groups.get(scope)
    roster = record.roster if record is not None else Roster()
    async with profile_locks(scope, speakers):
        stored = {m: await deps.stores.profile.read(scope, m) for m in speakers}
        targets: _Targets = {}
        for member_id in speakers:
            own = [
                (memcell_id, cell)
                for memcell_id, cell in segments
                if wrote_in(cell, member_id)
                and not already_read(stored[member_id], memcell_id)
            ]
            if own:
                targets[member_id] = own
        if not targets:
            logger.info(
                "member_profiles_nothing_to_read",
                group_id=scope.owner_id,
                pending=len(pending),
            )
            return None
        batch_language = _language_of(segments)
        by_language: dict[str | None, list[str]] = {}
        for member_id in targets:
            held = stored[member_id]
            language = (held.primary_language if held else None) or batch_language
            by_language.setdefault(language, []).append(member_id)
        shown = [label_cell(cell, roster) for _, cell in segments]
        forgotten = await deps.stores.forget_log.live_targets(
            target="image",
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
        )
        written: dict[str, tuple[Profile, ProfileUpdated]] = {}
        marks: dict[str, list[str]] = {}
        calls = 0
        try:
            for language, members in by_language.items():
                read = await _what_to_read(
                    deps,
                    scope,
                    segments=segments,
                    shown=shown,
                    members=members,
                    targets=targets,
                    forgotten=forgotten,
                    marks=marks,
                )
                if read is None:
                    continue
                calls += 1
                written |= await _call(
                    deps,
                    scope,
                    read,
                    stored=stored,
                    language=language,
                    forgotten=forgotten,
                )
        except UNUSABLE_CALL as err:
            return failure_reason(err)
        except InvalidInputError as err:
            # The input everalgo refused is this batch's: kept with the
            # back-off and dropped after ATTEMPTS, so it never blocks every
            # later batch of the group.
            logger.warning(
                "member_profiles_input_refused",
                group_id=scope.owner_id,
                error=str(err)[:300],
            )
            return "invalid_input"
        for member_id, (profile, updated) in written.items():
            await deps.stores.profile.write(scope, profile)
            _log_member(
                scope, member_id, targets[member_id], stored[member_id], updated
            )
        await _mark_read(deps, marks)
    logger.info(
        "member_profiles_batch_read",
        group_id=scope.owner_id,
        segments=len(segments),
        speakers=len(speakers),
        targets=len(targets),
        written=len(written),
        calls=calls,
    )
    return None


@dataclass(frozen=True)
class _Read:
    """What one call of a batch reads.

    Attributes:
        segments: ``(memcell_id, cell)`` the call shows, oldest first.
        shown: The same segments as the model is shown them.
        members: The members the call updates, in label order.
        targets: Each member's segments, of ``segments``.
        only: Attach only these pictures (content hashes); ``None``, every
            picture a member posted.
        as_text: Attach each picture as this text instead of its pixels.
    """

    segments: Sequence[tuple[str, MemCell]]
    shown: Sequence[MemCell]
    members: Sequence[str]
    targets: _Targets
    only: Collection[str] | None = None
    as_text: Mapping[str, str] | None = None


async def _what_to_read(
    deps: MemoryDeps,
    scope: Scope,
    *,
    segments: Sequence[tuple[str, MemCell]],
    shown: Sequence[MemCell],
    members: Sequence[str],
    targets: _Targets,
    forgotten: Collection[str],
    marks: dict[str, list[str]],
) -> _Read | None:
    """What the call for ``members`` reads; ``None`` for no call.

    ``full`` reads the whole batch. ``cues`` reads what the episode calls
    marked about each member (:func:`select_members_input`): a member with
    nothing marked is left out, and with nobody left there is no call. The
    segments each member's update reads with a cue record are added to
    ``marks``, to be marked read once the batch is written.
    """
    cues = deps.stores.profile_cues
    if deps.profile_input != "cues" or cues is None:
        return _Read(segments=segments, shown=shown, members=members, targets=targets)
    chosen = await select_members_input(
        segments,
        {m: [memcell_id for memcell_id, _ in targets[m]] for m in members},
        cues=cues,
        shown=shown,
        image_memory=deps.stores.image_memory,
        scope=scope,
        picture_pixels=deps.profile_picture_pixels,
        forgotten_pictures=forgotten,
    )
    for member_id, memcell_ids in chosen.read.items():
        marks.setdefault(member_id, []).extend(memcell_ids)
    if not chosen.people:
        logger.info(
            "member_profiles_no_cues",
            group_id=scope.owner_id,
            members=len(members),
            segments=len(segments),
        )
        return None
    cells_of = dict(chosen.segments)
    return _Read(
        segments=chosen.segments,
        shown=chosen.shown,
        members=[m for m in members if m in chosen.people],
        targets={
            m: [(memcell_id, cells_of[memcell_id]) for memcell_id in memcell_ids]
            for m, memcell_ids in chosen.people.items()
        },
        only=chosen.pictures,
        as_text=chosen.picture_text,
    )


async def _call(
    deps: MemoryDeps,
    scope: Scope,
    read: _Read,
    *,
    stored: dict[str, Profile | None],
    language: str | None,
    forgotten: Collection[str],
) -> dict[str, tuple[Profile, ProfileUpdated]]:
    """One ``aupdate_many`` call for ``read.members``, who share ``language``.

    Returns each member the reply named: the document to write (its forget
    queue kept as stored) and what the update took.
    """
    segments, targets = read.segments, read.targets
    cells, pictures, posted = await members_pictures(
        read.shown,
        read.members,
        images=deps.stores.images,
        limits=deps.picture_limits,
        forgotten=forgotten,
        only=read.only,
        as_text=read.as_text,
    )
    # A forgotten picture leaves the text with its message when the message
    # held nothing else: a member is read from the segments where a message
    # of theirs is left, and one with none is not read at all.
    read_as = {
        memcell_id: cell for (memcell_id, _), cell in zip(segments, cells, strict=True)
    }
    own = {
        m: [pair for pair in targets[m] if wrote_in(read_as[pair[0]], m)]
        for m in read.members
    }
    members = [m for m in read.members if own[m]]
    if not members:
        return {}
    olds = {m: _items_of(stored[m]) for m in members}
    algo_targets = [
        ProfileTarget(
            sender_id=member_id,
            old_profile=profile_mapper.to_algo(olds[member_id])
            if olds[member_id] is not None
            else None,
            segment_ids=[memcell_id for memcell_id, _ in own[member_id]],
            pictures=posted.get(member_id, ()),
        )
        for member_id in members
    ]

    async def attempt() -> ProfileUpdates:
        return await ProfileExtractor(llm=deps.llm).aupdate_many(
            cells,
            targets=algo_targets,
            segment_ids=[memcell_id for memcell_id, _ in segments],
            pictures=pictures,
            custom_instructions=deps.instructions.load("profile_extract"),
            output_language=language,
        )

    with memory_span(
        "everos.member_profiles.update",
        observation_type="generation",
        metadata={
            "group_id": scope.owner_id,
            "app_id": scope.app_id,
            "project_id": scope.project_id,
            "segments": len(segments),
            "members": len(members),
            "output_language": language,
        },
    ):
        result = await retry_llm_output(
            attempt, operator="member_profiles", group_id=scope.owner_id
        )
    for counts in result.attribution.values():
        for outcome in _OUTCOMES:
            if amount := getattr(counts, outcome):
                _attribution.labels(outcome=outcome).inc(amount)
    out: dict[str, tuple[Profile, ProfileUpdated]] = {}
    for member_id, update in result.updates.items():
        held = stored[member_id]
        updated = finish_update(
            olds[member_id],
            update,
            history=held.history if held is not None else [],
            segments=own[member_id],
            language=language,
            pictures=len(posted.get(member_id, ())),
        )
        queued = list(held.pending_memcell_ids) if held is not None else []
        out[member_id] = (
            updated.profile.model_copy(update={"pending_memcell_ids": queued}),
            updated,
        )
    dropped = sum(counts.dropped for counts in result.attribution.values())
    if dropped:
        logger.info(
            "member_profile_operations_left_out",
            group_id=scope.owner_id,
            dropped=dropped,
            by_member={
                m: counts.dropped
                for m, counts in result.attribution.items()
                if counts.dropped
            },
        )
    return out


async def _cue_read(deps: MemoryDeps, member_id: str, memcell_id: str) -> bool:
    """Whether the member's profile read the segment (its cue record says so)."""
    cues = deps.stores.profile_cues
    if cues is None:
        return False
    found = await cues.of_person(member_id, [memcell_id])
    return memcell_id in found and found[memcell_id].read


async def _mark_read(deps: MemoryDeps, marks: Mapping[str, Sequence[str]]) -> None:
    """Record that each member's profile read these segments."""
    cues = deps.stores.profile_cues
    if cues is None:
        return
    for member_id, memcell_ids in marks.items():
        if memcell_ids:
            await cues.mark_read(member_id, memcell_ids)


def _items_of(profile: Profile | None) -> Profile | None:
    """The profile the model updates: ``None`` while it holds no item."""
    if profile is None or not (profile.explicit_info or profile.implicit_traits):
        return None
    return profile.model_copy(update={"pending_memcell_ids": []})


def _log_member(
    scope: Scope,
    member_id: str,
    segments: Sequence[tuple[str, MemCell]],
    held: Profile | None,
    updated: ProfileUpdated,
) -> None:
    if updated.held:
        logger.info(
            "member_profile_held_items_kept",
            group_id=scope.owner_id,
            member_id=member_id,
            item_ids=updated.held,
        )
    logger.info(
        "member_profile_updated",
        group_id=scope.owner_id,
        member_id=member_id,
        segments=len(segments),
        pictures=updated.pictures,
        mode="UPDATE" if _items_of(held) is not None else "FIRST",
        replaced=updated.replaced,
        superseded=updated.superseded,
        consolidated=updated.consolidated,
        batched=True,
    )


async def _back_off(
    deps: MemoryDeps,
    scope: Scope,
    batch: PendingBatch,
    pending: list[str],
    *,
    reason: str,
    wait_seconds: int,
) -> str:
    """Count a failed batch; keep it for a later read, or drop it. Returns
    what became of the batch, for the run's error."""
    failures = batch.failures + 1
    if failures >= ATTEMPTS:
        _failures.labels(reason=reason, outcome="dropped").inc()
        logger.warning(
            "member_profiles_batch_dropped",
            group_id=scope.owner_id,
            segments=len(pending),
            failures=failures,
            reason=reason,
        )
        await deps.stores.group_profile.set_member_pending(scope, [])
        return (
            f"member profile update of {scope.owner_id!r} failed ({reason}) "
            f"{failures} times in a row: its batch of {len(pending)} segments "
            "was dropped"
        )
    wait = wait_seconds * 2 ** (failures - 1)
    _failures.labels(reason=reason, outcome="backed_off").inc()
    logger.warning(
        "member_profiles_update_failed",
        group_id=scope.owner_id,
        segments=len(pending),
        failures=failures,
        reason=reason,
        retry_in_seconds=wait,
    )
    await deps.stores.group_profile.set_member_pending(
        scope, pending, failures=failures, retry_at=_now_ms() + wait * 1000
    )
    return (
        f"member profile update of {scope.owner_id!r} failed ({reason}): its "
        f"batch of {len(pending)} segments is kept and read again in {wait} s "
        f"(failure {failures} of {ATTEMPTS})"
    )


def _language_of(segments: Sequence[tuple[str, MemCell]]) -> str | None:
    """The segments' conversation language, newest first, as a cell's is read."""
    found = conversation_language([cell for _, cell in reversed(segments)])
    return None if found is None else str(found)


def _has_messages(cell: MemCell) -> bool:
    return any(isinstance(item, ChatMessage) for item in cell.items)


def _now_ms() -> int:
    return to_timestamp_ms(get_utc_now())
