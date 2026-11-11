"""update_group_profile strategy -- the group profile (docs/group-memory.md, 7.5).

The group profile is an append-only set of items -- topics, decisions,
dissent, roles -- kept by one model call per batch of the group's segments
(``chat.group_profile_update``, :mod:`everos.memory.extract.group_profile`).

- **Batch.** Each :class:`GroupEpisodeExtracted` queues its segment on the
  group (``GroupProfileStore.pending``; a replay queues it once). The batch is
  read after ``[groups] group_profile_batch`` segments, on
  :class:`GroupFlushed`, and on the idle tick ``[groups]
  group_profile_idle_seconds`` after the group's last segment. A tick with
  nothing queued does nothing.
- **Input.** Only this group's queued segments, never another group's and
  never a member's personal memory; a segment that is gone (its session
  deleted) is not read. Speakers carry their roster names. Each segment is
  read without its forgotten messages (section 9.4), so no evidence can cite
  one: a forget that left an item stale queues the item's other segments,
  and this update rewrites it from them.
- **Checks.** Every operation the model returns is verified, and a failing
  one is dropped and counted by reason (:func:`verify`); a topic or item it
  names that resolves to nothing drops only the link, also counted; items are
  built from the verified evidence alone.
- **A call that yields nothing usable** (an invalid reply, a failed call) is
  not raised -- raising made the scheduler retry at once, and the idle tick,
  which fires on every scan while a group stays idle, call again and again.
  It is counted (``everos_group_profile_update_failures_total`` by reason),
  and the batch stays queued with a back-off: no update of the group tries
  again before ``group_profile_idle_seconds`` has passed, then twice that;
  new segments still queue meanwhile. The :data:`ATTEMPTS`-th failure in a
  row drops the batch: its operations are lost, its segments stay searchable.
- **Lock.** One per group (:func:`group_profile_lock`): an update reads the
  items the previous one wrote, and a segment queued meanwhile waits.
- **Language.** Named to the model, resolved in code from the batch's human
  messages, newest segment first; only a batch without a letter leaves it to
  the model.
- **Switch.** ``[groups] group_profile = false`` queues nothing and calls
  nothing.
"""

from __future__ import annotations

from everalgo.llm import parse_reply
from everalgo.types import ChatMessage

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Counter as MetricCounter
from everos.core.observability.tracing import memory_span
from everos.core.utils.datetime import get_utc_now, to_timestamp_ms
from everos.memory._partition_locks import group_profile_lock
from everos.memory.deps import MemoryDeps
from everos.memory.events import (
    GroupEpisodeExtracted,
    GroupFlushed,
    parse_group_key,
)
from everos.memory.extract.conversation_language import conversation_language
from everos.memory.extract.group_profile import (
    MAX_TOPICS,
    GroupProfileReply,
    Segment,
    build_messages,
    current_profile,
    drop_summary,
    verify,
)
from everos.memory.extract.llm_retry import (
    UNUSABLE_CALL,
    failure_reason,
    retry_llm_output,
)
from everos.memory.extract.pipeline.pictures import memcell_alive
from everos.memory.groups import readable_segments
from everos.memory.models import MemCell, Roster
from everos.memory.ports import (
    Event,
    Idle,
    IdleTick,
    Immediate,
    PendingBatch,
    Scope,
    StrategySpec,
)

logger = get_logger(__name__)

ATTEMPTS = 3
"""Updates of one batch whose call yields nothing usable before it is dropped."""

_failures = MetricCounter(
    name="group_profile_update_failures_total",
    description=(
        "Group-profile updates whose model call yielded nothing usable, by reason "
        "and by what became of the batch (backed_off, dropped)."
    ),
    labelnames=("reason", "outcome"),
    namespace="everos",
)


async def update_group_profile(
    event: GroupEpisodeExtracted | GroupFlushed | IdleTick, deps: MemoryDeps
) -> list[Event]:
    """Queue a new segment on the group's batch, or read the batch when due."""
    groups = load_settings().groups
    if not groups.group_profile:
        return []
    if isinstance(event, IdleTick):
        key = parse_group_key(event.bucket_key)
        if key is None:
            logger.warning("group_profile_bad_idle_key", key=event.bucket_key)
            return []
        app_id, project_id, group_id = key
    else:
        app_id, project_id, group_id = event.app_id, event.project_id, event.group_id
    scope = Scope(
        app_id=app_id, project_id=project_id, owner_id=group_id, owner_type="group"
    )
    async with group_profile_lock(app_id, project_id, group_id):
        batch = await deps.stores.group_profile.pending_batch(scope)
        pending = list(batch.memcell_ids)
        waiting = batch.retry_at > _now_ms()
        if isinstance(event, GroupEpisodeExtracted):
            if event.memcell_id in pending:
                return []
            pending = [*pending, event.memcell_id]
            if len(pending) < groups.group_profile_batch or waiting:
                await deps.stores.group_profile.set_pending(
                    scope, pending, failures=batch.failures, retry_at=batch.retry_at
                )
                return []
        if not pending or waiting:
            return []
        failure = await _update(deps, scope, pending)
        if failure is None:
            await deps.stores.group_profile.set_pending(scope, [])
        else:
            await _back_off(
                deps,
                scope,
                batch,
                pending,
                reason=failure,
                wait_seconds=groups.group_profile_idle_seconds,
            )
    return []


async def _update(deps: MemoryDeps, scope: Scope, pending: list[str]) -> str | None:
    """Read the batch into new items; the caller holds the group's lock.

    Returns:
        ``None`` once the batch is read (or there is nothing left to read);
        the failure's reason when the call yielded nothing usable.
    """
    # Forgotten messages are not read (section 9.4); a segment left with no
    # message is not shown at all.
    cells = await readable_segments(
        deps.stores.message_forget, await deps.stores.memcell.cells_by_id(pending)
    )
    segments = sorted(
        (
            Segment(memcell_id=m, cell=cells[m])
            for m in pending
            if m in cells and _has_messages(cells[m])
        ),
        key=lambda s: s.cell.timestamp,
    )
    if not segments:
        logger.info(
            "group_profile_nothing_to_read",
            group_id=scope.owner_id,
            pending=len(pending),
        )
        return None
    record = await deps.stores.groups.get(scope)
    roster = record.roster if record is not None else Roster()
    profile = current_profile(
        [stored.item for stored in await deps.stores.group_profile.items(scope)]
    )
    language = conversation_language([s.cell for s in reversed(segments)])
    messages = build_messages(
        roster=roster, profile=profile, segments=segments, language=language
    )

    async def attempt() -> GroupProfileReply:
        text = await deps.llm.achat(messages, response_format=GroupProfileReply)
        return parse_reply(GroupProfileReply, text, operator="group profile update")

    with memory_span(
        "everos.group_profile.update",
        observation_type="generation",
        metadata={
            "group_id": scope.owner_id,
            "app_id": scope.app_id,
            "project_id": scope.project_id,
            "segments": len(segments),
            "output_language": None if language is None else str(language),
        },
    ):
        try:
            reply = await retry_llm_output(
                attempt, operator="group_profile", group_id=scope.owner_id
            )
        except UNUSABLE_CALL as err:
            return failure_reason(err)
    checked = verify(
        reply.operations,
        group_id=scope.owner_id,
        segments=segments,
        roster=roster,
        profile=profile,
        observed_at=segments[-1].cell.timestamp,
        shown_topics=profile.topics[:MAX_TOPICS],
    )
    newest = segments[-1].memcell_id
    refs = await deps.stores.group_profile.append(
        scope, checked.items, only_if=memcell_alive(deps.stores.memcell, newest)
    )
    logger.info(
        "group_profile_updated",
        group_id=scope.owner_id,
        segments=len(segments),
        proposed=checked.proposed,
        written=len(refs),
        dropped=drop_summary(checked.dropped),
        links_dropped=drop_summary(checked.links_dropped),
    )
    return None


async def _back_off(
    deps: MemoryDeps,
    scope: Scope,
    batch: PendingBatch,
    pending: list[str],
    *,
    reason: str,
    wait_seconds: int,
) -> None:
    """Count a failed update; keep the batch for a later try, or drop it."""
    failures = batch.failures + 1
    if failures >= ATTEMPTS:
        _failures.labels(reason=reason, outcome="dropped").inc()
        logger.warning(
            "group_profile_batch_dropped",
            group_id=scope.owner_id,
            segments=len(pending),
            failures=failures,
            reason=reason,
        )
        await deps.stores.group_profile.set_pending(scope, [])
        return
    wait = wait_seconds * 2 ** (failures - 1)
    _failures.labels(reason=reason, outcome="backed_off").inc()
    logger.warning(
        "group_profile_update_failed",
        group_id=scope.owner_id,
        segments=len(pending),
        failures=failures,
        reason=reason,
        retry_in_seconds=wait,
    )
    await deps.stores.group_profile.set_pending(
        scope, pending, failures=failures, retry_at=_now_ms() + wait * 1000
    )


def _now_ms() -> int:
    return to_timestamp_ms(get_utc_now())


def _has_messages(cell: MemCell) -> bool:
    return any(isinstance(item, ChatMessage) for item in cell.items)


UPDATE_GROUP_PROFILE_SPEC = StrategySpec(
    name="update_group_profile",
    fn=update_group_profile,
    trigger=Immediate(on=[GroupEpisodeExtracted, GroupFlushed]),
    max_retries=2,
)

UPDATE_GROUP_PROFILE_IDLE_SPEC = StrategySpec(
    name="update_group_profile_idle",
    fn=update_group_profile,
    trigger=Idle(
        on=[GroupEpisodeExtracted],
        event_field="group_key",
        idle_seconds=600,
        scan_interval_seconds=60,
    ),
    max_retries=2,
)
"""The idle half: ``[groups] group_profile_idle_seconds`` after a group's last
segment (the bootstrap sets the configured value). The scheduler keeps firing
for a group that stays idle; with nothing queued, or while a failed batch
waits out its back-off, a tick returns at once."""
