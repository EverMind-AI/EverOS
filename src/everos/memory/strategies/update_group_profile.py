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
  deleted) is not read. Speakers carry their roster names.
- **Checks.** Every operation the model returns is verified, and a failing
  one is dropped and counted by reason (:func:`verify`); items are built from
  the verified evidence alone.
- **Lock.** One per group (:func:`group_profile_lock`): an update reads the
  items the previous one wrote, and a segment queued meanwhile waits.
- **Language.** Named to the model, resolved in code from the batch's human
  messages, newest segment first; only a batch without a letter leaves it to
  the model.
- **Switch.** ``[groups] group_profile = false`` queues nothing and calls
  nothing.
"""

from __future__ import annotations

from everalgo.llm import LLMOutputError, parse_reply

from everos.config import load_settings
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span
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
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.extract.pipeline.pictures import memcell_alive
from everos.memory.models import Roster
from everos.memory.ports import Event, Idle, IdleTick, Immediate, Scope, StrategySpec

logger = get_logger(__name__)


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
        pending = await deps.stores.group_profile.pending(scope)
        if isinstance(event, GroupEpisodeExtracted):
            if event.memcell_id in pending:
                return []
            pending = [*pending, event.memcell_id]
            if len(pending) < groups.group_profile_batch:
                await deps.stores.group_profile.set_pending(scope, pending)
                return []
        if not pending:
            return []
        await _update(deps, scope, pending)
        await deps.stores.group_profile.set_pending(scope, [])
    return []


async def _update(deps: MemoryDeps, scope: Scope, pending: list[str]) -> None:
    """Read the batch into new items; the caller holds the group's lock."""
    cells = await deps.stores.memcell.cells_by_id(pending)
    segments = sorted(
        (Segment(memcell_id=m, cell=cells[m]) for m in pending if m in cells),
        key=lambda s: s.cell.timestamp,
    )
    if not segments:
        logger.info(
            "group_profile_nothing_to_read",
            group_id=scope.owner_id,
            pending=len(pending),
        )
        return
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
        except LLMOutputError:
            # A reply that never validates loses this batch's operations, not
            # the strategy: the next batch reads the profile as it stands.
            logger.warning(
                "group_profile_reply_unusable",
                group_id=scope.owner_id,
                segments=len(segments),
            )
            return
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
    )


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
for a group that stays idle; with nothing queued a tick returns at once."""
