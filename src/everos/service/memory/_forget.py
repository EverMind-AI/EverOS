"""Forgetting: the use case behind ``POST /memory/forget`` and its undo.

One engine for a user's memory and a group's (docs/api.md; docs/group-memory.md,
sections 5.5 and 9.4): the target is planned first (:mod:`._forget_plan`,
reads only) and a dry run answers with the plan; a real run carries out
exactly that plan (:mod:`._forget_apply`), logs it in one format
(:mod:`._forget_log`) and answers with its ``forget_id``, which the undo takes
(:mod:`._forget_undo`). A forget **marks**: nothing leaves markdown, every
version it reaches is marked forgotten or superseded by one written without
what was forgotten, and its index row goes at once.

The run, in order:

1. a group's target messages are recorded in ``message_forget`` first, so
   every reader that starts afterwards skips them -- re-extraction, the
   profiles' next updates, a buffer read;
2. the conversations the target touches settle: a cut in progress finishes
   (its lock; 503 past ``session_wait_seconds``) and the extractions running
   for them end; the plan is then made again, so nothing they wrote escapes it;
3. every model call (re-extraction of a segment that keeps text, a group
   scene narrated again) is made before anything is written; a failure here
   leaves the memory as it was;
4. the writes, then the profiles and the log entry; a failure part-way
   reverses what the run did, or -- when that fails too -- logs the run as an
   incomplete forget that its undo can finish.

Who may: a user's memory is changed by its owner (the caller checks); inside
a group the asker must be a current member allowed to forget what the target
rests on (:func:`~everos.memory.access.authorize_forget`). One forget, undo or
update of one owner at a time (:func:`~everos.memory.forget_lock`).
"""

from __future__ import annotations

__all__ = ["ForgetManager"]

import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

from everos.core.errors import (
    ForbiddenError,
    InvalidInputError,
    NotFoundError,
    SessionBusyError,
)
from everos.core.locks import LOCKS, session_key
from everos.core.observability.logging import get_logger
from everos.memory import forget_lock
from everos.memory.access import Identity, admits, authorize_forget
from everos.memory.ports import ForgetEntry, Scope

from ._forget_apply import Applied, ForgetWriter, Reextraction, applied_record
from ._forget_models import ForgetResult, ForgetTarget, ForgottenPicture, UndoResult
from ._forget_plan import ForgetPlan, ForgetPlanner
from ._forget_undo import Undoer
from ._records import memory_id

if TYPE_CHECKING:
    from everos.memory.deps import Indexes, Stores
    from everos.memory.extract.scene import SceneNarrator
    from everos.memory.ports import GroupRecord, IndexEraser

    from ._extraction import ExtractionSupervisor

logger = get_logger("everos.service.memory")

_SETTLE_SECONDS = 120.0
"""How long a forget waits for the running extractions of its conversations."""


class ForgetManager:
    """Forgets in a user's or a group's memory, their dry runs and their undos.

    Args:
        stores: Markdown and system state.
        indexes: The read side (whether another owner still refers to a
            picture); ``None`` answers it from the owner's own files.
        eraser: Erases the index rows of what a forget takes out of default
            reads; hands what an undo puts back to the cascade.
        pipeline: The conversation pipeline's re-extraction; ``None`` refuses
            a forget that needs it.
        scenes: Writes a group's changed scene again; ``None`` marks it
            forgotten instead.
        extractions: The memorize extractions a forget waits for.
        session_wait_seconds: How long a forget waits for a busy
            conversation's lock before answering 503.
    """

    def __init__(
        self,
        *,
        stores: Stores,
        indexes: Indexes | None,
        eraser: IndexEraser,
        pipeline: Reextraction | None,
        scenes: SceneNarrator | None,
        extractions: ExtractionSupervisor,
        session_wait_seconds: float,
    ) -> None:
        self._stores = stores
        self._extractions = extractions
        self._wait = session_wait_seconds
        self._has_pipeline = pipeline is not None
        self._planner = ForgetPlanner(
            stores=stores,
            indexes=indexes,
            scene_members=scenes.min_members if scenes is not None else None,
        )
        self._writer = ForgetWriter(
            stores=stores, eraser=eraser, pipeline=pipeline, scenes=scenes
        )
        self._undoer = Undoer(stores=stores, eraser=eraser)

    # ── forget ───────────────────────────────────────────────────────────

    async def forget(
        self,
        *,
        scope: Scope,
        identity: Identity,
        target: ForgetTarget,
        reason: str | None = None,
        dry_run: bool = True,
        strict: bool = True,
    ) -> ForgetResult:
        """Forget ``target`` in ``scope``'s memory (or plan it, ``dry_run``).

        ``strict`` answers a record id that names nothing with 404; the
        delete-by-id alias passes ``False`` and skips it.

        Raises:
            InvalidInputError: No asker inside a group; a record id of no
                kind of the owner's.
            ForbiddenError: Not a current member, or not allowed this target.
            NotFoundError: The target names nothing the owner holds.
            SessionBusyError: A conversation of the forget stays busy (503).
        """
        group = scope.owner_type == "group"
        if group and identity.asker_id is None:
            raise InvalidInputError("asker_id is required for a group forget")
        async with forget_lock(scope):
            record = await self._member_record(scope, identity) if group else None
            plan = await self._planner.plan(scope, target, record, strict=strict)
            if record is not None:
                authorize_forget(record.roster, identity.asker_id, sources=plan.sources)
            if dry_run:
                return plan_result(plan, forget_id="", dry_run=True)
            if plan.reextracts and not self._has_pipeline:
                raise InvalidInputError("re-extraction is not available")
            forget_id = _mint_forget_id()
            already = dict(plan.already)
            await self._record_messages(plan, forget_id)
            try:
                if plan.sessions:
                    await self._settle(scope, plan.sessions)
                    plan = await self._planner.plan(
                        scope,
                        target,
                        record,
                        already=already if plan.messages else None,
                        strict=strict,
                    )
                    await self._record_messages(plan, forget_id)
                prepared = await self._writer.prepare(plan)
            except BaseException:
                await self._stores.message_forget.undo(forget_id)
                raise
            applied = Applied()
            try:
                await self._writer.write(plan, prepared, applied)
                async with self._writer.profile_guard(plan):
                    await self._writer.profiles(plan, applied)
                    await self._log(plan, applied, forget_id, reason, identity)
            except BaseException:
                await self._roll_back(plan, applied, forget_id)
                raise
        result = plan_result(plan, forget_id=forget_id, dry_run=False, applied=applied)
        logger.info(
            "memory.forgotten",
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_type=scope.owner_type,
            owner_id=scope.owner_id,
            forget_id=forget_id,
            target=target.kind,
            messages=len(result.messages),
            episodes=len(result.episodes),
            rewritten=result.rewritten,
            image_memories=len(result.image_memories),
            images=len(result.images),
            scenes=len(result.scenes),
            profile_items=len(result.profile_items) + len(result.stale_profile_items),
            group_items=len(result.group_profile_items)
            + len(result.stale_group_profile_items),
            reason=reason,
        )
        return result

    async def _log(
        self,
        plan: ForgetPlan,
        applied: Applied,
        forget_id: str,
        reason: str | None,
        identity: Identity,
    ) -> None:
        scope = plan.scope
        await self._stores.forget_log.record(
            target=plan.target.kind,
            target_id=plan.target.label(scope.owner_id),
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
            reason=reason,
            requested_by=identity.asker_id,
            undo=applied_record(plan, applied),
            forget_id=forget_id,
        )

    async def _roll_back(
        self, plan: ForgetPlan, applied: Applied, forget_id: str
    ) -> None:
        """Reverse a run that failed part-way, or keep what an undo needs.

        When the reversal fails too, the run is logged under its id as an
        incomplete forget, so ``/memory/forget/undo`` can finish the job.
        """
        scope = plan.scope
        record = applied_record(plan, applied)
        entry = ForgetEntry(
            forget_id=forget_id,
            target=plan.target.kind,
            target_id=plan.target.label(scope.owner_id),
            owner_id=scope.owner_id,
            owner_type=scope.owner_type,
            undo={**record, "segments": []},
            app_id=scope.app_id,
            project_id=scope.project_id,
        )
        try:
            await self._undoer.undo(scope, entry)
        except Exception:
            logger.exception("memory.forget_rollback_failed", forget_id=forget_id)
            await self._stores.forget_log.record(
                target=plan.target.kind,
                target_id=plan.target.label(scope.owner_id),
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
                owner_type=scope.owner_type,
                reason="incomplete forget",
                undo=record,
                forget_id=forget_id,
            )

    async def _record_messages(self, plan: ForgetPlan, forget_id: str) -> None:
        for session, ids in plan.messages.items():
            await self._stores.message_forget.record(session, ids, forget_id=forget_id)

    async def _member_record(self, scope: Scope, identity: Identity) -> GroupRecord:
        """The group, for a current member or its deliverer; a stranger is
        refused before anything."""
        record = await self._stores.groups.get(scope)
        if record is None or not admits(record.roster, identity.asker_id):
            raise ForbiddenError(
                f"the asker is not a current member of group {scope.owner_id!r}"
            )
        return record

    async def _settle(self, scope: Scope, sessions: list[str]) -> None:
        """Let a cut in progress finish, then wait for the conversations'
        extractions."""
        for session_id in sessions:
            key = session_key(scope.app_id, scope.project_id, session_id)
            try:
                async with LOCKS.hold("session", key, max_wait=self._wait):
                    pass
            except TimeoutError:
                raise SessionBusyError(
                    "a conversation of the forget is being memorized; nothing was "
                    "forgotten, retry later"
                ) from None
        left = await self._extractions.settle(sessions, _SETTLE_SECONDS)
        if left:
            logger.warning("memory.forget_extractions_running", running=left)

    # ── undo ─────────────────────────────────────────────────────────────

    async def undo(
        self, *, scope: Scope, identity: Identity, forget_id: str
    ) -> UndoResult:
        """Reverse the forget ``forget_id`` of ``scope``'s owner.

        Inside a group the undo takes the forget's own authorisation.

        Raises:
            InvalidInputError: No asker inside a group.
            ForbiddenError: Not allowed the forget it reverses.
            NotFoundError: The owner has no forget of that id.
            ConflictError: A later change replaced what this forget wrote.
        """
        group = scope.owner_type == "group"
        if group and identity.asker_id is None:
            raise InvalidInputError("asker_id is required for a group undo")
        async with forget_lock(scope):
            record = await self._member_record(scope, identity) if group else None
            entry = await self._stores.forget_log.get(forget_id)
            if (
                entry is None
                or entry.owner_type != scope.owner_type
                or entry.owner_id != scope.owner_id
                or entry.app_id != scope.app_id
                or entry.project_id != scope.project_id
                or (group and entry.undo is None)
            ):
                where = "in the group" if group else f"of user {scope.owner_id!r}"
                raise NotFoundError(f"no forget {forget_id!r} {where}")
            if record is not None:
                authorize_forget(
                    record.roster,
                    identity.asker_id,
                    sources=(entry.undo or {}).get("sources") or [],
                )
            return await self._undoer.undo(scope, entry)

    async def undo_picture(self, *, scope: Scope, image_id: str) -> UndoResult:
        """Undo the newest live forget of one picture of a user (the legacy
        undo body).

        A picture that was never forgotten is not an error: the answer says
        nothing was undone and names the occurrences that are current, which
        is what a caller retrying an undo after a timeout needs to see; the
        file's status is cleared all the same (a forget whose log write
        failed leaves it set).
        """
        undone = await self._undo_latest(scope, "image", image_id)
        if undone is not None:
            return undone
        await self._stores.images.restore(image_id)
        current = [
            row
            for row in await self._stores.image_memory.read_by_image(scope, image_id)
            if row.item.superseded_by is None
        ]
        return UndoResult(
            forget_id="",
            applied=False,
            pictures=tuple(
                ForgottenPicture(
                    entry_id=row.ref.entry_id, image_id=image_id, name=row.item.name
                )
                for row in current
            ),
        )

    async def undo_session(self, *, scope: Scope, session_id: str) -> UndoResult:
        """Undo the newest live forget of one conversation of a user (the
        legacy undo body); ``applied`` is false when there is none."""
        undone = await self._undo_latest(scope, "session", session_id)
        return undone or UndoResult(forget_id="", applied=False)

    async def _undo_latest(
        self, scope: Scope, target: Literal["image", "session"], target_id: str
    ) -> UndoResult | None:
        async with forget_lock(scope):
            entry = await self._stores.forget_log.latest_live(
                target=target,
                target_id=target_id,
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
                owner_type=scope.owner_type,
            )
            if entry is None:
                return None
            return await self._undoer.undo(scope, entry)


def plan_result(
    plan: ForgetPlan,
    *,
    forget_id: str,
    dry_run: bool,
    applied: Applied | None = None,
) -> ForgetResult:
    """What a forget answers: the plan, and what the run wrote."""
    scope = plan.scope

    def rows(entry_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(memory_id(scope, e) for e in entry_ids))

    if plan.is_group:
        scenes = [v.ref.entry_id for e in plan.scenes for v in e.versions]
    else:
        scenes = [e.current.ref.entry_id for e in plan.scenes if e.current]
        scenes += (
            [row[1] for row in applied.orphaned]
            if applied is not None
            else [s.ref.entry_id for s in plan.orphaned]
        )
    items, stale, count = _profile_answer(plan, applied)
    written = applied.written if applied is not None else {}
    return ForgetResult(
        forget_id=forget_id,
        dry_run=dry_run,
        target=plan.target,
        messages=tuple(
            (session.session_id, message_id)
            for session, ids in plan.messages.items()
            for message_id in ids
        ),
        episodes=rows(
            [
                *(r.old.ref.entry_id for r in plan.rewrites),
                *(e.ref.entry_id for e in plan.episodes),
            ]
        ),
        new_episodes=rows(row[3] for row in written.get("episode", [])),
        rewritten=len(plan.rewrites)
        if applied is None
        else len(written.get("episode", [])),
        image_memories=rows(m.ref.entry_id for m in plan.image_memories),
        pictures=tuple(
            ForgottenPicture(
                entry_id=m.ref.entry_id,
                image_id=m.item.image_id,
                name=m.item.name or "",
            )
            for m in plan.image_memories
        ),
        images=tuple(plan.files),
        scenes=rows(scenes),
        new_scenes=rows(row[3] for row in written.get("scene", [])),
        rewritten_scenes=sum(
            1 for e in plan.scenes if e.members or (e.image_id and e.current)
        )
        if applied is None
        else len(written.get("scene", [])),
        profiles=tuple(plan.members_removed),
        profile_items=items,
        profile_item_count=count,
        stale_profile_items=stale,
        group_profile_items=rows(
            s.ref.entry_id for s in [*plan.group_forgotten, *plan.group_rewrites]
        ),
        stale_group_profile_items=rows(s.ref.entry_id for s in plan.group_stale),
    )


def _profile_answer(
    plan: ForgetPlan, applied: Applied | None
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    """``(forgotten ids, stale ids, items)`` of the profile edits.

    A user's answer names every value taken out of ``user.md``, items and
    replaced values; a group's names the forgotten items (and the replaced
    values a record target named) apart from the stale ones.
    """
    forgotten: list[str] = []
    stale: list[str] = []
    count = 0
    if applied is not None:
        for change in applied.profiles:
            count += len(change["items"])
            if plan.is_group:
                forgotten += change.get("forgotten") or []
                stale += change.get("stale") or []
                continue
            values = [item for _b, _n, item in change["items"]]
            values += [entry for _n, entry in change["history"]]
            forgotten += [
                str(v["item_id"]) for v in values if isinstance(v.get("item_id"), str)
            ]
    else:
        for edit in plan.profiles:
            count += len(edit.items)
            forgotten += edit.forgotten
            if plan.is_group:
                stale += edit.stale
            else:
                forgotten += edit.stale + edit.history_ids
    return tuple(dict.fromkeys(forgotten)), tuple(dict.fromkeys(stale)), count


def _mint_forget_id() -> str:
    """A new forget's id (``fg_<16 hex>``), minted before anything is recorded."""
    return f"fg_{uuid.uuid4().hex[:16]}"
