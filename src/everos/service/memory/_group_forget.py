"""Forget and undo inside a group: the use case behind group-scoped ``/memory/forget``.

docs/group-memory.md, sections 5.5 and 9.4. What a target touches is planned
first (:mod:`._group_forget_plan`, reads only); ``dry_run`` answers with the
plan, and the real run carries out exactly that plan:

1. The target's messages are recorded in ``message_forget``, so every reader
   that starts afterwards skips them -- re-extraction, the profiles' next
   updates, a buffer read. The sessions they belong to are then settled: a
   cut in progress finishes (its lock) and the extractions already running
   for them end, so the plan, made again now, misses nothing they wrote.
2. Every model call is made before anything is written: each segment that
   lost messages is extracted again (the group episode call, labels and
   roster block included) over what is left of it, and each changed scene is
   narrated again from its remaining members. A failure here undoes step 1
   and leaves the group as it was.
3. The new versions are appended; then the old ones -- every record that may
   say what was forgotten -- are **taken out of markdown**, not marked: the
   forgotten text must appear nowhere under the memory root, and no index
   row may keep it. The forget log holds what was taken out, byte for byte,
   in ``system.db`` (``undo_json``), and the memcell payload keeps the
   messages themselves; nothing else does.
4. Profile items follow the source rule: forgotten when all their sources
   are, otherwise stale -- taken out too, and rewritten from their remaining
   segments by the profile's next update, which they are queued on.

An undo takes the ``forget_id``, re-checks the forget's own authorisation,
and reverses it exactly: what the forget wrote is taken out, what it took out
is put back under its own ids, the profiles get their items back, the queued
segments leave the queues, the messages' marks are stamped undone. An undo
needs the forget's own writes to still be there: a later forget that rewrote
them must be undone first (409).
"""

from __future__ import annotations

__all__ = [
    "GroupForgetManager",
    "GroupForgetResult",
    "GroupForgetTarget",
    "GroupReextraction",
    "GroupUndoResult",
]

import uuid
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from everos.core.errors import (
    ConflictError,
    ForbiddenError,
    InvalidInputError,
    NotFoundError,
    SessionBusyError,
)
from everos.core.locks import LOCKS, session_key
from everos.core.observability.logging import get_logger
from everos.memory import (
    forget_lock,
    group_profile_lock,
    profile_lock,
    scenes_lock,
)
from everos.memory.access import Identity, authorize_forget
from everos.memory.extract.pipeline.group_scene import scene_version
from everos.memory.extract.pipeline.pictures import memcell_alive
from everos.memory.models import (
    Cluster,
    Episode,
    MemberProfile,
)
from everos.memory.ports import (
    NO_SOURCE_CHECK,
    MemoryRef,
    Scope,
    SessionKey,
    Stored,
)

from ._group_forget_plan import (
    ForgetPlan,
    ForgetPlanner,
    GroupForgetTarget,
    member_edit,
)
from ._group_forget_records import (
    Applied,
    GroupForgetResult,
    GroupUndoResult,
    forget_result,
    merge_withdrawals,
    row_ids,
    undo_record,
    with_items,
    withdrawal_from,
    withdrawn_ref,
    without_items,
)

if TYPE_CHECKING:
    from everalgo.types import Episode as AlgoEpisode

    from everos.memory.deps import Indexes, Stores
    from everos.memory.models import MemCell
    from everos.memory.ports import IndexEraser

    from ._extraction import ExtractionSupervisor

logger = get_logger("everos.service.memory")

_SETTLE_SECONDS = 120.0
"""How long a forget waits for the running extractions of its sessions."""

_EPISODE_TABLE = "episode"
"""Group episodes and scene versions share this index table (section 7.7)."""


@runtime_checkable
class GroupReextraction(Protocol):
    """What a forget needs of the group pipeline: the episode call, and scenes'."""

    async def reextract(
        self, *, scope: Scope, session_id: str, memcell_id: str, cell: MemCell
    ) -> Episode | None: ...

    async def scene_narrative(
        self,
        *,
        owner_id: str,
        members: Sequence[Stored[Episode]],
        language: str | None,
    ) -> AlgoEpisode: ...


class GroupForgetManager:
    """Forgets inside a group and their undos.

    Args:
        stores: Markdown and system state.
        indexes: The read side (whether another owner still refers to a
            picture); ``None`` keeps every picture file in circulation.
        eraser: Erases the index rows of what goes, and queues what an undo
            puts back for the cascade.
        pipeline: The group pipeline: re-extraction and scene narratives.
        extractions: The memorize extractions, which a forget waits for.
        session_wait_seconds: How long a forget waits for a busy session's
            lock before answering 503.
    """

    def __init__(
        self,
        *,
        stores: Stores,
        indexes: Indexes | None,
        eraser: IndexEraser,
        pipeline: GroupReextraction | None,
        extractions: ExtractionSupervisor,
        session_wait_seconds: float,
    ) -> None:
        self._stores = stores
        self._eraser = eraser
        self._pipeline = pipeline
        self._extractions = extractions
        self._wait = session_wait_seconds
        self._planner = ForgetPlanner(stores=stores, indexes=indexes)

    # ── forget ───────────────────────────────────────────────────────────

    async def forget(
        self,
        *,
        scope: Scope,
        identity: Identity,
        target: GroupForgetTarget,
        reason: str | None = None,
        dry_run: bool = False,
    ) -> GroupForgetResult:
        """Forget ``target`` in the group ``scope`` (or plan it, ``dry_run``).

        Raises:
            InvalidInputError: No asker.
            ForbiddenError: The asker is not a current member, or may not
                forget what the target rests on (403).
            NotFoundError: The group does not hold what the target names.
            SessionBusyError: A session the forget touches stays busy (503).
        """
        if identity.asker_id is None:
            raise InvalidInputError("asker_id is required for a group forget")
        asker = identity.asker_id
        async with forget_lock(scope):
            record = await self._member_record(scope, asker)
            plan = await self._planner.plan(scope, target, record)
            authorize_forget(record.roster, asker, sources=plan.sources)
            if dry_run:
                return forget_result(plan, forget_id="", dry_run=True)
            if plan.rewrites and self._pipeline is None:
                raise InvalidInputError("group re-extraction is not available")
            forget_id = _mint_forget_id()
            already = dict(plan.already)
            for session, ids in plan.messages.items():
                await self._stores.message_forget.record(
                    session, ids, forget_id=forget_id
                )
            try:
                await self._settle(plan.messages)
                plan = await self._planner.plan(scope, target, record, already=already)
                # A message cut while the sessions settled is the target's too.
                for session, ids in plan.messages.items():
                    await self._stores.message_forget.record(
                        session, ids, forget_id=forget_id
                    )
                episodes = await self._reextract(plan)
            except BaseException:
                await self._stores.message_forget.undo(forget_id)
                raise
            applied = Applied()
            try:
                await self._apply(plan, episodes, applied)
            except BaseException:
                # All or nothing: what this run changed goes back, and a retry
                # starts from the group as it was.
                await self._roll_back(scope, forget_id, undo_record(plan, applied))
                raise
            await self._stores.forget_log.record(
                target=target.kind,
                target_id=target.label(),
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
                owner_type="group",
                reason=reason,
                requested_by=asker,
                undo=undo_record(plan, applied),
                forget_id=forget_id,
            )
        result = forget_result(
            plan, forget_id=forget_id, dry_run=False, applied=applied
        )
        logger.info(
            "memory.group.forgotten",
            app_id=scope.app_id,
            project_id=scope.project_id,
            group_id=scope.owner_id,
            forget_id=forget_id,
            target=target.kind,
            messages=len(result.messages),
            episodes=len(result.episodes),
            rewritten=result.rewritten,
            image_memories=len(result.image_memories),
            scenes=len(result.scenes),
            member_items=len(result.member_profile_items)
            + len(result.stale_member_profile_items),
            group_items=len(result.group_profile_items)
            + len(result.stale_group_profile_items),
            reason=reason,
        )
        return result

    async def _roll_back(
        self, scope: Scope, forget_id: str, undo: Mapping[str, Any]
    ) -> None:
        """Reverse a run that failed part-way, or keep what an undo needs.

        When the reversal itself fails, the run's undo record is logged under
        its id instead, so ``/memory/forget/undo`` can finish the job.
        """
        try:
            await self._reverse(scope, forget_id, undo)
        except Exception:
            logger.exception("memory.group.forget_rollback_failed", forget_id=forget_id)
            await self._stores.forget_log.record(
                target="message",
                target_id="rollback",
                app_id=scope.app_id,
                project_id=scope.project_id,
                owner_id=scope.owner_id,
                owner_type="group",
                reason="incomplete forget",
                undo=undo,
                forget_id=forget_id,
            )

    async def _member_record(self, scope: Scope, asker: str) -> Any:
        """The group, for a current member; a stranger is refused before anything."""
        record = await self._stores.groups.get(scope)
        if record is None or not record.roster.is_current_member(asker):
            raise ForbiddenError(
                f"the asker is not a current member of group {scope.owner_id!r}"
            )
        return record

    async def _settle(self, messages: Mapping[SessionKey, Sequence[str]]) -> None:
        """Let a cut in progress finish, then wait for the sessions' extractions."""
        for session in messages:
            key = session_key(session.app_id, session.project_id, session.session_id)
            try:
                async with LOCKS.hold("session", key, max_wait=self._wait):
                    pass
            except TimeoutError:
                raise SessionBusyError(
                    "a session of the forget is busy; nothing was forgotten, "
                    "retry later"
                ) from None
        left = await self._extractions.settle(
            [s.session_id for s in messages], _SETTLE_SECONDS
        )
        if left:
            logger.warning("memory.group.forget_extractions_running", running=left)

    async def _reextract(self, plan: ForgetPlan) -> list[Episode]:
        """The new episode version of every rewrite, in order (model calls)."""
        if not plan.rewrites:
            return []
        assert self._pipeline is not None
        pipeline = self._pipeline
        episodes: list[Episode] = []
        for rewrite in plan.rewrites:
            episode = await pipeline.reextract(
                scope=plan.scope,
                session_id=rewrite.session_id,
                memcell_id=rewrite.memcell_id,
                cell=rewrite.cell,
            )
            if episode is None:  # the plan checked; a cell cannot lose its text
                raise ConflictError("a segment changed while it was forgotten")
            episodes.append(episode)
        return episodes

    async def _apply(
        self, plan: ForgetPlan, episodes: Sequence[Episode], applied: Applied
    ) -> None:
        scope = plan.scope
        # The new versions first: a failure later leaves the old ones too,
        # never neither.
        replaced: dict[str, MemoryRef] = {}
        for rewrite, episode in zip(plan.rewrites, episodes, strict=True):
            ref = await self._stores.episode.append(
                scope,
                episode,
                only_if=memcell_alive(self._stores.memcell, rewrite.memcell_id),
            )
            if ref is not None:
                applied.written_episodes.append(ref)
                replaced[rewrite.old.ref.entry_id] = ref
        await self._scenes(plan, replaced, applied)
        leaving = [r.old.ref.entry_id for r in plan.rewrites] + [
            e.ref.entry_id for e in plan.dropped
        ]
        await self._withdraw("episode", self._stores.episode, scope, leaving, applied)
        await self._withdraw(
            "image_memory",
            self._stores.image_memory,
            scope,
            [m.ref.entry_id for m in plan.image_memories],
            applied,
        )
        await self._member_profiles(plan, applied)
        await self._group_profile(plan, applied)
        for image_id in plan.files:
            await self._stores.images.forget(image_id)
            applied.files.append(image_id)

    async def _withdraw(
        self,
        kind: str,
        store: Any,
        scope: Scope,
        entry_ids: Collection[str],
        applied: Applied,
    ) -> None:
        """Take entries out of markdown, erase their rows, keep them for an undo."""
        if not entry_ids:
            return
        withdrawal = await store.withdraw(scope, entry_ids)
        previous = applied.withdrawn.get(kind)
        applied.withdrawn[kind] = (
            withdrawal if previous is None else merge_withdrawals(previous, withdrawal)
        )
        table = {"scene": _EPISODE_TABLE, "group_profile": "group_profile_item"}.get(
            kind, kind
        )
        refs = [withdrawn_ref(scope, entry) for entry in withdrawal.entries]
        if refs:
            await self._eraser.erase(scope, {table: refs})
        if withdrawal.released:
            await self._eraser.requeue(
                kind, {md_path for md_path, _, _ in withdrawal.released}
            )

    async def _scenes(
        self, plan: ForgetPlan, replaced: Mapping[str, MemoryRef], applied: Applied
    ) -> None:
        """Each changed scene: its versions go, a new one is narrated afresh."""
        if not plan.scenes:
            return
        scope = plan.scope
        assert self._pipeline is not None or all(not e.members for e in plan.scenes)
        async with scenes_lock(scope):
            for edit in plan.scenes:
                ids = [
                    replaced[m].entry_id if m in replaced else m for m in edit.members
                ]
                members = sorted(
                    await self._stores.episode.read_many(scope, ids),
                    key=lambda s: s.item.timestamp,
                )
                if len(members) >= 2 and self._pipeline is not None:
                    newest = max(edit.versions, key=lambda v: v.item.version)
                    narrative = await self._pipeline.scene_narrative(
                        owner_id=scope.owner_id,
                        members=members,
                        language=newest.item.language,
                    )
                    scene = scene_version(
                        scope,
                        scene_id=edit.scene_id,
                        version=newest.item.version + 1,
                        members=members,
                        narrative=narrative,
                        language=newest.item.language,
                    )
                    ref = await self._stores.scene.append(
                        scope,
                        scene,
                        supersedes=edit.newest_kept,
                        only_if=NO_SOURCE_CHECK,
                    )
                    if ref is not None:
                        applied.written_scenes.append(ref)
                await self._withdraw(
                    "scene",
                    self._stores.scene,
                    scope,
                    [v.ref.entry_id for v in edit.versions],
                    applied,
                )
                await self._cluster(scope, edit.scene_id, replaced, plan, applied)

    async def _cluster(
        self,
        scope: Scope,
        scene_id: str,
        replaced: Mapping[str, MemoryRef],
        plan: ForgetPlan,
        applied: Applied,
    ) -> None:
        """The scene's cluster follows its episodes; a forgotten scene's goes."""
        cluster = await self._stores.cluster.get(scene_id)
        if cluster is None:
            return
        whole = any(e.scene_id == scene_id and e.whole for e in plan.scenes)
        dropped = {e.ref.entry_id for e in plan.dropped}
        applied.clusters.append(
            {"cluster": cluster.model_dump(mode="json"), "whole": whole}
        )
        if whole:
            await self._stores.cluster.delete(scene_id)
            return
        members = [
            replaced[m].entry_id if m in replaced else m
            for m in cluster.members
            if m not in dropped
        ]
        await self._stores.cluster.delete(scene_id)
        if members:
            await self._stores.cluster.upsert(
                scope,
                "group_scene",
                cluster.model_copy(
                    update={"members": members, "count": len(members), "preview": []}
                ),
                member_type="episode",
            )

    async def _member_profiles(self, plan: ForgetPlan, applied: Applied) -> None:
        """Whole profiles of a member target; the source rule on the others."""
        scope = plan.scope
        for member_id in plan.members_removed:
            async with profile_lock(scope, member_id):
                removed = await self._stores.member_profile.remove(scope, member_id)
            if removed is None:
                continue
            await self._eraser.erase(scope, {"user_profile": [removed.ref]})
            applied.members_removed.append(
                {
                    "member_id": member_id,
                    "profile": removed.item.model_dump(mode="json"),
                }
            )
        for planned in plan.members:
            member_id = planned.member_id
            async with profile_lock(scope, member_id):
                held = await self._stores.member_profile.read(scope, member_id)
                if held is None:
                    continue
                edit = member_edit(
                    held, plan.readable, set(plan.pictures), plan.record_items
                )
                if not edit.changes:
                    continue
                kept, record = without_items(held, edit)
                await self._stores.member_profile.write(scope, kept)
            applied.member_items.append(record)

    async def _group_profile(self, plan: ForgetPlan, applied: Applied) -> None:
        """Forgotten and stale items go; the stale ones' segments are queued."""
        ids = [s.ref.entry_id for s in (*plan.group_forgotten, *plan.group_stale)]
        if not ids and not plan.group_queue:
            return
        scope = plan.scope
        async with group_profile_lock(scope.app_id, scope.project_id, scope.owner_id):
            await self._withdraw(
                "group_profile", self._stores.group_profile, scope, ids, applied
            )
            if plan.group_queue:
                pending = await self._stores.group_profile.pending(scope)
                added = [m for m in plan.group_queue if m not in pending]
                if added:
                    await self._stores.group_profile.set_pending(
                        scope, [*pending, *added]
                    )
                    applied.group_queued.extend(added)

    # ── undo ─────────────────────────────────────────────────────────────

    async def undo(
        self, *, scope: Scope, identity: Identity, forget_id: str
    ) -> GroupUndoResult:
        """Reverse the forget ``forget_id`` of the group ``scope``.

        Raises:
            InvalidInputError: No asker.
            ForbiddenError: The asker may not forget what the forget addressed.
            NotFoundError: No forget of this group has that id.
            ConflictError: A later forget rewrote what this one wrote (409):
                undo the later one first.
        """
        if identity.asker_id is None:
            raise InvalidInputError("asker_id is required for a group undo")
        asker = identity.asker_id
        async with forget_lock(scope):
            record = await self._member_record(scope, asker)
            entry = await self._stores.forget_log.get(forget_id)
            if (
                entry is None
                or entry.owner_type != "group"
                or entry.owner_id != scope.owner_id
                or entry.app_id != scope.app_id
                or entry.project_id != scope.project_id
                or entry.undo is None
            ):
                raise NotFoundError(f"no forget {forget_id!r} in the group")
            undo = entry.undo
            authorize_forget(record.roster, asker, sources=undo.get("sources") or [])
            if entry.undone:
                return GroupUndoResult(forget_id=forget_id, applied=False)
            result = await self._reverse(scope, forget_id, undo)
            await self._stores.forget_log.mark_undone(forget_id)
        logger.info(
            "memory.group.undo_completed",
            app_id=scope.app_id,
            project_id=scope.project_id,
            group_id=scope.owner_id,
            forget_id=forget_id,
            episodes=len(result.episodes),
            image_memories=len(result.image_memories),
            scenes=len(result.scenes),
        )
        return result

    async def _reverse(
        self, scope: Scope, forget_id: str, undo: Mapping[str, Any]
    ) -> GroupUndoResult:
        written = undo.get("written") or {}
        episodes_written = [str(e) for e in written.get("episode") or []]
        scenes_written = [str(e) for e in written.get("scene") or []]
        present = {
            s.ref.entry_id
            for s in await self._stores.episode.read_many(scope, episodes_written)
        }
        versions = {v.ref.entry_id for v in await self._stores.scene.versions(scope)}
        if set(episodes_written) - present or set(scenes_written) - versions:
            raise ConflictError(
                "a later forget changed what this one wrote; undo that one first"
            )
        withdrawn_now: list[str] = []
        taken = await self._stores.episode.withdraw(scope, episodes_written)
        if taken.entries:
            await self._eraser.erase(
                scope,
                {_EPISODE_TABLE: [withdrawn_ref(scope, e) for e in taken.entries]},
            )
            withdrawn_now += [f"{scope.owner_id}_{e.entry_id}" for e in taken.entries]
        scene_taken = await self._stores.scene.withdraw(scope, scenes_written)
        if scene_taken.entries:
            await self._eraser.erase(
                scope,
                {
                    _EPISODE_TABLE: [
                        withdrawn_ref(scope, e) for e in scene_taken.entries
                    ]
                },
            )
            withdrawn_now += [
                f"{scope.owner_id}_{e.entry_id}" for e in scene_taken.entries
            ]
        stash = undo.get("withdrawn") or {}
        restored: dict[str, list[MemoryRef]] = {}
        stores = {
            "episode": self._stores.episode,
            "image_memory": self._stores.image_memory,
            "scene": self._stores.scene,
            "group_profile": self._stores.group_profile,
        }
        for kind, store in stores.items():
            if kind not in stash:
                continue
            withdrawal = withdrawal_from(stash[kind])
            refs = await store.reinstate(scope, withdrawal)
            restored[kind] = refs
            paths = {e.md_path for e in withdrawal.entries} | {
                md_path for md_path, _, _ in withdrawal.released
            }
            if paths:
                await self._eraser.requeue(kind, paths)
        members_back = await self._restore_members(scope, undo)
        items_back = await self._restore_member_items(scope, undo)
        await self._restore_group_queue(scope, undo)
        await self._restore_clusters(scope, undo)
        for image_id in undo.get("files") or []:
            await self._stores.images.restore(image_id)
        await self._stores.message_forget.undo(forget_id)
        messages = tuple(
            (str(session_id), str(message_id))
            for session_id, ids in undo.get("messages") or []
            for message_id in ids
        )
        return GroupUndoResult(
            forget_id=forget_id,
            applied=True,
            messages=messages,
            episodes=row_ids(scope, restored.get("episode", [])),
            image_memories=row_ids(scope, restored.get("image_memory", [])),
            images=tuple(str(i) for i in undo.get("files") or []),
            scenes=row_ids(scope, restored.get("scene", [])),
            member_profiles=tuple(members_back),
            member_profile_items=tuple(items_back),
            group_profile_items=row_ids(scope, restored.get("group_profile", [])),
            withdrawn=tuple(withdrawn_now),
        )

    async def _restore_members(
        self, scope: Scope, undo: Mapping[str, Any]
    ) -> list[str]:
        back: list[str] = []
        for removed in undo.get("members_removed") or []:
            member_id = str(removed["member_id"])
            profile = MemberProfile.model_validate(removed["profile"])
            async with profile_lock(scope, member_id):
                if await self._stores.member_profile.read(scope, member_id) is None:
                    await self._stores.member_profile.write(scope, profile)
                    back.append(member_id)
        return back

    async def _restore_member_items(
        self, scope: Scope, undo: Mapping[str, Any]
    ) -> list[str]:
        back: list[str] = []
        for change in undo.get("member_items") or []:
            member_id = str(change["member_id"])
            async with profile_lock(scope, member_id):
                held = await self._stores.member_profile.read(scope, member_id)
                if held is None:
                    continue
                restored, ids = with_items(held, change)
                await self._stores.member_profile.write(scope, restored)
            back += ids
        return back

    async def _restore_group_queue(self, scope: Scope, undo: Mapping[str, Any]) -> None:
        queued = set(undo.get("group_queued") or [])
        if not queued:
            return
        async with group_profile_lock(scope.app_id, scope.project_id, scope.owner_id):
            pending = await self._stores.group_profile.pending(scope)
            left = [m for m in pending if m not in queued]
            if left != pending:
                await self._stores.group_profile.set_pending(scope, left)

    async def _restore_clusters(self, scope: Scope, undo: Mapping[str, Any]) -> None:
        changes = undo.get("clusters") or []
        if not changes:
            return
        async with scenes_lock(scope):
            for change in reversed(changes):
                cluster = Cluster.model_validate(change["cluster"])
                if cluster.id is None:
                    continue
                await self._stores.cluster.delete(cluster.id)
                await self._stores.cluster.upsert(
                    scope, "group_scene", cluster, member_type="episode"
                )


def _mint_forget_id() -> str:
    """A new forget's id (``fg_<16 hex>``), minted before anything is recorded."""
    return f"fg_{uuid.uuid4().hex[:16]}"
