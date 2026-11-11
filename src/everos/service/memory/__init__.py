"""MemoryService -- the memory read / write use cases behind one facade.

One method per use case: :meth:`MemoryService.add` (``/add``),
:meth:`~MemoryService.flush` (``/flush``), :meth:`~MemoryService.search` and
:meth:`~MemoryService.get`; ``add`` and ``flush`` run inline or hand the
request to the ``async_mode`` queue.

The add cycle lives in ``_memorize`` (lock, cut, extract, publish);
``_extraction`` keeps an extraction alive past a cancelled request, and
``_queue`` is the ``async_mode`` worker (steps in ``_queue_steps``, data in
``_queue_models``).
Everything is injected by :func:`everos.service.build_application`; the
service owns only those two background collaborators (extraction supervisor,
queue worker) and their lifecycle (:meth:`~MemoryService.start` /
:meth:`~MemoryService.aclose`).
"""

from __future__ import annotations

__all__ = [
    "TASK_STATUSES",
    "DeleteResult",
    "ExtractionSupervisor",
    "FlushResult",
    "ForgetResult",
    "ForgottenPicture",
    "GroupForgetResult",
    "GroupForgetTarget",
    "GroupUndoResult",
    "GroupUpsertResult",
    "GroupView",
    "MemorizeResult",
    "MemoryService",
    "ProjectMember",
    "ProjectProfile",
    "ProjectView",
    "QueueHealth",
    "Queued",
    "RevisionResult",
    "RosterChangeResult",
    "RosterEventInput",
    "SessionForgetResult",
    "SessionUndoResult",
    "TaskItem",
    "TaskPage",
    "UndoResult",
    "to_task_item",
]

import asyncio
from typing import TYPE_CHECKING, Any

from everos.component.images import ImageLimits
from everos.core.errors import (
    ConflictError,
    ForbiddenError,
    InvalidInputError,
    NotFoundError,
    PathTraversalError,
    SessionBusyError,
)
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_utc_now, to_timestamp_ms
from everos.memory.access import (
    AccessPolicy,
    Identity,
    authorize_manage,
    resolve_identity,
)
from everos.memory.extract.ingest.multimodal import (
    PictureBudget,
    keep_question_pictures,
)
from everos.memory.ports import Scope, SessionKey, all_of, eq, is_null

from ._delete import Deleter
from ._delete import DeleteResult as DeleteResult
from ._extraction import ExtractionSupervisor as ExtractionSupervisor
from ._forget import ForgetManager
from ._forget import ForgetResult as ForgetResult
from ._forget import ForgottenPicture as ForgottenPicture
from ._forget_session import SessionForgetManager
from ._forget_session import SessionForgetResult as SessionForgetResult
from ._group_forget import GroupForgetManager, GroupReextraction
from ._group_forget import GroupForgetResult as GroupForgetResult
from ._group_forget import GroupForgetTarget as GroupForgetTarget
from ._group_forget import GroupUndoResult as GroupUndoResult
from ._groups import GroupUpsertResult as GroupUpsertResult
from ._groups import GroupUseCases
from ._groups import GroupView as GroupView
from ._groups import RosterChangeResult as RosterChangeResult
from ._groups import RosterEventInput as RosterEventInput
from ._memorize import Memorizer, flush_payload
from ._models import FlushResult as FlushResult
from ._models import MemorizeResult as MemorizeResult
from ._models import Queued as Queued
from ._projects import ProjectMember as ProjectMember
from ._projects import ProjectProfile as ProjectProfile
from ._projects import ProjectUseCases
from ._projects import ProjectView as ProjectView
from ._queue import MemorizeQueueWorker
from ._queue_models import TASK_STATUSES as TASK_STATUSES
from ._queue_models import QueueHealth as QueueHealth
from ._queue_models import TaskItem as TaskItem
from ._queue_models import TaskPage as TaskPage
from ._queue_models import to_task_item as to_task_item
from ._revise import RevisionManager
from ._revise import RevisionResult as RevisionResult
from ._undo import SessionUndoResult as SessionUndoResult
from ._undo import UndoManager
from ._undo import UndoResult as UndoResult

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from everalgo.llm import LLMClient

    from everos.component.parser import RawFileBuilder
    from everos.config import Settings
    from everos.memory.deps import Indexes, Stores
    from everos.memory.extract.boundary import BoundaryDetection
    from everos.memory.extract.pipeline import Pipeline
    from everos.memory.get import GetManager, GetRequest, GetResponse
    from everos.memory.groups import RosterManager
    from everos.memory.ports import EventBus, IndexEraser
    from everos.memory.search import SearchRequest, SearchResponse
    from everos.memory.search.manager import SearchManager

logger = get_logger(__name__)

_DRAIN_TIMEOUT_SECONDS = 30.0
"""How long :meth:`MemoryService.aclose` waits for extractions still running."""

_GROUP_FORGET_ANSWERS: tuple[type[BaseException], ...] = (
    ForbiddenError,
    NotFoundError,
    ConflictError,
    InvalidInputError,
    SessionBusyError,
)
"""What a group forget answers the caller with, rather than fails with."""


class MemoryService:
    """Memorize, flush, search and get, over injected domain objects and ports.

    Args:
        settings: Application settings (``[memorize]``).
        boundary: Cuts each session's buffered messages into memcells.
        multimodal_llm: Parses non-text content; ``None`` when
            ``[multimodal]`` is not configured.
        stores: Write-side stores (session buffer, memcells, memorize inbox).
        user_pipeline: The user-memory track, run on every cell.
        agent_pipeline: The agent-memory track, run on every cell in agent mode.
        bus: Where each cell's events are published.
        search_manager: Answers :meth:`search`.
        get_manager: Answers :meth:`get`.
        raw_file_builder: Host file hydration capability for ingestion.
        index_eraser: Removes deleted memories' index rows (:meth:`delete`).
    """

    def __init__(
        self,
        *,
        settings: Settings,
        boundary: BoundaryDetection,
        multimodal_llm: LLMClient | None,
        stores: Stores,
        user_pipeline: Pipeline,
        agent_pipeline: Pipeline,
        bus: EventBus,
        search_manager: SearchManager,
        get_manager: GetManager,
        raw_file_builder: RawFileBuilder,
        index_eraser: IndexEraser,
        indexes: Indexes | None = None,
        image_limits: ImageLimits | None = None,
        rosters: RosterManager | None = None,
        group_pipeline: Pipeline | None = None,
        access: AccessPolicy | None = None,
    ) -> None:
        self._search_manager = search_manager
        self._indexes = indexes
        self._access = access or AccessPolicy(
            stores.groups, forgotten=stores.message_forget
        )
        self._get_manager = get_manager
        self._extractions = ExtractionSupervisor()
        self._stores = stores
        self._raw_file_builder = raw_file_builder
        self._image_limits = image_limits or _image_limits_from(settings)
        self._deleter = Deleter(
            stores=stores, eraser=index_eraser, settings=settings.memorize
        )
        # Forgetting a picture has to find what referred to it, which is an
        # index lookup, so it needs the read side as well as the write side.
        self._forgetter = (
            ForgetManager(
                stores=stores,
                indexes=indexes,
                images=stores.images,
                eraser=index_eraser,
            )
            if indexes is not None
            else None
        )
        # Undo needs no index read: what a forget changed is in the log and in
        # markdown, which is where the picture's occurrences are read from.
        self._undoer = UndoManager(
            stores=stores, images=stores.images, eraser=index_eraser
        )
        self._reviser = RevisionManager(stores=stores)
        self._session_forgetter = SessionForgetManager(
            stores=stores, images=stores.images, eraser=index_eraser
        )
        # Forget inside a group (docs/group-memory.md, 9.4): re-extraction
        # runs the group pipeline's own episode call.
        self._group_forgetter = GroupForgetManager(
            stores=stores,
            indexes=indexes,
            eraser=index_eraser,
            pipeline=group_pipeline
            if isinstance(group_pipeline, GroupReextraction)
            else None,
            extractions=self._extractions,
            session_wait_seconds=settings.memorize.delete_wait_seconds,
        )
        self._memorizer = Memorizer(
            settings=settings,
            boundary=boundary,
            multimodal_llm=multimodal_llm,
            user_pipeline=user_pipeline,
            agent_pipeline=agent_pipeline,
            bus=bus,
            extractions=self._extractions,
            raw_file_builder=raw_file_builder,
            image_store=stores.images,
            vision=settings.extraction.vision,
            rosters=rosters,
            group_pipeline=group_pipeline,
        )
        self._groups = (
            GroupUseCases(
                rosters=rosters,
                memorizer=self._memorizer,
                clock=lambda: to_timestamp_ms(get_utc_now()),
            )
            if rosters is not None
            else None
        )
        self._projects = ProjectUseCases(
            self._access, indexes.group_profile if indexes is not None else None
        )
        queue_cfg = settings.memorize.async_queue
        self._queue = MemorizeQueueWorker(
            queue_cfg,
            inbox=stores.memorize_inbox,
            deliver=self._memorize,
            accept_async=queue_cfg.enabled,
        )

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start delivering the ``async_mode`` queue."""
        if not self._queue.accepting:
            logger.info("memorize_queue_not_accepting_async")
        await self._queue.start()

    async def aclose(self) -> None:
        """Stop the queue, then wait (bounded) for extractions still running."""
        await self._queue.stop()
        await self._extractions.shutdown(_DRAIN_TIMEOUT_SECONDS)

    # ── read use cases ─────────────────────────────────────────────────────

    async def search(
        self, req: SearchRequest, *, asserted_asker: str | None = None
    ) -> SearchResponse:
        """Multi-round retrieval over a user's or a group's memory.

        Pictures attached to the question are stored first (content-addressed,
        no memory, no reference) and handed to the manager as content hashes,
        so the decider can look at them. A picture the store rejects is dropped
        with a log line: the text question is still answerable.

        ``asserted_asker`` is the asker a gateway asserted through the
        identity header (``[access] identity_header``); the read scope is
        decided by the access policy either way.
        """
        if not req.images:
            return await self._search_manager.search(req, asserted_asker=asserted_asker)
        stored = await keep_question_pictures(
            req.images,
            store=self._stores.images,
            limits=self._image_limits,
            raw_file_builder=self._raw_file_builder,
            budget=PictureBudget(
                max_bytes=self._image_limits.max_request_bytes,
                max_pictures=self._image_limits.max_per_request,
            ),
        )
        return await self._search_manager.search(
            req,
            question_image_ids=[image.image_id for image in stored],
            asserted_asker=asserted_asker,
        )

    async def picture_visible(
        self, image_id: str, *, asker_id: str | None, asserted_asker: str | None
    ) -> bool:
        """Whether a picture's bytes may be served to this asker (row 7).

        A hash only personal memories refer to is served as before; one a
        live group picture memory refers to needs an asker who can see one of
        them. The route answers an invisible picture exactly as a missing one.

        Raises:
            ForbiddenError: The identity header and ``asker_id`` disagree.
        """
        identity = resolve_identity(header=asserted_asker, body_asker=asker_id)
        if self._indexes is None:
            return True
        hits = await self._indexes.image_memory.find(
            all_of(eq("image_id", image_id), is_null("deprecated_by")), limit=1000
        )
        references = [hit.stored for hit in hits]
        return await self._access.picture_visible(
            identity,
            references,
            # Asked only of a hash nothing refers to any more: whether a group
            # forget took its last memory out (docs/group-memory.md, 9.4).
            forgotten_in_group=not references
            and await self._stores.forget_log.holds_picture(image_id),
        )

    async def get(
        self, req: GetRequest, *, asserted_asker: str | None = None
    ) -> GetResponse:
        """Paginated listing over the requested ``memory_type``, in the read scope."""
        return await self._get_manager.get(req, asserted_asker=asserted_asker)

    # ── group use cases ────────────────────────────────────────────────────

    def _group_cases(self) -> GroupUseCases:
        if self._groups is None:
            raise InvalidInputError("group memory is not available")
        return self._groups

    async def upsert_group(self, **request: Any) -> GroupUpsertResult:
        """Create a group or update its meta (``PUT /groups/{id}``).

        Raises:
            ForbiddenError: The asker may not manage the existing group.
            RosterConflictError: The roster does not fold.
            InvalidInputError: No asker.
        """
        return await self._group_cases().upsert(**request)

    async def change_roster(self, **request: Any) -> RosterChangeResult:
        """Append roster events (``POST /groups/{id}/roster``).

        The group's sessions' open buffers are cut first (D18).

        Raises:
            ForbiddenError: Not a current member, or not allowed this event.
            RosterConflictError: The events do not fold.
        """
        return await self._group_cases().change(**request)

    async def read_group(self, **request: Any) -> GroupView:
        """A group's meta and roster, for a current member (``GET /groups/{id}``)."""
        return await self._group_cases().read(**request)

    async def read_project(self, **request: Any) -> ProjectView:
        """The asker's groups of a project and their members (``GET /projects/{id}``).

        Raises:
            ForbiddenError: The asker is in none of the project's groups.
            InvalidInputError: No asker.
        """
        return await self._projects.read(**request)

    async def project_profile(self, **request: Any) -> ProjectProfile:
        """The group profiles the asker may see across a project (section 8.7).

        Raises:
            ForbiddenError: The asker is in none of the project's groups.
            InvalidInputError: No asker.
        """
        return await self._projects.profile(**request)

    # ── group forget use cases ─────────────────────────────────────────────

    async def forget_in_group(
        self,
        *,
        app_id: str,
        project_id: str,
        group_id: str,
        asker_id: str | None,
        target: GroupForgetTarget,
        reason: str | None = None,
        dry_run: bool = False,
        asserted_asker: str | None = None,
    ) -> GroupForgetResult:
        """Forget messages, a member, a session, records or a picture of a group.

        docs/group-memory.md, sections 5.5 and 9.4: the target travels to
        everything derived from it, and the forget is reversible by its
        ``forget_id``. ``dry_run`` touches nothing and reports what the real
        run would.

        Raises:
            ForbiddenError: Not a current member, or not allowed this target.
            NotFoundError: The group holds no such message, session, member,
                record or picture.
            SessionBusyError: A session of the forget stays busy.
            InvalidInputError: No asker.
        """
        return await self._run_to_completion(
            self._group_forgetter.forget(
                scope=_group_scope(app_id, project_id, group_id),
                identity=resolve_identity(header=asserted_asker, body_asker=asker_id),
                target=target,
                reason=reason,
                dry_run=dry_run,
            ),
            label="memory_group_forget",
        )

    async def undo_group_forget(
        self,
        *,
        app_id: str,
        project_id: str,
        group_id: str,
        asker_id: str | None,
        forget_id: str,
        asserted_asker: str | None = None,
    ) -> GroupUndoResult:
        """Undo one forget of a group by its id, with the forget's authorisation.

        Raises:
            ForbiddenError: Not a current member, or not allowed the forget.
            NotFoundError: No forget of this group has that id.
            ConflictError: A later forget rewrote what this one wrote.
        """
        return await self._run_to_completion(
            self._group_forgetter.undo(
                scope=_group_scope(app_id, project_id, group_id),
                identity=resolve_identity(header=asserted_asker, body_asker=asker_id),
                forget_id=forget_id,
            ),
            label="memory_group_forget_undo",
        )

    # ── delete use cases ───────────────────────────────────────────────────

    async def delete_in_group(
        self,
        *,
        app_id: str,
        project_id: str,
        group_id: str,
        asker_id: str | None,
        session_id: str | None,
        member_id: str | None,
        asserted_asker: str | None = None,
    ) -> tuple[DeleteResult, GroupForgetResult]:
        """``/delete`` with a group scope: an owner's or admin's forget.

        docs/group-memory.md, section 15: in Step 2 a group-scoped delete is
        owner/admin only and behaves as a forget -- soft, propagated, and
        reversible by the returned forget's id -- not as a hard delete.

        Raises:
            ForbiddenError: The asker is not an owner or admin of the group.
            InvalidInputError: Neither a session nor a member is named.
        """
        if (session_id is None) == (member_id is None):
            raise InvalidInputError(
                "a group-scoped delete names one session_id or one user_id (member)"
            )
        identity = resolve_identity(header=asserted_asker, body_asker=asker_id)
        if identity.asker_id is None:
            raise InvalidInputError("asker_id is required for a group-scoped delete")
        target = (
            GroupForgetTarget(kind="session", session_id=session_id)
            if session_id is not None
            else GroupForgetTarget(kind="member", member_id=member_id)
        )
        forgotten = await self._run_to_completion(
            self._delete_in_group(
                _group_scope(app_id, project_id, group_id), identity, target
            ),
            label="memory_group_delete",
        )
        count = (
            len(forgotten.episodes)
            + len(forgotten.image_memories)
            + len(forgotten.scenes)
            + len(forgotten.member_profile_items)
            + len(forgotten.stale_member_profile_items)
            + len(forgotten.group_profile_items)
            + len(forgotten.stale_group_profile_items)
            + len(forgotten.member_profiles)
        )
        filters = ["group_id", "session_id" if session_id is not None else "user_id"]
        logger.info(
            "memory_deleted_in_group",
            app_id=app_id,
            project_id=project_id,
            group_id=group_id,
            filters=filters,
            count=count,
            forget_id=forgotten.forget_id,
        )
        return DeleteResult(filters=filters, count=count), forgotten

    async def _delete_in_group(
        self, scope: Scope, identity: Identity, target: GroupForgetTarget
    ) -> GroupForgetResult:
        """The owner/admin check, then the forget; one supervised step."""
        record = await self._stores.groups.get(scope)
        if record is None:
            raise ForbiddenError(
                f"the asker is not a current member of group {scope.owner_id!r}"
            )
        authorize_manage(record.roster, identity.asker_id)
        return await self._group_forgetter.forget(
            scope=scope, identity=identity, target=target, reason="delete"
        )

    async def _run_to_completion[T](
        self, step: Coroutine[Any, Any, T], *, label: str
    ) -> T:
        """Run a forget (or its undo) to its end even if the caller goes away.

        Cut in half by a disconnect, a forget would leave a group half
        forgotten; like a deletion it is supervised and only shielded here.
        Its own answers (403, 404, 409, 503) are not logged as failures.
        """
        task = self._extractions.supervise(
            step,
            session_id=None,
            memcell_ids=[],
            label=label,
            expected=_GROUP_FORGET_ANSWERS,
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            logger.warning(f"{label}_continues_after_cancel")
            raise

    async def delete(
        self,
        *,
        app_id: str,
        project_id: str,
        user_id: str | None,
        agent_id: str | None,
        session_id: str | None,
    ) -> DeleteResult:
        """Delete one scope of memories (``/delete``).

        Runs to completion even if the caller goes away: a deletion cut in
        half would leave the scope half deleted.

        Raises:
            InvalidInputError: The session belongs to a group. Inside a group
                a delete is an owner's or admin's propagated forget
                (:meth:`delete_in_group`, which needs ``group_id`` and
                ``asker_id``); deleting the session's buffer and memcells
                alone would leave the group's memory of it half removed.
        """
        task = self._extractions.supervise(
            self._delete_outside_groups(
                app_id=app_id,
                project_id=project_id,
                user_id=user_id,
                agent_id=agent_id,
                session_id=session_id,
            ),
            session_id=session_id,
            memcell_ids=[],
            label="memory_delete",
            expected=(SessionBusyError, PathTraversalError, InvalidInputError),
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            logger.warning(
                "memory_delete_continues_after_cancel", session_id=session_id
            )
            raise

    async def _delete_outside_groups(
        self,
        *,
        app_id: str,
        project_id: str,
        user_id: str | None,
        agent_id: str | None,
        session_id: str | None,
    ) -> DeleteResult:
        """Refuse a session of a group, then delete; one supervised step.

        The session's binding is read inside the shielded task, not before
        it: a client that goes away while that read is in flight would
        otherwise cancel a database call half way, and the session's
        connection could be closed under the rollback that follows.
        """
        if session_id is not None:
            bound = await self._stores.groups.binding(
                SessionKey(app_id, project_id, session_id)
            )
            if bound is not None:
                raise InvalidInputError(
                    "the session belongs to a group; name group_id and asker_id "
                    "to delete it inside the group"
                )
        return await self._deleter.delete(
            app_id=app_id,
            project_id=project_id,
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
        )

    async def delete_by_ids(
        self,
        *,
        app_id: str,
        project_id: str,
        memory_type: str,
        ids: list[str],
        reason_code: str | None,
        note: str | None,
    ) -> None:
        """Delete single memories by id (``/delete_by_ids``), shielded like
        :meth:`delete`."""
        task = self._extractions.supervise(
            self._deleter.delete_by_ids(
                app_id=app_id,
                project_id=project_id,
                memory_type=memory_type,
                ids=ids,
                reason_code=reason_code,
                note=note,
            ),
            session_id=None,
            memcell_ids=[],
            label="memory_delete_by_ids",
        )
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            logger.warning("memory_delete_by_ids_continues_after_cancel")
            raise

    async def forget_image(
        self,
        *,
        app_id: str,
        project_id: str,
        user_id: str,
        image_id: str,
        reason: str | None = None,
    ) -> ForgetResult:
        """Forget one picture for one user: its memories, its vector, its file.

        The picture's memories are marked forgotten, its index row is erased,
        its original is taken out of circulation, and every episode whose body
        referred to it is written again without that reference. Nothing is
        deleted: the forget marks are what keep the material out of a default
        result, and undoing them is a status change.

        Raises:
            RuntimeError: The service was built without read-side indexes, so
                there is no way to find what referred to the picture.
        """
        if self._forgetter is None:
            raise RuntimeError("forget needs the read-side indexes")
        return await self._forgetter.forget_image(
            scope=Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=user_id,
                owner_type="user",
            ),
            image_id=image_id,
            reason=reason,
        )

    async def undo_forget_image(
        self, *, app_id: str, project_id: str, user_id: str, image_id: str
    ) -> UndoResult:
        """Undo the newest forget of one picture for one user.

        A picture that was never forgotten is not an error: the answer says
        nothing was undone and reports the occurrences that are current, which
        is what a caller retrying an undo after a timeout needs to see.
        """
        return await self._undoer.undo_image(
            scope=Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=user_id,
                owner_type="user",
            ),
            image_id=image_id,
        )

    async def revise_image_memory(
        self,
        *,
        app_id: str,
        project_id: str,
        user_id: str,
        image_id: str,
        description: str | None = None,
        ocr: str | None = None,
        entry_id: str | None = None,
    ) -> RevisionResult:
        """Write a new version of one picture memory, superseding the old one.

        The index follows from markdown: the new entry's tokens and vector are
        computed by the cascade when it reads the file, so a revision never
        touches the index directly.

        Raises:
            NotFoundError: The picture has no such current occurrence.
            ValueError: Neither field to change was given.
        """
        return await self._reviser.revise_image(
            scope=Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=user_id,
                owner_type="user",
            ),
            image_id=image_id,
            description=description,
            ocr=ocr,
            entry_id=entry_id,
        )

    async def forget_session(
        self,
        *,
        app_id: str,
        project_id: str,
        user_id: str,
        session_id: str,
        reason: str | None = None,
    ) -> SessionForgetResult:
        """Forget everything one conversation established for one user.

        Marks rather than deletes, so :meth:`undo_forget_session` reverses it.
        A picture file survives while any other live memory references it, and
        a profile item while a segment it rests on is remembered.
        """
        return await self._session_forgetter.forget_session(
            scope=Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=user_id,
                owner_type="user",
            ),
            session_id=session_id,
            reason=reason,
        )

    async def undo_forget_session(
        self, *, app_id: str, project_id: str, user_id: str, session_id: str
    ) -> SessionUndoResult:
        """Undo the newest forget of one conversation for one user."""
        return await self._undoer.undo_session(
            scope=Scope(
                app_id=app_id,
                project_id=project_id,
                owner_id=user_id,
                owner_type="user",
            ),
            session_id=session_id,
        )

    async def finish_interrupted_deletions(self) -> int:
        """Finish whole-owner deletions a crash cut short (startup)."""
        return await self._stores.owner_dir.sweep_trash()

    # ── async_mode queue ───────────────────────────────────────────────────

    @property
    def accepts_async(self) -> bool:
        """Whether new ``async_mode`` requests are accepted onto the queue."""
        return self._queue.accepting

    async def has_queued(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> bool:
        """True when the session has queued requests not yet delivered."""
        return await self._queue.has_pending(
            session_id=session_id, app_id=app_id, project_id=project_id
        )

    async def enqueue(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        is_final: bool,
        defer_extraction: bool,
    ) -> int:
        """Persist one request for the queue; return the session's backlog.

        Raises:
            PayloadTooLargeError: The body is over the configured limit.
            QueueSaturatedError: The session or the whole queue is at its cap.
            InvalidInputError: The body is not valid UTF-8.
        """
        return await self._queue.enqueue(
            payload,
            request_id=request_id,
            is_final=is_final,
            defer_extraction=defer_extraction,
        )

    async def queue_health(self) -> QueueHealth | None:
        """The queue's health probe; ``None`` while the queue is not started."""
        if not self._queue.started:
            return None
        return await self._queue.health()

    # ── write use cases ────────────────────────────────────────────────────

    async def add(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        async_mode: bool,
        defer_extraction: bool,
    ) -> MemorizeResult | Queued:
        """Run one ``/add`` inline, or queue it.

        Queued when the caller asked for ``async_mode`` and the queue takes
        async work, **or** the session already has queued requests: running
        inline then would overtake requests the client was told are
        accepted.

        Raises:
            PayloadTooLargeError: Queued, and the body is over the limit.
            QueueSaturatedError: Queued, and the session or queue is full.
        """
        if (async_mode and self.accepts_async) or await self._has_backlog(payload):
            return await self._queue_request(
                payload,
                request_id=request_id,
                is_final=False,
                defer_extraction=defer_extraction,
            )
        return await self._memorize(payload, defer_extraction=defer_extraction)

    async def flush(
        self,
        *,
        session_id: str,
        request_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> FlushResult | Queued:
        """Run one ``/flush`` inline, or queue it behind the session's backlog.

        A flush has no ``async_mode`` (the Cloud API has none either); it is
        queued only when earlier requests of the session still are.

        Raises:
            QueueSaturatedError: Queued, and the session or queue is full.
        """
        payload = flush_payload(session_id, app_id, project_id)
        if await self._has_backlog(payload):
            return await self._queue_request(
                payload, request_id=request_id, is_final=True, defer_extraction=False
            )
        return await self._flush_inline(
            session_id=session_id, app_id=app_id, project_id=project_id
        )

    async def _has_backlog(self, payload: dict[str, Any]) -> bool:
        return await self.has_queued(
            session_id=payload["session_id"],
            app_id=payload.get("app_id", "default"),
            project_id=payload.get("project_id", "default"),
        )

    async def _queue_request(
        self,
        payload: dict[str, Any],
        *,
        request_id: str,
        is_final: bool,
        defer_extraction: bool,
    ) -> Queued:
        pending = await self.enqueue(
            payload,
            request_id=request_id,
            is_final=is_final,
            defer_extraction=defer_extraction,
        )
        return Queued(message_count=len(payload["messages"]), pending=pending)

    async def _flush_inline(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> FlushResult:
        """Force boundary detection over the session's whole buffer."""
        result = await self._memorize(
            flush_payload(session_id, app_id, project_id), is_final=True
        )
        return FlushResult(
            status="extracted" if result.status == "extracted" else "no_extraction"
        )

    async def _memorize(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        """Execute one add cycle; see :meth:`Memorizer.run`.

        Args:
            payload: ``{"session_id", "messages": [...]}`` — entrypoints DTO
                dumped to dict.
            is_final: ``True`` only for flush (algo guarantees ``tail=[]``).
            defer_extraction: Persist fresh messages to the unprocessed buffer
                but skip boundary detection and both extraction pipelines.
        """
        return await self._memorizer.run(
            payload, is_final=is_final, defer_extraction=defer_extraction
        )


def _group_scope(app_id: str, project_id: str, group_id: str) -> Scope:
    return Scope(
        app_id=app_id, project_id=project_id, owner_id=group_id, owner_type="group"
    )


def _image_limits_from(settings: Settings) -> ImageLimits:
    """``[images]`` as the request bounds the question-picture path enforces.

    A service built without explicit limits (a test double, a CLI path that
    does not inject them) still has to bound what a search may carry; reading
    the loaded settings gives it the same numbers ingest uses.
    """
    images = settings.images
    return ImageLimits(
        max_bytes=images.max_bytes,
        max_per_message=images.max_per_message,
        max_request_bytes=images.max_request_bytes,
        max_per_request=images.max_per_request,
        formats=tuple(images.formats),
        model_edge=images.model_edge,
        cache_quality=images.cache_quality,
    )
