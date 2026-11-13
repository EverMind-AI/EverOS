"""The memorize use case: cut a session's messages into memcells, extract them.

Memorize, end to end::

    payload { session_id, messages[] }
        -> the session's space: its group, or one person's own
        -> ingest -> BoundaryDetection (buffer / merge / boundary / tail)
        -> for each cell: ConversationPipeline.run_cell
                          -> publish that cell's events on the EventBus
        -> the pipeline's status -> MemorizeResult

Boundary detection owns buffer / merge / boundary / tail. Publishing is this
use case's job:
pipelines and strategies return events, and each cell's are published as
soon as the cell is done, so a failing cell never takes an earlier cell's
events with it.

A session is remembered in one space, decided under the session lock before
anything is written (docs/group-memory.md, sections 7.1 and 7.9):

* a session bound to a group, or a batch naming ``group_id``: the batch is
  admitted into its group (binding, inline roster, implicit joins; 409 / 403
  / 422 before anything is written), and an inline roster change first cuts
  the open buffer of every session of the group -- their locks held with the
  batch's, in sorted order (D18). A group with a deliverer takes batches and
  flushes from its deliverer only (403 otherwise, section 5.1);
* a session of three or more people (role ``user`` senders, in the batch and
  the open buffer together) without a ``group_id``: it becomes the implicit
  group of those people, created on first sight and bound like a declared
  one; a person who joins its session later is refused (409);
* otherwise one or two people's conversation with the assistant: their own
  memory, each person keeping a copy of every segment (D14, amended
  2026-11-13: two people are not a group).

The conversation pipeline writes each cell in that space: once in a group's,
once per person in their own; group content is never filed as a sender's
personal memory.

A group takes each message once (section 5.2): the batch's messages the group
already holds -- in any of its sessions, forgotten or not -- are skipped
before ingest (:class:`~everos.memory.groups.GroupDedupe`), and the rest are
recorded as held once they are stored. Explicit groups and implicit groups
(three or more people) only; a session of one or two people never.
"""

from __future__ import annotations

__all__ = ["Memorizer", "flush_payload"]

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from everos.core.context import resolve_request_id
from everos.core.errors import InvalidInputError, RosterConflictError
from everos.core.locks import LOCKS, session_key
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span
from everos.core.utils.datetime import epoch_ms, get_utc_now, to_timestamp_ms
from everos.memory import IngestResult as _IngestResult
from everos.memory.access import authorize_delivery, resolve_identity
from everos.memory.events import GroupFlushed, SessionFlushed
from everos.memory.extract.ingest import generated_ids
from everos.memory.extract.ingest import process as ingest_process
from everos.memory.extract.pipeline import PipelineOutcome, PreparedCell
from everos.memory.groups import (
    IMPLICIT_GROUP_MIN_PEOPLE,
    BatchMessage,
    Claim,
    GroupDedupe,
    implicit_group_id,
    is_implicit_group,
)
from everos.memory.ports import Scope, SessionKey
from everos.service.memory._extraction import ExtractionSupervisor
from everos.service.memory._groups import to_events
from everos.service.memory._models import MemorizeResult
from everos.service.memory._retry import SegmentRetries, held_positions

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.component.parser import RawFileBuilder
    from everos.config import Settings
    from everos.memory import IngestResult
    from everos.memory.extract.boundary import BoundaryDetection, BoundaryOutcome
    from everos.memory.extract.pipeline import CellOutcome, Pipeline
    from everos.memory.groups import RosterManager
    from everos.memory.ports import (
        EventBus,
        FailedSegment,
        ForgetLogStore,
        MemcellStore,
        SessionBufferStore,
    )
    from everos.memory.ports.images import ImageStore

logger = get_logger("everos.service.memory")

TrackStatus = Literal["accumulated", "extracted", "skipped"]


class Memorizer:
    """Runs memorize calls, one session at a time, over injected domain objects.

    Args:
        settings: Application settings (``[memorize]``).
        boundary: Cuts each session's buffered messages into memcells.
        multimodal_llm: Parses non-text content; ``None`` when
            ``[multimodal]`` is not configured.
        image_store: Where picture content items are stored, by content hash;
            ``None`` disables the picture path (pictures are reported as
            rejected rather than silently dropped).
        conversation_pipeline: The conversation track, run on every cell:
            one episode per cell, in its session's space.
        bus: Where each cell's events are published.
        extractions: Keeps extractions alive past a cancelled request.
        raw_file_builder: Host file hydration capability for ingestion.
        session_buffer: The sessions' open buffers: whose messages a session
            without a group already holds.
        rosters: Admits a batch into its group, declared or implicit; ``None``
            refuses ``group_id`` and makes no implicit group (a build without
            group memory).
        dedupe: Which of a group batch's messages the group already holds;
            ``None`` takes every message (a build without the ledger).
        memcells: Where a segment whose extraction failed is marked, for the
            session's next add or flush to extract again (:mod:`._retry`);
            ``None`` retries nothing.
        forget_log: What a retry checks for a forget standing over its
            segment; ``None`` checks nothing.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        boundary: BoundaryDetection,
        multimodal_llm: LLMClient | None,
        conversation_pipeline: Pipeline,
        bus: EventBus,
        extractions: ExtractionSupervisor,
        raw_file_builder: RawFileBuilder,
        session_buffer: SessionBufferStore,
        image_store: ImageStore | None = None,
        vision: bool = True,
        rosters: RosterManager | None = None,
        dedupe: GroupDedupe | None = None,
        memcells: MemcellStore | None = None,
        forget_log: ForgetLogStore | None = None,
    ) -> None:
        self._rosters = rosters
        self._dedupe = dedupe
        self._retries = SegmentRetries(memcells, forget_log)
        self._session_buffer = session_buffer
        self._raw_file_builder = raw_file_builder
        self._image_store = image_store
        self._vision = vision
        self._settings = settings
        self._boundary = boundary
        self._multimodal_llm = multimodal_llm
        self._conversation_pipeline = conversation_pipeline
        self._bus = bus
        self._extractions = extractions

    async def run(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool = False,
        defer_extraction: bool = False,
    ) -> MemorizeResult:
        """Execute one add cycle. Dispatched concurrently across pipelines.

        Args:
            payload: ``{"session_id", "messages": [...]}`` — entrypoints DTO
                dumped to dict.
            is_final: ``True`` only for flush (algo guarantees ``tail=[]``).
            defer_extraction: Persist fresh messages to the unprocessed buffer
                but skip boundary detection and both extraction pipelines.
                Intended for clients that batch work and call ``/flush`` on
                idle or thresholds.

        Concurrency: serialised per ``(app_id, project_id, session_id)`` on
        the ``"session"`` lock of :data:`everos.core.locks.LOCKS`. The lock
        spans the read-merge-boundary-write cycle: two unserialised calls
        would both read the same buffer rows, cut their own slice, and the
        later buffer write would drop the earlier call's tail messages
        (they never reached a memcell either).

        ``settings.memorize.session_lock_timeout_seconds`` bounds the two
        places a call can hang while something else waits on the lock: the
        wait for the lock itself, and the boundary LLM call. It no longer
        cancels the call as a whole: once the boundary has decided, the
        writes that make the cells durable and the extraction of those cells
        run to completion. A timeout that landed after the memcell row was
        committed and the buffer consumed used to cancel the extraction, and
        nothing retried it -- the cell stayed without an episode for good
        (found by the concurrent-memorize property test). Extraction runs
        after the lock is released; it only reads memcells, never the buffer.
        """
        session_id = payload["session_id"]

        span_name = (
            "everos.memory.flush"
            if is_final
            else "everos.memory.buffer"
            if defer_extraction
            else "everos.memory.add"
        )
        with memory_span(
            span_name,
            observation_type="span",
            session_id=session_id,
            metadata={
                "is_final": is_final,
                "defer_extraction": defer_extraction,
                "request_id": resolve_request_id(),
            },
        ):
            # Buffers and memcells are scoped by app / project, so two sessions
            # that merely share a session_id across projects must not wait on
            # each other (the queue runs them side by side).
            app_id = payload.get("app_id") or "default"
            project_id = payload.get("project_id") or "default"
            # A batch that changes its group's roster cuts every session of
            # the group first (D18), so it holds their locks as well as its
            # own -- all of them in sorted order, like the roster route.
            others = await self._sessions_cut_with(payload, is_final=is_final)
            async with contextlib.AsyncExitStack() as held:
                held_others = await self._hold_sessions(
                    held,
                    app_id=app_id,
                    project_id=project_id,
                    own=session_id,
                    others=others,
                )
                elsewhere = await self._cut_buffers(app_id, project_id, held_others)
                prepared, claim, retries = await self._cut(
                    payload,
                    is_final=is_final,
                    defer_extraction=defer_extraction,
                )
            if isinstance(prepared, MemorizeResult) and not elsewhere and not retries:
                await self._settle(claim, payload)
                await self._announce_flush(payload, is_final=is_final)
                return prepared
            # The cells exist, so their extraction must not die with a
            # cancelled request (a client that went away). The task is owned
            # by the supervisor, which keeps it, retrieves its outcome and
            # waits for it at shutdown; the request only shields it.
            memcell_ids = [
                *(r.memcell_id for r in retries),
                *(
                    mid
                    for _, b in [*elsewhere, *_own_cuts(prepared)]
                    for mid in b.memcell_ids
                ),
            ]
            extraction = self._extractions.supervise(
                self._extract_cut(
                    payload,
                    prepared,
                    elsewhere,
                    claim=claim,
                    is_final=is_final,
                    retries=retries,
                ),
                session_id=session_id,
                memcell_ids=memcell_ids,
            )
            try:
                return await asyncio.shield(extraction)
            except asyncio.CancelledError:
                logger.warning(
                    "memorize_extraction_continues_after_cancel",
                    session_id=session_id,
                    memcell_ids=memcell_ids,
                )
                raise

    async def check_delivery(self, payload: dict[str, Any]) -> None:
        """Refuse a batch or a flush from anyone but its group's deliverer (403).

        Asked before a request is run or queued, so a refused one is answered
        at once; :meth:`_admit_group_batch` asks again under the session lock,
        which is what a queued request meets when it runs. A session of no
        group, an implicit group and a group without a deliverer take anyone's
        batches, as before.

        Raises:
            ForbiddenError: The group has a deliverer and the asker (the
                gateway's header, else ``asker_id``) is not it.
        """
        if self._rosters is None:
            return
        session = SessionKey(
            app_id=payload.get("app_id") or "default",
            project_id=payload.get("project_id") or "default",
            session_id=payload["session_id"],
        )
        group_id = payload.get("group_id") or await self._rosters.store.binding(session)
        if group_id is None or is_implicit_group(group_id):
            return
        await self._check_delivery(
            payload,
            Scope(
                app_id=session.app_id,
                project_id=session.project_id,
                owner_id=group_id,
                owner_type="group",
            ),
        )

    async def _check_delivery(self, payload: dict[str, Any], scope: Scope) -> None:
        """:meth:`check_delivery` for the group ``scope``, already resolved."""
        assert self._rosters is not None
        record = await self._rosters.store.get(scope)
        if record is None or record.roster.deliverer_id is None:
            return
        identity = resolve_identity(
            header=payload.get("asserted_asker"), body_asker=payload.get("asker_id")
        )
        authorize_delivery(record.roster, identity.asker_id, group_id=scope.owner_id)

    async def _sessions_cut_with(
        self, payload: dict[str, Any], *, is_final: bool
    ) -> list[str]:
        """The other sessions of the batch's group, when the batch changes its roster.

        Read before any lock is taken, to know which locks to take; the
        session's binding is decided again under its lock. A session that
        binds between the two reads is not cut: the windows keep reads right,
        the cut only keeps a new member's first segment theirs (section 7.1).
        """
        if self._rosters is None or is_final or not payload.get("roster"):
            return []
        session = SessionKey(
            app_id=payload.get("app_id") or "default",
            project_id=payload.get("project_id") or "default",
            session_id=payload["session_id"],
        )
        group_id = payload.get("group_id") or await self._rosters.store.binding(session)
        if group_id is None:
            return []
        record = await self._rosters.store.get(
            Scope(
                app_id=session.app_id,
                project_id=session.project_id,
                owner_id=group_id,
                owner_type="group",
            )
        )
        if record is None:
            return []
        return [s for s in record.sessions if s != session.session_id]

    async def _hold_sessions(
        self,
        held: contextlib.AsyncExitStack,
        *,
        app_id: str,
        project_id: str,
        own: str | None,
        others: Sequence[str],
    ) -> list[str]:
        """Take the batch's session lock and the others', in sorted order.

        The batch's own lock (``own``) is required: waiting past the session
        lock timeout raises, as it always has. Another session whose lock
        stays busy is skipped with a warning -- the membership windows keep
        reads correct; the cut only keeps a new member's first segment theirs.

        Returns:
            The other sessions whose locks are held.
        """
        wait = self._settings.memorize.session_lock_timeout_seconds
        mine = {own} if own is not None else set()
        keys = {session_key(app_id, project_id, s): s for s in {*mine, *others}}
        got: list[str] = []
        for key in sorted(keys):
            session_id = keys[key]
            if session_id == own:
                await held.enter_async_context(
                    LOCKS.hold("session", key, max_wait=wait)
                )
                continue
            try:
                await held.enter_async_context(
                    LOCKS.hold("session", key, max_wait=wait)
                )
            except TimeoutError:
                logger.warning(
                    "groups.buffer_cut_skipped_busy",
                    app_id=app_id,
                    project_id=project_id,
                    session_id=session_id,
                )
                continue
            got.append(session_id)
        return got

    async def _cut_buffers(
        self, app_id: str, project_id: str, sessions: Sequence[str]
    ) -> list[tuple[IngestResult, BoundaryOutcome]]:
        """Close each listed session's open buffer into cells (a flush's cut).

        Runs under the sessions' locks; the cells are extracted after the
        locks are released.
        """
        cut: list[tuple[IngestResult, BoundaryOutcome]] = []
        for session_id in sessions:
            session = SessionKey(app_id, project_id, session_id)
            group_id = (
                await self._rosters.store.binding(session) if self._rosters else None
            )
            empty = _IngestResult(
                session_id=session_id,
                app_id=app_id,
                project_id=project_id,
                messages=[],
                group_id=group_id,
            )
            closed = await self._boundary.detect(empty, is_final=True)
            if closed.cells:
                cut.append((empty, closed))
        return cut

    async def _extract_cut(
        self,
        payload: dict[str, Any],
        prepared: MemorizeResult | tuple[IngestResult, list[BoundaryOutcome]],
        elsewhere: Sequence[tuple[IngestResult, BoundaryOutcome]],
        *,
        claim: Claim | None = None,
        is_final: bool,
        retries: Sequence[FailedSegment] = (),
    ) -> MemorizeResult:
        """Extract the session's failed segments again, then the batch's own
        cells, then those cut from the group's others.

        The batch's messages are recorded as held once it is stored and its
        own cells extracted (``claim``). A cell whose extraction failed is
        stored and marked, and the session's next add or flush extracts it
        again, so its messages are recorded as held then too: a re-send of
        the batch is a duplicate, never a second segment. Only a cancelled
        extraction (shutdown) records nothing, and a re-send is taken in
        full, as before the ledger.
        """
        try:
            retried = await self._extract_retries(payload, retries)
            result = (
                prepared
                if isinstance(prepared, MemorizeResult)
                else await self._extract(payload, *prepared)
            )
        except Exception:
            await self._settle(claim, payload)
            raise
        except BaseException:
            if claim is not None:
                claim.release()
            raise
        if retried and result.status == "accumulated":
            result = result.model_copy(update={"status": "extracted"})
        await self._settle(claim, payload)
        for ingested, closed in elsewhere:
            await self._extract(
                flush_payload(
                    ingested.session_id, ingested.app_id, ingested.project_id
                ),
                ingested,
                [closed],
            )
        await self._announce_flush(payload, is_final=is_final)
        return result

    async def _announce_flush(self, payload: dict[str, Any], *, is_final: bool) -> None:
        """Publish ``GroupFlushed`` after a flush of a session bound to a group,
        ``SessionFlushed`` after a flush of any other session.

        Published whether or not the flush cut a segment: the profiles
        waiting on ``[groups] member_profile_every`` (a group's members,
        section 7.4) or on ``[profile] update_every`` (the users of a
        one-to-one session) are brought up to date on it.
        """
        if not is_final:
            return
        session = SessionKey(
            app_id=payload.get("app_id") or "default",
            project_id=payload.get("project_id") or "default",
            session_id=payload["session_id"],
        )
        group_id = (
            await self._rosters.store.binding(session)
            if self._rosters is not None
            else None
        )
        flushed: GroupFlushed | SessionFlushed = (
            SessionFlushed(
                session_id=session.session_id,
                app_id=session.app_id,
                project_id=session.project_id,
            )
            if group_id is None
            else GroupFlushed(
                group_id=group_id,
                session_id=session.session_id,
                app_id=session.app_id,
                project_id=session.project_id,
            )
        )
        await self._bus.publish([flushed])

    async def _cut(
        self,
        payload: dict[str, Any],
        *,
        is_final: bool,
        defer_extraction: bool,
    ) -> tuple[
        MemorizeResult | tuple[IngestResult, list[BoundaryOutcome]],
        Claim | None,
        list[FailedSegment],
    ]:
        """Inner critical section — runs under the per-session lock.

        Returns the result when nothing reached the boundary, or the
        ingested batch and the boundary outcomes for :meth:`_extract` to run
        once the lock is released (two for a group batch whose roster change
        cut the open buffer first, one otherwise) -- and, for a group batch,
        the claim on its messages (section 5.2), which the caller settles
        once the batch's own cells are extracted -- and the session's failed
        segments this call extracts again (:mod:`._retry`; none for a
        deferred add, which makes no model call). A message such a segment
        holds is a duplicate: the segment's retry extracts it.
        """
        session = SessionKey(
            app_id=payload.get("app_id") or "default",
            project_id=payload.get("project_id") or "default",
            session_id=payload["session_id"],
        )
        group_id = await self._group_of(payload, session)
        buffered: list[BatchMessage] = []
        if group_id is None:
            group_id, buffered = await self._people_of(payload, session)
        cut_first: list[BoundaryOutcome] = []
        claim: Claim | None = None
        if group_id is not None:
            payload = {**payload, "group_id": group_id}
            cut_first = await self._admit_group_batch(
                payload,
                session,
                group_id,
                buffered=buffered,
                is_final=is_final,
            )
            claim = await self._claim(payload, session, group_id)
        try:
            pending = await self._retries.pending(session)
            skip = claim.duplicates if claim is not None else frozenset()
            if pending and payload.get("messages"):
                skip |= held_positions(generated_ids(payload), pending)
            stored = await self._store(
                payload,
                cut_first,
                is_final=is_final,
                defer_extraction=defer_extraction,
                skip=skip,
            )
        except BaseException:
            if claim is not None:
                claim.release()
            raise
        retries = (
            self._retries.claim(pending, is_final=is_final)
            if is_final or not defer_extraction
            else []
        )
        return stored, claim, retries

    async def _store(
        self,
        payload: dict[str, Any],
        cut_first: list[BoundaryOutcome],
        *,
        is_final: bool,
        defer_extraction: bool,
        skip: frozenset[int] = frozenset(),
    ) -> MemorizeResult | tuple[IngestResult, list[BoundaryOutcome]]:
        """Ingest the batch (but the messages at ``skip``) and run its boundary."""
        ingested = await ingest_process(
            payload,
            multimodal_llm=self._multimodal_llm,
            raw_file_builder=self._raw_file_builder,
            image_store=self._image_store,
            vision=self._vision,
            skip=skip,
        )
        if defer_extraction and not is_final:
            buffered = await self._boundary.buffer(ingested)
            ingested = _held_too(ingested, buffered)
            if cut_first:
                return ingested, cut_first
            return _result(payload, ingested, _status(buffered.status))

        boundary = await self._boundary.detect(ingested, is_final=is_final)
        ingested = _held_too(ingested, boundary)
        boundaries = [*cut_first, boundary] if boundary.cells else cut_first

        if not boundaries:
            # Nothing went past the boundary stage — no pipelines to dispatch.
            return _result(payload, ingested, _status(boundary.status))
        return ingested, boundaries

    async def _claim(
        self, payload: dict[str, Any], session: SessionKey, group_id: str
    ) -> Claim | None:
        """Which of the batch's messages the group already holds (section 5.2).

        ``None`` -- every message taken -- for a flush, a build without the
        ledger and an implicit group of fewer than three people: two people
        are no group (D14), and a session bound to two people's implicit
        group by an earlier build is taken as it was. Skipped messages are
        logged.
        """
        messages = payload.get("messages") or []
        if self._dedupe is None or not messages:
            return None
        scope = Scope(
            app_id=session.app_id,
            project_id=session.project_id,
            owner_id=group_id,
            owner_type="group",
        )
        if is_implicit_group(group_id) and not await self._three_or_more(scope):
            return None
        claim = await self._dedupe.claim(scope, session, messages)
        if claim.duplicates:
            logger.info(
                "memorize.duplicates_skipped",
                app_id=session.app_id,
                project_id=session.project_id,
                group_id=group_id,
                session_id=session.session_id,
                skipped=len(claim.duplicates),
                messages=len(messages),
            )
        return claim

    async def _three_or_more(self, scope: Scope) -> bool:
        """Whether an implicit group has three or more people (its identity).

        Every implicit group made now has (:data:`IMPLICIT_GROUP_MIN_PEOPLE`);
        one of two people is an earlier build's.
        """
        assert self._rosters is not None
        record = await self._rosters.store.get(scope)
        people = (
            {i.member_id for i in record.roster.intervals if i.kind == "human"}
            if record is not None
            else set()
        )
        return len(people) >= IMPLICIT_GROUP_MIN_PEOPLE

    async def _settle(self, claim: Claim | None, payload: dict[str, Any]) -> None:
        """Record the batch's messages as held by its group, and drop the claim.

        Called once the batch is stored and its own cells extracted. A failed
        write is logged, not raised: the messages are kept, and a client told
        to send them again would have them taken twice.
        """
        if claim is None:
            return
        try:
            await claim.commit()
        except Exception as exc:
            logger.warning(
                "memorize.duplicates_ledger_write_failed",
                session_id=payload["session_id"],
                group_id=payload.get("group_id"),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            claim.release()

    async def _group_of(
        self, payload: dict[str, Any], session: SessionKey
    ) -> str | None:
        """The group this batch belongs to: its ``group_id``, or the binding.

        Raises:
            RosterConflictError: The batch names another group than the one
                the session is bound to (409, D18).
            InvalidInputError: ``group_id`` on a build without group memory,
                or an implicit group's id (reserved: a session of three or
                more people becomes its group without naming it).
        """
        named = payload.get("group_id") or None
        if named is not None and is_implicit_group(named):
            raise InvalidInputError(
                f"group_id {named!r} is an implicit group's: send the batch "
                "without group_id"
            )
        if self._rosters is None:
            if named is not None:
                raise InvalidInputError("group memory is not available")
            return None
        bound = await self._rosters.store.binding(session)
        if named is not None and bound is not None and named != bound:
            raise RosterConflictError(
                f"session {session.session_id!r} belongs to another group"
            )
        return named or bound

    async def _people_of(
        self, payload: dict[str, Any], session: SessionKey
    ) -> tuple[str | None, list[BatchMessage]]:
        """The implicit group of a session without one, when three or more people talk.

        The people are the role ``user`` senders of the batch and of the
        session's open buffer together (a buffer is never cut for it: what it
        holds goes with the group). Cells already cut stay what they were:
        a two-person session that a third person joins becomes the group of
        the three from that batch, and its earlier segments stay the two
        people's own copies.

        Returns:
            The implicit group's id and the buffer's messages (founders of a
            group created now), or ``None`` and nothing for one or two people.
        """
        if self._rosters is None:
            return None, []
        buffered = [
            BatchMessage(
                sender_id=m.sender_id,
                role=m.role,
                at=to_timestamp_ms(m.timestamp),
                sender_name=m.sender_name,
            )
            for m in await self._session_buffer.list(session)
        ]
        people = {
            m.sender_id for m in [*buffered, *_batch_messages(payload)] if _person(m)
        }
        if len(people) < IMPLICIT_GROUP_MIN_PEOPLE:
            return None, []
        return implicit_group_id(people), buffered

    async def _admit_group_batch(
        self,
        payload: dict[str, Any],
        session: SessionKey,
        group_id: str,
        *,
        buffered: Sequence[BatchMessage],
        is_final: bool,
    ) -> list[BoundaryOutcome]:
        """Admit a group batch; cut the open buffer first when it changes the roster.

        Runs under the session lock, before the batch's messages are buffered.
        Returns the outcome of that cut (D18), or nothing. An implicit group
        takes no roster event and so never cuts: its batch -- with
        ``buffered``, the open buffer's messages when the session becomes
        the group's -- only creates, joins and binds.
        """
        assert self._rosters is not None
        messages = _batch_messages(payload)
        scope = Scope(
            app_id=session.app_id,
            project_id=session.project_id,
            owner_id=group_id,
            owner_type="group",
        )
        if is_implicit_group(group_id):
            if messages or buffered:
                await self._rosters.admit_implicit(
                    scope, session=session, messages=[*buffered, *messages]
                )
            return []
        raw_roster = payload.get("roster") or []
        # Before anything is cut or written: a group with a deliverer takes
        # batches and flushes from it alone.
        await self._check_delivery(payload, scope)
        if not messages and not raw_roster:
            return []  # a flush: the binding already names the group
        await self._rosters.check_binding(session, group_id)
        identity = resolve_identity(
            header=payload.get("asserted_asker"), body_asker=payload.get("asker_id")
        )
        first = min((m.at for m in messages), default=None)
        events = to_events(
            raw_roster,
            source="inline",
            now=first if first is not None else to_timestamp_ms(get_utc_now()),
        )
        cut: list[BoundaryOutcome] = []
        if events and not is_final:
            closed = await self._boundary.detect(
                _IngestResult(
                    session_id=session.session_id,
                    app_id=session.app_id,
                    project_id=session.project_id,
                    messages=[],
                    group_id=group_id,
                ),
                is_final=True,
            )
            if closed.cells:
                cut.append(closed)
        await self._rosters.admit_batch(
            scope,
            session=session,
            asker_id=identity.asker_id,
            roster=events,
            messages=messages,
        )
        return cut

    async def with_buffers_closed[T](
        self,
        *,
        app_id: str,
        project_id: str,
        sessions: Sequence[str],
        change: Callable[[], Awaitable[T]],
    ) -> T:
        """Cut every listed session's open buffer, then run ``change`` (D18).

        The sessions' locks are taken in sorted order and held while
        ``change`` runs, so no message can slip into a buffer between the cut
        and the roster change. A session whose lock stays busy past the
        session lock timeout is skipped: the membership windows keep reads
        correct, the cut only keeps a new member's first segment theirs.
        The cut cells are extracted after the locks are released.
        """
        async with contextlib.AsyncExitStack() as held:
            locked = await self._hold_sessions(
                held, app_id=app_id, project_id=project_id, own=None, others=sessions
            )
            cut = await self._cut_buffers(app_id, project_id, locked)
            result = await change()
        for ingested, closed in cut:
            await self._extract(
                flush_payload(ingested.session_id, app_id, project_id),
                ingested,
                [closed],
            )
        return result

    async def _extract_retries(
        self, payload: dict[str, Any], retries: Sequence[FailedSegment]
    ) -> bool:
        """Extract the claimed failed segments again, each on its own.

        A segment that fails again is marked again (or left, past
        :data:`~._retry.MAX_EXTRACTION_FAILURES`) and logged; it never fails
        the call that retried it. Returns whether one was extracted.
        """
        session = SessionKey(
            app_id=payload.get("app_id") or "default",
            project_id=payload.get("project_id") or "default",
            session_id=payload["session_id"],
        )
        extracted = False
        try:
            for segment in retries:
                extracted = await self._extract_again(session, segment) or extracted
        finally:
            self._retries.release(retries)
        return extracted

    async def _extract_again(self, session: SessionKey, segment: FailedSegment) -> bool:
        """One failed segment, extracted again; whether it was. Never raises."""
        ingested = _IngestResult(
            session_id=session.session_id,
            app_id=session.app_id,
            project_id=session.project_id,
            messages=[],
            group_id=segment.group_id,
        )
        prepared = PreparedCell(
            cell=segment.cell,
            memcell_id=segment.memcell_id,
            sender_ids=segment.sender_ids,
            picture_names=segment.picture_names,
            retry=True,
        )
        try:
            outcome = await self._conversation_pipeline.run_cell(ingested, prepared)
        except Exception as exc:
            _log_cell_failure(ingested, prepared, "conversation", exc)
            await self._retries.failed(
                memcell_id=segment.memcell_id,
                cell=segment.cell,
                sender_ids=segment.sender_ids,
                message_ids=segment.message_ids,
                group_id=segment.group_id,
                picture_names=segment.picture_names,
                failures=segment.failures + 1,
            )
            return False
        logger.info(
            "memorize_cell_extraction_retried",
            session_id=session.session_id,
            memcell_id=segment.memcell_id,
            failures=segment.failures,
        )
        try:
            await self._retries.extracted(segment)
            await self._bus.publish(outcome.events)
        except Exception as exc:
            logger.error(
                "memorize_cell_retry_not_settled",
                memcell_id=segment.memcell_id,
                error=f"{type(exc).__name__}: {exc}",
            )
        return True

    async def _extract(
        self,
        payload: dict[str, Any],
        ingested: IngestResult,
        boundaries: Sequence[BoundaryOutcome],
    ) -> MemorizeResult:
        """Run the pipelines over the cells the boundary stage wrote, cell by cell.

        The conversation pipeline writes each cell's episode in its session's
        space (``ingested.group_id``: a group's, else the person's own).

        Each pipeline's events for a cell are published as soon as that cell
        is done. A pipeline that fails on a cell is logged and the loop goes
        on, so the cells that did succeed still
        reach their strategies; the first failure is re-raised once every
        cell has been tried, so the caller still sees the extraction failed.

        Raises:
            Exception: The first pipeline failure, after all cells ran.
        """
        pipelines: dict[str, Pipeline] = {"conversation": self._conversation_pipeline}
        outcomes: dict[str, list[CellOutcome]] = {track: [] for track in pipelines}
        first_error: BaseException | None = None
        cells = [c for boundary in boundaries for c in _prepared_cells(boundary)]
        message_ids = {
            memcell_id: ids
            for boundary in boundaries
            for memcell_id, ids in zip(
                boundary.memcell_ids, boundary.per_cell_message_ids, strict=True
            )
        }
        for prepared in cells:
            results = await asyncio.gather(
                *(p.run_cell(ingested, prepared) for p in pipelines.values()),
                return_exceptions=True,
            )
            for track, result in zip(pipelines, results, strict=True):
                if isinstance(result, BaseException):
                    _log_cell_failure(ingested, prepared, track, result)
                    first_error = first_error or result
                    if isinstance(result, Exception):
                        await self._retries.failed(
                            memcell_id=prepared.memcell_id,
                            cell=prepared.cell,
                            sender_ids=prepared.sender_ids,
                            message_ids=message_ids.get(prepared.memcell_id, ()),
                            group_id=ingested.group_id,
                            picture_names=prepared.picture_names,
                            failures=1,
                        )
                    continue
                outcomes[track].append(result)
                await self._bus.publish(result.events)
        if first_error is not None:
            raise first_error

        conversation = PipelineOutcome.from_cells(
            "conversation", outcomes["conversation"]
        )
        return _result(payload, ingested, _status(conversation.status))


def _result(
    payload: dict[str, Any],
    ingested: IngestResult,
    status: Literal["accumulated", "extracted"],
) -> MemorizeResult:
    """The batch's result: its messages, its status and what ingest dropped."""
    return MemorizeResult(
        message_count=len(payload.get("messages", [])),
        status=status,
        rejected=ingested.rejected_count,
        failed=ingested.failed_count,
        duplicates=ingested.skipped_count,
    )


def _held_too(ingested: IngestResult, boundary: BoundaryOutcome) -> IngestResult:
    """``ingested`` counting, as skipped, the messages its session already held."""
    if not boundary.duplicates:
        return ingested
    return ingested.model_copy(
        update={"skipped_count": ingested.skipped_count + boundary.duplicates}
    )


def _batch_messages(payload: dict[str, Any]) -> list[BatchMessage]:
    """The part of the batch's messages the roster cares about."""
    return [
        BatchMessage(
            sender_id=m["sender_id"],
            role=m["role"],
            at=epoch_ms(int(m["timestamp"])),
            sender_name=m.get("sender_name"),
        )
        for m in payload.get("messages") or []
    ]


def _person(message: BatchMessage) -> bool:
    """Whether the message is a person's: a role ``user`` sender."""
    return message.role == "user" and bool(message.sender_id)


def _own_cuts(
    prepared: MemorizeResult | tuple[IngestResult, list[BoundaryOutcome]],
) -> list[tuple[IngestResult, BoundaryOutcome]]:
    """The batch's own ``(ingested, boundary)`` pairs; none for a result."""
    if isinstance(prepared, MemorizeResult):
        return []
    ingested, boundaries = prepared
    return [(ingested, boundary) for boundary in boundaries]


def _prepared_cells(boundary: BoundaryOutcome) -> list[PreparedCell]:
    return [
        PreparedCell(
            cell=cell,
            memcell_id=memcell_id,
            sender_ids=senders,
            picture_names=names,
        )
        for cell, memcell_id, senders, names in zip(
            boundary.cells,
            boundary.memcell_ids,
            boundary.per_cell_all_senders,
            boundary.per_cell_picture_names,
            strict=True,
        )
    ]


def _log_cell_failure(
    ingested: IngestResult, prepared: PreparedCell, track: str, exc: BaseException
) -> None:
    logger.error(
        "memorize_cell_extraction_failed",
        session_id=ingested.session_id,
        memcell_id=prepared.memcell_id,
        track=track,
        error=f"{type(exc).__name__}: {exc}",
    )


def _status(track: TrackStatus) -> Literal["accumulated", "extracted"]:
    """``extracted`` when the track extracted; otherwise ``accumulated``."""
    if track == "extracted":
        return "extracted"
    return "accumulated"


def flush_payload(
    session_id: str,
    app_id: str,
    project_id: str,
    *,
    asker_id: str | None = None,
    asserted_asker: str | None = None,
) -> dict[str, Any]:
    """The payload a flush runs with: the session, and no new messages.

    The flush's asker (``asker_id``, the gateway's ``asserted_asker``) rides
    along only when given, so a flush that names none is the payload it
    always was; a group with a deliverer reads it (section 5.1).
    """
    payload: dict[str, Any] = {
        "session_id": session_id,
        "app_id": app_id,
        "project_id": project_id,
        "messages": [],
    }
    if asker_id is not None:
        payload["asker_id"] = asker_id
    if asserted_asker is not None:
        payload["asserted_asker"] = asserted_asker
    return payload
