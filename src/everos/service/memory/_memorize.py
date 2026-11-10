"""The memorize use case: cut a session's messages into memcells, extract them.

Memorize, end to end::

    payload { session_id, messages[] }
        -> ingest -> BoundaryDetection (buffer / merge / boundary / tail)
        -> for each cell: UserMemoryPipeline.run_cell
                          (+ AgentMemoryPipeline.run_cell in agent mode)
                          -> publish that cell's events on the EventBus
        -> merge the pipelines' statuses -> MemorizeResult

Boundary detection owns buffer / merge / boundary / tail, so the same cells
feed both pipelines in agent mode. Publishing is this use case's job:
pipelines and strategies return events, and each cell's are published as
soon as the cell is done, so a failing cell never takes an earlier cell's
events with it.

A session bound to a group (or a batch naming ``group_id``) takes the group
path (docs/group-memory.md, section 7.1): under the session lock the batch is
admitted into its group (binding, inline roster, implicit joins; 409 / 403 /
422 before anything is written), an inline roster change first cuts the open
buffer of every session of the group -- their locks held with the batch's, in
sorted order (D18) -- and the cells go to the group pipeline -- never
to the one-to-one pipelines, which would file group content as each sender's
personal memory. The bootstrap wires
:class:`~everos.memory.extract.pipeline.GroupMemoryPipeline`; a build without
one (``group_pipeline=None``) cuts group cells and keeps them as memcells,
unextracted.
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
from everos.core.utils.datetime import get_utc_now, to_timestamp_ms
from everos.memory import IngestResult as _IngestResult
from everos.memory.access import resolve_identity
from everos.memory.extract.ingest import process as ingest_process
from everos.memory.extract.pipeline import PipelineOutcome, PreparedCell
from everos.memory.groups import BatchMessage
from everos.memory.ports import Scope, SessionKey
from everos.service.memory._extraction import ExtractionSupervisor
from everos.service.memory._groups import to_events
from everos.service.memory._models import MemorizeResult

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.component.parser import RawFileBuilder
    from everos.config import Settings
    from everos.memory import IngestResult
    from everos.memory.extract.boundary import BoundaryDetection, BoundaryOutcome
    from everos.memory.extract.pipeline import CellOutcome, Pipeline
    from everos.memory.groups import RosterManager
    from everos.memory.ports import EventBus
    from everos.memory.ports.images import ImageStore

logger = get_logger("everos.service.memory")

Mode = Literal["chat", "agent"]
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
        user_pipeline: The user-memory track, run on every cell.
        agent_pipeline: The agent-memory track, run on every cell in agent mode.
        bus: Where each cell's events are published.
        extractions: Keeps extractions alive past a cancelled request.
        raw_file_builder: Host file hydration capability for ingestion.
        rosters: Admits a group batch into its group; ``None`` refuses
            ``group_id`` (a build without group memory).
        group_pipeline: Extracts group cells (the group episode); ``None``
            keeps them as memcells until it is wired.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        boundary: BoundaryDetection,
        multimodal_llm: LLMClient | None,
        user_pipeline: Pipeline,
        agent_pipeline: Pipeline,
        bus: EventBus,
        extractions: ExtractionSupervisor,
        raw_file_builder: RawFileBuilder,
        image_store: ImageStore | None = None,
        vision: bool = True,
        rosters: RosterManager | None = None,
        group_pipeline: Pipeline | None = None,
    ) -> None:
        self._rosters = rosters
        self._group_pipeline = group_pipeline
        self._multi_sender_warned: set[tuple[str, str, str]] = set()
        self._raw_file_builder = raw_file_builder
        self._image_store = image_store
        self._vision = vision
        self._settings = settings
        self._boundary = boundary
        self._multimodal_llm = multimodal_llm
        self._user_pipeline = user_pipeline
        self._agent_pipeline = agent_pipeline
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
        mode = self._settings.memorize.mode
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
                "mode": mode,
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
                elsewhere = await self._cut_buffers(
                    app_id, project_id, held_others, mode=mode
                )
                prepared = await self._cut(
                    payload,
                    mode=mode,
                    is_final=is_final,
                    defer_extraction=defer_extraction,
                )
            if isinstance(prepared, MemorizeResult) and not elsewhere:
                return prepared
            # The cells exist, so their extraction must not die with a
            # cancelled request (a client that went away). The task is owned
            # by the supervisor, which keeps it, retrieves its outcome and
            # waits for it at shutdown; the request only shields it.
            memcell_ids = [
                mid
                for _, b in [*elsewhere, *_own_cuts(prepared)]
                for mid in b.memcell_ids
            ]
            extraction = self._extractions.supervise(
                self._extract_cut(payload, prepared, elsewhere, mode=mode),
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
        self, app_id: str, project_id: str, sessions: Sequence[str], *, mode: Mode
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
            closed = await self._boundary.detect(empty, mode=mode, is_final=True)
            if closed.cells:
                cut.append((empty, closed))
        return cut

    async def _extract_cut(
        self,
        payload: dict[str, Any],
        prepared: MemorizeResult | tuple[IngestResult, list[BoundaryOutcome]],
        elsewhere: Sequence[tuple[IngestResult, BoundaryOutcome]],
        *,
        mode: Mode,
    ) -> MemorizeResult:
        """Extract the batch's own cells, then those cut from the group's others."""
        result = (
            prepared
            if isinstance(prepared, MemorizeResult)
            else await self._extract(payload, *prepared, mode=mode)
        )
        for ingested, closed in elsewhere:
            await self._extract(
                flush_payload(
                    ingested.session_id, ingested.app_id, ingested.project_id
                ),
                ingested,
                [closed],
                mode=mode,
            )
        return result

    async def _cut(
        self,
        payload: dict[str, Any],
        *,
        mode: Mode,
        is_final: bool,
        defer_extraction: bool,
    ) -> MemorizeResult | tuple[IngestResult, list[BoundaryOutcome]]:
        """Inner critical section — runs under the per-session lock.

        Returns the result when nothing reached the boundary, or the
        ingested batch and the boundary outcomes for :meth:`_extract` to run
        once the lock is released (two for a group batch whose roster change
        cut the open buffer first, one otherwise).
        """
        session = SessionKey(
            app_id=payload.get("app_id") or "default",
            project_id=payload.get("project_id") or "default",
            session_id=payload["session_id"],
        )
        group_id = await self._group_of(payload, session)
        cut_first: list[BoundaryOutcome] = []
        if group_id is not None:
            payload = {**payload, "group_id": group_id}
            cut_first = await self._admit_group_batch(
                payload, session, group_id, mode=mode, is_final=is_final
            )
        else:
            self._note_multi_sender(payload, session)
        ingested = await ingest_process(
            payload,
            multimodal_llm=self._multimodal_llm,
            raw_file_builder=self._raw_file_builder,
            image_store=self._image_store,
            vision=self._vision,
        )
        if defer_extraction and not is_final:
            buffered = await self._boundary.buffer(ingested, mode=mode)
            if cut_first:
                return ingested, cut_first
            return MemorizeResult(
                message_count=len(payload.get("messages", [])),
                status=_merge_status(buffered.status, "skipped"),
            )

        boundary = await self._boundary.detect(ingested, mode=mode, is_final=is_final)
        boundaries = [*cut_first, boundary] if boundary.cells else cut_first

        if not boundaries:
            # Nothing went past the boundary stage — no pipelines to dispatch.
            return MemorizeResult(
                message_count=len(payload.get("messages", [])),
                status=_merge_status(boundary.status, "skipped"),
            )
        return ingested, boundaries

    async def _group_of(
        self, payload: dict[str, Any], session: SessionKey
    ) -> str | None:
        """The group this batch belongs to: its ``group_id``, or the binding.

        Raises:
            RosterConflictError: The batch names another group than the one
                the session is bound to (409, D18).
            InvalidInputError: ``group_id`` on a build without group memory.
        """
        named = payload.get("group_id") or None
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

    async def _admit_group_batch(
        self,
        payload: dict[str, Any],
        session: SessionKey,
        group_id: str,
        *,
        mode: Mode,
        is_final: bool,
    ) -> list[BoundaryOutcome]:
        """Admit a group batch; cut the open buffer first when it changes the roster.

        Runs under the session lock, before the batch's messages are buffered.
        Returns the outcome of that cut (D18), or nothing.
        """
        assert self._rosters is not None
        messages = payload.get("messages") or []
        raw_roster = payload.get("roster") or []
        if not messages and not raw_roster:
            return []  # a flush: the binding already names the group
        await self._rosters.check_binding(session, group_id)
        identity = resolve_identity(
            header=payload.get("asserted_asker"), body_asker=payload.get("asker_id")
        )
        first = min((int(m["timestamp"]) for m in messages), default=None)
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
                mode=mode,
                is_final=True,
            )
            if closed.cells:
                cut.append(closed)
        await self._rosters.admit_batch(
            Scope(
                app_id=session.app_id,
                project_id=session.project_id,
                owner_id=group_id,
                owner_type="group",
            ),
            session=session,
            asker_id=identity.asker_id,
            roster=events,
            messages=[
                BatchMessage(
                    sender_id=m["sender_id"],
                    role=m["role"],
                    at=int(m["timestamp"]),
                    sender_name=m.get("sender_name"),
                )
                for m in messages
            ],
        )
        return cut

    def _note_multi_sender(self, payload: dict[str, Any], session: SessionKey) -> None:
        """Log once per session a one-to-one batch with several human senders.

        Such a batch keeps the per-sender behaviour (section 10); a group chat
        must send ``group_id`` to become group memory.
        """
        senders = {
            m.get("sender_id")
            for m in payload.get("messages") or []
            if m.get("role") == "user"
        }
        key = (session.app_id, session.project_id, session.session_id)
        if len(senders) < 2 or key in self._multi_sender_warned:
            return
        if len(self._multi_sender_warned) > 10_000:
            self._multi_sender_warned.clear()
        self._multi_sender_warned.add(key)
        logger.info(
            "memorize.multi_sender_without_group",
            app_id=session.app_id,
            project_id=session.project_id,
            session_id=session.session_id,
            senders=len(senders),
        )

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
        mode = self._settings.memorize.mode
        async with contextlib.AsyncExitStack() as held:
            locked = await self._hold_sessions(
                held, app_id=app_id, project_id=project_id, own=None, others=sessions
            )
            cut = await self._cut_buffers(app_id, project_id, locked, mode=mode)
            result = await change()
        for ingested, closed in cut:
            await self._extract(
                flush_payload(ingested.session_id, app_id, project_id),
                ingested,
                [closed],
                mode=mode,
            )
        return result

    async def _extract(
        self,
        payload: dict[str, Any],
        ingested: IngestResult,
        boundaries: Sequence[BoundaryOutcome],
        *,
        mode: Mode,
    ) -> MemorizeResult:
        """Run the pipelines over the cells the boundary stage wrote, cell by cell.

        Each pipeline's events for a cell are published as soon as that cell
        is done. A pipeline that fails on a cell is logged and the loop goes
        on, so the cells (and the other pipeline) that did succeed still
        reach their strategies; the first failure is re-raised once every
        cell has been tried, so the caller still sees the extraction failed.

        Raises:
            Exception: The first pipeline failure, after all cells ran.
        """
        if ingested.group_id is not None:
            return await self._extract_group(payload, ingested, boundaries)
        pipelines: dict[str, Pipeline] = {"user_memory": self._user_pipeline}
        if mode == "agent":
            pipelines["agent_memory"] = self._agent_pipeline
        outcomes: dict[str, list[CellOutcome]] = {track: [] for track in pipelines}
        first_error: BaseException | None = None
        cells = [c for boundary in boundaries for c in _prepared_cells(boundary)]
        for prepared in cells:
            results = await asyncio.gather(
                *(p.run_cell(ingested, prepared) for p in pipelines.values()),
                return_exceptions=True,
            )
            for track, result in zip(pipelines, results, strict=True):
                if isinstance(result, BaseException):
                    _log_cell_failure(ingested, prepared, track, result)
                    first_error = first_error or result
                    continue
                outcomes[track].append(result)
                await self._bus.publish(result.events)
        if first_error is not None:
            raise first_error

        user = PipelineOutcome.from_cells("user_memory", outcomes["user_memory"])
        agent_status: TrackStatus = (
            PipelineOutcome.from_cells("agent_memory", outcomes["agent_memory"]).status
            if mode == "agent"
            else "skipped"
        )
        return MemorizeResult(
            message_count=len(payload.get("messages", [])),
            status=_merge_status(user.status, agent_status),
        )

    async def _extract_group(
        self,
        payload: dict[str, Any],
        ingested: IngestResult,
        boundaries: Sequence[BoundaryOutcome],
    ) -> MemorizeResult:
        """Group cells go to the group pipeline, and to nothing else.

        The group write path (section 7): the group pipeline writes one group
        episode per cell under ``groups/<g>/``. Without one the cells stay
        memcells, unextracted, and the batch reads as accumulated -- never as
        the one-to-one per-sender copies.
        """
        cells = [c for boundary in boundaries for c in _prepared_cells(boundary)]
        count = len(payload.get("messages", []))
        if self._group_pipeline is None:
            logger.info(
                "memorize.group_extraction_pending",
                session_id=ingested.session_id,
                group_id=ingested.group_id,
                memcell_ids=[c.memcell_id for c in cells],
            )
            return MemorizeResult(message_count=count, status="accumulated")
        done: list[CellOutcome] = []
        first_error: BaseException | None = None
        for prepared in cells:
            try:
                result = await self._group_pipeline.run_cell(ingested, prepared)
            except Exception as exc:  # one cell's failure; the next still runs
                _log_cell_failure(ingested, prepared, "group_memory", exc)
                first_error = first_error or exc
                continue
            done.append(result)
            await self._bus.publish(result.events)
        if first_error is not None:
            raise first_error
        outcome = PipelineOutcome.from_cells("group_memory", done)
        return MemorizeResult(
            message_count=count, status=_merge_status(outcome.status, "skipped")
        )


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


def _merge_status(
    user: TrackStatus, agent: TrackStatus
) -> Literal["accumulated", "extracted"]:
    """Either ``extracted`` wins; otherwise ``accumulated``."""
    if user == "extracted" or agent == "extracted":
        return "extracted"
    return "accumulated"


def flush_payload(session_id: str, app_id: str, project_id: str) -> dict[str, Any]:
    """The payload a flush runs with: the session, and no new messages."""
    return {
        "session_id": session_id,
        "app_id": app_id,
        "project_id": project_id,
        "messages": [],
    }
