"""MemoryService -- the memory read / write use cases behind one facade.

One method per use case: :meth:`MemoryService.memorize` (``/add``),
:meth:`~MemoryService.flush` (``/flush``), :meth:`~MemoryService.search` and
:meth:`~MemoryService.get`, plus the ``async_mode`` queue the ``/add`` route
can hand a request to instead of running it inline.

Memorize, end to end::

    payload { session_id, messages[] }
        -> ingest -> boundary stage (buffer / merge / boundary / tail)
        -> for each cell: UserMemoryPipeline.run_cell
                          (+ AgentMemoryPipeline.run_cell in agent mode)
                          -> publish that cell's events on the EventBus
        -> merge the pipelines' statuses -> MemorizeResult

The boundary stage owns buffer / merge / boundary / tail, so the same cells
feed both pipelines in agent mode. Publishing is this service's job:
pipelines and strategies return events, and each cell's are published as
soon as the cell is done, so a failing cell never takes an earlier cell's
events with it.

Everything is injected by :func:`everos.service.build_application`; the
service owns only its two background collaborators -- the extraction
supervisor and the ``async_mode`` queue worker -- and their lifecycle
(:meth:`~MemoryService.start` / :meth:`~MemoryService.aclose`).
"""

from __future__ import annotations

__all__ = ["ExtractionSupervisor", "MemorizeResult", "MemoryService"]

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from everos.core.context import resolve_request_id
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span
from everos.memory.extract.ingest import process as ingest_process
from everos.memory.extract.pipeline import PipelineOutcome, PreparedCell
from everos.service._boundary import BoundaryOutcome, buffer_messages, prepare_cells
from everos.service._session_lock import get_session_lock
from everos.service.memorize_queue import MemorizeQueueWorker, QueueHealth

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from everalgo.llm import LLMClient

    from everos.config import Settings
    from everos.memory import IngestResult
    from everos.memory.deps import Stores
    from everos.memory.extract.pipeline import CellOutcome, Pipeline
    from everos.memory.get import GetManager, GetRequest, GetResponse
    from everos.memory.ports import EventBus
    from everos.memory.prompt_slots import PromptLoader
    from everos.memory.search import SearchRequest, SearchResponse
    from everos.memory.search.manager import SearchManager

logger = get_logger(__name__)

_DRAIN_TIMEOUT_SECONDS = 30.0
"""How long :meth:`MemoryService.aclose` waits for extractions still running."""

Mode = Literal["chat", "agent"]
TrackStatus = Literal["accumulated", "extracted", "skipped"]


class MemorizeResult(BaseModel):
    """What memorize returns to the caller (route serialises it)."""

    message_count: int
    status: Literal["accumulated", "extracted"]


class ExtractionSupervisor:
    """Owns the extraction tasks that may outlive their request.

    A request whose client went away is cancelled while its extraction
    runs; the memcell is already committed, so the extraction must finish.
    ``supervise`` keeps the task referenced until it ends and retrieves its
    outcome: a failure is logged with the memcell ids (it is not retried
    here -- the pipelines write markdown and emit events and are not
    idempotent, the same reason the async queue does not retry a failed
    delivery), and ``drain`` lets shutdown wait for what is still running
    instead of dropping it.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[MemorizeResult]] = set()

    def supervise(
        self,
        coro: Coroutine[Any, Any, MemorizeResult],
        *,
        session_id: str,
        memcell_ids: list[str],
    ) -> asyncio.Task[MemorizeResult]:
        task: asyncio.Task[MemorizeResult] = asyncio.ensure_future(coro)
        self._tasks.add(task)

        def _done(t: asyncio.Task[MemorizeResult]) -> None:
            self._tasks.discard(t)
            if t.cancelled():
                logger.error(
                    "memorize_extraction_cancelled",
                    session_id=session_id,
                    memcell_ids=memcell_ids,
                )
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    "memorize_extraction_failed",
                    session_id=session_id,
                    memcell_ids=memcell_ids,
                    error=f"{type(exc).__name__}: {exc}",
                )

        task.add_done_callback(_done)
        return task

    @property
    def in_flight(self) -> int:
        return len(self._tasks)

    async def drain(self, wait_seconds: float) -> int:
        """Wait up to ``wait_seconds`` for running extractions; return what is left."""
        pending = {t for t in self._tasks if not t.done()}
        if pending:
            await asyncio.wait(pending, timeout=wait_seconds)
        return sum(1 for t in self._tasks if not t.done())


class MemoryService:
    """Memorize, flush, search and get, over injected domain objects and ports.

    Args:
        settings: Application settings (``[memorize]``, ``[boundary_detection]``).
        llm: Boundary-detection LLM; ``None`` leaves messages buffered.
        multimodal_llm: Parses non-text content; ``None`` when
            ``[multimodal]`` is not configured.
        prompt_loader: Source of the boundary-detection prompt.
        stores: Write-side stores (session buffer, memcells, memorize inbox).
        user_pipeline: The user-memory track, run on every cell.
        agent_pipeline: The agent-memory track, run on every cell in agent mode.
        bus: Where each cell's events are published.
        search_manager: Answers :meth:`search`.
        get_manager: Answers :meth:`get`.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMClient | None,
        multimodal_llm: LLMClient | None,
        prompt_loader: PromptLoader,
        stores: Stores,
        user_pipeline: Pipeline,
        agent_pipeline: Pipeline,
        bus: EventBus,
        search_manager: SearchManager,
        get_manager: GetManager,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._multimodal_llm = multimodal_llm
        self._prompt_loader = prompt_loader
        self._stores = stores
        self._user_pipeline = user_pipeline
        self._agent_pipeline = agent_pipeline
        self._bus = bus
        self._search_manager = search_manager
        self._get_manager = get_manager
        self._extractions = ExtractionSupervisor()
        queue_cfg = settings.memorize.async_queue
        self._queue = MemorizeQueueWorker(
            queue_cfg,
            inbox=stores.memorize_inbox,
            deliver=self.memorize,
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
        running = self._extractions.in_flight
        if not running:
            return
        logger.info("memorize_extractions_draining", in_flight=running)
        left = await self._extractions.drain(_DRAIN_TIMEOUT_SECONDS)
        if left:
            logger.error(
                "memorize_extractions_abandoned",
                in_flight=left,
                waited_seconds=_DRAIN_TIMEOUT_SECONDS,
            )

    # ── read use cases ─────────────────────────────────────────────────────

    async def search(self, req: SearchRequest) -> SearchResponse:
        """Hybrid retrieval across the configured memory backends."""
        return await self._search_manager.search(req)

    async def get(self, req: GetRequest) -> GetResponse:
        """Paginated listing over the requested ``memory_type``."""
        return await self._get_manager.get(req)

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

    async def flush(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> MemorizeResult:
        """Force boundary detection over the session's whole buffer."""
        payload: dict[str, Any] = {
            "session_id": session_id,
            "app_id": app_id,
            "project_id": project_id,
            "messages": [],
        }
        return await self.memorize(payload, is_final=True)

    async def memorize(
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

        Concurrency: serialised per ``(app_id, project_id, session_id)`` via
        :func:`everos.service._session_lock.get_session_lock`. The lock
        spans the read-merge-boundary-write cycle so concurrent /add calls
        on the same session cannot lose-update each other's tail.

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
            lock_key = (
                f"{payload.get('app_id', 'default')}\x1f"
                f"{payload.get('project_id', 'default')}\x1f{session_id}"
            )
            timeout_s = self._settings.memorize.session_lock_timeout_seconds
            lock = get_session_lock(lock_key)
            async with asyncio.timeout(timeout_s):
                await lock.acquire()
            try:
                prepared = await self._memorize_locked(
                    payload,
                    mode=mode,
                    is_final=is_final,
                    defer_extraction=defer_extraction,
                    boundary_timeout=timeout_s,
                )
            finally:
                lock.release()
            if isinstance(prepared, MemorizeResult):
                return prepared
            ingested, boundary = prepared
            # The cells exist, so their extraction must not die with a
            # cancelled request (a client that went away). The task is owned
            # by the supervisor, which keeps it, retrieves its outcome and
            # waits for it at shutdown; the request only shields it.
            extraction = self._extractions.supervise(
                self._extract(payload, ingested, boundary, mode=mode),
                session_id=session_id,
                memcell_ids=boundary.memcell_ids,
            )
            try:
                return await asyncio.shield(extraction)
            except asyncio.CancelledError:
                logger.warning(
                    "memorize_extraction_continues_after_cancel",
                    session_id=session_id,
                    memcell_ids=boundary.memcell_ids,
                )
                raise

    async def _memorize_locked(
        self,
        payload: dict[str, Any],
        *,
        mode: Mode,
        is_final: bool,
        defer_extraction: bool,
        boundary_timeout: float | None = None,
    ) -> MemorizeResult | tuple[IngestResult, BoundaryOutcome]:
        """Inner critical section — runs under the per-session lock.

        Returns the result when nothing reached the boundary, or the
        ingested batch and the boundary outcome for :meth:`_extract` to run
        once the lock is released.
        """
        ingested = await ingest_process(payload, multimodal_llm=self._multimodal_llm)
        if defer_extraction and not is_final:
            buffered = await buffer_messages(
                ingested, mode=mode, session_buffer=self._stores.session_buffer
            )
            return MemorizeResult(
                message_count=len(payload.get("messages", [])),
                status=_merge_status(buffered.status, "skipped"),
            )

        # Boundary detection runs an LLM (everalgo detect_boundaries). Wrap it
        # in a generation span so its token usage lands on a GENERATION
        # observation (costed by Langfuse) instead of the SPAN-typed request
        # root (dropped).
        boundary_cfg = self._settings.boundary_detection
        with memory_span("everos.memcell.boundary", observation_type="generation"):
            boundary = await prepare_cells(
                ingested,
                mode=mode,
                is_final=is_final,
                llm_client=self._llm,
                prompt_loader=self._prompt_loader,
                hard_token_limit=boundary_cfg.hard_token_limit,
                hard_msg_limit=boundary_cfg.hard_msg_limit,
                session_buffer=self._stores.session_buffer,
                memcell_store=self._stores.memcell,
                timeout_seconds=boundary_timeout,
            )

        if not boundary.cells:
            # Nothing went past the boundary stage — no pipelines to dispatch.
            return MemorizeResult(
                message_count=len(payload.get("messages", [])),
                status=_merge_status(boundary.status, "skipped"),
            )
        return ingested, boundary

    async def _extract(
        self,
        payload: dict[str, Any],
        ingested: IngestResult,
        boundary: BoundaryOutcome,
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
        pipelines: dict[str, Pipeline] = {"user_memory": self._user_pipeline}
        if mode == "agent":
            pipelines["agent_memory"] = self._agent_pipeline
        outcomes: dict[str, list[CellOutcome]] = {track: [] for track in pipelines}
        first_error: BaseException | None = None
        for prepared in _prepared_cells(boundary):
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


def _prepared_cells(boundary: BoundaryOutcome) -> list[PreparedCell]:
    return [
        PreparedCell(cell=cell, memcell_id=memcell_id, sender_ids=senders)
        for cell, memcell_id, senders in zip(
            boundary.cells,
            boundary.memcell_ids,
            boundary.per_cell_all_senders,
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
