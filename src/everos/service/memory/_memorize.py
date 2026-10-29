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
"""

from __future__ import annotations

__all__ = ["Memorizer"]

import asyncio
from typing import TYPE_CHECKING, Any, Literal

from everos.core.context import resolve_request_id
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span
from everos.memory.extract.ingest import process as ingest_process
from everos.memory.extract.pipeline import PipelineOutcome, PreparedCell
from everos.service.memory._extraction import ExtractionSupervisor
from everos.service.memory._models import MemorizeResult
from everos.service.memory._session_locks import SessionLocks

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.config import Settings
    from everos.memory import IngestResult
    from everos.memory.extract.boundary import BoundaryDetection, BoundaryOutcome
    from everos.memory.extract.pipeline import CellOutcome, Pipeline
    from everos.memory.ports import EventBus

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
        user_pipeline: The user-memory track, run on every cell.
        agent_pipeline: The agent-memory track, run on every cell in agent mode.
        bus: Where each cell's events are published.
        locks: The per-session locks.
        extractions: Keeps extractions alive past a cancelled request.
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
        locks: SessionLocks,
        extractions: ExtractionSupervisor,
    ) -> None:
        self._settings = settings
        self._boundary = boundary
        self._multimodal_llm = multimodal_llm
        self._user_pipeline = user_pipeline
        self._agent_pipeline = agent_pipeline
        self._bus = bus
        self._locks = locks
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

        Concurrency: serialised per ``(app_id, project_id, session_id)`` via
        :class:`SessionLocks`. The lock
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
            lock = self._locks.get(lock_key)
            async with asyncio.timeout(timeout_s):
                await lock.acquire()
            try:
                prepared = await self._cut(
                    payload,
                    mode=mode,
                    is_final=is_final,
                    defer_extraction=defer_extraction,
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

    async def _cut(
        self,
        payload: dict[str, Any],
        *,
        mode: Mode,
        is_final: bool,
        defer_extraction: bool,
    ) -> MemorizeResult | tuple[IngestResult, BoundaryOutcome]:
        """Inner critical section — runs under the per-session lock.

        Returns the result when nothing reached the boundary, or the
        ingested batch and the boundary outcome for :meth:`_extract` to run
        once the lock is released.
        """
        ingested = await ingest_process(payload, multimodal_llm=self._multimodal_llm)
        if defer_extraction and not is_final:
            buffered = await self._boundary.buffer(ingested, mode=mode)
            return MemorizeResult(
                message_count=len(payload.get("messages", [])),
                status=_merge_status(buffered.status, "skipped"),
            )

        boundary = await self._boundary.detect(ingested, mode=mode, is_final=is_final)

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
