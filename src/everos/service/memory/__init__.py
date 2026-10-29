"""MemoryService -- the memory read / write use cases behind one facade.

One method per use case: :meth:`MemoryService.memorize` (``/add``),
:meth:`~MemoryService.flush` (``/flush``), :meth:`~MemoryService.search` and
:meth:`~MemoryService.get`, plus the ``async_mode`` queue the ``/add`` route
can hand a request to instead of running it inline.

The memorize use case itself lives in ``_memorize`` (the private steps:
lock, cut, extract, publish); ``_extraction`` keeps an extraction alive past
a cancelled request and ``_session_locks`` holds the per-session locks.

Everything is injected by :func:`everos.service.build_application`; the
service owns only its two background collaborators -- the extraction
supervisor and the ``async_mode`` queue worker -- and their lifecycle
(:meth:`~MemoryService.start` / :meth:`~MemoryService.aclose`).
"""

from __future__ import annotations

__all__ = ["ExtractionSupervisor", "MemorizeResult", "MemoryService", "Queued"]

from typing import TYPE_CHECKING, Any

from everos.core.observability.logging import get_logger
from everos.service.memorize_queue import MemorizeQueueWorker, QueueHealth

from ._extraction import ExtractionSupervisor as ExtractionSupervisor
from ._memorize import Memorizer
from ._models import MemorizeResult as MemorizeResult
from ._models import Queued as Queued
from ._session_locks import SessionLocks

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

    from everos.config import Settings
    from everos.memory.deps import Stores
    from everos.memory.extract.boundary import BoundaryDetection
    from everos.memory.extract.pipeline import Pipeline
    from everos.memory.get import GetManager, GetRequest, GetResponse
    from everos.memory.ports import EventBus
    from everos.memory.search import SearchRequest, SearchResponse
    from everos.memory.search.manager import SearchManager

logger = get_logger(__name__)

_DRAIN_TIMEOUT_SECONDS = 30.0
"""How long :meth:`MemoryService.aclose` waits for extractions still running."""


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
    ) -> None:
        self._search_manager = search_manager
        self._get_manager = get_manager
        self._extractions = ExtractionSupervisor()
        self._memorizer = Memorizer(
            settings=settings,
            boundary=boundary,
            multimodal_llm=multimodal_llm,
            user_pipeline=user_pipeline,
            agent_pipeline=agent_pipeline,
            bus=bus,
            locks=SessionLocks(),
            extractions=self._extractions,
        )
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

    async def submit_add(
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
        return await self.memorize(payload, defer_extraction=defer_extraction)

    async def submit_flush(
        self,
        *,
        session_id: str,
        request_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> MemorizeResult | Queued:
        """Run one ``/flush`` inline, or queue it behind the session's backlog.

        A flush has no ``async_mode`` (the Cloud API has none either); it is
        queued only when earlier requests of the session still are.

        Raises:
            QueueSaturatedError: Queued, and the session or queue is full.
        """
        payload = _flush_payload(session_id, app_id, project_id)
        if await self._has_backlog(payload):
            return await self._queue_request(
                payload, request_id=request_id, is_final=True, defer_extraction=False
            )
        return await self.flush(
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

    async def flush(
        self,
        *,
        session_id: str,
        app_id: str = "default",
        project_id: str = "default",
    ) -> MemorizeResult:
        """Force boundary detection over the session's whole buffer."""
        return await self.memorize(
            _flush_payload(session_id, app_id, project_id), is_final=True
        )

    async def memorize(
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


def _flush_payload(session_id: str, app_id: str, project_id: str) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "app_id": app_id,
        "project_id": project_id,
        "messages": [],
    }
