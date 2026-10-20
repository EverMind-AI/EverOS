"""Memorize use case — ingest + boundary + dual pipeline dispatch.

End-to-end orchestration:

    POST /api/v2/memory/add { session_id, messages[] }
        → ingest.process → IngestResult
        → _boundary.prepare_cells(mode=settings.memorize.mode) → cells
        → asyncio.gather(
            UserMemoryPipeline.run(cells, ...),
            AgentMemoryPipeline.run(cells, ...) if mode == "agent",
          )
        → merge outcome.status → {message_count, status}

The boundary stage owns buffer / merge / boundary / tail — so the same
``cells`` feed both pipelines in agent mode (chat mode skips the agent
pipeline entirely).

Lazy singletons: writer / loader / pipelines / LLM client are all
constructed on first use (service module imports run before lifespan
resolves the memory-root and reads env vars).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from everos.component.llm import get_llm_client
from everos.config import load_settings
from everos.core.context import resolve_request_id
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import memory_span
from everos.core.persistence import MemoryRoot
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.engine import OfflineEngine
from everos.infra.persistence.markdown import EpisodeWriter
from everos.memory import IngestResult
from everos.memory.extract.ingest import process as ingest_process
from everos.memory.extract.pipeline import (
    AgentMemoryPipeline,
    UserMemoryPipeline,
)
from everos.memory.prompt_slots import PromptLoader
from everos.memory.strategies import (
    extract_agent_case,
    extract_agent_skill,
    extract_atomic_facts,
    extract_foresight,
    extract_user_profile,
    reflect_episodes,
    trigger_profile_clustering,
    trigger_skill_clustering,
)
from everos.service._boundary import BoundaryOutcome, buffer_messages, prepare_cells
from everos.service._session_lock import get_session_lock

logger = get_logger(__name__)


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
    delivery), and ``drain`` lets the lifespan wait for what is still
    running at shutdown instead of dropping it.
    """

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task[MemorizeResult]] = set()

    def supervise(
        self,
        coro: Any,
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


extractions = ExtractionSupervisor()


# Lazy singletons ────────────────────────────────────────────────────────────


_episode_writer: EpisodeWriter | None = None
_prompt_loader: PromptLoader | None = None
_user_pipeline: UserMemoryPipeline | None = None
_agent_pipeline: AgentMemoryPipeline | None = None
_ome_engine: OfflineEngine | None = None


def _config_root() -> Path:
    """Return the directory holding bundled prompt slots (``config/``)."""
    # ``src/everos/config/`` ships in the wheel alongside this service module.
    return Path(__file__).resolve().parent.parent / "config"


def _get_episode_writer() -> EpisodeWriter:
    global _episode_writer
    if _episode_writer is None:
        _episode_writer = EpisodeWriter(MemoryRoot.resolve())
    return _episode_writer


def _get_prompt_loader() -> PromptLoader:
    global _prompt_loader
    if _prompt_loader is None:
        _prompt_loader = PromptLoader(_config_root())
    return _prompt_loader


def _get_user_pipeline() -> UserMemoryPipeline:
    global _user_pipeline
    if _user_pipeline is None:
        _user_pipeline = UserMemoryPipeline(
            episode_writer=_get_episode_writer(),
            prompt_loader=_get_prompt_loader(),
            llm_client=get_llm_client(),
            engine=_get_engine(),
        )
    return _user_pipeline


def _get_agent_pipeline() -> AgentMemoryPipeline:
    global _agent_pipeline
    if _agent_pipeline is None:
        _agent_pipeline = AgentMemoryPipeline(engine=_get_engine())
    return _agent_pipeline


_STRATEGIES_ALWAYS = (
    extract_atomic_facts,
    extract_foresight,
    extract_agent_case,
    extract_user_profile,
)
"""LLM-only strategies — no embed dependency, always registered."""

_STRATEGIES_REQUIRE_EMBED = (
    trigger_skill_clustering,
    extract_agent_skill,
    trigger_profile_clustering,
    reflect_episodes,
)
"""Strategies whose body re-embeds or consumes a re-embedded cluster.

All are registered unconditionally; each guards its own body via
:func:`everos.component.embedding.get_embedding_capability` at execution
time. When capability is unavailable the strategy no-ops (logged at
debug level) rather than raising deep inside its body.

Why body-guard: the capability singleton at
:mod:`everos.component.embedding.accessor` is process-wide and never
invalidated at runtime — a tier upgrade (Tier 1 → Tier 2 by editing
everos.toml) is applied only after a server restart. Body-guard is
NOT a hot-reload mechanism; it exists for defensive degradation so
that direct-emit paths (unit tests, future features that bypass the
upstream clustering gate) still fail cleanly instead of crashing
inside recall.
"""


def _get_engine() -> OfflineEngine:
    """Return the singleton OfflineEngine; constructed + registered on first call.

    All strategies (both always-on and embed-requiring) are registered
    unconditionally. Embed-requiring strategies guard their own bodies
    with :func:`everos.component.embedding.get_embedding_capability`.

    Tier changes require a server restart to take effect: the accessor's
    cached capability is not invalidated at runtime. Body-guard keeps
    dispatches clean when capability was absent at start; it does not
    substitute for restart on config changes.

    Lifecycle (start/stop) is wired by ``OmeLifespanProvider``.
    """
    global _ome_engine
    if _ome_engine is None:
        root = MemoryRoot.resolve()
        jobstore_path = root.ome_db
        jobstore_path.parent.mkdir(parents=True, exist_ok=True)
        engine = OfflineEngine(
            config=OMEConfig(
                jobstore_path=jobstore_path,
                config_path=root.ome_config,
            )
        )
        for strategy in (*_STRATEGIES_ALWAYS, *_STRATEGIES_REQUIRE_EMBED):
            engine.register(strategy)
        _ome_engine = engine
    return _ome_engine


# Public entry ───────────────────────────────────────────────────────────────


async def memorize(
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
        defer_extraction: Persist fresh messages to the unprocessed buffer but
            skip boundary detection and both extraction pipelines. Intended for
            clients that batch work and call ``/flush`` on idle or thresholds.

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
    settings = load_settings()
    mode = settings.memorize.mode
    boundary_cfg = settings.boundary_detection
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
        timeout_s = settings.memorize.session_lock_timeout_seconds
        lock = get_session_lock(lock_key)
        async with asyncio.timeout(timeout_s):
            await lock.acquire()
        try:
            prepared = await _memorize_locked(
                payload,
                mode=mode,
                boundary_cfg=boundary_cfg,
                is_final=is_final,
                defer_extraction=defer_extraction,
                boundary_timeout=timeout_s,
            )
        finally:
            lock.release()
        if isinstance(prepared, MemorizeResult):
            return prepared
        ingested, boundary = prepared
        # The cells exist, so their extraction must not die with a cancelled
        # request (a client that went away). The task is owned by the
        # module's supervisor, which keeps it, retrieves its outcome and
        # waits for it at shutdown; the request only shields it.
        extraction = extractions.supervise(
            _extract(payload, ingested, boundary, mode=mode),
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
    payload: dict[str, Any],
    *,
    mode: Literal["chat", "agent"],
    boundary_cfg: Any,
    is_final: bool,
    defer_extraction: bool,
    boundary_timeout: float | None = None,
) -> MemorizeResult | tuple[IngestResult, BoundaryOutcome]:
    """Inner critical section — runs under the per-session lock.

    Returns the result when nothing reached the boundary, or the
    ingested batch and the boundary outcome for :func:`_extract` to run
    once the lock is released.
    """
    ingested = await ingest_process(payload)
    if defer_extraction and not is_final:
        buffered = await buffer_messages(ingested, mode=mode)
        return MemorizeResult(
            message_count=len(payload.get("messages", [])),
            status=_merge_status(buffered.status, "skipped"),
        )

    # Boundary detection runs an LLM (everalgo detect_boundaries). Wrap it in a
    # generation span so its token usage lands on a GENERATION observation
    # (costed by Langfuse) instead of the SPAN-typed request root (dropped).
    with memory_span("everos.memcell.boundary", observation_type="generation"):
        boundary = await prepare_cells(
            ingested,
            mode=mode,
            is_final=is_final,
            llm_client=get_llm_client(),
            prompt_loader=_get_prompt_loader(),
            hard_token_limit=boundary_cfg.hard_token_limit,
            hard_msg_limit=boundary_cfg.hard_msg_limit,
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
    payload: dict[str, Any],
    ingested: IngestResult,
    boundary: BoundaryOutcome,
    *,
    mode: Literal["chat", "agent"],
) -> MemorizeResult:
    """Run the extraction pipelines for the cells the boundary stage wrote."""
    user_task = _get_user_pipeline().run(
        ingested,
        cells=boundary.cells,
        memcell_ids=boundary.memcell_ids,
        per_cell_all_senders=boundary.per_cell_all_senders,
    )
    if mode == "agent":
        agent_task = _get_agent_pipeline().run(
            ingested,
            cells=boundary.cells,
            memcell_ids=boundary.memcell_ids,
        )
        user_outcome, agent_outcome = await asyncio.gather(user_task, agent_task)
        merged_status = _merge_status(user_outcome.status, agent_outcome.status)
    else:
        user_outcome = await user_task
        merged_status = _merge_status(user_outcome.status, "skipped")

    return MemorizeResult(
        message_count=len(payload.get("messages", [])),
        status=merged_status,
    )


def _merge_status(
    user: Literal["accumulated", "extracted", "skipped"],
    agent: Literal["accumulated", "extracted", "skipped"],
) -> Literal["accumulated", "extracted"]:
    """Either ``extracted`` wins; otherwise ``accumulated``."""
    if user == "extracted" or agent == "extracted":
        return "extracted"
    return "accumulated"
