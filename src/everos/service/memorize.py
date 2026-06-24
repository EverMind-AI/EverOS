"""Memorize use case — ingest + boundary + dual pipeline dispatch.

End-to-end orchestration:

    POST /api/v1/memory/add { session_id, messages[] }
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
from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.engine import OfflineEngine
from everos.infra.persistence.markdown import EpisodeWriter
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
from everos.service._boundary import prepare_cells
from everos.service._session_lock import get_session_lock

logger = get_logger(__name__)


class MemorizeResult(BaseModel):
    """What memorize returns to the caller (route serialises it)."""

    message_count: int
    status: Literal["accumulated", "extracted"]


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
        _episode_writer = EpisodeWriter(MemoryRoot.default())
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


def _get_engine() -> OfflineEngine:
    """Return the singleton OfflineEngine; constructed + registered on first call.

    Lifecycle (start/stop) is wired by ``OmeLifespanProvider``.
    """
    global _ome_engine
    if _ome_engine is None:
        root = MemoryRoot.default()
        jobstore_path = root.ome_db
        jobstore_path.parent.mkdir(parents=True, exist_ok=True)
        engine = OfflineEngine(
            config=OMEConfig(
                jobstore_path=jobstore_path,
                config_path=root.ome_config,
            )
        )
        engine.register(extract_atomic_facts)
        engine.register(extract_foresight)
        engine.register(extract_agent_case)
        engine.register(trigger_skill_clustering)
        engine.register(extract_agent_skill)
        engine.register(trigger_profile_clustering)
        engine.register(extract_user_profile)
        engine.register(reflect_episodes)
        _ome_engine = engine
    return _ome_engine


# Public entry ───────────────────────────────────────────────────────────────


async def memorize(
    payload: dict[str, Any],
    *,
    is_final: bool = False,
) -> MemorizeResult:
    """Execute one add cycle. Dispatched concurrently across pipelines.

    Args:
        payload: ``{"session_id", "messages": [...]}`` — entrypoints DTO
            dumped to dict.
        is_final: ``True`` only for flush (algo guarantees ``tail=[]``).

    Concurrency: serialised per ``session_id`` via
    :func:`everos.service._session_lock.get_session_lock`. The lock
    spans the entire read-merge-boundary-write cycle so concurrent /add
    calls on the same session cannot lose-update each other's tail.
    An outer ``asyncio.timeout`` (configured by
    ``settings.memorize.session_lock_timeout_seconds``) ensures a stuck
    LLM cannot hold the lock indefinitely — on timeout the task is
    cancelled and ``async with`` auto-releases the lock.
    """
    settings = load_settings()
    mode = settings.memorize.mode
    boundary_cfg = settings.boundary_detection
    session_id = payload["session_id"]

    async with asyncio.timeout(settings.memorize.session_lock_timeout_seconds):
        async with get_session_lock(session_id):
            return await _memorize_locked(
                payload,
                mode=mode,
                boundary_cfg=boundary_cfg,
                is_final=is_final,
            )


async def _memorize_locked(
    payload: dict[str, Any],
    *,
    mode: Literal["chat", "agent"],
    boundary_cfg: Any,
    is_final: bool,
) -> MemorizeResult:
    """Inner critical section — runs under the per-session lock."""
    ingested = await ingest_process(payload)
    boundary = await prepare_cells(
        ingested,
        mode=mode,
        is_final=is_final,
        llm_client=get_llm_client(),
        prompt_loader=_get_prompt_loader(),
        hard_token_limit=boundary_cfg.hard_token_limit,
        hard_msg_limit=boundary_cfg.hard_msg_limit,
    )

    if not boundary.cells:
        # Nothing went past the boundary stage — no pipelines to dispatch.
        return MemorizeResult(
            message_count=len(payload.get("messages", [])),
            status=_merge_status(boundary.status, "skipped"),
        )

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
