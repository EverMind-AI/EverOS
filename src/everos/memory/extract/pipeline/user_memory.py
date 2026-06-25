"""User memory pipeline — per-sender Episode fan-out on pre-cut cells.

Cells / memcell_ids / message_id-mapping / sender lists are produced by
:mod:`everos.service._boundary` (which also writes the single
``memcell`` sqlite row per cell). This pipeline only handles the
user-perspective output: Episode md + ``UserPipelineStarted`` emit (one
per cell, fired at the start of ``run`` so atomic_fact / foresight /
clustering strategies run in parallel with the in-pipeline Episode work).

Run inside ``service.memorize`` via ``asyncio.gather`` alongside
:class:`AgentMemoryPipeline` (the latter only in ``mode="agent"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.types import MemCell as AlgoMemCell
from everalgo.user_memory import EpisodeExtractor

from everos.component.utils.datetime import from_timestamp, to_iso_format
from everos.core.observability.logging import get_logger
from everos.memory import Episode, IngestResult, PipelineOutcome
from everos.memory.events import EpisodeExtracted, UserPipelineStarted
from everos.memory.prompt_slots import PromptLoader

if TYPE_CHECKING:
    from everalgo.llm.protocols import LLMClient

    from everos.infra.ome.engine import OfflineEngine
    from everos.infra.persistence.markdown import EpisodeWriter

logger = get_logger(__name__)

_TRACK = "user_memory"


class UserMemoryPipeline:
    """Per-sender Episode extraction on a list of pre-cut MemCells."""

    def __init__(
        self,
        episode_writer: EpisodeWriter,
        prompt_loader: PromptLoader,
        llm_client: LLMClient | None,
        engine: OfflineEngine,
    ) -> None:
        # EpisodeExtractor requires `llm` at construction. Skip-with-warning
        # when no LLM is configured — the boundary stage will have skipped
        # the run already; this is just a defensive null check.
        self._ep_ext = (
            EpisodeExtractor(llm=llm_client) if llm_client is not None else None
        )
        self._episode_writer = episode_writer
        self._prompt_loader = prompt_loader
        self._engine = engine

    async def run(
        self,
        ingested: IngestResult,
        cells: list[AlgoMemCell],
        memcell_ids: list[str],
        per_cell_all_senders: list[list[str]],
    ) -> PipelineOutcome:
        """Emit UserPipelineStarted per cell, then extract Episodes + write md."""
        if not cells:
            return PipelineOutcome(track=_TRACK, status="accumulated", message_count=0)
        if self._ep_ext is None:
            logger.warning(
                "user_memory_pipeline_no_llm_client",
                extra={"session_id": ingested.session_id, "cells": len(cells)},
            )
            return PipelineOutcome(track=_TRACK, status="skipped", message_count=0)

        # Emit upfront so OME-async strategies (atomic_fact / foresight /
        # cluster) start in parallel with the in-pipeline Episode work; they
        # consume the MemCell directly and do not depend on Episode output.
        for cell, memcell_id in zip(cells, memcell_ids, strict=True):
            await self._emit_pipeline_started(
                memcell_id=memcell_id,
                session_id=ingested.session_id,
                app_id=ingested.app_id,
                project_id=ingested.project_id,
                cell=cell,
            )

        episode_prompt = self._prompt_loader.load("episode_extract")
        md_paths: list[str] = []
        msg_count = 0
        for cell, memcell_id, all_senders in zip(
            cells, memcell_ids, per_cell_all_senders, strict=True
        ):
            msg_count += len(cell.items)
            user_senders = _unique_user_senders(cell)
            if not user_senders:
                continue
            # One generic LLM call per cell (sender_id=None drives the algo's
            # whole-memcell EPISODE_GENERATION_PROMPT — explicitly cheaper
            # than the per-user fan-out per the algo's docstring). Fan-out
            # is then md-only: every user sender owns a copy of the same
            # narrative under its own owner_id path.
            algo_ep = await self._ep_ext.aextract(
                cell, sender_id=None, prompt=episode_prompt
            )
            for sender_id in user_senders:
                ep = Episode.from_algo(
                    algo_ep,
                    owner_id=sender_id,
                    session_id=ingested.session_id,
                    sender_ids=all_senders,
                    parent_id=memcell_id,
                )
                inline, sections = _episode_to_entry_body(ep)
                eid = await self._episode_writer.append_entry(
                    ep.owner_id,
                    inline=inline,
                    sections=sections,
                    app_id=ingested.app_id,
                    project_id=ingested.project_id,
                )
                md_paths.append(
                    str(
                        self._episode_writer.path_for(
                            ep.owner_id,
                            eid.date,
                            app_id=ingested.app_id,
                            project_id=ingested.project_id,
                        )
                    )
                )
                await self._engine.emit(
                    EpisodeExtracted(
                        memcell_id=memcell_id,
                        episode_entry_id=eid.format(),
                        episode_text=ep.episode,
                        episode_timestamp_ms=ep.timestamp,
                        owner_id=ep.owner_id,
                        session_id=ingested.session_id,
                        app_id=ingested.app_id,
                        project_id=ingested.project_id,
                        source="pipeline",
                    )
                )

        return PipelineOutcome(
            track=_TRACK,
            status="extracted",
            message_count=msg_count,
            extracted_md_paths=md_paths,
        )

    async def _emit_pipeline_started(
        self,
        memcell_id: str,
        session_id: str,
        app_id: str,
        project_id: str,
        cell: AlgoMemCell,
    ) -> None:
        await self._engine.emit(
            UserPipelineStarted(
                memcell_id=memcell_id,
                session_id=session_id,
                app_id=app_id,
                project_id=project_id,
                memcell=cell,
            )
        )


# ── Helpers ───────────────────────────────────────────────────────────────


def _unique_user_senders(cell: AlgoMemCell) -> list[str]:
    """Distinct role=user sender_ids in a cell, preserving order.

    Drives per-sender Episode fan-out: each user perspective gets its own
    Episode for the cell. Skips non-``ChatMessage`` items (agent
    trajectories' ``ToolCallResult`` has no ``role``).
    """
    senders: list[str] = []
    for item in cell.items:
        if getattr(item, "role", None) != "user":
            continue
        sid = getattr(item, "sender_id", None)
        if sid and sid not in senders:
            senders.append(sid)
    return senders


def _episode_to_entry_body(
    episode: Episode,
) -> tuple[dict[str, object], dict[str, str]]:
    """Split a domain Episode into ``(inline, sections)`` for md rendering.

    Lives in the pipeline (memory) layer rather than the writer (infra)
    because it depends on :class:`everos.memory.Episode` — infra is not
    allowed to import memory per the layered architecture contract.

    Inline persists the audit / scope fields cascade needs to rebuild
    the LanceDB row: ``owner_id`` / ``session_id`` / ``timestamp`` /
    ``parent_id`` / ``sender_ids``. ``parent_id`` is the source memcell
    id (minted by the boundary stage), and the cascade handler reads it
    back so the LanceDB ``episode`` row keeps its back-link to the source.

    The md entry's ``entry_id`` (managed by the chassis writer) is the
    single source of *entry* identity; cascade derives a global episode
    id from ``<owner_id>_<entry_id>`` on the fly.
    """
    ts_iso = (
        to_iso_format(from_timestamp(episode.timestamp))
        if isinstance(episode.timestamp, int)
        else str(episode.timestamp)
    )

    inline: dict[str, object] = {
        "owner_id": episode.owner_id,
        "session_id": episode.session_id,
        "timestamp": ts_iso,
        "parent_type": "memcell",
        "parent_id": episode.parent_id,
    }
    if episode.sender_ids:
        inline["sender_ids"] = list(episode.sender_ids)

    extra = episode.model_dump(
        exclude={
            "owner_id",
            "episode",
            "timestamp",
            "session_id",
            "sender_ids",
            "parent_id",
        }
    )
    subject = extra.pop("subject", None)
    summary = extra.pop("summary", None)

    sections: dict[str, str] = {}
    if subject:
        sections["Subject"] = str(subject)
    if summary:
        sections["Summary"] = str(summary)
    sections["Content"] = episode.episode
    return inline, sections
