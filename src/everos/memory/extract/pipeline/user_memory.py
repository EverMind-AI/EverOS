"""User memory pipeline — per-sender Episode fan-out on pre-cut cells.

Cells / memcell_ids / message_id-mapping / sender lists are produced by
:mod:`everos.memory.extract.boundary` (which also writes the single
``memcell`` sqlite row per cell). This pipeline only handles the
user-perspective output of one cell: Episode md, plus the
``UserPipelineStarted`` and ``EpisodeExtracted`` events that announce it.

``MemoryService.add`` runs it cell by cell alongside
:class:`AgentMemoryPipeline` (the latter only in ``mode="agent"``) and
publishes each cell's events as soon as the cell is done.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everalgo.types import Episode as AlgoEpisode
from everalgo.types import MemCell as AlgoMemCell
from everalgo.user_memory import EpisodeExtractor

from everos.core.observability.tracing import capture_output, memory_span
from everos.core.utils.language import detect_output_language
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.events import EpisodeExtracted, UserPipelineStarted
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.extract.pipeline.base import CellOutcome, PreparedCell
from everos.memory.models import Episode, IngestResult
from everos.memory.ports import EpisodeStore, Event, MemoryRef, Scope

if TYPE_CHECKING:
    from everalgo.llm import LLMClient


class UserMemoryPipeline:
    """Per-sender Episode extraction, one pre-cut MemCell at a time.

    Args:
        episode_store: Where extracted episodes are written.
        instructions: Custom instructions for the episode operator.
        llm_client: LLM for extraction.
    """

    def __init__(
        self,
        episode_store: EpisodeStore,
        instructions: CustomInstructions,
        llm_client: LLMClient,
    ) -> None:
        self._ep_ext = EpisodeExtractor(llm=llm_client)
        self._episode_store = episode_store
        self._instructions = instructions

    async def run_cell(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> CellOutcome:
        """Extract the cell's Episode, write one copy per user sender.

        The returned events start with the cell's ``UserPipelineStarted`` so
        the strategies that consume the MemCell directly (atomic_fact /
        foresight / clustering) are published first, followed by one
        ``EpisodeExtracted`` per written episode.
        """
        cell, memcell_id = prepared.cell, prepared.memcell_id
        events: list[Event] = [
            UserPipelineStarted(
                memcell_id=memcell_id,
                session_id=ingested.session_id,
                app_id=ingested.app_id,
                project_id=ingested.project_id,
                memcell=cell,
            )
        ]
        md_paths: list[str] = []
        user_senders = _unique_user_senders(cell)
        if user_senders:
            algo_ep = await self._extract(ingested, prepared)
            # The model writes in the conversation's language; what it chose
            # pins every extraction downstream of this episode.
            language = detect_output_language(algo_ep.episode)
            for sender_id in user_senders:
                ep = Episode.from_algo(
                    algo_ep,
                    owner_id=sender_id,
                    session_id=ingested.session_id,
                    sender_ids=list(prepared.sender_ids),
                    parent_id=memcell_id,
                    language=language,
                )
                ref = await self._persist(ingested, ep)
                md_paths.append(ref.md_path)
                events.append(
                    EpisodeExtracted(
                        memcell_id=memcell_id,
                        episode_entry_id=ref.entry_id,
                        episode_text=ep.episode,
                        episode_timestamp_ms=ep.timestamp,
                        owner_id=ep.owner_id,
                        session_id=ingested.session_id,
                        app_id=ingested.app_id,
                        project_id=ingested.project_id,
                        source="pipeline",
                        language=language,
                    )
                )
        return CellOutcome(
            message_count=len(cell.items),
            events=events,
            md_paths=md_paths,
        )

    async def _extract(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> AlgoEpisode:
        """One whole-cell episode, shared by every user sender of the cell.

        ``sender_id=None`` binds the episode to no user; the fan-out to one
        copy per sender happens in markdown only, so a cell costs one call.
        An unusable reply is tried again by :func:`retry_llm_output`.
        """
        extractor = self._ep_ext
        custom_instructions = self._instructions.load("episode_extract")

        async def attempt() -> AlgoEpisode:
            return await extractor.aextract(
                prepared.cell,
                sender_id=None,
                custom_instructions=custom_instructions,
            )

        with memory_span(
            "everos.extract",
            observation_type="generation",
            session_id=ingested.session_id,
            metadata={
                "app_id": ingested.app_id,
                "project_id": ingested.project_id,
                "memcell_id": prepared.memcell_id,
            },
        ) as extract_span:
            # The LLM client records token usage onto this span.
            algo_ep = await retry_llm_output(
                attempt, operator="episode", memcell_id=prepared.memcell_id
            )
            capture_output(extract_span, algo_ep.episode)
        return algo_ep

    async def _persist(self, ingested: IngestResult, ep: Episode) -> MemoryRef:
        """Write one episode under its owner and trace the write."""
        with memory_span(
            "everos.persist.markdown",
            observation_type="span",
            session_id=ingested.session_id,
            metadata={
                "owner_id": ep.owner_id,
                "app_id": ingested.app_id,
                "project_id": ingested.project_id,
            },
        ) as persist_span:
            ref = await self._episode_store.append(
                Scope(
                    app_id=ingested.app_id,
                    project_id=ingested.project_id,
                    owner_id=ep.owner_id,
                    owner_type="user",
                ),
                ep,
            )
            # Written .md path, memory-root-relative (only when
            # capture_content is on) — never the host absolute path.
            capture_output(persist_span, ref.md_path)
        return ref


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
