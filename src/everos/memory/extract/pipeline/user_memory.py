"""User memory pipeline — per-sender Episode fan-out on pre-cut cells.

Cells / memcell_ids / message_id-mapping / sender lists are produced by
:mod:`everos.memory.extract.boundary` (which also writes the single
``memcell`` sqlite row per cell). This pipeline only handles the
user-perspective output of one cell: Episode md, plus the
``UserPipelineStarted`` and ``EpisodeExtracted`` events that announce it.

Each cell costs one model call (``EpisodeExtractor.aextract``). Every copy
is written with :func:`~everos.memory.episode_summary.placeholder_summary`
as its summary; the ``summarize_episode`` strategy writes the model's preview
afterwards, from the episode text and in its language.

``MemoryService.add`` runs it cell by cell alongside
:class:`AgentMemoryPipeline` (the latter only in ``mode="agent"``) and
publishes each cell's events as soon as the cell is done.
"""

from __future__ import annotations

from datetime import UTC, tzinfo
from typing import TYPE_CHECKING

from everalgo.types import Episode as AlgoEpisode
from everalgo.types import MemCell as AlgoMemCell
from everalgo.user_memory import EpisodeExtractor

from everos.core.observability.tracing import capture_output, memory_span
from everos.core.utils.language import detect_output_language
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.episode_summary import placeholder_summary
from everos.memory.events import EpisodeCopy, EpisodeExtracted, UserPipelineStarted
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.extract.pipeline.base import CellOutcome, PreparedCell
from everos.memory.models import Episode, IngestResult
from everos.memory.ports import EpisodeStore, Event, MemcellStore, MemoryRef, Scope

if TYPE_CHECKING:
    from everalgo.llm import LLMClient


class UserMemoryPipeline:
    """Per-sender Episode extraction, one pre-cut MemCell at a time.

    Args:
        episode_store: Where extracted episodes are written.
        memcell_store: Whether the cell still exists when its episodes are
            written (a session deletion removes it first).
        instructions: Custom instructions for the episode operator.
        episode_llm: LLM for the episode extraction.
        tz: The zone message times are shown to the model in.
    """

    def __init__(
        self,
        episode_store: EpisodeStore,
        memcell_store: MemcellStore,
        instructions: CustomInstructions,
        *,
        episode_llm: LLMClient,
        tz: tzinfo = UTC,
    ) -> None:
        self._ep_ext = EpisodeExtractor(llm=episode_llm)
        self._episode_store = episode_store
        self._memcell_store = memcell_store
        self._instructions = instructions
        self._tz = tz

    async def run_cell(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> CellOutcome:
        """Extract the cell's Episode, write one copy per user sender.

        The returned events start with the cell's ``UserPipelineStarted``, so
        the strategies that read the memcell itself start first, followed by
        one ``EpisodeExtracted`` listing every copy written (none when no
        copy was).
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
            algo_ep = await self._extract_episode(ingested, prepared)
            # The model writes in the conversation's language; what it chose
            # pins every extraction downstream of this episode.
            language = detect_output_language(algo_ep.episode)
            written = await self._write_copies(
                ingested, prepared, algo_ep, senders=user_senders, language=language
            )
            md_paths = [md_path for _, md_path in written]
            if written:
                events.append(
                    EpisodeExtracted(
                        memcell_id=memcell_id,
                        episode_text=algo_ep.episode,
                        episode_timestamp_ms=algo_ep.timestamp,
                        copies=[episode_copy for episode_copy, _ in written],
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

    async def _extract_episode(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> AlgoEpisode:
        """One whole-cell episode, shared by every user sender of the cell.

        ``sender_id=None`` binds the episode to no user; the fan-out to one
        copy per sender happens in markdown only, so a cell costs one model
        call. An unusable reply is tried again by :func:`retry_llm_output`.
        """
        extractor = self._ep_ext
        custom_instructions = self._instructions.load("episode_extract")

        async def attempt() -> AlgoEpisode:
            return await extractor.aextract(
                prepared.cell,
                sender_id=None,
                custom_instructions=custom_instructions,
                tz=self._tz,
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

    async def _write_copies(
        self,
        ingested: IngestResult,
        prepared: PreparedCell,
        algo_ep: AlgoEpisode,
        *,
        senders: list[str],
        language: str | None,
    ) -> list[tuple[EpisodeCopy, str]]:
        """Write one copy of the episode per sender; each written one, with its path.

        Every copy carries the placeholder summary. A copy whose write was
        skipped (the memcell was deleted meanwhile) is left out.
        """
        episode = algo_ep.model_copy(
            update={"summary": placeholder_summary(algo_ep.episode)}
        )
        written: list[tuple[EpisodeCopy, str]] = []
        for sender_id in senders:
            ep = Episode.from_algo(
                episode,
                owner_id=sender_id,
                session_id=ingested.session_id,
                sender_ids=list(prepared.sender_ids),
                parent_id=prepared.memcell_id,
                language=language,
            )
            ref = await self._persist(ingested, ep, prepared.memcell_id)
            if ref is None:
                continue
            episode_copy = EpisodeCopy(
                owner_id=sender_id,
                entry_id=ref.entry_id,
                owner_name=_sender_label(prepared.cell, sender_id),
            )
            written.append((episode_copy, ref.md_path))
        return written

    async def _persist(
        self, ingested: IngestResult, ep: Episode, memcell_id: str
    ) -> MemoryRef | None:
        """Write one episode under its owner and trace the write.

        ``None`` when the store skipped the write: the memcell was deleted
        (by a session deletion) while the episode was being extracted.
        """

        async def memcell_alive() -> bool:
            return bool(await self._memcell_store.find_by_ids([memcell_id]))

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
                only_if=memcell_alive,
            )
            if ref is None:
                return None
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


def _sender_label(cell: AlgoMemCell, sender_id: str) -> str:
    """How the episode operator labelled the sender's lines.

    Each chat line is rendered under its ``sender_name``, else its
    ``sender_id``; ingestion gives every line of one sender the same name.
    """
    for item in cell.items:
        name = getattr(item, "sender_name", None)
        if getattr(item, "sender_id", None) == sender_id and name and name.strip():
            return name.strip()
    return sender_id
