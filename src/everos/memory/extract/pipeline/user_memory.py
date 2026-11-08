"""User memory pipeline — per-sender Episode fan-out, plus the pictures beside it.

Cells / memcell_ids / message_id-mapping / sender lists are produced by
:mod:`everos.memory.extract.boundary` (which also writes the single ``memcell``
sqlite row per cell). This pipeline only handles the user-perspective output of
one cell: the Episode md, one image-memory md per picture the cell carried, and
the ``UserPipelineStarted`` / ``EpisodeExtracted`` events that announce them.

Two extraction routes, chosen per cell by whether it carries a picture:

- **Text only** — everalgo's :class:`EpisodeExtractor`, unchanged. This is the
  overwhelming majority of cells and nothing about it moves.
- **With pictures** — one call of our own (:mod:`.image_memory`) that renders
  the cell the same way, hands the model the pictures, and asks for the episode
  *and* one record per picture in a single reply. A picture cannot travel
  through ``EpisodeExtractor`` because its message type is text by contract.

Both routes produce an algo ``Episode``, so everything downstream (the md
writer, the events, the strategies keyed on ``EpisodeExtracted``) is unaware of
which one ran. The picture memories are written before the episode so that an
episode body's ``img-N`` references name records that exist.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from everalgo.types import Episode as AlgoEpisode
from everalgo.types import MemCell as AlgoMemCell
from everalgo.user_memory import EpisodeExtractor

from everos.component.images import ImageLimits
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import capture_output, memory_span
from everos.core.utils.language import detect_output_language
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.events import EpisodeExtracted, UserPipelineStarted
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.extract.pipeline import image_memory as image_extract
from everos.memory.extract.pipeline.base import CellOutcome, PreparedCell
from everos.memory.models import Episode, ImageMemory, IngestResult
from everos.memory.ports import (
    EpisodeStore,
    Event,
    ImageMemoryStore,
    MemcellStore,
    MemoryRef,
    Scope,
)
from everos.memory.ports.images import ImageStore

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

logger = get_logger(__name__)

_IMAGE_REF_PREFIX = "[IMAGE: "


class UserMemoryPipeline:
    """Per-sender Episode extraction, one pre-cut MemCell at a time.

    Args:
        episode_store: Where extracted episodes are written.
        memcell_store: Whether the cell still exists when its episodes are
            written (a session deletion removes it first).
        instructions: Custom instructions for the episode operator.
        llm_client: LLM for extraction.
        image_store: Where pictures are read from; ``None`` disables the
            picture route (a cell's pictures then contribute nothing).
        image_memory_store: Where a picture's memory is written; required with
            ``image_store``.
        image_limits: ``[images]`` — the size a picture is scaled to before the
            model sees it.
    """

    def __init__(
        self,
        episode_store: EpisodeStore,
        memcell_store: MemcellStore,
        instructions: CustomInstructions,
        llm_client: LLMClient,
        *,
        image_store: ImageStore | None = None,
        image_memory_store: ImageMemoryStore | None = None,
        image_limits: ImageLimits | None = None,
    ) -> None:
        self._ep_ext = EpisodeExtractor(llm=llm_client)
        self._episode_store = episode_store
        self._memcell_store = memcell_store
        self._instructions = instructions
        self._llm = llm_client
        self._images = image_store
        self._image_memory_store = image_memory_store
        self._image_limits = image_limits

    async def run_cell(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> CellOutcome:
        """Extract the cell's Episode (and pictures), write one copy per sender.

        The returned events start with the cell's ``UserPipelineStarted`` so
        the strategies that consume the MemCell directly (foresight / clustering)
        are published first, followed by one ``EpisodeExtracted`` per written
        episode.
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
            pictures = await self._pictures_of(cell)
            algo_ep, described = await self._extract(ingested, prepared, pictures)
            # The model writes in the conversation's language; what it chose
            # pins every extraction downstream of this episode.
            language = detect_output_language(algo_ep.episode)
            for sender_id in user_senders:
                md_paths += await self._persist_images(
                    ingested,
                    prepared,
                    pictures=pictures,
                    described=described,
                    owner_id=sender_id,
                    session_id=ingested.session_id,
                )
                ep = Episode.from_algo(
                    algo_ep,
                    owner_id=sender_id,
                    session_id=ingested.session_id,
                    sender_ids=list(prepared.sender_ids),
                    parent_id=memcell_id,
                    language=language,
                )
                ref = await self._persist(ingested, ep, memcell_id)
                if ref is None:
                    continue
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
        self,
        ingested: IngestResult,
        prepared: PreparedCell,
        pictures: Sequence[_Picture],
    ) -> tuple[AlgoEpisode, dict[str, tuple[str, str]]]:
        """One whole-cell episode (and picture descriptions) for the cell.

        ``sender_id=None`` binds the episode to no user; the fan-out to one copy
        per sender happens in markdown only, so a cell costs one call. An
        unusable reply is tried again by :func:`retry_llm_output`.
        """
        custom_instructions = self._instructions.load("episode_extract")

        with memory_span(
            "everos.extract",
            observation_type="generation",
            session_id=ingested.session_id,
            metadata={
                "app_id": ingested.app_id,
                "project_id": ingested.project_id,
                "memcell_id": prepared.memcell_id,
                "pictures": len(pictures),
            },
        ) as extract_span:
            if pictures:
                algo_ep, described = await self._extract_with_pictures(
                    prepared, pictures, custom_instructions
                )
            else:
                algo_ep, described = await self._extract_text(
                    prepared, custom_instructions
                )
            capture_output(extract_span, algo_ep.episode)
        return algo_ep, described

    async def _extract_text(
        self, prepared: PreparedCell, custom_instructions: str | None
    ) -> tuple[AlgoEpisode, dict[str, tuple[str, str]]]:
        """The text-only route: everalgo's extractor, exactly as before."""

        async def attempt() -> AlgoEpisode:
            return await self._ep_ext.aextract(
                prepared.cell,
                sender_id=None,
                custom_instructions=custom_instructions,
            )

        return (
            await retry_llm_output(
                attempt, operator="episode", memcell_id=prepared.memcell_id
            ),
            {},
        )

    async def _extract_with_pictures(
        self,
        prepared: PreparedCell,
        pictures: Sequence[_Picture],
        custom_instructions: str | None,
    ) -> tuple[AlgoEpisode, dict[str, tuple[str, str]]]:
        """The picture route: one call for the episode and every description."""
        labels = [(picture.label, picture.data, picture.mime) for picture in pictures]

        async def attempt() -> tuple[AlgoEpisode, dict[str, tuple[str, str]]]:
            reply = await image_extract.extract(
                self._llm,
                prepared.cell,
                pictures=labels,
                custom_instructions=custom_instructions,
                output_language=None,
            )
            return (
                image_extract.to_episode(
                    reply,
                    prepared.cell,
                    labels={p.label: p.image_id for p in pictures},
                ),
                {fact.ref: (fact.description, fact.ocr) for fact in reply.images},
            )

        return await retry_llm_output(
            attempt, operator="image_memory", memcell_id=prepared.memcell_id
        )

    async def _pictures_of(self, cell: AlgoMemCell) -> list[_Picture]:
        """The cell's pictures, in conversation order, read and labelled.

        Labels (``img-1``, ``img-2``) are assigned per cell: they are local
        names a reply can refer to, deliberately distinct from the content hash
        that identifies the file. A reply that names ``img-1`` therefore cannot
        be confused with one from another cell.
        """
        if self._images is None or self._image_limits is None:
            return []
        pictures: list[_Picture] = []
        seen: dict[str, str] = {}
        for item in cell.items:
            text = str(getattr(item, "content", "") or "")
            for image_id in _image_ids_in(text):
                if image_id in seen:
                    continue
                label = f"img-{len(seen) + 1}"
                seen[image_id] = label
                try:
                    data = await self._images.read_derived(
                        image_id,
                        edge=self._image_limits.model_edge,
                        quality=self._image_limits.cache_quality,
                    )
                except (FileNotFoundError, OSError, ValueError) as err:
                    # The picture is gone (forgotten, or a store rebuilt without
                    # its cache). The conversation is still worth remembering:
                    # keep the reference in the text and extract without it.
                    logger.warning(
                        "memory.images.unreadable",
                        image_id=image_id,
                        error=f"{type(err).__name__}: {err}"[:160],
                    )
                    continue
                pictures.append(
                    _Picture(
                        label=label,
                        image_id=image_id,
                        data=data,
                        mime="image/jpeg",
                    )
                )
        return pictures

    async def _persist_images(
        self,
        ingested: IngestResult,
        prepared: PreparedCell,
        *,
        pictures: Sequence[_Picture],
        described: dict[str, tuple[str, str]],
        owner_id: str,
        session_id: str | None,
    ) -> list[str]:
        """Write one image memory per picture, under ``owner_id``.

        Called once per sender, like the episode: the same picture gives each
        participant of the cell their own copy of the memory, because "what did
        this user send me" and "what did I send this user" are different
        questions with the same picture as the answer.
        """
        if not described or self._image_memory_store is None:
            return []
        memories: list[ImageMemory] = []
        for picture in pictures:
            described_pair = described.get(picture.label)
            if not described_pair:
                continue
            description, ocr = described_pair
            original = await self._images.relpath_of(picture.image_id)
            info = await self._images.metadata(picture.image_id)
            memories.append(
                ImageMemory(
                    owner_id=owner_id,
                    description=description,
                    # ``MemCell.timestamp`` is already epoch milliseconds.
                    timestamp=prepared.cell.timestamp,
                    ocr=ocr,
                    image_id=picture.image_id,
                    name=prepared.picture_names.get(picture.image_id, ""),
                    original_relpath=original,
                    media_type=info.media_type if info else "jpeg",
                    width=info.width if info else 0,
                    height=info.height if info else 0,
                    session_id=session_id,
                    parent_id=prepared.memcell_id,
                )
            )
        if not memories:
            return []
        refs = await self._image_memory_store.append_many(
            Scope(
                app_id=ingested.app_id,
                project_id=ingested.project_id,
                owner_id=owner_id,
                owner_type="user",
            ),
            memories,
            only_if=_memcell_alive(self._memcell_store, prepared.memcell_id),
        )
        return [ref.md_path for ref in refs]

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


class _Picture:
    """One picture of a cell, ready for the model.

    Args:
        label: Local name the reply refers to it by (``img-1``).
        image_id: Content hash — what the memory records and a forget takes.
        data: Bytes of the model-scaled copy.
        mime: MIME type of those bytes.
    """

    __slots__ = ("data", "image_id", "label", "mime")

    def __init__(self, *, label: str, image_id: str, data: bytes, mime: str) -> None:
        self.label = label
        self.image_id = image_id
        self.data = data
        self.mime = mime


def _image_ids_in(text: str) -> list[str]:
    """Every ``[IMAGE: <id>]`` reference in ``text``, in order."""
    found: list[str] = []
    start = 0
    while True:
        index = text.find(_IMAGE_REF_PREFIX, start)
        if index < 0:
            return found
        end = text.find("]", index)
        if end < 0:
            return found
        found.append(text[index + len(_IMAGE_REF_PREFIX) : end])
        start = end + 1


def _memcell_alive(memcell_store: MemcellStore, memcell_id: str):
    """The ``only_if`` shared by every write derived from this cell."""

    async def check() -> bool:
        return bool(await memcell_store.find_by_ids([memcell_id]))

    return check


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
