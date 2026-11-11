"""User memory pipeline — per-sender Episode fan-out, plus the pictures beside it.

Cells / memcell_ids / message_id-mapping / sender lists are produced by
:mod:`everos.memory.extract.boundary` (which also writes the single ``memcell``
sqlite row per cell). This pipeline only handles the user-perspective output of
one cell: the Episode md, one image-memory md per picture the cell carried, the
profile items the pictures established, and the ``EpisodeExtracted`` events
that announce the episodes.

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

Both routes are told the language to write in. It is resolved before the call
from the session's human messages
(:mod:`everos.memory.extract.conversation_language`), never left to the model,
which answered English picture cells in Chinese; the same language then pins
the episode, its event and so every extraction downstream. Only a session
without a letter anywhere leaves the model the choice, and the language its
episode was written in is pinned instead, as before.

A picture is evidence about its sender as well as a memory of its own, so the
picture route also merges the reply's profile items into ``user.md`` before the
episode is published -- into the poster's ``user.md`` only (D17): a picture one
sender posted says nothing about the other senders of the cell. That runs
here rather than in the ``extract_user_profile`` strategy because a
conversation carried by photos would otherwise never move the portrait, and
it shares that strategy's partition lock so the two writers of ``user.md``
cannot lose each other's items. The strategy still runs
afterwards and reads the merged profile, so picture items and text items end up
in one document with one merge rule between them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from everalgo.types import Episode as AlgoEpisode
from everalgo.types import MemCell as AlgoMemCell
from everalgo.user_memory import EpisodeExtractor, OutputLanguage

from everos.component.images import ImageLimits
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import capture_output, memory_span
from everos.core.utils.language import detect_output_language
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.conversation_language import resolve_cell_language
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.extract.pipeline import image_memory as image_extract
from everos.memory.extract.pipeline import profile_delta
from everos.memory.extract.pipeline.base import CellOutcome, PreparedCell
from everos.memory.extract.pipeline.pictures import (
    Picture,
    memcell_alive,
    ops_by_poster,
    pictures_of,
)
from everos.memory.models import Episode, ImageMemory, IngestResult
from everos.memory.ports import (
    EpisodeStore,
    Event,
    ImageMemoryStore,
    MemcellStore,
    MemoryRef,
    ProfileStore,
    Scope,
    SessionKey,
)
from everos.memory.ports.images import ImageStore

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

logger = get_logger(__name__)

_Extraction = tuple[
    AlgoEpisode,
    dict[str, tuple[str, str]],
    list[profile_delta.ProfileItemOp],
]
"""What one cell's extraction answers: the episode, each picture's
``(description, ocr)`` by label, and the profile operations the pictures
produced (empty on the text route). One tuple because the three are readings of
the same call and splitting them would let them drift apart."""


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
        profile_store: Where the profile items the pictures established are
            merged. ``None`` keeps the picture route's profile half off.
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
        profile_store: ProfileStore | None = None,
    ) -> None:
        self._ep_ext = EpisodeExtractor(llm=llm_client)
        self._episode_store = episode_store
        self._memcell_store = memcell_store
        self._instructions = instructions
        self._llm = llm_client
        self._images = image_store
        self._image_memory_store = image_memory_store
        self._image_limits = image_limits
        self._profile_store = profile_store

    async def run_cell(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> CellOutcome:
        """Extract the cell's Episode (and pictures), write one copy per sender.

        The returned events are one ``EpisodeExtracted`` per written episode.
        """
        cell, memcell_id = prepared.cell, prepared.memcell_id
        events: list[Event] = []
        md_paths: list[str] = []
        user_senders = _unique_user_senders(cell)
        if user_senders:
            # Named to the model, never left to it: the language is the
            # conversation's, resolved from its human messages.
            named = await resolve_cell_language(
                cell,
                memcell_id=memcell_id,
                session=SessionKey(
                    ingested.app_id, ingested.project_id, ingested.session_id
                ),
                memcells=self._memcell_store,
            )
            pictures = await self._pictures_of(cell)
            algo_ep, described, profile_ops = await self._extract(
                ingested, prepared, pictures, named
            )
            # That language pins every extraction downstream of this episode.
            # Only a conversation without a letter in it left the choice to
            # the model; then what it chose is read off the episode.
            language = (
                str(named)
                if named is not None
                else detect_output_language(algo_ep.episode)
            )
            by_poster = ops_by_poster(profile_ops, pictures)
            for sender_id in user_senders:
                md_paths += await self._persist_images(
                    ingested,
                    prepared,
                    pictures=pictures,
                    described=described,
                    owner_id=sender_id,
                    session_id=ingested.session_id,
                )
                # D17: what a picture says about a person is about its poster;
                # another sender of the cell gets none of it.
                await self._persist_profile(
                    ingested,
                    prepared,
                    ops=by_poster.get(sender_id, []),
                    pictures=[p for p in pictures if p.poster == sender_id],
                    owner_id=sender_id,
                    language=language,
                )
                ep = Episode.from_algo(
                    algo_ep,
                    owner_id=sender_id,
                    session_id=ingested.session_id,
                    sender_ids=list(prepared.sender_ids),
                    parent_id=memcell_id,
                    language=language,
                    image_ids=[picture.image_id for picture in pictures],
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
        pictures: Sequence[Picture],
        language: OutputLanguage | None,
    ) -> _Extraction:
        """One whole-cell episode (and picture descriptions, and profile items).

        ``sender_id=None`` binds the episode to no user; the fan-out to one copy
        per sender happens in markdown only, so a cell costs one call. An
        unusable reply is tried again by :func:`retry_llm_output`. ``language``
        is named to the model on either route; ``None`` leaves it the choice.
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
                "output_language": None if language is None else str(language),
            },
        ) as extract_span:
            if pictures:
                algo_ep, described, ops = await self._extract_with_pictures(
                    prepared, pictures, custom_instructions, language
                )
            else:
                algo_ep, described, ops = (
                    *await self._extract_text(prepared, custom_instructions, language),
                    [],
                )
            capture_output(extract_span, algo_ep.episode)
        return algo_ep, described, ops

    async def _extract_text(
        self,
        prepared: PreparedCell,
        custom_instructions: str | None,
        language: OutputLanguage | None,
    ) -> tuple[AlgoEpisode, dict[str, tuple[str, str]]]:
        """The text-only route: everalgo's extractor, told the language."""

        async def attempt() -> AlgoEpisode:
            return await self._ep_ext.aextract(
                prepared.cell,
                sender_id=None,
                custom_instructions=custom_instructions,
                output_language=language,
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
        pictures: Sequence[Picture],
        custom_instructions: str | None,
        language: OutputLanguage | None,
    ) -> _Extraction:
        """The picture route: one call for the episode, the descriptions, the profile.

        The reply's profile operations name a picture by its per-cell label
        (``img-1``); :func:`profile_delta.retarget` turns those into content
        hashes before anything is written, because the label means nothing
        outside this cell and a later forget knows only the hash.
        """
        labels = [(picture.label, picture.data, picture.mime) for picture in pictures]
        by_label = {picture.label: picture.image_id for picture in pictures}

        async def attempt() -> _Extraction:
            reply = await image_extract.extract(
                self._llm,
                prepared.cell,
                pictures=labels,
                custom_instructions=custom_instructions,
                output_language=language,
            )
            return (
                image_extract.to_episode(reply, prepared.cell, labels=by_label),
                {fact.ref: (fact.description, fact.ocr) for fact in reply.images},
                profile_delta.retarget(reply.profile, by_label),
            )

        return await retry_llm_output(
            attempt, operator="image_memory", memcell_id=prepared.memcell_id
        )

    async def _persist_profile(
        self,
        ingested: IngestResult,
        prepared: PreparedCell,
        *,
        ops: Sequence[profile_delta.ProfileItemOp],
        pictures: Sequence[Picture],
        owner_id: str,
        language: str | None,
    ) -> int:
        """Merge the pictures' profile items into ``owner_id``'s profile.

        Skipped when the route produced none (every text-only cell) and when no
        store was injected. The write is serialised with the
        ``extract_user_profile`` strategy on the owner's partition lock, so the
        strategy that runs just after this cell sees the merged items instead
        of overwriting them.
        """
        if self._profile_store is None or not ops:
            return 0
        return await profile_delta.write_picture_profile(
            store=self._profile_store,
            scope=Scope(
                app_id=ingested.app_id,
                project_id=ingested.project_id,
                owner_id=owner_id,
                owner_type="user",
            ),
            ops=ops,
            observed_at=prepared.cell.timestamp,
            image_ids=[picture.image_id for picture in pictures],
            language=language,
        )

    async def _pictures_of(self, cell: AlgoMemCell) -> list[Picture]:
        """The cell's pictures, in conversation order, read and labelled.

        Labels (``img-1``, ``img-2``) are assigned per cell: they are local
        names a reply can refer to, deliberately distinct from the content hash
        that identifies the file. A reply that names ``img-1`` therefore cannot
        be confused with one from another cell.
        """
        return await pictures_of(cell, images=self._images, limits=self._image_limits)

    async def _persist_images(
        self,
        ingested: IngestResult,
        prepared: PreparedCell,
        *,
        pictures: Sequence[Picture],
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
            only_if=memcell_alive(self._memcell_store, prepared.memcell_id),
        )
        return [ref.md_path for ref in refs]

    async def _persist(
        self, ingested: IngestResult, ep: Episode, memcell_id: str
    ) -> MemoryRef | None:
        """Write one episode under its owner and trace the write.

        ``None`` when the store skipped the write: the memcell was deleted
        (by a session deletion) while the episode was being extracted.
        """

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
                only_if=memcell_alive(self._memcell_store, memcell_id),
            )
            if ref is None:
                return None
            # Written .md path, memory-root-relative (only when
            # capture_content is on) — never the host absolute path.
            capture_output(persist_span, ref.md_path)
        return ref


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
