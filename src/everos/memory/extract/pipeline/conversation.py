"""The conversation pipeline -- one reading per segment, in the segment's space.

Cells / memcell_ids / message_id-mapping / sender lists are produced by
:mod:`everos.memory.extract.boundary` (which also writes the single ``memcell``
sqlite row per cell). This pipeline writes what one cell is remembered as:
**one** extraction call, its episode, one picture memory per picture the cell
carried (written first, so an episode body's ``img-N`` references name records
that exist), and the :class:`EpisodeExtracted` event that announces the
episode. A segment is remembered in exactly one kind of space
(docs/group-memory.md, D3):

* **A group's** -- the session is bound to a group (``ingested.group_id``): a
  declared group, or the implicit group of a session's people (three or more
  people talking without a ``group_id``). The episode goes under
  ``groups/<group_id>/``. The model is shown the segment with its speakers
  labelled by their roster names and the roster block after the operator's
  custom instructions (:mod:`.group_render`, section 7.2); after the call,
  code sets the attribution (``sender_ids`` / ``subject_ids`` / ``origin``)
  and the segment's span, which every read compares with the asker's
  membership windows. A picture memory is dated by the message that carried
  it and attributed to its poster (section 7.3).
* **A user's own** -- one person's conversation with the assistant, or two
  people's without a ``group_id`` (D14, amended 2026-11-13): the prompt is
  everalgo's episode prompt byte for byte and the conversation alone, and
  each human sender keeps a copy -- the episode, its picture memories (dated
  by the segment, section 10) and its event -- under ``users/<sender>/``.

In either space a message forgotten while it waited in the buffer is not read
(section 9.4), and a forget of a segment already extracted re-runs the same
call through :meth:`ConversationPipeline.reextract`, which writes nothing.

Every cell is extracted by one call (:mod:`.episode`) and one prompt: the
episode and, when the cell carries pictures, one record per picture, in a
single reply. The call is told the language to write in, resolved before the
call from the session's human messages
(:mod:`everos.memory.extract.conversation_language`), never left to the model;
the same language then pins the episode, its event and so every extraction
downstream. Only a session without a letter anywhere leaves the model the
choice, and the language its episode was written in is pinned instead.

A picture is evidence about its poster as well as a memory of its own; what it
says about its poster reaches that person's profile through the profile
update, which reads the poster's pictures itself (D17), so this pipeline
writes no profile item. The event's owner type routes it: a user's episode to
the personal strategies, a group's -- with its human speakers -- to the group
strategies (member-in-group profiles, the group profile, scenes). Group
content never feeds personal memory.
"""

from __future__ import annotations

__all__ = ["ConversationPipeline"]

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from everalgo.types import Episode as AlgoEpisode
from everalgo.user_memory import OutputLanguage

from everos.component.images import ImageLimits
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import capture_output, memory_span
from everos.core.utils.language import detect_output_language
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.events import EpisodeExtracted
from everos.memory.extract.conversation_language import resolve_cell_language
from everos.memory.extract.pipeline.base import CellOutcome, PreparedCell
from everos.memory.extract.pipeline.episode import extract_cell
from everos.memory.extract.pipeline.group_render import (
    attribution,
    label_cell,
    roster_block,
    with_roster_block,
)
from everos.memory.extract.pipeline.pictures import (
    Picture,
    memcell_alive,
    pictures_of,
)
from everos.memory.groups import named_members, roster_names, without_forgotten
from everos.memory.models import (
    Episode,
    Group,
    ImageMemory,
    IngestResult,
    MemCell,
    Roster,
)
from everos.memory.ports import (
    EpisodeStore,
    Event,
    GroupStore,
    ImageMemoryStore,
    MemcellStore,
    MemoryRef,
    MessageForgetStore,
    Scope,
    SessionKey,
)
from everos.memory.ports.images import ImageStore

if TYPE_CHECKING:
    from everalgo.llm import LLMClient

logger = get_logger(__name__)

_Extraction = tuple[AlgoEpisode, dict[str, tuple[str, str]]]
"""What one cell's extraction answers: the episode, and each picture's
``(description, ocr)`` by label (empty for a cell without pictures). One tuple
because the two are readings of the same call and splitting them would let
them drift apart."""


@dataclass(frozen=True)
class _Space:
    """Where a segment is remembered: a group's space, or one user's own.

    Attributes:
        scope: The owner the records are written under.
        group: The group's meta; ``None`` in a user's own space.
        roster: The group's roster (labels, roster block, attribution);
            empty in a user's own space.
    """

    scope: Scope
    group: Group | None = None
    roster: Roster = field(default_factory=Roster)


@dataclass(frozen=True)
class _Segment:
    """One segment's extraction: the episode and what the call read.

    Attributes:
        episode: The episode (in a group, attribution and span set), not
            written.
        pictures: The pictures the call was shown.
        described: Each picture's ``(description, ocr)`` by label.
        language: The episode's language (the conversation's).
    """

    episode: Episode
    pictures: list[Picture]
    described: dict[str, tuple[str, str]]
    language: str | None


class ConversationPipeline:
    """One episode per pre-cut memcell, in its session's space.

    Args:
        episode_store: Where episodes are written.
        memcell_store: Whether the cell still exists when its records are
            written (a session deletion removes it first).
        groups: A group's meta and roster (labels, roster block, attribution).
        instructions: Custom instructions for the episode operator.
        llm_client: LLM for extraction.
        image_store: Where pictures are read from; ``None`` attaches no
            picture to the call (a cell's pictures then contribute nothing).
        image_memory_store: Where a picture's memory is written; required with
            ``image_store``.
        image_limits: ``[images]`` — the size a picture is scaled to before the
            model sees it.
        message_forget: Which messages of a session are forgotten; a cell is
            extracted without them. ``None`` reads every message.
    """

    def __init__(
        self,
        episode_store: EpisodeStore,
        memcell_store: MemcellStore,
        groups: GroupStore,
        instructions: CustomInstructions,
        llm_client: LLMClient,
        *,
        image_store: ImageStore | None = None,
        image_memory_store: ImageMemoryStore | None = None,
        image_limits: ImageLimits | None = None,
        message_forget: MessageForgetStore | None = None,
    ) -> None:
        self._episode_store = episode_store
        self._memcell_store = memcell_store
        self._groups = groups
        self._instructions = instructions
        self._llm = llm_client
        self._images = image_store
        self._image_memory_store = image_memory_store
        self._image_limits = image_limits
        self._message_forget = message_forget

    async def run_cell(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> CellOutcome:
        """Extract the cell's episode (and pictures) and write it in its space.

        A group's cell is written once, under the group. A cell outside any
        group is written once per human sender, each a copy in that person's
        own memory: one person, or two people talking without a ``group_id``
        (D14, amended 2026-11-13) -- one call either way. Skipped, with
        nothing written: a user's cell without a human sender, a cell left
        without a chat message once its forgotten messages are taken out.
        """
        message_count = len(prepared.cell.items)
        session = SessionKey(ingested.app_id, ingested.project_id, ingested.session_id)
        if ingested.group_id is not None:
            spaces = [
                await self._group_space(
                    Scope(
                        app_id=ingested.app_id,
                        project_id=ingested.project_id,
                        owner_id=ingested.group_id,
                        owner_type="group",
                    ),
                    prepared.cell,
                )
            ]
            cell = await self._readable(prepared.memcell_id, prepared.cell)
            if not _has_text(cell):
                return CellOutcome(message_count=message_count)
        else:
            humans = _human_senders(prepared.cell)
            if not humans:
                return CellOutcome(message_count=message_count)
            spaces = [
                _Space(
                    scope=Scope(
                        app_id=ingested.app_id,
                        project_id=ingested.project_id,
                        owner_id=human,
                        owner_type="user",
                    )
                )
                for human in humans
            ]
            # A message its owner forgot while it waited in the buffer is not
            # read either; a cell that lost every text message to it is left.
            cell = await self._readable(prepared.memcell_id, prepared.cell)
            if cell is not prepared.cell and not _has_text(cell):
                return CellOutcome(message_count=message_count)
        if prepared.retry:
            spaces = await self._unwritten(spaces, prepared.memcell_id)
            if not spaces:
                return CellOutcome(message_count=message_count)
        found = await self._extract_segment(
            spaces[0], session, prepared.memcell_id, cell, senders=prepared.sender_ids
        )
        md_paths: list[str] = []
        events: list[Event] = []
        for space in spaces:
            # Each person's copy: the same reading of the segment, filed under
            # its own owner, pictures first, as for one person.
            mine = _owned_by(found, space.scope.owner_id)
            if not (prepared.retry and await self._pictures_written(space, prepared)):
                md_paths += await self._persist_pictures(
                    space, ingested, prepared, mine
                )
            ref = await self._persist(
                space, ingested, mine.episode, prepared.memcell_id
            )
            if ref is None:
                continue
            md_paths.append(ref.md_path)
            events.append(_announce(space, ingested, prepared, cell, mine, ref))
        return CellOutcome(
            message_count=message_count, events=events, md_paths=md_paths
        )

    async def reextract(
        self,
        *,
        scope: Scope,
        session_id: str,
        memcell_id: str,
        cell: MemCell,
        senders: Sequence[str] = (),
    ) -> Episode | None:
        """A segment's episode, extracted again from ``cell``; not written.

        The call :meth:`run_cell` makes in the segment's space -- a group's
        (speaker labels, the roster block) or a user's own (the one-to-one
        call) -- in the conversation's language, with the pictures ``cell``
        still carries, over a segment a forget changed: ``cell`` is the
        stored segment without what was forgotten
        (:func:`~everos.memory.groups.without_forgotten`). ``senders`` are a
        user's episode's senders, which it keeps. The pictures' descriptions
        the call returns again are dropped: their memories were never made
        of the forgotten message.

        Returns:
            The new episode version, or ``None`` when ``cell`` holds no text
            message any more (nothing is left to remember).
        """
        if not _has_text(cell):
            return None
        space = (
            await self._group_space(scope, cell)
            if scope.owner_type == "group"
            else _Space(scope=scope)
        )
        session = SessionKey(scope.app_id, scope.project_id, session_id)
        found = await self._extract_segment(
            space, session, memcell_id, cell, senders=senders
        )
        return found.episode

    async def _unwritten(
        self, spaces: Sequence[_Space], memcell_id: str
    ) -> list[_Space]:
        """The spaces that hold no episode of the segment yet.

        A retried segment's earlier extraction may have failed after writing
        some owners' copies (and their pictures); those are not written again.
        """
        return [
            space
            for space in spaces
            if not await self._episode_store.read_for_segments(
                space.scope, [memcell_id]
            )
        ]

    async def _pictures_written(self, space: _Space, prepared: PreparedCell) -> bool:
        """Whether the space holds a picture memory of the segment already."""
        if self._image_memory_store is None:
            return False
        return bool(
            await self._image_memory_store.read_for_segments(
                space.scope, [prepared.memcell_id]
            )
        )

    async def _group_space(self, scope: Scope, cell: MemCell) -> _Space:
        """The group's space: its meta and roster as stored now."""
        record = await self._groups.get(scope)
        if record is None:
            return _Space(
                scope=scope,
                group=Group(group_id=scope.owner_id, created_at=cell.timestamp),
                roster=Roster(),
            )
        return _Space(scope=scope, group=record.group, roster=record.roster)

    async def _readable(self, memcell_id: str, cell: MemCell) -> MemCell:
        """A cell without the messages a live forget names (``cell`` itself
        when none is)."""
        if self._message_forget is None:
            return cell
        forgotten = await self._message_forget.forgotten_in_segments([memcell_id])
        return without_forgotten(cell, forgotten.get(memcell_id, ()))

    async def _extract_segment(
        self,
        space: _Space,
        session: SessionKey,
        memcell_id: str,
        cell: MemCell,
        *,
        senders: Sequence[str] = (),
    ) -> _Segment:
        """Run the episode call over ``cell`` and build the space's episode.

        ``senders`` are the boundary's senders of the cell, which a user's
        episode records; a group's records the attribution's.
        """
        # Named to the model, never left to it: the language is the
        # conversation's, resolved from its human messages.
        named = await resolve_cell_language(
            cell,
            memcell_id=memcell_id,
            session=session,
            memcells=self._memcell_store,
        )
        # Labels (``img-1``, ``img-2``) are assigned per cell: local names a
        # reply can refer to, deliberately distinct from the content hash.
        pictures = await pictures_of(
            cell, images=self._images, limits=self._image_limits
        )
        custom = self._instructions.load("episode_extract")
        shown = cell
        if space.group is not None:
            custom = with_roster_block(
                custom, roster_block(space.group, space.roster, cell)
            )
            shown = label_cell(cell, space.roster)
        algo_ep, described = await self._extract(
            space, session, memcell_id, shown, pictures, custom, named
        )
        # That language pins every extraction downstream of this episode.
        # Only a conversation without a letter in it left the choice to the
        # model; then what it chose is read off the episode.
        language = (
            str(named) if named is not None else detect_output_language(algo_ep.episode)
        )
        image_ids = [picture.image_id for picture in pictures]
        if space.group is None:
            episode = Episode.from_algo(
                algo_ep,
                owner_id=space.scope.owner_id,
                session_id=session.session_id,
                sender_ids=list(senders),
                parent_id=memcell_id,
                language=language,
                image_ids=image_ids,
            )
        else:
            found = attribution(
                cell, space.roster, texts=(algo_ep.subject, algo_ep.episode)
            )
            episode = Episode.from_algo(
                algo_ep,
                owner_id=space.scope.owner_id,
                session_id=session.session_id,
                sender_ids=found.sender_ids,
                parent_id=memcell_id,
                language=language,
                image_ids=image_ids,
            ).model_copy(
                update={
                    "subject_ids": found.subject_ids,
                    "origin": found.origin,
                    "source_start": found.source_start,
                }
            )
        return _Segment(
            episode=episode,
            pictures=list(pictures),
            described=described,
            language=language,
        )

    async def _extract(
        self,
        space: _Space,
        session: SessionKey,
        memcell_id: str,
        shown: MemCell,
        pictures: Sequence[Picture],
        custom_instructions: str | None,
        language: OutputLanguage | None,
    ) -> _Extraction:
        """One whole-cell episode (and picture descriptions).

        An unusable reply is tried again (:func:`.episode.extract_cell`).
        ``language`` is named to the model; ``None`` leaves it the choice.
        """
        metadata: dict[str, Any] = {
            "app_id": space.scope.app_id,
            "project_id": space.scope.project_id,
            "memcell_id": memcell_id,
        }
        if space.group is not None:
            metadata["group_id"] = space.scope.owner_id
        metadata["pictures"] = len(pictures)
        metadata["output_language"] = None if language is None else str(language)
        with memory_span(
            "everos.extract",
            observation_type="generation",
            session_id=session.session_id,
            metadata=metadata,
        ) as extract_span:
            extraction = await extract_cell(
                self._llm,
                shown,
                pictures=pictures,
                custom_instructions=custom_instructions,
                output_language=language,
                memcell_id=memcell_id,
            )
            capture_output(extract_span, extraction[0].episode)
        return extraction

    async def _persist_pictures(
        self,
        space: _Space,
        ingested: IngestResult,
        prepared: PreparedCell,
        found: _Segment,
    ) -> list[str]:
        """One picture memory per picture the reply described, in the space.

        A user's is dated by the segment. A group's is dated by the message
        that carried it -- a membership window is checked against when it was
        posted -- and names its poster (``sender_ids``), whom it is about and
        its origin (section 7.3).
        """
        images = self._images
        if not found.described or self._image_memory_store is None or images is None:
            return []
        memories: list[ImageMemory] = []
        for picture in found.pictures:
            pair = found.described.get(picture.label)
            if not pair:
                continue
            description, ocr = pair
            info = await images.metadata(picture.image_id)
            memories.append(
                ImageMemory(
                    owner_id=space.scope.owner_id,
                    description=description,
                    ocr=ocr,
                    image_id=picture.image_id,
                    name=prepared.picture_names.get(picture.image_id, ""),
                    original_relpath=await images.relpath_of(picture.image_id),
                    media_type=info.media_type if info else "jpeg",
                    width=info.width if info else 0,
                    height=info.height if info else 0,
                    session_id=ingested.session_id,
                    parent_id=prepared.memcell_id,
                    **_picture_fields(
                        space, picture, f"{description}\n{ocr}", prepared.cell
                    ),
                )
            )
        if not memories:
            return []
        refs = await self._image_memory_store.append_many(
            space.scope,
            memories,
            only_if=memcell_alive(self._memcell_store, prepared.memcell_id),
        )
        return [ref.md_path for ref in refs]

    async def _persist(
        self, space: _Space, ingested: IngestResult, episode: Episode, memcell_id: str
    ) -> MemoryRef | None:
        """Write the episode and trace the write.

        ``None`` when the store skipped the write: the memcell was deleted
        (by a session deletion) while the episode was being extracted.
        """
        owner = "group_id" if space.group is not None else "owner_id"
        with memory_span(
            "everos.persist.markdown",
            observation_type="span",
            session_id=ingested.session_id,
            metadata={
                owner: space.scope.owner_id,
                "app_id": ingested.app_id,
                "project_id": ingested.project_id,
            },
        ) as persist_span:
            ref = await self._episode_store.append(
                space.scope,
                episode,
                only_if=memcell_alive(self._memcell_store, memcell_id),
            )
            if ref is None:
                return None
            # Written .md path, memory-root-relative (only when
            # capture_content is on) — never the host absolute path.
            capture_output(persist_span, ref.md_path)
        return ref


def _announce(
    space: _Space,
    ingested: IngestResult,
    prepared: PreparedCell,
    cell: MemCell,
    found: _Segment,
    ref: MemoryRef,
) -> EpisodeExtracted:
    """The event of a written episode; a group's names its human speakers."""
    group: dict[str, Any] = {}
    if space.group is not None:
        group = {
            "owner_type": "group",
            "speaker_ids": _human_speakers(cell, space.roster),
        }
    return EpisodeExtracted(
        memcell_id=prepared.memcell_id,
        episode_entry_id=ref.entry_id,
        episode_text=found.episode.episode,
        episode_timestamp_ms=found.episode.timestamp,
        owner_id=space.scope.owner_id,
        session_id=ingested.session_id,
        app_id=ingested.app_id,
        project_id=ingested.project_id,
        language=found.language,
        **group,
    )


def _owned_by(found: _Segment, owner_id: str) -> _Segment:
    """``found`` as ``owner_id``'s copy (``found`` itself when it is theirs)."""
    if found.episode.owner_id == owner_id:
        return found
    return replace(
        found, episode=found.episode.model_copy(update={"owner_id": owner_id})
    )


def _picture_fields(
    space: _Space, picture: Picture, said: str, cell: MemCell
) -> dict[str, Any]:
    """A picture memory's date and, in a group, its attribution.

    In a group: ``sender_ids`` is the poster; ``subject_ids`` the human
    members its description or text names, else its human poster;
    ``origin`` whether an assistant posted it.
    """
    if space.group is None:
        # ``MemCell.timestamp`` is already epoch milliseconds.
        return {"timestamp": cell.timestamp}
    poster = picture.sender_id
    at = picture.at or cell.timestamp
    interval = space.roster.interval_at(poster, at)
    assistant = (
        interval.kind == "assistant" if interval is not None else not picture.poster
    )
    named = named_members(said, roster_names(space.roster, until=at, kind="human"))
    return {
        "timestamp": at,
        "sender_ids": [poster] if poster else [],
        "subject_ids": named or ([] if assistant or not poster else [poster]),
        "origin": "assistant" if assistant else "member",
    }


def _has_text(cell: MemCell) -> bool:
    """Whether the cell holds a chat message (``kind == "text"``) to remember."""
    return any(getattr(item, "kind", None) == "text" for item in cell.items)


def _human_senders(cell: MemCell) -> list[str]:
    """Distinct role=user sender_ids in a cell, preserving order.

    Skips non-``ChatMessage`` items (agent trajectories' ``ToolCallResult``
    has no ``role``).
    """
    senders: list[str] = []
    for item in cell.items:
        if getattr(item, "role", None) != "user":
            continue
        sid = getattr(item, "sender_id", None)
        if sid and sid not in senders:
            senders.append(sid)
    return senders


def _human_speakers(cell: MemCell, roster: Roster) -> list[str]:
    """The members whose own (role ``user``) messages are in the cell, in order.

    The member-in-group profile reads a member's own lines, so only a human
    member who spoke as a user has a profile to update from this segment.
    """
    speakers: list[str] = []
    for item in cell.items:
        sender = getattr(item, "sender_id", None)
        if not sender or sender in speakers or getattr(item, "role", None) != "user":
            continue
        interval = roster.interval_at(sender, int(getattr(item, "timestamp", 0) or 0))
        if interval is not None and interval.kind == "assistant":
            continue
        speakers.append(sender)
    return speakers
