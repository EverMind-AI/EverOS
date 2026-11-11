"""Group memory pipeline -- one group episode per segment (docs/group-memory.md, 7).

A cell cut from a session bound to a group is the group's memory, not each
sender's: the pipeline writes **one** episode under ``groups/<group_id>/``
(D3), one picture memory per picture the segment carries (7.3), and the
profile items a picture establishes into its poster's member-in-group profile
only (D17). It never writes under ``users/``.

The extraction call is the one-to-one call -- everalgo's
:class:`EpisodeExtractor` for text, the EverOS picture call for a segment with
pictures -- and only what it is given changes (:mod:`.group_render`): speakers
labelled by their roster names, assistants as ``"<name> (assistant)"``, and the
roster block appended after the operator's custom instructions. After it, the
code sets the attribution fields (``sender_ids`` / ``subject_ids`` /
``origin``) and the segment's span (``source_start`` / ``timestamp``), which
every read compares with the asker's membership windows.

The language is the conversation's, resolved in code before the call exactly
as on the one-to-one path (:mod:`everos.memory.extract.conversation_language`):
the segment's human messages, and its session's earlier segments when it has
too little text of its own. It is named to the model on both routes, so the
picture descriptions and the poster's picture profile items get it too, and it
pins the episode and its event, which carry it on to the member-in-group
profiles and the scenes. Only a session without a letter anywhere leaves the
choice to the model; then the language its episode was written in is pinned.

Every cell announces its episode as :class:`GroupEpisodeExtracted`, which the
group strategies consume (member-in-group profiles, scenes). None of the
one-to-one events is published: those would run the personal strategies on
group content.

A message forgotten while it waited in the buffer is not read: the cell is
extracted without the messages a live forget names (section 9.4). A forget
of a segment already extracted re-runs the same call through
:meth:`GroupMemoryPipeline.reextract`, over the segment without what was
forgotten, and writes nothing itself: the forget decides what replaces what.
"""

from __future__ import annotations

__all__ = ["GroupMemoryPipeline"]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everalgo.types import Episode as AlgoEpisode
from everalgo.user_memory import EpisodeExtractor, OutputLanguage

from everos.component.images import ImageLimits
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import capture_output, memory_span
from everos.core.utils.language import detect_output_language
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.events import GroupEpisodeExtracted
from everos.memory.extract.conversation_language import resolve_cell_language
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.extract.pipeline import image_memory as image_extract
from everos.memory.extract.pipeline import profile_delta
from everos.memory.extract.pipeline.base import CellOutcome, PreparedCell
from everos.memory.extract.pipeline.group_render import (
    attribution,
    label_cell,
    roster_block,
    with_roster_block,
)
from everos.memory.extract.pipeline.group_scene import narrate_scene
from everos.memory.extract.pipeline.member_profile import (
    write_member_picture_profile,
)
from everos.memory.extract.pipeline.pictures import (
    Picture,
    memcell_alive,
    ops_by_poster,
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
    MemberProfileStore,
    MemcellStore,
    MemoryRef,
    MessageForgetStore,
    Scope,
    SessionKey,
    Stored,
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
"""The episode, each picture's ``(description, ocr)`` by label, and the
profile operations the pictures produced -- readings of one call."""


@dataclass(frozen=True)
class _Segment:
    """One segment's extraction: the group episode and what the call read.

    Attributes:
        episode: The group episode, attribution and span set, not written.
        pictures: The pictures the call was shown.
        described: Each picture's ``(description, ocr)`` by label.
        ops: The profile operations the pictures produced.
        language: The episode's language (the conversation's).
        roster: The group's roster the call was rendered with.
    """

    episode: Episode
    pictures: list[Picture]
    described: dict[str, tuple[str, str]]
    ops: list[profile_delta.ProfileItemOp]
    language: str | None
    roster: Roster


class GroupMemoryPipeline:
    """One group episode per pre-cut memcell of a group session.

    Args:
        episode_store: Where the group episode is written.
        memcell_store: Whether the cell still exists when it is written.
        groups: The group's meta and roster (labels, roster block, attribution).
        instructions: Custom instructions for the episode operator.
        llm_client: LLM for extraction.
        image_store: Where pictures are read from; ``None`` disables the
            picture route.
        image_memory_store: Where a picture's memory is written.
        image_limits: ``[images]`` -- the size a picture is scaled to.
        member_profiles: Where a picture's profile items reach its poster's
            member-in-group profile; ``None`` keeps that half off.
        message_forget: Which messages are forgotten; a cell is extracted
            without them. ``None`` reads every message.
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
        member_profiles: MemberProfileStore | None = None,
        message_forget: MessageForgetStore | None = None,
    ) -> None:
        self._message_forget = message_forget
        self._ep_ext = EpisodeExtractor(llm=llm_client)
        self._episode_store = episode_store
        self._memcell_store = memcell_store
        self._groups = groups
        self._instructions = instructions
        self._llm = llm_client
        self._images = image_store
        self._image_memory_store = image_memory_store
        self._image_limits = image_limits
        self._member_profiles = member_profiles

    async def run_cell(
        self, ingested: IngestResult, prepared: PreparedCell
    ) -> CellOutcome:
        """Extract the cell's group episode (and pictures) and write it once."""
        if ingested.group_id is None:
            raise ValueError("the group pipeline runs only on a group session")
        scope = Scope(
            app_id=ingested.app_id,
            project_id=ingested.project_id,
            owner_id=ingested.group_id,
            owner_type="group",
        )
        session = SessionKey(ingested.app_id, ingested.project_id, ingested.session_id)
        cell = await self._readable(prepared.memcell_id, prepared.cell)
        if not _has_text(cell):
            return CellOutcome(message_count=len(prepared.cell.items))
        found = await self._extract_segment(scope, session, prepared.memcell_id, cell)
        md_paths = await self._persist_images(
            scope,
            ingested,
            prepared,
            found.roster,
            pictures=found.pictures,
            described=found.described,
        )
        await self._persist_member_profiles(
            scope,
            prepared,
            ops=found.ops,
            pictures=found.pictures,
            language=found.language,
        )
        episode = found.episode
        ref = await self._persist(scope, ingested, episode, prepared.memcell_id)
        events: list[Event] = []
        if ref is not None:
            md_paths.append(ref.md_path)
            events.append(
                GroupEpisodeExtracted(
                    memcell_id=prepared.memcell_id,
                    episode_entry_id=ref.entry_id,
                    episode_text=episode.episode,
                    episode_timestamp_ms=episode.timestamp,
                    group_id=scope.owner_id,
                    session_id=ingested.session_id,
                    app_id=ingested.app_id,
                    project_id=ingested.project_id,
                    speaker_ids=_human_speakers(cell, found.roster),
                    language=found.language,
                )
            )
        return CellOutcome(
            message_count=len(prepared.cell.items), events=events, md_paths=md_paths
        )

    async def reextract(
        self, *, scope: Scope, session_id: str, memcell_id: str, cell: MemCell
    ) -> Episode | None:
        """The segment's group episode, extracted again from ``cell``; not written.

        The call :meth:`run_cell` makes -- speaker labels, the roster block,
        the conversation's language, the picture route when ``cell`` still
        carries pictures -- over a segment a forget changed: ``cell`` is the
        stored segment without what was forgotten
        (:func:`~everos.memory.groups.without_forgotten`). The pictures'
        descriptions and profile items the call returns again are dropped:
        their memories were never made of the forgotten message.

        Returns:
            The new episode version, or ``None`` when ``cell`` holds no text
            message any more (nothing is left to remember).
        """
        if not _has_text(cell):
            return None
        session = SessionKey(scope.app_id, scope.project_id, session_id)
        found = await self._extract_segment(scope, session, memcell_id, cell)
        return found.episode

    async def scene_narrative(
        self,
        *,
        owner_id: str,
        members: Sequence[Stored[Episode]],
        language: str | None,
    ) -> AlgoEpisode:
        """A scene's narrative written from scratch over ``members``.

        What a forget writes when an episode of a scene changed: the INIT of
        the scene strategy's reflector, never an UPDATE -- the previous
        narrative may say what was forgotten.
        """
        return await narrate_scene(
            self._llm,
            owner_id=owner_id,
            members=members,
            previous=None,
            custom_instructions=self._instructions.load("reflect"),
            language=language,
        )

    async def _readable(self, memcell_id: str, cell: MemCell) -> MemCell:
        """``cell`` without the messages a live forget names."""
        if self._message_forget is None:
            return cell
        forgotten = await self._message_forget.forgotten_in_segments([memcell_id])
        return without_forgotten(cell, forgotten.get(memcell_id, ()))

    async def _extract_segment(
        self, scope: Scope, session: SessionKey, memcell_id: str, cell: MemCell
    ) -> _Segment:
        """Run the episode call over ``cell`` and set the attribution fields."""
        record = await self._groups.get(scope)
        roster = record.roster if record is not None else Roster()
        group = (
            record.group
            if record is not None
            else Group(group_id=scope.owner_id, created_at=cell.timestamp)
        )
        # Named to the model, never left to it: the conversation's language,
        # resolved from the human messages as on the one-to-one path.
        named = await resolve_cell_language(
            cell,
            memcell_id=memcell_id,
            session=session,
            memcells=self._memcell_store,
        )
        pictures = await pictures_of(
            cell, images=self._images, limits=self._image_limits
        )
        custom = with_roster_block(
            self._instructions.load("episode_extract"),
            roster_block(group, roster, cell),
        )
        algo_ep, described, ops = await self._extract(
            scope,
            session,
            memcell_id,
            label_cell(cell, roster),
            pictures,
            custom,
            named,
        )
        # That language pins the episode, its picture profile items and its
        # event, so the member-in-group profiles and the scenes follow it.
        # Only a conversation without a letter in it left the choice to the
        # model; then what it chose is read off the episode.
        language = (
            str(named) if named is not None else detect_output_language(algo_ep.episode)
        )
        found = attribution(cell, roster, texts=(algo_ep.subject, algo_ep.episode))
        episode = Episode.from_algo(
            algo_ep,
            owner_id=scope.owner_id,
            session_id=session.session_id,
            sender_ids=found.sender_ids,
            parent_id=memcell_id,
            language=language,
            image_ids=[picture.image_id for picture in pictures],
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
            ops=list(ops),
            language=language,
            roster=roster,
        )

    async def _extract(
        self,
        scope: Scope,
        session: SessionKey,
        memcell_id: str,
        labelled: MemCell,
        pictures: Sequence[Picture],
        custom_instructions: str,
        language: OutputLanguage | None,
    ) -> _Extraction:
        """The one-to-one extraction call over the labelled cell.

        ``language`` is named to the model on either route; ``None`` leaves
        it the choice.
        """
        with memory_span(
            "everos.extract",
            observation_type="generation",
            session_id=session.session_id,
            metadata={
                "app_id": scope.app_id,
                "project_id": scope.project_id,
                "memcell_id": memcell_id,
                "group_id": scope.owner_id,
                "pictures": len(pictures),
                "output_language": None if language is None else str(language),
            },
        ) as extract_span:
            if pictures:
                extraction = await self._extract_with_pictures(
                    memcell_id, labelled, pictures, custom_instructions, language
                )
            else:

                async def attempt() -> AlgoEpisode:
                    return await self._ep_ext.aextract(
                        labelled,
                        sender_id=None,
                        custom_instructions=custom_instructions,
                        output_language=language,
                    )

                episode = await retry_llm_output(
                    attempt, operator="episode", memcell_id=memcell_id
                )
                extraction = (episode, {}, [])
            capture_output(extract_span, extraction[0].episode)
        return extraction

    async def _extract_with_pictures(
        self,
        memcell_id: str,
        labelled: MemCell,
        pictures: Sequence[Picture],
        custom_instructions: str,
        language: OutputLanguage | None,
    ) -> _Extraction:
        labels = [(picture.label, picture.data, picture.mime) for picture in pictures]
        by_label = {picture.label: picture.image_id for picture in pictures}

        async def attempt() -> _Extraction:
            reply = await image_extract.extract(
                self._llm,
                labelled,
                pictures=labels,
                custom_instructions=custom_instructions,
                output_language=language,
            )
            return (
                image_extract.to_episode(reply, labelled, labels=by_label),
                {fact.ref: (fact.description, fact.ocr) for fact in reply.images},
                profile_delta.retarget(reply.profile, by_label),
            )

        return await retry_llm_output(
            attempt, operator="image_memory", memcell_id=memcell_id
        )

    async def _persist_images(
        self,
        scope: Scope,
        ingested: IngestResult,
        prepared: PreparedCell,
        roster: Roster,
        *,
        pictures: Sequence[Picture],
        described: dict[str, tuple[str, str]],
    ) -> list[str]:
        """One picture memory per picture, owned by the group (section 7.3).

        ``sender_ids`` is the poster; the memory is dated by the message that
        carried the picture, so a membership window is checked against when
        it was posted.
        """
        if not described or self._image_memory_store is None or self._images is None:
            return []
        memories: list[ImageMemory] = []
        for picture in pictures:
            pair = described.get(picture.label)
            if not pair:
                continue
            description, ocr = pair
            info = await self._images.metadata(picture.image_id)
            poster = picture.sender_id
            at = picture.at or prepared.cell.timestamp
            interval = roster.interval_at(poster, at)
            assistant = (
                interval.kind == "assistant"
                if interval is not None
                else not picture.poster
            )
            named = named_members(
                f"{description}\n{ocr}", roster_names(roster, until=at, kind="human")
            )
            memories.append(
                ImageMemory(
                    owner_id=scope.owner_id,
                    description=description,
                    timestamp=at,
                    ocr=ocr,
                    image_id=picture.image_id,
                    name=prepared.picture_names.get(picture.image_id, ""),
                    original_relpath=await self._images.relpath_of(picture.image_id),
                    media_type=info.media_type if info else "jpeg",
                    width=info.width if info else 0,
                    height=info.height if info else 0,
                    session_id=ingested.session_id,
                    parent_id=prepared.memcell_id,
                    sender_ids=[poster] if poster else [],
                    subject_ids=named or ([] if assistant or not poster else [poster]),
                    origin="assistant" if assistant else "member",
                )
            )
        if not memories:
            return []
        refs = await self._image_memory_store.append_many(
            scope,
            memories,
            only_if=memcell_alive(self._memcell_store, prepared.memcell_id),
        )
        return [ref.md_path for ref in refs]

    async def _persist_member_profiles(
        self,
        scope: Scope,
        prepared: PreparedCell,
        *,
        ops: Sequence[profile_delta.ProfileItemOp],
        pictures: Sequence[Picture],
        language: str | None,
    ) -> None:
        """Each poster's picture profile items, into their profile in this group.

        D17: an item goes to the poster of the picture it names and to no one
        else; an assistant's picture reaches no profile.
        """
        if self._member_profiles is None or not ops:
            return
        for poster, routed in ops_by_poster(ops, pictures).items():
            if not poster:
                continue
            own = [p for p in pictures if p.poster == poster]
            await write_member_picture_profile(
                store=self._member_profiles,
                scope=scope,
                member_id=poster,
                ops=routed,
                observed_at=prepared.cell.timestamp,
                image_ids=[p.image_id for p in own],
                memcell_id=prepared.memcell_id,
                span=(
                    min(p.at or prepared.cell.timestamp for p in own),
                    max(p.at or prepared.cell.timestamp for p in own),
                ),
                language=language,
            )

    async def _persist(
        self, scope: Scope, ingested: IngestResult, episode: Episode, memcell_id: str
    ) -> MemoryRef | None:
        """Write the group episode; ``None`` when the cell was deleted meanwhile."""
        with memory_span(
            "everos.persist.markdown",
            observation_type="span",
            session_id=ingested.session_id,
            metadata={
                "group_id": scope.owner_id,
                "app_id": ingested.app_id,
                "project_id": ingested.project_id,
            },
        ) as persist_span:
            ref = await self._episode_store.append(
                scope,
                episode,
                only_if=memcell_alive(self._memcell_store, memcell_id),
            )
            if ref is None:
                return None
            capture_output(persist_span, ref.md_path)
        return ref


def _has_text(cell: MemCell) -> bool:
    """Whether the cell holds a chat message (``kind == "text"``) to remember."""
    return any(getattr(item, "kind", None) == "text" for item in cell.items)


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
