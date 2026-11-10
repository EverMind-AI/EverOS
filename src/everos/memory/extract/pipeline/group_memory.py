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

Every cell announces its episode as :class:`GroupEpisodeExtracted`, which the
group strategies consume (member-in-group profiles, scenes). None of the
one-to-one events is published: those would run the personal strategies on
group content.
"""

from __future__ import annotations

__all__ = ["GroupMemoryPipeline"]

from collections.abc import Sequence
from typing import TYPE_CHECKING

from everalgo.types import Episode as AlgoEpisode
from everalgo.user_memory import EpisodeExtractor

from everos.component.images import ImageLimits
from everos.core.observability.logging import get_logger
from everos.core.observability.tracing import capture_output, memory_span
from everos.core.utils.language import detect_output_language
from everos.memory.custom_instructions import CustomInstructions
from everos.memory.events import GroupEpisodeExtracted
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
from everos.memory.extract.pipeline.member_profile import (
    write_member_picture_profile,
)
from everos.memory.extract.pipeline.pictures import (
    Picture,
    memcell_alive,
    ops_by_poster,
    pictures_of,
)
from everos.memory.groups import named_members, roster_names
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
    Scope,
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
    ) -> None:
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
        cell = prepared.cell
        if not any(getattr(i, "kind", None) == "text" for i in cell.items):
            return CellOutcome(message_count=len(cell.items))
        record = await self._groups.get(scope)
        roster = record.roster if record is not None else Roster()
        group = (
            record.group
            if record is not None
            else Group(group_id=scope.owner_id, created_at=cell.timestamp)
        )
        pictures = await pictures_of(
            cell, images=self._images, limits=self._image_limits
        )
        custom = with_roster_block(
            self._instructions.load("episode_extract"),
            roster_block(group, roster, cell),
        )
        algo_ep, described, ops = await self._extract(
            ingested, prepared, label_cell(cell, roster), pictures, custom
        )
        language = detect_output_language(algo_ep.episode)
        found = attribution(cell, roster, texts=(algo_ep.subject, algo_ep.episode))
        md_paths = await self._persist_images(
            scope, ingested, prepared, roster, pictures=pictures, described=described
        )
        await self._persist_member_profiles(
            scope, prepared, ops=ops, pictures=pictures, language=language
        )
        episode = Episode.from_algo(
            algo_ep,
            owner_id=scope.owner_id,
            session_id=ingested.session_id,
            sender_ids=found.sender_ids,
            parent_id=prepared.memcell_id,
            language=language,
            image_ids=[picture.image_id for picture in pictures],
        ).model_copy(
            update={
                "subject_ids": found.subject_ids,
                "origin": found.origin,
                "source_start": found.source_start,
            }
        )
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
                    speaker_ids=_human_speakers(cell, roster),
                    language=language,
                )
            )
        return CellOutcome(
            message_count=len(cell.items), events=events, md_paths=md_paths
        )

    async def _extract(
        self,
        ingested: IngestResult,
        prepared: PreparedCell,
        labelled: MemCell,
        pictures: Sequence[Picture],
        custom_instructions: str,
    ) -> _Extraction:
        """The one-to-one extraction call over the labelled cell."""
        with memory_span(
            "everos.extract",
            observation_type="generation",
            session_id=ingested.session_id,
            metadata={
                "app_id": ingested.app_id,
                "project_id": ingested.project_id,
                "memcell_id": prepared.memcell_id,
                "group_id": ingested.group_id,
                "pictures": len(pictures),
            },
        ) as extract_span:
            if pictures:
                extraction = await self._extract_with_pictures(
                    prepared, labelled, pictures, custom_instructions
                )
            else:

                async def attempt() -> AlgoEpisode:
                    return await self._ep_ext.aextract(
                        labelled,
                        sender_id=None,
                        custom_instructions=custom_instructions,
                    )

                episode = await retry_llm_output(
                    attempt, operator="episode", memcell_id=prepared.memcell_id
                )
                extraction = (episode, {}, [])
            capture_output(extract_span, extraction[0].episode)
        return extraction

    async def _extract_with_pictures(
        self,
        prepared: PreparedCell,
        labelled: MemCell,
        pictures: Sequence[Picture],
        custom_instructions: str,
    ) -> _Extraction:
        labels = [(picture.label, picture.data, picture.mime) for picture in pictures]
        by_label = {picture.label: picture.image_id for picture in pictures}

        async def attempt() -> _Extraction:
            reply = await image_extract.extract(
                self._llm,
                labelled,
                pictures=labels,
                custom_instructions=custom_instructions,
                output_language=None,
            )
            return (
                image_extract.to_episode(reply, labelled, labels=by_label),
                {fact.ref: (fact.description, fact.ocr) for fact in reply.images},
                profile_delta.retarget(reply.profile, by_label),
            )

        return await retry_llm_output(
            attempt, operator="image_memory", memcell_id=prepared.memcell_id
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
