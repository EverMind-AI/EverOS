"""What a profile update reads: the cues of its segments, or every message.

A person's profile update reads the segments queued on it (the strategies
choose them: a user's own space, a group's members). ``[profile] input``
decides what of them the model is shown:

- ``full`` (the default) -- every message of every segment, and every picture
  the person posted in them: the update of 2026-11-13, request for request.
  A paired judge on 2026-11-17 preferred the profiles it keeps to the cue-only
  ones 18 to 2 on LoCoMo at the same update cadence (p 4e-4): the whole
  conversation carries how the person acts, which a cue list of self-
  statements does not.
- ``cues`` -- only what the segments' episode calls marked as
  saying something lasting about the person (:mod:`.profile_cues`), each
  marked message with its timestamp and, before it, the message before it by
  someone else, cut to :data:`CONTEXT_CHARS` characters, so the model can
  read what the message answers. A marked picture's own message is shown too
  (it is where the picture was sent), and the picture is attached as its
  picture memory's text -- the description and visible text the episode call
  wrote while looking at it -- or, with ``[profile] picture_pixels``, as
  itself. A segment that marked nothing for the person is read without being
  shown, and an update with nothing to show makes no call: its segments are
  marked read, so they are not queued again.

Measured on the 2026-11-14 builds (DeepSeek V4.1 Flash): the operations of an
update quoted 18% of a LoCoMo conversation, 4-5% of a group's, and 92-99% of
the lines they quoted were the person's own; 44-48% of a group member's
updates changed nothing.

A segment without any cue record -- extracted before cues existed, or while
no profile of its space read them -- is read in full in ``cues`` mode too
(every message; its pictures attached as this mode attaches pictures):
nothing it holds is lost to the switch. What a forget took out is never read
in either mode: the segments come in readable (without forgotten messages),
a cue of a message that is gone selects nothing, and a forgotten picture is
taken out before anything is chosen (and by the picture read,
:func:`~.pictures.profile_pictures`).

:func:`select_profile_input` is the selection of one person's update: the
one-to-one strategy calls it (:mod:`everos.memory.strategies.
extract_user_profile`). :func:`select_members_input` applies the same rules
to the several people one batched update of a group reads
(:mod:`everos.memory.strategies._member_batch`), and renders the
conversation once: in each segment, the union of what every person reads
there.
"""

from __future__ import annotations

__all__ = [
    "CONTEXT_CHARS",
    "MembersInput",
    "ProfileInput",
    "picture_texts",
    "select_members_input",
    "select_profile_input",
]

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from everalgo.types import ChatMessage

from everos.memory.extract.pipeline.pictures import image_ids_in, without_pictures
from everos.memory.models import MemCell
from everos.memory.ports import ImageMemoryStore, ProfileCueStore, Scope

CONTEXT_CHARS = 200
"""A marked message's context -- the message before it by someone else -- is
cut to this many characters."""

_CUT = "..."
_IMAGE_REF_PREFIX = "[IMAGE: "


@dataclass(frozen=True)
class ProfileInput:
    """What one update reads.

    Attributes:
        segments: ``(memcell_id, cell)`` the update reads, oldest first, each
            cell readable and whole: an item's sources and span are read off
            them. Empty when nothing is to be shown: no call.
        shown: The same segments as the model is shown them, one per entry
            of ``segments``.
        pictures: The pictures to attach, by content hash; ``None`` attaches
            every picture the person posted in ``shown``.
        picture_text: Content hash -> the text to attach a picture as;
            ``None`` attaches the pictures themselves.
        read: Every segment the update reads, shown or not, that carries a
            cue record: the ones to mark read once the update is done.
        skipped: Of ``read``, those not shown (no cue of the person left).
    """

    segments: list[tuple[str, MemCell]]
    shown: list[MemCell]
    pictures: frozenset[str] | None = None
    picture_text: Mapping[str, str] | None = None
    read: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MembersInput:
    """What one update of several people's profiles reads (``cues`` mode).

    Attributes:
        segments: ``(memcell_id, cell)`` the update shows, oldest first, each
            cell readable and whole. Empty when nothing is to be shown: no
            call.
        shown: The same segments as the model is shown them, one per entry
            of ``segments``: in each, what every person reads there.
        people: Each person the update reads, with the segments they are read
            from (ids, oldest first). A person with nothing to read is not
            here: no operation of theirs, no token.
        pictures: The pictures to attach, by content hash.
        picture_text: Content hash -> the text to attach a picture as;
            ``None`` attaches the pictures themselves.
        read: Per person, every segment with a cue record the update reads
            for them, shown or not: the ones to mark read once it is done.
    """

    segments: list[tuple[str, MemCell]]
    shown: list[MemCell]
    people: dict[str, list[str]]
    pictures: frozenset[str] = frozenset()
    picture_text: Mapping[str, str] | None = None
    read: dict[str, list[str]] = field(default_factory=dict)


async def select_profile_input(
    segments: Sequence[tuple[str, MemCell]],
    person_id: str,
    *,
    mode: Literal["cues", "full"],
    cues: ProfileCueStore | None,
    shown: Sequence[MemCell] | None = None,
    image_memory: ImageMemoryStore | None = None,
    scope: Scope | None = None,
    picture_pixels: bool = False,
    forgotten_pictures: Collection[str] = (),
) -> ProfileInput:
    """What an update of ``person_id``'s profile reads of ``segments``.

    Args:
        segments: ``(memcell_id, cell)`` to read, oldest first, readable
            (without the messages a live forget names).
        person_id: Whom the profile is about.
        mode: ``[profile] input``.
        cues: The cue records; ``None`` reads every segment in full.
        shown: ``segments`` as the model is shown them (a group's speakers
            labelled); the cells themselves when omitted.
        image_memory: Where the pictures' memories are read, for pictures
            attached as text.
        scope: The profile's space, where those memories are.
        picture_pixels: Attach the marked pictures themselves.
        forgotten_pictures: Pictures a live forget names (content hashes):
            taken out of the segments before anything is chosen, so a cue of
            a forgotten picture selects nothing, and a message that held
            nothing else is not shown (``cues`` only; ``full`` leaves them to
            the picture read, as before).
    """
    views = list(shown) if shown is not None else [cell for _, cell in segments]
    if mode == "full" or cues is None or not segments:
        return ProfileInput(segments=list(segments), shown=views)
    if forgotten_pictures:
        views = [without_pictures(view, forgotten_pictures) for view in views]
    records = await cues.of_person(person_id, [m for m, _ in segments])
    kept: list[tuple[str, MemCell]] = []
    kept_views: list[MemCell] = []
    attach: set[str] = set()
    read: list[str] = []
    skipped: list[str] = []
    for (memcell_id, cell), view in zip(segments, views, strict=True):
        record = records.get(memcell_id)
        if record is None:
            # Never marked: every message, and every picture the person
            # posted, attached as this mode attaches pictures.
            kept.append((memcell_id, cell))
            kept_views.append(view)
            attach.update(_posted(view, person_id))
            continue
        read.append(memcell_id)
        chosen = _chosen_messages(view, person_id, record.message_ids, record.image_ids)
        if not chosen:
            skipped.append(memcell_id)
            continue
        reduced = _cued_cell(view, {person_id: chosen})
        kept.append((memcell_id, cell))
        kept_views.append(reduced)
        attach.update(i for i in record.image_ids if i in _posted(reduced, person_id))
    if picture_pixels or not attach:
        return ProfileInput(
            segments=kept,
            shown=kept_views,
            pictures=frozenset(attach),
            read=read,
            skipped=skipped,
        )
    texts = (
        await picture_texts(image_memory, scope, [m for m, _ in kept], attach)
        if image_memory is not None and scope is not None
        else {}
    )
    return ProfileInput(
        segments=kept,
        shown=kept_views,
        pictures=frozenset(texts),
        picture_text=texts,
        read=read,
        skipped=skipped,
    )


async def select_members_input(
    segments: Sequence[tuple[str, MemCell]],
    people: Mapping[str, Sequence[str]],
    *,
    cues: ProfileCueStore,
    shown: Sequence[MemCell] | None = None,
    image_memory: ImageMemoryStore | None = None,
    scope: Scope | None = None,
    picture_pixels: bool = False,
    forgotten_pictures: Collection[str] = (),
) -> MembersInput:
    """What one update of several people's profiles reads of ``segments``.

    :func:`select_profile_input`'s ``cues`` rules for each person, over the
    segments they are read from, with the conversation rendered once: a
    segment shows every message a cue of one of its people marks, each after
    its context -- the nearest earlier message by someone other than the
    person it was marked for, cut to :data:`CONTEXT_CHARS`, pictures bare --
    and a message one person's cue marks while it is another's context is
    shown whole. A segment never marked for one of its people is shown whole
    and read for them in full, their pictures included; a segment with
    nothing to show for anyone is not shown. A person whose segments marked
    nothing for them is not read at all. A segment a person's profile has
    read already (its record says so: a batch read again) is not read for
    them again.

    Args:
        segments: ``(memcell_id, cell)`` of the batch, oldest first, readable
            (without the messages a live forget names).
        people: Each person to read, with the ids of the segments of
            ``segments`` holding a message of theirs that their profile has
            not read.
        cues: The cue records.
        shown: ``segments`` as the model is shown them (speakers labelled);
            the cells themselves when omitted.
        image_memory: Where the pictures' memories are read, for pictures
            attached as text.
        scope: The space the profiles belong to, where those memories are.
        picture_pixels: Attach the marked pictures themselves.
        forgotten_pictures: Pictures a live forget names (content hashes),
            taken out of the segments before anything is chosen.
    """
    views = list(shown) if shown is not None else [cell for _, cell in segments]
    if forgotten_pictures:
        views = [without_pictures(view, forgotten_pictures) for view in views]
    view_of = {
        memcell_id: view for (memcell_id, _), view in zip(segments, views, strict=True)
    }
    whole: set[str] = set()
    marked: dict[str, dict[str, set[str]]] = {}
    chosen: dict[str, list[str]] = {}
    read: dict[str, list[str]] = {}
    attach: set[str] = set()
    for person_id, memcell_ids in people.items():
        records = await cues.of_person(person_id, memcell_ids)
        for memcell_id in memcell_ids:
            view = view_of[memcell_id]
            record = records.get(memcell_id)
            if record is None:
                # Never marked: every message, and every picture the person
                # posted, as select_profile_input reads such a segment.
                whole.add(memcell_id)
                chosen.setdefault(person_id, []).append(memcell_id)
                attach.update(_posted(view, person_id))
                continue
            if record.read:
                continue
            read.setdefault(person_id, []).append(memcell_id)
            mine = _chosen_messages(
                view, person_id, record.message_ids, record.image_ids
            )
            if not mine:
                continue
            marked.setdefault(memcell_id, {})[person_id] = mine
            chosen.setdefault(person_id, []).append(memcell_id)
            reduced = _cued_cell(view, {person_id: mine})
            attach.update(
                i for i in record.image_ids if i in _posted(reduced, person_id)
            )
    kept: list[tuple[str, MemCell]] = []
    kept_views: list[MemCell] = []
    for (memcell_id, cell), view in zip(segments, views, strict=True):
        if memcell_id in whole:
            kept.append((memcell_id, cell))
            kept_views.append(view)
        elif memcell_id in marked:
            kept.append((memcell_id, cell))
            kept_views.append(_cued_cell(view, marked[memcell_id]))
    order = {memcell_id: n for n, (memcell_id, _) in enumerate(kept)}
    targets = {
        person_id: sorted(ids, key=order.__getitem__)
        for person_id, ids in chosen.items()
    }
    if picture_pixels or not attach:
        return MembersInput(
            segments=kept,
            shown=kept_views,
            people=targets,
            pictures=frozenset(attach),
            read=read,
        )
    texts = (
        await picture_texts(image_memory, scope, [m for m, _ in kept], attach)
        if image_memory is not None and scope is not None
        else {}
    )
    return MembersInput(
        segments=kept,
        shown=kept_views,
        people=targets,
        pictures=frozenset(texts),
        picture_text=texts,
        read=read,
    )


async def picture_texts(
    store: ImageMemoryStore,
    scope: Scope,
    memcell_ids: Sequence[str],
    image_ids: Collection[str],
) -> dict[str, str]:
    """Each picture's current memory, in ``scope``, as the text it is attached as.

    A picture memory written from one of ``memcell_ids`` and not replaced or
    forgotten: its description, and the text visible in the picture when
    there is any. A picture with no such memory is left out (it is not
    attached).
    """
    wanted = set(image_ids)
    found: dict[str, str] = {}
    for stored in await store.read_for_segments(scope, list(memcell_ids)):
        memory = stored.item
        if memory.superseded_by is not None or memory.image_id not in wanted:
            continue
        text = f"Shown in words (the picture is not attached): {memory.description}"
        if memory.ocr.strip():
            text += f"\nText in the picture: {memory.ocr}"
        found[memory.image_id] = text
    return found


def _posted(cell: MemCell, person_id: str) -> set[str]:
    """The pictures ``person_id``'s own messages in ``cell`` carry."""
    return {
        image_id
        for item in cell.items
        if _own(item, person_id)
        for image_id in image_ids_in(str(item.content or ""))
    }


def _own(item: object, person_id: str) -> bool:
    return (
        isinstance(item, ChatMessage)
        and item.sender_id == person_id
        and item.role == "user"
    )


def _chosen_messages(
    cell: MemCell,
    person_id: str,
    message_ids: Collection[str],
    image_ids: Collection[str],
) -> set[str]:
    """The person's messages still in ``cell`` that a cue marks, by id: the
    marked ones, and the ones that carry a marked picture."""
    marked = set(message_ids)
    pictures = set(image_ids)
    return {
        item.id
        for item in cell.items
        if _own(item, person_id)
        and isinstance(item, ChatMessage)
        and item.content
        and (
            item.id in marked
            or any(i in pictures for i in image_ids_in(str(item.content)))
        )
    }


def _cued_cell(cell: MemCell, chosen: Mapping[str, Collection[str]]) -> MemCell:
    """``cell`` with only the chosen messages (by person, by id), each after
    its context: the nearest earlier message by someone other than the person
    it was chosen for, cut to :data:`CONTEXT_CHARS`, its pictures shown as
    bare references (they are not the person's). A message chosen for one
    person and another's context is shown whole."""
    messages = [item for item in cell.items if isinstance(item, ChatMessage)]
    whole: set[int] = set()
    context: set[int] = set()
    for person_id, ids in chosen.items():
        for index, message in enumerate(messages):
            if message.id not in ids:
                continue
            whole.add(index)
            before = next(
                (
                    position
                    for position in range(index - 1, -1, -1)
                    if messages[position].sender_id != person_id
                    and messages[position].content
                ),
                None,
            )
            if before is not None:
                context.add(before)
    shown = {i: messages[i] for i in whole} | {
        i: _context(messages[i]) for i in context - whole
    }
    return cell.model_copy(update={"items": [shown[i] for i in sorted(shown)]})


def _context(message: ChatMessage) -> ChatMessage:
    """Another speaker's message as context: no picture, at most
    :data:`CONTEXT_CHARS` characters."""
    text = str(message.content or "")
    for image_id in image_ids_in(text):
        text = text.replace(f"{_IMAGE_REF_PREFIX}{image_id}]", "[IMAGE]")
    if len(text) > CONTEXT_CHARS:
        text = text[:CONTEXT_CHARS].rstrip() + _CUT
    return message.model_copy(update={"content": text})
