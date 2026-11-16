"""The profile cues an episode call marks: the block, the entries, their mapping.

A profile update reads, by default (``[profile] input = "cues"``), only what
the segment says about the person: the messages and pictures the episode call
marked as lasting facts about their speaker (:mod:`.profile_input`). The
marking rides on the episode call -- the model is reading the segment
anyway -- as a ``<profile_cues>`` block appended to the episode request, after
what the cell adds (a group's roster block, the ``<pictures>`` block) and
before the ``<output_language>`` block, and only when a profile of the cell's
space will read it, exactly as the picture block reaches only picture cells.
With no profile reading cues the episode request is the one of 2026-11-14,
byte for byte.

The block (:data:`CHAT_PROFILE_CUES`, ``chat.episode_profile_cues``) reuses
the profile update's own admission wording (everalgo ``profile`` prompts,
``<what_a_profile_is>``): a profile is a portrait, not a log, so a cue is a
statement still true the next time -- a role, a constraint, an environment,
the people in one's life, a preference, a habit -- never a one-off event,
another person's opinion, a joke or a hypothetical. Its last lines extend the
reply format the way the picture block does: one more field, ``profile_cues``,
beside ``title`` and ``content`` (and ``images``). An empty list is a valid
answer, and the common one.

Each entry names a speaker as the conversation renders them, a short
verbatim ``quote`` of one of that speaker's own messages and, in a picture
cell, the ``pictures`` (labels) that speaker posted which show something
lasting about them. :func:`map_cues` reads the entries strictly in meaning:

- a quote is mapped to the first message of that speaker that contains it,
  compared without case, with runs of whitespace as one space and every
  quotation mark read as a plain one; a label to its picture only when that
  speaker posted it;
- only a human speaker of the cell is mapped (in a group, never the
  assistant); an entry that maps to nothing is dropped and counted, and so is
  an entry that is not an object with a speaker;
- every human speaker of the cell gets a record, an empty one when nothing
  was marked: that is what tells "nothing lasting was said" from "never
  marked".

The entries are read leniently in form, as every episode reply is: a key an
entry does not define is ignored, ``null`` reads as empty, one label given
bare reads as a list of it, and an entry that cannot be read is dropped
without failing the episode.
"""

from __future__ import annotations

__all__ = [
    "CHAT_PROFILE_CUES",
    "CellCues",
    "CueConsumers",
    "ProfileCue",
    "cue_block",
    "map_cues",
    "readable_cues",
]

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from everalgo.llm import StrictOutputModel
from everalgo.types import ChatMessage
from pydantic import Field, ValidationError, model_validator

from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Counter as MetricCounter
from everos.memory.ports import PersonCues
from everos.memory.prompts import PromptTemplate

if TYPE_CHECKING:
    from everos.memory.extract.pipeline.pictures import Picture
    from everos.memory.models import MemCell

logger = get_logger(__name__)

_cue_entries = MetricCounter(
    name="profile_cue_entries_total",
    description=(
        "Profile cue entries episode replies held, by outcome: mapped to a "
        "message or picture of their speaker, or dropped (they mapped to "
        "nothing, or could not be read)."
    ),
    labelnames=("outcome",),
    namespace="everos",
)

_CUE_RULES = """\
Also mark what each person's profile should read. A profile is a portrait of a
person, not a log of what happened: nothing enters it because it occurred, only
because it is still true the next time you meet this person.
- In "profile_cues", add one entry for each message in which a human speaker
  states something about themselves that will still hold later: their role, the
  constraints they work under, their environment, the people and pets in their
  life, the preferences they voiced, their habits. "I'm tired today" or "I'm in
  a hurry right now" was genuinely said and does not belong.
- Leave out one-off events, other people's opinions, jokes and hypotheticals.
  An assistant's messages are never cues.
- "speaker" names the speaker exactly as the conversation does; "quote" copies
  a short part of that speaker's own message, word for word."""

_CUE_PICTURES = """\
- "pictures" lists the labels of the pictures that speaker posted which show
  something lasting about them -- a pet, a home, a hobby, a workplace, a
  possession: a picture is evidence about its poster like a sentence they
  wrote. An entry may name pictures only, its "quote" empty."""

_CUE_EMPTY = """\
- An empty list is the right answer when the conversation says nothing lasting
  about anyone, which is common."""

_CUE_REPLY_TEXT = """\
The JSON object carries a third field, "profile_cues":
{"title": "...", "content": "...", "profile_cues": [{"speaker": "...", \
"quote": "..."}]}"""

_CUE_REPLY_PICTURES = """\
The JSON object carries a fourth field, "profile_cues":
{"title": "...", "content": "...", "images": [...], "profile_cues": \
[{"speaker": "...", "quote": "...", "pictures": ["img-1"]}]}"""

CHAT_PROFILE_CUES = PromptTemplate(
    id="chat.episode_profile_cues",
    version=1,
    parts=(
        # 0: what a cue is; 1: the pictures a cue names (picture cells);
        # 2: an empty list is an answer; 3 / 4: the reply field, for a cell
        # without pictures / with them.
        _CUE_RULES,
        _CUE_PICTURES,
        _CUE_EMPTY,
        _CUE_REPLY_TEXT,
        _CUE_REPLY_PICTURES,
    ),
    variants=(("text", (0, 2, 3)), ("pictures", (0, 1, 2, 4))),
)
"""The ``<profile_cues>`` block appended to ``chat.episode`` when a profile of
the cell's space reads cues (:func:`cue_block`): ``text`` for a cell without
pictures, ``pictures`` for one with them."""


def cue_block(*, pictures: bool) -> str:
    """The ``<profile_cues>`` block's text for a cell with or without pictures."""
    parts = dict(CHAT_PROFILE_CUES.variants)["pictures" if pictures else "text"]
    return "\n".join(CHAT_PROFILE_CUES.parts[i] for i in parts)


@dataclass(frozen=True)
class CueConsumers:
    """Which spaces' profiles read the episode call's cues.

    Decided by the bootstrap from the profile strategies it registers and
    their settings, and given to the conversation pipeline: a cell of a space
    whose profile reads no cue is sent the episode request without the
    ``<profile_cues>`` block.

    Attributes:
        users: A user's own space: ``extract_user_profile`` is enabled and
            ``[profile] input = "cues"``.
        groups: A group's space: ``update_member_profiles`` is enabled in
            ``[groups] member_profile_mode = "batched"`` and ``[profile]
            input = "cues"``.
    """

    users: bool = False
    groups: bool = False


class ProfileCue(StrictOutputModel):
    """One message, or picture, that says something lasting about its speaker.

    Attributes:
        speaker: The speaker as the conversation names them.
        quote: A short verbatim part of one of that speaker's own messages;
            empty for an entry that names pictures only.
        pictures: Labels of pictures that speaker posted (``img-1``).
    """

    speaker: str
    quote: str = ""
    pictures: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _read_leniently(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        known = {k: v for k, v in value.items() if k in cls.model_fields}
        quote = known.get("quote")
        known["quote"] = "" if quote is None else str(quote)
        pictures = known.get("pictures")
        if pictures is None:
            known["pictures"] = []
        elif isinstance(pictures, str):
            known["pictures"] = [pictures]
        elif isinstance(pictures, list):
            known["pictures"] = [str(p) for p in pictures if isinstance(p, str | int)]
        return known


def readable_cues(value: object) -> list[Any]:
    """The entries of a reply's ``profile_cues`` that can be read; the rest dropped.

    For a ``field_validator(mode="before")``: ``null`` reads as no entry and
    one entry given bare as a list of it. An entry that is not an object with
    a non-blank speaker is dropped (logged at debug level), so a malformed cue
    never fails the episode it rides on.
    """
    if value is None:
        return []
    entries = [value] if isinstance(value, dict) else value
    if not isinstance(entries, list):
        return []
    kept: list[Any] = []
    for entry in entries:
        try:
            cue = ProfileCue.model_validate(entry)
        except ValidationError:
            continue
        if cue.speaker.strip():
            kept.append(entry)
    if len(kept) < len(entries):
        logger.debug(
            "memory.episode.profile_cues_unreadable", dropped=len(entries) - len(kept)
        )
    return kept


@dataclass(frozen=True)
class CellCues:
    """One cell's cues, mapped to its messages and pictures.

    Attributes:
        by_person: One record per human speaker of the cell (``sender_id``),
            an empty one when nothing was marked for them.
        given: Entries the reply held that could be read.
        mapped: Of those, entries that mapped to a message or a picture.
        dropped: Entries that mapped to nothing.
    """

    by_person: dict[str, PersonCues]
    given: int = 0
    mapped: int = 0
    dropped: int = 0


_QUOTE_MARKS = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "\uff07": "'",  # fullwidth apostrophe
        "`": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "″": '"',
        "«": '"',
        "»": '"',
        "\uff02": '"',  # fullwidth quotation mark
    }
)
"""Every quotation mark read as a plain one, so a quote a model retyped with
curly marks still finds its message."""

_ELLIPSIS = re.compile(r"^(?:\.\.\.|…)+|(?:\.\.\.|…)+$")


def _normal(text: str) -> str:
    """``text`` compared as a quote is: no case, whitespace runs as one space,
    every quotation mark a plain one."""
    return " ".join(text.translate(_QUOTE_MARKS).casefold().split())


def _bare_quote(quote: str) -> str:
    """The quote without the marks and ellipses a model wraps it in."""
    text = _normal(quote)
    for _ in range(2):
        text = _ELLIPSIS.sub("", text.strip()).strip().strip("'\"").strip()
    return text


def map_cues(
    cues: Sequence[ProfileCue],
    cell: MemCell,
    *,
    people: Sequence[str],
    pictures: Sequence[Picture],
) -> CellCues:
    """The entries of one reply as records of the cell's human speakers.

    Args:
        cues: The reply's readable entries.
        cell: The cell as the model was shown it (a group's speakers carry
            their roster labels): a speaker is matched as it was rendered.
        people: The cell's human speakers, by ``sender_id``; only their
            messages and pictures are ever mapped, and each gets a record.
        pictures: The pictures the call attached, with their posters.
    """
    eligible = set(people)
    by_speaker: dict[str, list[ChatMessage]] = {}
    for item in cell.items:
        if not isinstance(item, ChatMessage) or item.sender_id not in eligible:
            continue
        if not item.content:
            continue
        name = _normal(item.sender_name or item.sender_id)
        by_speaker.setdefault(name, []).append(item)
    order = {
        item.id: index
        for index, item in enumerate(cell.items)
        if isinstance(item, ChatMessage)
    }
    labels = {picture.label: picture for picture in pictures}
    messages: dict[str, list[str]] = {person: [] for person in people}
    images: dict[str, list[str]] = {person: [] for person in people}
    mapped = dropped = 0
    for cue in cues:
        found = _map_one(cue, by_speaker, labels)
        if found is None:
            dropped += 1
            continue
        person, message_id, image_ids = found
        mapped += 1
        if message_id is not None and message_id not in messages[person]:
            messages[person].append(message_id)
        for image_id in image_ids:
            if image_id not in images[person]:
                images[person].append(image_id)
    if mapped:
        _cue_entries.labels(outcome="mapped").inc(mapped)
    if dropped:
        _cue_entries.labels(outcome="dropped").inc(dropped)
    image_order = {picture.image_id: n for n, picture in enumerate(pictures)}
    return CellCues(
        by_person={
            person: PersonCues(
                message_ids=tuple(
                    sorted(messages[person], key=lambda m: order.get(m, 0))
                ),
                image_ids=tuple(
                    sorted(images[person], key=lambda i: image_order.get(i, 0))
                ),
            )
            for person in people
        },
        given=len(cues),
        mapped=mapped,
        dropped=dropped,
    )


def _map_one(
    cue: ProfileCue,
    by_speaker: Mapping[str, Sequence[ChatMessage]],
    labels: Mapping[str, Picture],
) -> tuple[str, str | None, list[str]] | None:
    """``(person, message id, picture ids)`` an entry names; ``None`` for nothing."""
    candidates = by_speaker.get(_normal(cue.speaker), ())
    if not candidates:
        return None
    senders = {message.sender_id for message in candidates}
    quote = _bare_quote(cue.quote)
    message = (
        next((m for m in candidates if quote in _normal(m.content)), None)
        if quote
        else None
    )
    person = message.sender_id if message is not None else None
    image_ids: list[str] = []
    for label in cue.pictures:
        picture = labels.get(label.strip())
        if picture is None or picture.poster not in senders:
            continue
        if person is None:
            person = picture.poster
        if picture.poster == person and picture.image_id not in image_ids:
            image_ids.append(picture.image_id)
    if person is None:
        return None
    return person, (message.id if message is not None else None), image_ids
