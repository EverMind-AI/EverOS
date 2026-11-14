"""The episode call -- one prompt, assembled by rule, one call path for every chat cell.

A chat cell, one-to-one or group, with pictures or without, is remembered by
one model call that returns its episode and, when the cell carries pictures,
one record per picture. Every such call sends the one registered prompt,
:data:`CHAT_EPISODE` (``chat.episode``), and its core is the original episode
request, everalgo-user-memory 0.3.1's, byte for byte, as EverOS 1.2.0 sent it:
one user message holding the whole-memcell prompt with the conversation start
time, one pseudo-JSON block per message, and the operator's custom
instructions in their slot -- the original's five default principles when
none are set (everalgo ``episode._render._render_original_prompt``). What a
cell adds follows the original's text, in this order, after one blank line:

* **A group cell** -- its speakers already labelled by their roster names in
  the messages (:mod:`.group_render`) -- adds the roster block, in a
  ``<group_chat>`` block.
* **A cell with pictures** adds the ``<pictures>`` block: the picture rules,
  the ``images`` field the reply carries beside ``title`` and ``content``, and
  the labels in order. Each picture then follows the text as an image part
  after a ``<picture>`` element naming its label.
* **Every cell** ends with the ``<output_language>`` block naming the
  conversation's language.

The reply is ``title`` and ``content`` (and ``images`` with pictures); the
summary is the content's first 200 characters, cut in code, as the original
filled it. The two renderings are the template's variants ``text`` and
``pictures``, each with its own fingerprint. A picture cannot travel through
``EpisodeExtractor`` (its message content is text by contract), which is why
the call is EverOS's own.

History. Until 2026-11-12 a picture cell's prompt carried everalgo's whole
system prompt -- its reply format and an empty language block included --
plus the picture rules, a second language block and a second reply format.
Version 1 of ``chat.episode`` (2026-11-12) sent the picture block to every
cell; two of 43 LoCoMo text cells then wrote picture records with nothing
attached, and text cells paid for 769 prompt tokens they never used. Version 2
assembled the prompt by rule around everalgo's rewritten rules (2026-10).
Those rules forbid copying a message's time into the content: with
GPT-4.1-mini only 1% of LoCoMo episodes carried a date, and temporal
questions fell to 53% against 90% on the stores the original built. Version 3
(2026-11-14) sent text cells the original again, version 4 every cell.

Two rules keep a picture's record and its episode apart:

- A picture's record describes **only what the picture shows**. Anything the
  conversation already says belongs to the episode; duplicating it would
  create two memories that drift apart.
- The episode body **refers to** a picture by its label and never restates
  its description, so replacing a description, or forgetting a picture,
  costs one entry, not a rewrite of the episode that mentions it.

What a picture says about its sender reaches the profile through the profile
update, which reads the segment's pictures itself
(:func:`~everos.memory.extract.pipeline.pictures.profile_pictures`); this
call writes no profile item.

The reply is read leniently in form and strictly in meaning
(:mod:`everos.memory.extract.reply_form`): a top-level key the reply model does
not define is ignored, an unknown key left empty in a picture entry reads as
absent, ``null`` reads as empty for ``ocr`` and as no entries for ``images``
(one entry given bare reads as a list of it), and a reply with an array
element left open or split in two objects is mended before it is validated. A
cell is never given a picture it did not send: a text cell's reply loses any
``images`` it writes, and a reference ``(img-N)`` to a label the cell did not
attach is taken out of the episode. A reply that leaves ``title`` or
``content`` blank is rejected (the guard ``EpisodeExtractor`` applies,
imported from everalgo so the two cannot drift apart); any other top-level key
it writes -- a ``summary``, or a key of the model's own -- is ignored, as the
original read its reply. The reply models' names
and docstrings are part of the JSON schema a ``json_schema``-mode call sends.

The call's output is capped by what it reads (:func:`reply_token_cap`), and a
reply cut at that cap is sampled again once, warmer, before the cell fails:
at temperature 0 a long transcription in ``ocr`` can repeat itself until the
output limit (on 2026-11-11 two or three of MemEye's 168 picture calls did,
in each of three builds, ran to the builds' 32,768-token cap and left their
cells with no memory at all).
"""

from __future__ import annotations

__all__ = [
    "CHAT_EPISODE",
    "REPLY_CHARS_PER_TOKEN",
    "REPLY_TOKENS_BASE",
    "REPLY_TOKENS_CEILING",
    "REPLY_TOKENS_PER_PICTURE",
    "EpisodeReply",
    "ImageEpisodeReply",
    "ImageFact",
    "build_messages",
    "build_original_prompt",
    "extract",
    "extract_cell",
    "reply_token_cap",
    "to_episode",
]

import json
import re
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Self

from everalgo.llm import (
    LLMMessage,
    StrictOutputModel,
    image_url_part_from_bytes,
    parse_reply,
)
from everalgo.types import Episode as AlgoEpisode
from everalgo.types import MemCell
from everalgo.user_memory._language import (
    PARTICIPANT_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everalgo.user_memory.episode._render import (
    _original_conversation,
    _render_original_prompt,
)
from everalgo.user_memory.episode._replies import _check_fields_written
from everalgo.user_memory.episode._summary import _original_summary
from everalgo.user_memory.prompts.en.episode_original import (
    DEFAULT_CUSTOM_INSTRUCTIONS,
    EPISODE_GENERATION_PROMPT,
)
from everalgo.utils import paragraphs, tagged
from pydantic import Field, ValidationError, field_validator, model_validator

from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Counter as MetricCounter
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.extract.reply_form import drop_empty_unknown_keys, repair_json_form
from everos.memory.prompts import PromptTemplate

if TYPE_CHECKING:
    from everos.memory.extract.pipeline.pictures import Picture

logger = get_logger(__name__)

_ignored_keys = MetricCounter(
    name="episode_reply_keys_ignored_total",
    description=(
        "Top-level keys an episode reply carried beyond its model's fields, "
        "dropped unread, by reply model."
    ),
    labelnames=("reply",),
    namespace="everos",
)


_PICTURE_RULES = """\
This message carries pictures: they follow this text, each image after a
<picture> element naming its label. In the conversation content a picture is
an [IMAGE: ...] reference where it was sent, and the labels follow those
references in order: the first is img-1, the second img-2, and so on.
- Write the episode as you would without the pictures: the conversation is the
  record.
- In "content", refer to a picture by its label — "the user sent a photo of a
  latte (img-1)" — and never restate what the picture shows. The picture has its
  own memory; the episode only points at it.
- This holds for every fact you can only know by looking: a year, a price, a
  name or brand, an address, a menu line, a label on an object. None of them
  belongs in "content". Say "a photo of the shop's signboard
  (img-2)" — not what the signboard says. The reason is exact: forgetting that
  picture must remove its content from the memory, and text you copied into the
  episode is no longer the picture's.
- For every label, add one entry to "images" describing what the picture itself
  shows: the subject, the setting, the people visible, and the one detail that
  tells this picture apart from another. The description must be readable on its
  own, months later, by someone who cannot see the picture.
- In each entry's "ocr", copy only text that is actually visible in the picture:
  signs, labels, headlines, menu lines, buttons, table headers. Transcribe it,
  never translate it, and keep it short. Write an empty string when the picture
  carries no text. Text the conversation already quotes is not picture text.
- A picture the conversation does not mention still gets its own "images" entry.
- If two pictures show the same thing, describe how they differ, not the thing
  twice.
- Every picture you were given must be referenced in "content" as "(img-N)" at
  the place it was sent, even when you also describe it.

Worked example. Conversation:
  [2026-10-30 09:12 UTC (Friday)] Wang Lei: went to that cafe, here is my latte
  [IMAGE: img-1]
  [2026-10-30 09:15 UTC (Friday)] Wang Lei: here is the sign [IMAGE: img-2]
Pictures: img-1 a latte with heart-shaped foam; img-2 a chalkboard reading
"DOU SHE COFFEE / EST. 2019 / blend latte 22".
Right answer for "content": "On October 30, 2026 at 9:12 AM UTC, Wang Lei went
to a new cafe and shared a photo of her latte (img-1), then a photo of the
cafe's chalkboard sign (img-2)."
Wrong answer, and why: "she shared a photo of the sign, which says DOU SHE
COFFEE, EST. 2019, blend latte 22" - it copied the sign's text into the episode,
so the words would survive after the photo is forgotten. The sign's text belongs
in that picture's own "ocr", nowhere else."""

_PICTURE_REPLY_FORMAT = """\
The JSON object carries a third field, "images", one entry per picture:
{"title": "...", "content": "...", "images": [{"ref": "img-1", \
"description": "...", "ocr": "..."}]}"""

CHAT_EPISODE = PromptTemplate(
    id="chat.episode",
    version=4,
    parts=(
        # 0: the original request (everalgo-user-memory 0.3.1) every cell is
        # sent; 1: the default principles it fills its custom instructions
        # with when the operator sets none.
        EPISODE_GENERATION_PROMPT,
        DEFAULT_CUSTOM_INSTRUCTIONS,
        # 2: the picture rules; 3: the "images" field of the reply; 4: the
        # labels in order, the <pictures> block's last line; 5: the <picture>
        # element before each image.
        _PICTURE_RULES,
        _PICTURE_REPLY_FORMAT,
        "Labels in order: {labels}. Read every attached picture.",
        "label: {label}",
    ),
    variants=(("text", (0, 1)), ("pictures", (0, 1, 2, 3, 4, 5))),
)
"""The episode prompt of every chat cell, by rule (:func:`build_messages`): one
user message, the original (parts 0-1), then a group's ``<group_chat>`` block,
then -- in the ``pictures`` variant -- the ``<pictures>`` block (parts 2-4) and
each image after its ``<picture>`` element (part 5), and last the
``<output_language>`` block."""

REPLY_TOKENS_BASE = 2_048
"""Output budget of an episode call before its pictures and its conversation
(:func:`reply_token_cap`)."""

REPLY_TOKENS_PER_PICTURE = 1_024
"""Output budget each attached picture adds (:func:`reply_token_cap`)."""

REPLY_CHARS_PER_TOKEN = 4
"""Characters of rendered conversation per token of output budget
(:func:`reply_token_cap`)."""

REPLY_TOKENS_CEILING = 32_768
"""The cap never exceeds this: the output limit the 2.0 store builds set."""

_LABEL_REFERENCE = re.compile(r"\s*(?:\((img-\d+)\)|\[IMAGE: (img-\d+)\])")
"""A picture reference by label, "(img-1)" or "[IMAGE: img-1]", with the space
before it."""


def reply_token_cap(*, pictures: int, conversation_chars: int) -> int:
    """``max_tokens`` of an episode call reading ``pictures`` and its conversation.

    ``2,048``, plus ``1,024`` per picture, plus one token per four characters
    of the rendered ``<conversation>``, at most :data:`REPLY_TOKENS_CEILING`.

    A reply that reaches the cap is a runaway (a transcription repeating
    itself), not a long answer. Picture replies grow with their pictures: of
    the 914 complete ones DeepSeek V4.1 Flash wrote on 2026-11-11 (MemEye,
    CUE-Mem and MemLens, temperatures 0 and 1) the longest was 2,423 tokens,
    for 8 pictures, and 1,486 for one picture. Text replies grow with their
    conversation: of the 78,597 complete text-episode replies in the build
    logs of 2026-10-24 to 11-11 (PersonaMem, Time-Dialog, LoCoMo, MemEye,
    MemLens, CUE-Mem, and a 10,000-word audio transcript) none was cut, and
    the longest, 2,613 tokens, came from a 24,120-character transcript cell
    (cap 8,078) -- three of them, all transcript cells, were above 2,048, the
    cap a cell without pictures would get from pictures alone. Without a cap a
    runaway ran to the builds' 32,768 tokens, 57 times the average picture
    reply's output. A ``max_tokens`` the deployment sets lower still wins
    (``[llm] extra``).
    """
    cap = (
        REPLY_TOKENS_BASE
        + REPLY_TOKENS_PER_PICTURE * pictures
        + conversation_chars // REPLY_CHARS_PER_TOKEN
    )
    return min(cap, REPLY_TOKENS_CEILING)


class ImageFact(StrictOutputModel):
    """One picture, as the model described it while looking at it.

    Attributes:
        ref: The picture's label in the conversation text (``img-1``), which is
            how the episode body refers to it.
        description: What the picture shows — who, where, which object, which
            detail distinguishes this picture from any other.
        ocr: Text visible in the picture, verbatim but bounded. Empty when the
            picture holds none, which is the usual case for a photograph.
    """

    ref: str
    description: str
    ocr: str = ""
    """Text visible in the picture (signs, labels, buttons, menu lines).
    Empty for a picture with no text in it — a photograph of a scene, which
    is the common case rather than a defect."""

    @model_validator(mode="before")
    @classmethod
    def _empty_unknown_keys_read_as_absent(cls, value: object) -> object:
        return drop_empty_unknown_keys(cls, value)

    @field_validator("ocr", mode="before")
    @classmethod
    def _null_is_empty(cls, value: object) -> object:
        return "" if value is None else value


class _LenientReply(StrictOutputModel):
    """How every episode reply is read: lenient in form, strict in meaning.

    Only the fields the reply model defines are read. Any other top-level
    key, empty or not -- a ``summary`` included, since the summary is cut in
    code -- is dropped, as everalgo-user-memory 0.3.1 read the original's
    reply (it required ``title`` and ``content`` and ignored the rest): on
    the LoCoMo probe of 2026-11-14 DeepSeek V4.1 Flash added keys of its own
    (``type``, ``_meta``, ``_comment``, ``emotional_tone``,
    ``_time_resolution``, ...) to 123 of its 127 replies. The names are
    logged at debug level and counted. A picture entry is still read
    strictly (:class:`ImageFact`).
    """

    @model_validator(mode="before")
    @classmethod
    def _read_only_known_keys(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        known = set(cls.model_fields) | {
            field.alias for field in cls.model_fields.values() if field.alias
        }
        ignored = sorted(key for key in value if key not in known)
        if not ignored:
            return value
        _ignored_keys.labels(reply=cls.__name__).inc(len(ignored))
        logger.debug(
            "memory.episode.reply_keys_ignored", reply=cls.__name__, keys=ignored
        )
        return {key: item for key, item in value.items() if key in known}

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: Any
    ) -> Self:
        """Validate a reply, mending its JSON form first when that is all that fails.

        The provider checks a reply and the operator parses it through this
        method, so both read the same mended text. Only a reply that is not
        valid JSON is mended (:func:`repair_json_form`); one that parses but
        breaks the model is not touched, and a slip the repair does not know
        keeps its original error.
        """
        try:
            return super().model_validate_json(json_data, **kwargs)
        except ValidationError as err:
            if any(error["type"] != "json_invalid" for error in err.errors()):
                raise
            text = (
                json_data if isinstance(json_data, str) else bytes(json_data).decode()
            )
            mended = repair_json_form(text)
            if mended is None:
                raise
            return super().model_validate_json(mended, **kwargs)


class EpisodeReply(_LenientReply):
    """The episode of a cell without pictures: title and content."""

    title: str
    content: str


class ImageEpisodeReply(_LenientReply):
    """The one reply that carries the episode and every picture's memory."""

    title: str
    content: str
    images: list[ImageFact] = Field(default_factory=list)

    @field_validator("images", mode="before")
    @classmethod
    def _one_or_none_reads_as_a_list(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value


def build_original_prompt(
    memcell: MemCell,
    *,
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
    roster: str | None = None,
    labels: Sequence[str] = (),
) -> str:
    """A cell's prompt text: the original, then what this cell adds.

    everalgo-user-memory 0.3.1's request byte for byte, as EverOS 1.2.0 sent
    it (the operator's custom instructions in their slot, or the five default
    principles when none are set), then one blank line and the blocks that
    follow it: a group's roster block in ``<group_chat>``; with pictures, the
    ``<pictures>`` block (the picture rules, the ``images`` field, the labels
    in order); then the ``<output_language>`` block every cell is sent -- the
    conversation's language, resolved in code, or the participant rule when
    none is.

    Args:
        memcell: The cell being extracted.
        custom_instructions: The operator's instructions.
        output_language: The language to write in.
        roster: A group cell's roster block; ``None`` for a user's own cell.
        labels: The labels of the pictures the cell attaches, in order.
    """
    original = _render_original_prompt(memcell, custom_instructions=custom_instructions)
    blocks = [tagged("group_chat", roster)] if roster is not None else []
    if labels:
        rules, reply_field, labels_line = CHAT_EPISODE.parts[2:5]
        names = labels_line.format(labels=", ".join(labels))
        blocks.append(tagged("pictures", paragraphs(rules, reply_field, names)))
    language = build_language_rule(output_language, fallback=PARTICIPANT_LANGUAGE_RULE)
    blocks.append(tagged("output_language", language))
    return f"{original}\n{paragraphs(*blocks)}"


def build_messages(
    memcell: MemCell,
    *,
    pictures: Sequence[tuple[str, bytes, str]],
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
    roster: str | None = None,
    detail: str = "high",
) -> list[LLMMessage]:
    """The call's one user message: the prompt text, then each picture after its label.

    Args:
        memcell: The cell being extracted.
        pictures: ``(label, bytes, mime)`` per picture, in conversation order.
            The label is the one the reply refers to the picture by
            (``img-1``). Empty for a cell without pictures, whose message is
            the prompt text alone.
        custom_instructions: The host's instructions.
        output_language: Language to write in. The pipeline names the
            conversation's, resolved in code; ``None`` leaves the model to
            follow the speakers, which it does not reliably do on a picture
            cell (whole replies drift into Chinese).
        roster: A group cell's roster block (:func:`.group_render.roster_block`);
            ``None`` for a user's own cell.
        detail: ``detail`` hint on each image part. ``high`` because this call
            has to read small text out of a picture; the decider's
            thumbnail-only default is set separately.
    """
    prompt = build_original_prompt(
        memcell,
        custom_instructions=custom_instructions,
        output_language=output_language,
        roster=roster,
        labels=[label for label, _, _ in pictures],
    )
    if not pictures:
        return [LLMMessage(role="user", content=prompt)]
    label_line = CHAT_EPISODE.parts[5]
    parts: list[Any] = [{"type": "text", "text": prompt}]
    for label, data, mime in pictures:
        parts.append(
            {"type": "text", "text": tagged("picture", label_line.format(label=label))}
        )
        part = image_url_part_from_bytes(data, mime)
        if part.image_url.detail is None and detail:
            part = part.model_copy(
                update={
                    "image_url": part.image_url.model_copy(update={"detail": detail})
                }
            )
        parts.append(part.model_dump())
    return [LLMMessage(role="user", content=parts)]


def _without_unattached(text: str, attached: Collection[str]) -> tuple[str, int]:
    """``text`` without its references to labels not in ``attached``, and how many."""
    removed = 0

    def drop(match: re.Match[str]) -> str:
        nonlocal removed
        if (match.group(1) or match.group(2)) in attached:
            return match.group(0)
        removed += 1
        return ""

    return _LABEL_REFERENCE.sub(drop, text), removed


def to_episode(
    reply: EpisodeReply | ImageEpisodeReply,
    memcell: MemCell,
    *,
    labels: Mapping[str, str] | None = None,
) -> AlgoEpisode:
    """The algo Episode the reply describes.

    The mapping the original extractor applied (``title`` → ``subject``,
    ``content`` → ``episode`` verbatim, ``timestamp`` from the cell), so an
    episode reads the same downstream whichever cell it came from. The model
    writes no summary: it is the content's first 200 characters, as the
    original filled it, taken before the labels gain their ids.

    ``owner_id`` is ``None``: the episode is extracted once for the whole
    cell and fanned out to one md per user sender by the caller, which is
    what keeps a cell at one model call.

    ``labels`` maps each per-cell label the cell attached (``img-1``) to the
    picture's durable id — its content hash. The body keeps the label as the
    model wrote it *and* gains the id, so an episode reads naturally while
    still naming the picture a forget or a result can address: the label means
    nothing outside this cell, and the hash is the only thing that survives it.
    A reference to a label the cell did not attach names no picture and is
    taken out of the content and the title -- every reference, on a cell
    without pictures.

    A label the reply never referenced is appended as a reference of its own.
    The prompt asks for one, but a model that omits it would otherwise leave a
    picture memory with nothing in the episode pointing at it — and the forget
    path finds what to rewrite *through* that reference, so the omission would
    make the picture's content survive in the episode body for ever.
    """
    attached = dict(labels or {})
    body, _ = _without_unattached(reply.content, attached)
    summary = _original_summary(body)
    title, _ = _without_unattached(reply.title, attached)
    if attached:
        for label, image_id in sorted(attached.items(), key=lambda kv: -len(kv[0])):
            body = body.replace(f"({label})", f"({label}: {image_id})")
            body = body.replace(f"[IMAGE: {label}]", f"[IMAGE: {image_id}]")
        missing = [
            f"({label}: {image_id})"
            for label, image_id in sorted(attached.items())
            if f"({label}: {image_id})" not in body
        ]
        if missing:
            body = f"{body.rstrip()} {' '.join(missing)}"
    return AlgoEpisode(
        owner_id=None,
        episode=body,
        subject=title,
        summary=summary,
        timestamp=memcell.timestamp,
    )


def _picture_entries_written(text: str) -> int:
    """How many ``images`` entries a reply's text holds; 0 when unreadable."""
    value: object = None
    for candidate in (text, repair_json_form(text)):
        if candidate is None:
            continue
        try:
            value = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(value, dict):
        return 0
    images = value.get("images")
    return len(images) if isinstance(images, list) else int(isinstance(images, dict))


async def extract(
    llm: Any,
    memcell: MemCell,
    *,
    pictures: Sequence[tuple[str, bytes, str]],
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
    max_tokens: int | None = None,
    roster: str | None = None,
) -> EpisodeReply | ImageEpisodeReply:
    """One call: the episode and every picture's memory, from the same reading.

    A cell without pictures asks for :class:`EpisodeReply`, one with pictures
    for :class:`ImageEpisodeReply`. The call asks for at most ``max_tokens``
    output tokens, by default :func:`reply_token_cap` of its pictures and
    conversation, and a reply cut there is retried like an invalid one
    (``retry_cut``): once more by default, sampled at
    ``[llm] retry_temperature``, with the same cap. The first attempt keeps
    the configured temperature.

    Raises:
        everalgo.llm.LLMOutputError: The reply leaves ``content`` or
            ``title`` blank.
        everos.component.llm.errors.LLMCallError: The upstream failed, no
            attempt produced a reply that validates, or every attempt was cut
            at the cap (``cut_short``).
    """
    messages = build_messages(
        memcell,
        pictures=pictures,
        custom_instructions=custom_instructions,
        output_language=output_language,
        roster=roster,
    )
    conversation = _original_conversation(memcell)
    cap = (
        reply_token_cap(pictures=len(pictures), conversation_chars=len(conversation))
        if max_tokens is None
        else max_tokens
    )
    model: type[EpisodeReply | ImageEpisodeReply] = (
        ImageEpisodeReply if pictures else EpisodeReply
    )
    text = await llm.achat(
        messages, response_format=model, max_tokens=cap, retry_cut=True
    )
    reply = parse_reply(model, text, operator="episode")
    _check_fields_written(reply, ("content", "title"), operator="episode")
    dropped = 0 if pictures else _picture_entries_written(text)
    if dropped:
        logger.warning("memory.episode.unattached_pictures_dropped", entries=dropped)
    logger.info(
        "memory.episode.extracted",
        pictures=len(pictures),
        images=len(reply.images) if isinstance(reply, ImageEpisodeReply) else 0,
        reply_chars=len(text),
        max_tokens=cap,
    )
    return reply


async def extract_cell(
    llm: Any,
    memcell: MemCell,
    *,
    pictures: Sequence[Picture],
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
    memcell_id: str,
    roster: str | None = None,
) -> tuple[AlgoEpisode, dict[str, tuple[str, str]]]:
    """The cell's episode and each picture's ``(description, ocr)`` by label.

    One :func:`extract` call, tried again by
    :func:`~everos.memory.extract.llm_retry.retry_llm_output` while its reply
    is unusable. The two results are readings of the same call and are never
    split. Only a picture the cell attached is described or referenced.
    ``roster`` is a group cell's roster block (:func:`build_messages`).
    """
    sent = [(picture.label, picture.data, picture.mime) for picture in pictures]
    by_label = {picture.label: picture.image_id for picture in pictures}

    async def attempt() -> tuple[AlgoEpisode, dict[str, tuple[str, str]]]:
        reply = await extract(
            llm,
            memcell,
            pictures=sent,
            custom_instructions=custom_instructions,
            output_language=output_language,
            roster=roster,
        )
        described = (
            {
                fact.ref: (fact.description, fact.ocr)
                for fact in reply.images
                if fact.ref in by_label
            }
            if isinstance(reply, ImageEpisodeReply)
            else {}
        )
        unattached = sum(
            _without_unattached(text, by_label)[1]
            for text in (reply.content, reply.title)
        )
        if unattached:
            logger.warning(
                "memory.episode.unattached_references_dropped",
                references=unattached,
                pictures=len(pictures),
                memcell_id=memcell_id,
            )
        return to_episode(reply, memcell, labels=by_label), described

    return await retry_llm_output(
        attempt, operator="episode", memcell_id=memcell_id, pictures=len(pictures)
    )
