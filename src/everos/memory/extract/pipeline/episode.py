"""The episode call -- one prompt, one call and one reader for every chat cell.

A chat cell, one-to-one or group, with pictures or without, is remembered by
one model call that returns its episode and, when the cell carries pictures,
one record per picture. Every such call sends the same prompt, registered as
:data:`CHAT_EPISODE` (``chat.episode``). Its system message, in order:

1. **everalgo's episode rules, verbatim** (:data:`EPISODE_RULES`):
   ``EPISODE_SYSTEM_PROMPT`` up to its ``<reply_format>`` block -- the role,
   what to write, staying faithful, details, concision, time references and
   the two worked examples. They are imported, never restated, so a cell is
   held to exactly the rules everalgo's ``EpisodeExtractor`` holds a text
   cell to.
2. **the picture rules**, a ``<pictures>`` block that applies only when the
   user message carries pictures. A cell without them is written by the
   rules above alone.
3. **one reply format**: ``content``, ``summary`` and ``title`` as everalgo
   asks for them, then ``images`` -- one entry per attached picture, an empty
   list when none is attached.
4. **the language block**, after every static block, and last the caller's
   custom instructions when there are any (the host's text; for a group
   segment, the roster block). Everything before the language block is the
   same for every cell, so a provider's prefix cache shares it across text
   and picture cells, one-to-one and group alike.

The user message is the cell's ``<conversation>``, rendered as everalgo
renders it; a cell with pictures adds a ``<pictures>`` element naming their
labels, then each picture, after a ``<picture>`` element with its label. A
picture cannot travel through ``EpisodeExtractor`` (its message content is
text by contract), which is why the call is EverOS's own.

Before 2026-11-12 a text cell went through ``EpisodeExtractor`` and a picture
cell through a prompt of this module's that carried everalgo's whole system
prompt -- its reply format and an empty language block included -- plus a
second language block and a second reply format.

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
(:mod:`everos.memory.extract.reply_form`): an unknown key left empty reads as
absent, ``null`` reads as empty for ``ocr`` and as no entries for ``images``
(one entry given bare reads as a list of it, and a text cell may leave
``images`` out), and a reply with an array element left open or split in two
objects is mended before it is validated. A reply that leaves ``content``,
``summary`` or ``title`` blank is rejected, and a ``summary`` wider than 400
units is cut at a sentence boundary: the two guards ``EpisodeExtractor``
applies, imported from everalgo so the two cannot drift apart. The reply
models' docstrings are part of the JSON schema a ``json_schema``-mode call
sends.

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
    "EPISODE_RULES",
    "REPLY_CHARS_PER_TOKEN",
    "REPLY_TOKENS_BASE",
    "REPLY_TOKENS_CEILING",
    "REPLY_TOKENS_PER_PICTURE",
    "EpisodeReply",
    "ImageFact",
    "build_messages",
    "build_system_prompt",
    "extract",
    "extract_cell",
    "render_conversation",
    "reply_token_cap",
    "to_episode",
]

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

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
from everalgo.user_memory.episode._replies import _check_fields_written
from everalgo.user_memory.episode._summary import _cap_summary
from everalgo.user_memory.prompts.en.episode import EPISODE_SYSTEM_PROMPT
from everalgo.utils import (
    append_custom_instructions,
    paragraphs,
    render_chat_line,
    tagged,
)
from pydantic import Field, ValidationError, field_validator, model_validator

from everos.core.observability.logging import get_logger
from everos.memory.extract.llm_retry import retry_llm_output
from everos.memory.extract.reply_form import drop_empty_unknown_keys, repair_json_form
from everos.memory.prompts import PromptTemplate

if TYPE_CHECKING:
    from everos.memory.extract.pipeline.pictures import Picture

logger = get_logger(__name__)


def _without_reply_format(system: str) -> str:
    """``system`` without its last block, which must be its ``<reply_format>``.

    Raises:
        RuntimeError: The prompt does not end with exactly one such block --
            everalgo changed its layout, and :data:`CHAT_EPISODE` needs review.
    """
    rules, opening, reply_format = system.rpartition("\n\n<reply_format>\n")
    if not opening or reply_format.count("</reply_format>") != 1:
        raise RuntimeError(
            "everalgo's EPISODE_SYSTEM_PROMPT no longer ends with one "
            "<reply_format> block; review chat.episode"
        )
    if not reply_format.endswith("\n</reply_format>"):
        raise RuntimeError(
            "everalgo's EPISODE_SYSTEM_PROMPT has text after its <reply_format> "
            "block; review chat.episode"
        )
    return rules


EPISODE_RULES = _without_reply_format(EPISODE_SYSTEM_PROMPT)
"""everalgo's episode rules as it sends them, the role through the worked
examples: ``EPISODE_SYSTEM_PROMPT`` without its reply format."""

_PICTURE_RULES = """\
Apply these rules only when the user message carries pictures: a <pictures>
element after <conversation> listing their labels, then each image after a
<picture> element naming its label. In <conversation> a picture is an
[IMAGE: ...] reference where it was sent, and the labels follow those
references in order: the first is img-1, the second img-2, and so on.
- Write the episode as you would without the pictures: the conversation is the
  record.
- In "content", refer to a picture by its label — "the user sent a photo of a
  latte (img-1)" — and never restate what the picture shows. The picture has its
  own memory; the episode only points at it.
- This holds for every fact you can only know by looking: a year, a price, a
  name or brand, an address, a menu line, a label on an object. None of them
  belongs in "content" or "summary". Say "a photo of the shop's signboard
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
Right answer for "content": "Wang Lei went to a new cafe and shared a photo of
her latte (img-1), then a photo of the cafe's chalkboard sign (img-2)."
Wrong answer, and why: "she shared a photo of the sign, which says DOU SHE
COFFEE, EST. 2019, blend latte 22" - it copied the sign's text into the episode,
so the words would survive after the photo is forgotten. The sign's text belongs
in that picture's own "ocr", nowhere else."""

_REPLY_FORMAT = """\
Reply with one JSON object, nothing else.
{"content": "...", "summary": "...", "title": "...", "images": [{"ref": "img-1", \
"description": "...", "ocr": "..."}]}
"images" has one entry per attached picture, in label order; with no picture \
attached it is []."""

CHAT_EPISODE = PromptTemplate(
    id="chat.episode",
    version=1,
    parts=(
        # 0: everalgo's rules; 1: the picture rules; 2: the reply format.
        EPISODE_RULES,
        _PICTURE_RULES,
        _REPLY_FORMAT,
        # 3: the user message's <pictures> element; 4: the <picture> element
        # before each image.
        "Labels in order: {labels}. Read every attached picture.",
        "label: {label}",
    ),
)
"""The episode prompt of every chat cell. Its system message is parts 0-2,
each block tagged, then the ``<output_language>`` block and the caller's
``<custom_instructions>``; parts 3 and 4 are the user message's picture
elements, sent only with pictures."""

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


class EpisodeReply(StrictOutputModel):
    """The cell's episode, and one memory per picture attached (none without)."""

    content: str
    summary: str
    title: str
    images: list[ImageFact] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _empty_unknown_keys_read_as_absent(cls, value: object) -> object:
        return drop_empty_unknown_keys(cls, value)

    @field_validator("images", mode="before")
    @classmethod
    def _one_or_none_reads_as_a_list(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, dict):
            return [value]
        return value

    @classmethod
    def model_validate_json(
        cls, json_data: str | bytes | bytearray, **kwargs: Any
    ) -> EpisodeReply:
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


def build_system_prompt(
    *, custom_instructions: str | None, output_language: OutputLanguage | None
) -> str:
    """The system message: the static blocks, the language, the instructions.

    The same for every cell a host sends with the same language and custom
    instructions, pictures or not.
    """
    rules, pictures, reply_format, _, _ = CHAT_EPISODE.parts
    system = paragraphs(
        rules,
        tagged("pictures", pictures),
        tagged("reply_format", reply_format),
        tagged(
            "output_language",
            build_language_rule(output_language, fallback=PARTICIPANT_LANGUAGE_RULE),
        ),
    )
    return append_custom_instructions(system, custom_instructions)


def render_conversation(memcell: MemCell) -> str:
    """The cell as ``[time UTC (weekday)] speaker: text`` lines.

    Everalgo's episode rendering, byte for byte: the chat messages with text,
    in a ``<conversation count="N">`` element. A message that carried a
    picture keeps its ``[IMAGE: <id>]`` reference in its text, which ties the
    picture to its time and speaker.
    """
    lines = [
        render_chat_line(message)
        for message in memcell.items
        if getattr(message, "kind", None) == "text" and getattr(message, "content", "")
    ]
    return tagged("conversation", "\n".join(lines), count=str(len(lines)))


def build_messages(
    memcell: MemCell,
    *,
    pictures: Sequence[tuple[str, bytes, str]],
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
    detail: str = "high",
) -> list[LLMMessage]:
    """The ``[system, user]`` pair: the rendered cell, then any pictures.

    Args:
        memcell: The cell being extracted.
        pictures: ``(label, bytes, mime)`` per picture, in conversation order.
            The label is the one the reply refers to the picture by
            (``img-1``). Empty for a cell without pictures, whose user
            message is the conversation alone, as everalgo sends it.
        custom_instructions: The host's instructions.
        output_language: Language to write in. The pipeline names the
            conversation's, resolved in code; ``None`` leaves the model to
            follow the speakers, which it does not reliably do on a picture
            cell (whole replies drift into Chinese).
        detail: ``detail`` hint on each image part. ``high`` because this call
            has to read small text out of a picture; the decider's
            thumbnail-only default is set separately.
    """
    system = LLMMessage(
        role="system",
        content=build_system_prompt(
            custom_instructions=custom_instructions, output_language=output_language
        ),
    )
    conversation = render_conversation(memcell)
    if not pictures:
        return [system, LLMMessage(role="user", content=conversation)]
    _, _, _, labels_line, label_line = CHAT_EPISODE.parts
    names = ", ".join(label for label, _, _ in pictures)
    parts: list[Any] = [
        {
            "type": "text",
            "text": paragraphs(
                conversation, tagged("pictures", labels_line.format(labels=names))
            ),
        }
    ]
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
    return [system, LLMMessage(role="user", content=parts)]


def to_episode(
    reply: EpisodeReply,
    memcell: MemCell,
    *,
    labels: Mapping[str, str] | None = None,
) -> AlgoEpisode:
    """The algo Episode the reply describes.

    The mapping everalgo's extractor applies (``title`` → ``subject``,
    ``content`` → ``episode`` verbatim, ``summary`` through its width guard,
    ``timestamp`` from the cell), so an episode reads the same downstream
    whichever cell it came from.

    ``owner_id`` is ``None``: the episode is extracted once for the whole
    cell and fanned out to one md per user sender by the caller, which is
    what keeps a cell at one model call.

    ``labels`` maps each per-cell label the model used (``img-1``) to the
    picture's durable id — its content hash. The body keeps the label as the
    model wrote it *and* gains the id, so an episode reads naturally while
    still naming the picture a forget or a result can address: the label means
    nothing outside this cell, and the hash is the only thing that survives it.

    A label the reply never referenced is appended as a reference of its own.
    The prompt asks for one, but a model that omits it would otherwise leave a
    picture memory with nothing in the episode pointing at it — and the forget
    path finds what to rewrite *through* that reference, so the omission would
    make the picture's content survive in the episode body for ever.
    """
    body = reply.content
    if labels:
        for label, image_id in sorted(labels.items(), key=lambda kv: -len(kv[0])):
            body = body.replace(f"({label})", f"({label}: {image_id})")
            body = body.replace(f"[IMAGE: {label}]", f"[IMAGE: {image_id}]")
        missing = [
            f"({label}: {image_id})"
            for label, image_id in sorted(labels.items())
            if f"({label}: {image_id})" not in body
        ]
        if missing:
            body = f"{body.rstrip()} {' '.join(missing)}"
    return AlgoEpisode(
        owner_id=None,
        episode=body,
        subject=reply.title,
        summary=_cap_summary(reply.summary),
        timestamp=memcell.timestamp,
    )


async def extract(
    llm: Any,
    memcell: MemCell,
    *,
    pictures: Sequence[tuple[str, bytes, str]],
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
    max_tokens: int | None = None,
) -> EpisodeReply:
    """One call: the episode and every picture's memory, from the same reading.

    The call asks for at most ``max_tokens`` output tokens, by default
    :func:`reply_token_cap` of its pictures and conversation, and a reply cut
    there is retried like an invalid one (``retry_cut``): once more by
    default, sampled at ``[llm] retry_temperature``, with the same cap. The
    first attempt keeps the configured temperature.

    Raises:
        everalgo.llm.LLMOutputError: The reply leaves ``content``,
            ``summary`` or ``title`` blank.
        everos.component.llm.errors.LLMCallError: The upstream failed, no
            attempt produced a reply that validates, or every attempt was cut
            at the cap (``cut_short``).
    """
    messages = build_messages(
        memcell,
        pictures=pictures,
        custom_instructions=custom_instructions,
        output_language=output_language,
    )
    cap = (
        reply_token_cap(
            pictures=len(pictures),
            conversation_chars=len(render_conversation(memcell)),
        )
        if max_tokens is None
        else max_tokens
    )
    text = await llm.achat(
        messages,
        response_format=EpisodeReply,
        max_tokens=cap,
        retry_cut=True,
    )
    reply = parse_reply(EpisodeReply, text, operator="episode")
    _check_fields_written(reply, ("content", "summary", "title"), operator="episode")
    logger.info(
        "memory.episode.extracted",
        pictures=len(pictures),
        images=len(reply.images),
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
) -> tuple[AlgoEpisode, dict[str, tuple[str, str]]]:
    """The cell's episode and each picture's ``(description, ocr)`` by label.

    One :func:`extract` call, tried again by
    :func:`~everos.memory.extract.llm_retry.retry_llm_output` while its reply
    is unusable. The two results are readings of the same call and are never
    split. A cell without pictures describes none.
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
        )
        described = {
            fact.ref: (fact.description, fact.ocr)
            for fact in reply.images
            if fact.ref in by_label
        }
        return to_episode(reply, memcell, labels=by_label), described

    return await retry_llm_output(
        attempt, operator="episode", memcell_id=memcell_id, pictures=len(pictures)
    )
