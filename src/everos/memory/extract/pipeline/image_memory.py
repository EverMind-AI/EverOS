"""Image-memory extraction — one call that reads the pictures it is asked about.

Why this exists next to everalgo's ``EpisodeExtractor``: that extractor's
conversation type is text by contract (``ChatMessage.content: str``), so a
picture cannot travel through it. Rather than transcribe pictures into prose
before extraction — which is exactly the lossy round-trip the design rejects —
the pipeline makes one call of its own when a cell carries pictures, and that
call does everything the episode call does plus one thing more.

The call is *additive*, not a replacement:

- **When a cell has no picture**, the pipeline runs everalgo's extractor
  unchanged. Nothing about the text-only path moves.
- **When a cell has pictures**, this module renders the cell the way everalgo
  renders it (same ``[time UTC] speaker: text`` lines, same
  ``<conversation>`` element), attaches the pictures as ``image_url`` parts
  after the text, and asks for a reply carrying the episode the extractor would
  have written *and* one short record per picture.

Two rules keep the results coherent with the rest of the store:

- A picture's record describes **only what the picture shows**. Anything the
  conversation already says belongs to the episode; duplicating it here would
  create two memories that drift apart.
- The episode body **refers to** a picture by its id and never restates its
  description. Replacing a picture's description then costs one entry, not a
  rewrite of the episode that mentions it.

What a picture says about its sender reaches the profile through the profile
update, which reads the segment's pictures itself
(:func:`~everos.memory.extract.pipeline.pictures.profile_pictures`): the 2.0
design has one profile writer, and every prompt that reads raw conversation
takes its pictures. This call writes no profile item.

Numbers on the reply side are the same as the episode prompt's (``content`` /
``summary`` / ``title``), so the caller can hand the episode half to
:class:`everos.memory.Episode` unchanged.

The reply is read leniently in form and strictly in meaning
(:mod:`everos.memory.extract.reply_form`): an unknown key left empty reads as
absent, ``null`` reads as empty for ``ocr`` and as no entries for ``images``
(one entry given bare reads as a list of it), and a reply with an
array element left open or split in two objects is mended before it is
validated. Each of those cost a second call carrying every picture again
(2026-11-11, 4.2% of MemEye's picture calls). The JSON schema a call sends is
unchanged; the reply models' docstrings are part of it, so they stay as they
are.

The call's output is capped by the number of pictures it reads
(:func:`reply_token_cap`), and a reply cut at that cap is sampled again once,
warmer, before the cell fails. At temperature 0 a long transcription in
``ocr`` can repeat itself until the output limit: on 2026-11-11 two or three
of MemEye's 168 picture calls (text-dense comic pages) did so in each of three
builds, ran to the 32,768-token cap the builds set, and left their cells with
neither an episode nor picture memories.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from everalgo.llm import (
    LLMMessage,
    StrictOutputModel,
    image_url_part_from_bytes,
    parse_reply,
)
from everalgo.types import MemCell
from everalgo.user_memory._language import (
    PARTICIPANT_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everalgo.user_memory.prompts.en.episode import assemble_system_prompt
from everalgo.utils import paragraphs, render_chat_line, tagged
from pydantic import Field, ValidationError, field_validator, model_validator

from everos.core.observability.logging import get_logger
from everos.memory.extract.reply_form import drop_empty_unknown_keys, repair_json_form

logger = get_logger(__name__)

REPLY_TOKENS_BASE = 2_048
"""Output budget of a picture call before its pictures (:func:`reply_token_cap`)."""

REPLY_TOKENS_PER_PICTURE = 1_024
"""Output budget each picture adds (:func:`reply_token_cap`)."""

REPLY_TOKENS_CEILING = 32_768
"""The cap never exceeds this: the output limit the 2.0 store builds set."""


def reply_token_cap(pictures: int) -> int:
    """``max_tokens`` of a picture call reading ``pictures`` pictures.

    ``2,048 + 1,024`` per picture, at most :data:`REPLY_TOKENS_CEILING`. The
    numbers come from the 914 complete picture replies DeepSeek V4.1 Flash
    wrote on 2026-11-11 (MemEye, CUE-Mem and MemLens, temperatures 0 and 1):
    the longest was 2,423 tokens, for 8 pictures (cap 10,240), the longest for
    one picture 1,486 (cap 3,072), and none came within 1,586 tokens of its
    cap -- so a reply that reaches the cap is a runaway (an ``ocr``
    transcription repeating itself), not a long answer. Without the cap a
    runaway ran to the builds' 32,768 tokens, 57 times the average reply's
    output (573). A ``max_tokens`` the deployment sets lower still wins
    (``[llm] extra``).
    """
    return min(
        REPLY_TOKENS_BASE + REPLY_TOKENS_PER_PICTURE * pictures, REPLY_TOKENS_CEILING
    )


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


class ImageEpisodeReply(StrictOutputModel):
    """The one reply that carries the episode and every picture's memory."""

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
    ) -> ImageEpisodeReply:
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


_IMAGE_RULES = """\
The user message carries <pictures>: one or more images, in the order their
labels appear in <conversation>. Each label is written in the conversation text
as [IMAGE: img-1], [IMAGE: img-2] and so on, at the position where the picture
was sent.
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

_IMAGE_REPLY_FORMAT = """\
Reply with one JSON object, nothing else:
{"content": "...", "summary": "...", "title": "...",
 "images": [{"ref": "img-1", "description": "...", "ocr": "..."}]}"""


def build_system_prompt(
    *, custom_instructions: str | None, output_language: OutputLanguage | None
) -> str:
    """The system message: everalgo's episode rules plus the picture rules.

    Reusing ``EPISODE_SYSTEM_PROMPT`` rather than restating it is deliberate:
    a picture-carrying cell must produce an episode of the same shape and
    quality as a text-only one, so the rules that shape it stay in one place.
    """
    base = assemble_system_prompt(
        custom_instructions=custom_instructions,
        language_rule="",  # placed by the host prompt below, once
    )
    return paragraphs(
        base,
        tagged("pictures", _IMAGE_RULES),
        tagged(
            "output_language",
            build_language_rule(output_language, fallback=PARTICIPANT_LANGUAGE_RULE),
        ),
        tagged("reply_format", _IMAGE_REPLY_FORMAT),
    )


def render_conversation(memcell: MemCell) -> str:
    """The cell as ``[time UTC (weekday)] speaker: text`` lines.

    Same rendering as everalgo's episode path, including dropping a message
    whose text is empty — with one exception: a message that carries only a
    picture keeps its line, because its ``[IMAGE: ...]`` reference is the
    only thing tying the picture to its time and speaker.
    """
    lines = [
        render_chat_line(message)
        for message in memcell.items
        if getattr(message, "kind", None) == "text"
        and (getattr(message, "content", "") or _has_image(message))
    ]
    body = tagged("conversation", "\n".join(lines), count=str(len(lines)))
    return body


def _has_image(message: Any) -> bool:
    return "[IMAGE:" in str(getattr(message, "content", "") or "")


def build_messages(
    memcell: MemCell,
    *,
    pictures: Sequence[tuple[str, bytes, str]],
    custom_instructions: str | None,
    output_language: OutputLanguage | None,
    detail: str = "high",
) -> list[LLMMessage]:
    """The ``[system, user]`` pair: the rendered cell, then the pictures.

    Args:
        memcell: The cell being extracted.
        pictures: ``(label, bytes, mime)`` per picture, in conversation order.
            The label is the one the conversation text uses (``img-1``).
        custom_instructions: The host's instructions.
        output_language: Language to write in. The pipeline names the
            conversation's, resolved in code; ``None`` leaves the model to
            follow the speakers, which it does not reliably do on a picture
            cell (whole replies drift into Chinese).
        detail: ``detail`` hint on each image part. ``high`` because this call
            has to read small text out of a picture; the decider's
            thumbnail-only default is set separately.
    """
    system = build_system_prompt(
        custom_instructions=custom_instructions, output_language=output_language
    )
    names = ", ".join(label for label, _, _ in pictures)
    user_text = paragraphs(
        render_conversation(memcell),
        tagged("pictures", f"Labels in order: {names}. Read every attached picture."),
    )
    parts: list[Any] = [{"type": "text", "text": user_text}]
    for label, data, mime in pictures:
        parts.append({"type": "text", "text": tagged("picture", f"label: {label}")})
        part = image_url_part_from_bytes(data, mime)
        if part.image_url.detail is None and detail:
            part = part.model_copy(
                update={
                    "image_url": part.image_url.model_copy(update={"detail": detail})
                }
            )
        parts.append(part.model_dump())
    return [
        LLMMessage(role="system", content=system),
        LLMMessage(role="user", content=parts),
    ]


def to_episode(
    reply: ImageEpisodeReply,
    memcell: MemCell,
    *,
    labels: Mapping[str, str] | None = None,
) -> Any:
    """The algo Episode the picture route's reply describes.

    Deliberately the same mapping everalgo's extractor applies (``title`` →
    ``subject``, ``content`` → ``episode`` verbatim, ``summary`` as written,
    ``timestamp`` from the cell), so a picture-carrying cell yields an Episode
    indistinguishable from a text-only one downstream.

    ``owner_id`` is ``None``: like the text route, the episode is extracted
    once for the whole cell and fanned out to one md per user sender by the
    caller, which is what keeps a cell at one model call.

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
    from everalgo.types import Episode as AlgoEpisode

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
        summary=reply.summary,
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
) -> ImageEpisodeReply:
    """One call: the episode and every picture's memory, from the same reading.

    The call asks for at most ``max_tokens`` output tokens, by default
    :func:`reply_token_cap` of its pictures, and a reply cut there is retried
    like an invalid one (``retry_cut``): once more by default, sampled at
    ``[llm] retry_temperature``, with the same cap. The first attempt keeps
    the configured temperature.

    Raises:
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
    cap = reply_token_cap(len(pictures)) if max_tokens is None else max_tokens
    text = await llm.achat(
        messages,
        response_format=ImageEpisodeReply,
        max_tokens=cap,
        retry_cut=True,
    )
    logger.info(
        "memory.images.extracted",
        pictures=len(pictures),
        reply_chars=len(text),
        max_tokens=cap,
    )
    return parse_reply(ImageEpisodeReply, text, operator="image memory extraction")


__all__ = [
    "REPLY_TOKENS_BASE",
    "REPLY_TOKENS_CEILING",
    "REPLY_TOKENS_PER_PICTURE",
    "ImageEpisodeReply",
    "ImageFact",
    "build_messages",
    "build_system_prompt",
    "extract",
    "render_conversation",
    "reply_token_cap",
    "to_episode",
]
