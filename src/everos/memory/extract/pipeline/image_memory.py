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
- What the pictures say about **their sender** leaves the call as profile
  operations (see :mod:`.profile_delta`), so a conversation carried by photos
  still moves the portrait. Those operations are add-only: this call never sees
  the stored profile, and an index it cannot see is an index it cannot address.

Numbers on the reply side are the same as the episode prompt's (``content`` /
``summary`` / ``title``), so the caller can hand the episode half to
:class:`everos.memory.Episode` unchanged.
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
from pydantic import Field

from everos.core.observability.logging import get_logger
from everos.memory.extract.pipeline.profile_delta import (
    PICTURE_PROFILE_RULES,
    ProfileItemOp,
)

logger = get_logger(__name__)


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


class ImageEpisodeReply(StrictOutputModel):
    """The one reply that carries the episode, every picture's memory, and the profile.

    ``profile`` is the write-time half of the design's "the picture call also
    emits a profile change": a photo of a pet, a home or a hobby reaches the
    portrait in the request that produced it, not in a later text-driven update
    that a picture-heavy conversation may never trigger.
    """

    content: str
    summary: str
    title: str
    images: list[ImageFact] = Field(default_factory=list)
    profile: list[ProfileItemOp] = Field(default_factory=list)


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
  twice."""

_IMAGE_REPLY_FORMAT = """\
Reply with one JSON object, nothing else:
{"content": "...", "summary": "...", "title": "...",
 "images": [{"ref": "img-1", "description": "...", "ocr": "..."}],
 "profile": [{"action": "add", "type": "explicit_info",
              "source_image_id": "img-1",
              "data": {"category": "...", "description": "...", "evidence": "..."}}]}"""


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
        tagged("profile", PICTURE_PROFILE_RULES),
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
        output_language: Language to write in; ``None`` follows the speakers.
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
    """
    from everalgo.types import Episode as AlgoEpisode

    body = reply.content
    if labels:
        for label, image_id in sorted(labels.items(), key=lambda kv: -len(kv[0])):
            body = body.replace(f"({label})", f"({label}: {image_id})")
            body = body.replace(f"[IMAGE: {label}]", f"[IMAGE: {image_id}]")
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

    Raises:
        everos.component.llm.errors.LLMCallError: The upstream failed, or no
            attempt produced a reply that validates.
    """
    messages = build_messages(
        memcell,
        pictures=pictures,
        custom_instructions=custom_instructions,
        output_language=output_language,
    )
    text = await llm.achat(
        messages,
        response_format=ImageEpisodeReply,
        max_tokens=max_tokens,
    )
    logger.info(
        "memory.images.extracted",
        pictures=len(pictures),
        reply_chars=len(text),
    )
    return parse_reply(ImageEpisodeReply, text, operator="image memory extraction")


__all__ = [
    "ImageEpisodeReply",
    "ImageFact",
    "build_messages",
    "build_system_prompt",
    "extract",
    "render_conversation",
    "to_episode",
]
