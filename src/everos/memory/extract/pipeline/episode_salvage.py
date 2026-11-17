"""The episode call's last resort: a reply salvaged, or one fallback request.

The episode call is tried twice by the provider (the second time warmer), and
a reply that fails both times leaves its cell with no episode and no picture
memory -- however much of it was right. On the multimodal builds of
2026-11-17 (DeepSeek V4.1 Flash) two cells lost everything that way, each
again on every one of the six re-sends that followed:

- a Mem-Gallery cell whose second picture's ``ocr`` ran away ("1 2 3 4 5 6 7
  8 9 0" repeated) until the cap, at temperature 0 and at the warmer retry;
- an H2HMem cell whose replies, 20 of 20, closed an ``ocr`` string with a
  typographic quote (``”``), so the string ran on into the next line and the
  reply was not JSON -- its title, its episode and all three pictures'
  entries were there, complete.

So when every attempt failed in form -- each one cut at the cap or not JSON
even after the mending of :mod:`everos.memory.extract.reply_form` -- the
replies are read once more, leniently
(:func:`~everos.memory.extract.reply_form.read_leniently`), before the cell
fails, and the first one that is best kept is used (:func:`salvage_reply`):

- ``title`` and ``content`` must have been read whole and not blank; nothing
  is salvaged from a reply without them;
- every picture entry whose ``ref`` and ``description`` were read whole is
  kept; an ``ocr`` that was cut, that runs away (one unit repeated at least
  four times over at least 100 characters) or that is longer than
  :data:`SALVAGED_OCR_CHARS` is trimmed (:func:`trim_ocr`): each run cut to
  one copy, then the text cut to that length. Any other entry is dropped,
  and its picture has no memory, as it would have none without the salvage;
- every profile cue read whole is kept (its entries are read leniently in
  meaning as always, :mod:`.profile_cues`).

When no reply keeps a ``title`` and a ``content``, a picture cell sends one
fallback request: the same request with :data:`CHAT_EPISODE_FALLBACK`
appended as one more text part, sent once (``achat_raw`` when the client has
it), its reply validated as any, or salvaged the same way when it is cut or
not JSON. A text cell sends none: the instruction is about ``ocr``, and no
text reply on record was ever cut. A failure for any other reason -- a reply
that is JSON but breaks the reply model in meaning, a refusal, the transport
-- is never salvaged: the call fails as it did.

Each salvage and each fallback is logged with the cell's id and counted
(``everos_episode_reply_salvage_total``). A reply that validates is never
read here: this runs only once the provider has given up.
"""

from __future__ import annotations

__all__ = [
    "CHAT_EPISODE_FALLBACK",
    "SALVAGED_OCR_CHARS",
    "Salvage",
    "collapse_runs",
    "fallback_messages",
    "recover",
    "salvage_reply",
    "trim_ocr",
]

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from everalgo.llm import LLMMessage
from pydantic import BaseModel, ValidationError

from everos.component.llm import LLMCallError
from everos.core.observability.logging import get_logger
from everos.core.observability.metrics import Counter as MetricCounter
from everos.memory.extract.reply_form import CutText, OpenObject, read_leniently
from everos.memory.prompts import PromptTemplate

if TYPE_CHECKING:
    from everos.component.llm import RawReply

logger = get_logger(__name__)

_salvages = MetricCounter(
    name="episode_reply_salvage_total",
    description=(
        "Episode calls whose every attempt failed in form, by step: the failed "
        "replies salvaged, or the one fallback request; and by outcome."
    ),
    labelnames=("step", "outcome"),
    namespace="everos",
)

SALVAGED_OCR_CHARS = 4_000
"""Longest ``ocr`` a salvaged picture entry keeps. Above the longest one of
the 13,562 picture entries the multimodal builds of 2026-11-17 wrote in
replies that validated (3,434 characters; 99.9% are under 2,000)."""

_SCANNED_OCR_CHARS = 8 * SALVAGED_OCR_CHARS
"""How much of a runaway ``ocr`` :func:`trim_ocr` reads: a reply cut at the
32,768-token ceiling holds over 100,000 characters of it."""

_RUN_UNIT_CHARS = 200
_RUN_COPIES = 4
_RUN_CHARS = 100

CHAT_EPISODE_FALLBACK = PromptTemplate(
    id="chat.episode_fallback",
    version=1,
    parts=(
        'Keep each "ocr" short, and never repeat a sequence: write a repeated '
        "word, number or line once.",
    ),
)
"""The instruction a picture cell's one fallback request appends, as one more
text part of its message (:func:`fallback_messages`)."""


@runtime_checkable
class _RawClient(Protocol):
    """A client that sends one request and hands its reply back unchecked."""

    async def achat_raw(
        self,
        messages: list[LLMMessage],
        *,
        response_format: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> RawReply: ...


@dataclass(frozen=True)
class Salvage:
    """A reply read back from a failed one, and what it kept.

    Attributes:
        reply: The reply, validated by the reply model it was asked for.
        images_kept: Picture entries kept.
        images_dropped: Picture entries dropped (``ref`` or ``description``
            not read whole).
        ocr_trimmed: Labels of the kept entries whose ``ocr`` was trimmed.
        cues_dropped: Profile cue entries not read whole.
    """

    reply: Any
    images_kept: int
    images_dropped: int
    ocr_trimmed: tuple[str, ...]
    cues_dropped: int


def collapse_runs(text: str) -> str:
    """``text`` with each run of one unit repeated, cut to one copy of it.

    A run is a stretch that repeats a unit of at most 200 characters at least
    four times and spans at least 100 characters -- a transcription that ran
    away. Of overlapping runs the longest is cut, the shorter unit on a tie.
    A run inside the text keeps the partial copy that leads into what follows
    it; one the text ends in does not.
    """
    size = len(text)
    found: list[tuple[int, int, int]] = []
    for unit in range(1, min(_RUN_UNIT_CHARS, size // _RUN_COPIES) + 1):
        at = 0
        while at + 2 * unit <= size:
            inside = next((end for start, end, _ in found if start <= at < end), None)
            if inside is not None:
                at = inside
                continue
            if text[at : at + unit] != text[at + unit : at + 2 * unit]:
                at += unit
                continue
            start = at
            while start > 0 and text[start - 1] == text[start - 1 + unit]:
                start -= 1
            end = at + 2 * unit
            while end < size and text[end] == text[end - unit]:
                end += 1
            if end - start >= max(_RUN_COPIES * unit, _RUN_CHARS):
                found.append((start, end, unit))
            at = end
    kept: list[tuple[int, int, int]] = []
    for start, end, unit in sorted(found, key=lambda r: (r[0] - r[1], r[2], r[0])):
        if all(end <= s or start >= e for s, e, _ in kept):
            kept.append((start, end, unit))
    pieces: list[str] = []
    at = 0
    for start, end, unit in sorted(kept):
        # One copy, and the partial copy that leads into what follows the run;
        # a run the text ends in (a reply cut by its cap) loses its partial one.
        rest = (end - start) % unit if end < size else 0
        pieces.append(text[at : start + unit] + text[end - rest : end])
        at = end
    pieces.append(text[at:])
    return "".join(pieces)


def trim_ocr(ocr: str) -> str:
    """``ocr`` with its runs cut to one copy, then cut to :data:`SALVAGED_OCR_CHARS`."""
    return collapse_runs(str(ocr)[:_SCANNED_OCR_CHARS])[:SALVAGED_OCR_CHARS].strip()


def _whole(value: object) -> bool:
    """A string read whole and not blank: closed, or its lost quote at a line end."""
    if not isinstance(value, str) or not value.strip():
        return False
    return not isinstance(value, CutText) or value.how == "line"


def _picture_entries(
    value: object,
) -> tuple[list[dict[str, str]], int, tuple[str, ...]]:
    """The entries kept from ``images``, how many were dropped, the trimmed labels."""
    items = [value] if isinstance(value, dict) else value
    if not isinstance(items, list):
        return [], 0, ()
    kept: list[dict[str, str]] = []
    trimmed: list[str] = []
    dropped = 0
    for entry in items:
        if not isinstance(entry, dict) or not (
            _whole(entry.get("ref")) and _whole(entry.get("description"))
        ):
            dropped += 1
            continue
        ocr = entry.get("ocr")
        ocr = ocr if isinstance(ocr, str) else ""
        if (
            isinstance(ocr, CutText)
            or len(ocr) > SALVAGED_OCR_CHARS
            or collapse_runs(ocr) != ocr
        ):
            ocr = trim_ocr(ocr)
            trimmed.append(str(entry["ref"]))
        kept.append(
            {
                "ref": str(entry["ref"]),
                "description": str(entry["description"]),
                "ocr": str(ocr),
            }
        )
    return kept, dropped, tuple(trimmed)


def _whole_cues(value: object) -> tuple[list[dict[str, object]], int]:
    """The cue entries read whole, and how many were not."""
    if not isinstance(value, list):
        return [], 0
    kept = [
        {
            key: str(item) if isinstance(item, str) else item
            for key, item in entry.items()
        }
        for entry in value
        if isinstance(entry, dict)
        and not isinstance(entry, OpenObject)
        and not any(isinstance(item, CutText) for item in entry.values())
    ]
    return kept, len(value) - len(kept)


def salvage_reply(text: str, model: type[BaseModel]) -> Salvage | None:
    """What a reply that failed in form still holds, read as ``model``.

    ``None`` when ``title`` or ``content`` was not read whole, or what was
    read does not validate as ``model``.
    """
    value = read_leniently(text)
    if not isinstance(value, dict):
        return None
    title, content = value.get("title"), value.get("content")
    if not (_whole(title) and _whole(content)):
        return None
    fields: dict[str, object] = {"title": str(title), "content": str(content)}
    images: list[dict[str, str]] = []
    dropped, trimmed, cues_dropped = 0, (), 0
    if "images" in model.model_fields:
        images, dropped, trimmed = _picture_entries(value.get("images"))
        fields["images"] = images
    if "profile_cues" in model.model_fields:
        fields["profile_cues"], cues_dropped = _whole_cues(value.get("profile_cues"))
    try:
        reply = model.model_validate(fields)
    except ValidationError:
        return None
    return Salvage(
        reply=reply,
        images_kept=len(images),
        images_dropped=dropped,
        ocr_trimmed=trimmed,
        cues_dropped=cues_dropped,
    )


def _failed_in_form(err: LLMCallError, model: type[BaseModel]) -> list[str] | None:
    """The replies of a call whose every attempt was cut or not JSON; else ``None``."""
    attempts = err.failed_attempts
    if not attempts:
        return None
    replies: list[str] = []
    for attempt in attempts:
        if attempt.reply is None:
            return None
        if attempt.reason == "invalid_reply" and not _not_json(attempt.reply, model):
            return None
        if attempt.reason not in ("invalid_reply", "cut_short"):
            return None
        replies.append(attempt.reply)
    return replies


def _not_json(text: str, model: type[BaseModel]) -> bool:
    """Whether ``text`` is not JSON, even after its form is mended."""
    try:
        model.model_validate_json(text)
    except ValidationError as err:
        return any(error["type"] == "json_invalid" for error in err.errors())
    return False


def _best(replies: Sequence[str], model: type[BaseModel]) -> tuple[int, Salvage] | None:
    """The attempt (from 1) whose salvage keeps most, the earliest on a tie."""
    best: tuple[int, Salvage] | None = None
    for attempt, text in enumerate(replies, start=1):
        found = salvage_reply(text, model)
        if found is None:
            continue
        if best is None or _rank(found) > _rank(best[1]):
            best = (attempt, found)
    return best


def _rank(found: Salvage) -> tuple[int, int, int]:
    return (found.images_kept, -found.images_dropped, -len(found.ocr_trimmed))


def fallback_messages(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    """A picture cell's request with :data:`CHAT_EPISODE_FALLBACK` appended.

    The one user message keeps every part, in order, and gains one text part
    after its last picture.
    """
    *head, last = messages
    content = last.content
    parts: list[Any] = (
        [*content] if isinstance(content, list) else [{"type": "text", "text": content}]
    )
    parts.append({"type": "text", "text": CHAT_EPISODE_FALLBACK.parts[0]})
    return [*head, LLMMessage(role=last.role, content=parts)]


async def recover(
    llm: Any,
    err: LLMCallError,
    *,
    messages: Sequence[LLMMessage],
    model: type[BaseModel],
    max_tokens: int,
    pictures: int,
    memcell_id: str | None,
) -> tuple[Any, str]:
    """The reply a failed episode call still yields, and its text.

    Salvages the failed replies when every attempt failed in form; when none
    keeps a title and a content, a picture cell sends its one fallback
    request (module docstring).

    Raises:
        LLMCallError: ``err``, when nothing could be recovered.
    """
    replies = _failed_in_form(err, model)
    if replies is None:
        raise err
    best = _best(replies, model)
    if best is not None:
        attempt, found = best
        _report(found, step="salvage", attempt=attempt, memcell_id=memcell_id)
        return found.reply, replies[attempt - 1]
    _salvages.labels(step="salvage", outcome="failed").inc()
    logger.warning(
        "memory.episode.salvage_failed",
        step="salvage",
        attempts=len(replies),
        fallback=bool(pictures),
        memcell_id=memcell_id,
    )
    if not pictures:
        raise err
    return await _fallback(
        llm,
        err,
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        memcell_id=memcell_id,
    )


async def _fallback(
    llm: Any,
    err: LLMCallError,
    *,
    messages: Sequence[LLMMessage],
    model: type[BaseModel],
    max_tokens: int,
    memcell_id: str | None,
) -> tuple[Any, str]:
    """The one fallback request, its reply validated or salvaged; else ``err``."""
    sent = fallback_messages(messages)
    try:
        if isinstance(llm, _RawClient):
            raw = await llm.achat_raw(
                sent, response_format=model, max_tokens=max_tokens
            )
            text, finish = raw.text, raw.finish_reason
        else:
            text = await llm.achat(sent, response_format=model, max_tokens=max_tokens)
            finish = "stop"
    except LLMCallError as failed:
        _fallback_failed(memcell_id, reason=failed.reason)
        raise err from err.__cause__
    if finish in (None, "stop"):
        try:
            reply = model.model_validate_json(text)
        except ValidationError:
            reply = None
        if reply is not None:
            _salvages.labels(step="fallback", outcome="recovered").inc()
            logger.warning(
                "memory.episode.fallback_answered",
                memcell_id=memcell_id,
                salvaged=False,
                reply_chars=len(text),
            )
            return reply, text
    if finish == "length" or _not_json(text, model):
        found = salvage_reply(text, model)
        if found is not None:
            _report(found, step="fallback", attempt=1, memcell_id=memcell_id)
            return found.reply, text
    _fallback_failed(memcell_id, reason=str(finish))
    raise err


def _fallback_failed(memcell_id: str | None, *, reason: str) -> None:
    _salvages.labels(step="fallback", outcome="failed").inc()
    logger.warning(
        "memory.episode.salvage_failed",
        step="fallback",
        reason=reason,
        memcell_id=memcell_id,
    )


def _report(found: Salvage, *, step: str, attempt: int, memcell_id: str | None) -> None:
    _salvages.labels(step=step, outcome="recovered").inc()
    logger.warning(
        "memory.episode.reply_salvaged",
        step=step,
        attempt=attempt,
        images_kept=found.images_kept,
        images_dropped=found.images_dropped,
        ocr_trimmed=list(found.ocr_trimmed),
        cues_dropped=found.cues_dropped,
        memcell_id=memcell_id,
    )
