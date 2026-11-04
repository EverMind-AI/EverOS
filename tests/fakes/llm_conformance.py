"""Conformance checks for ``LLMClient`` implementations.

An implementation provides an ``LLMClientTestHarness`` and runs every check in ``REQUIRED_CHECKS`` against it, a
fresh harness per check, since a check may leave stubs in the queue. Each check is a coroutine function that raises
``AssertionError`` on a violation, so it runs under any test runner; its docstring names the section of that
document it verifies. ``OPTIONAL_CHECKS`` cover recommendations; failing one is not a breach of the contract.
"""

from __future__ import annotations

__all__ = [
    "NOT_SENT",
    "OPTIONAL_CHECKS",
    "REQUIRED_CHECKS",
    "ConformanceCheck",
    "EffectiveRequest",
    "FinishReason",
    "LLMClientTestHarness",
    "NotSent",
    "UpstreamStub",
    "check_cancellation_propagates",
    "check_explicit_values_win_over_defaults",
    "check_invalid_replies_are_not_returned",
    "check_missing_finish_reason_is_accepted",
    "check_non_text_parts_reach_upstream",
    "check_none_means_default_or_omitted",
    "check_refusal_raises",
    "check_retries_are_bounded",
    "check_retries_never_raise_max_tokens",
    "check_truncation_is_never_returned",
]

import asyncio
import enum
import json
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError

from everos.algo.llm.output import StrictOutputModel
from everos.algo.llm.types import ImageUrlInner, ImageUrlPart, LLMMessage, TextPart

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient


class _NotSent(enum.Enum):
    NOT_SENT = "not sent"


NOT_SENT: Final = _NotSent.NOT_SENT
"""Marks a parameter the request omitted, as opposed to one sent as a literal ``null``."""

NotSent = Literal[_NotSent.NOT_SENT]
"""The type of ``NOT_SENT``."""

FinishReason = Literal["stop", "length", "content_filter", "abort", "error"]
"""A finish reason an ``UpstreamStub`` can report, in OpenAI vocabulary plus the vLLM-style abnormal stops."""


class EffectiveRequest(BaseModel):
    """Semantic view of one request the client sent upstream, normalised from the captured wire request.

    It carries only what the contract binds; the reply-format parameter is the implementation's own choice and is
    left out.

    Attributes:
        messages: The messages as delivered, translated back into ``LLMMessage`` form.
        temperature: The value sent; ``None`` if sent as a literal null, ``NOT_SENT`` if omitted.
        max_tokens: The output-token cap sent under whatever name the upstream uses; ``None`` if sent as a
            literal null, ``NOT_SENT`` if omitted.
    """

    model_config = ConfigDict(frozen=True)

    messages: list[LLMMessage]
    temperature: float | None | NotSent = NOT_SENT
    max_tokens: int | None | NotSent = NOT_SENT


class UpstreamStub(BaseModel):
    """One scripted upstream answer, in OpenAI Chat Completions vocabulary.

    A harness for another upstream translates it into that upstream's equivalent, e.g. Anthropic's
    ``stop_reason: "max_tokens"`` for ``finish_reason="length"`` and ``stop_reason: "refusal"`` for a ``refusal``.

    Attributes:
        content: Reply text; ``None`` for absent content.
        refusal: Refusal the upstream flags, if any; a flagged refusal may still come with JSON content.
        finish_reason: Reported finish reason; ``None`` for an upstream that reports none. ``abort`` and
            ``error`` are what vLLM-style gateways report for an abnormal stop.
        hang: Never answer, so a check can cancel an in-flight call; the other fields are ignored.
    """

    model_config = ConfigDict(frozen=True)

    content: str | None = None
    refusal: str | None = None
    finish_reason: FinishReason | None = "stop"
    hang: bool = False


class LLMClientTestHarness(Protocol):
    """Factory an implementation provides so the conformance checks can drive its client."""

    def build(self, **defaults: Any) -> LLMClient:
        """Build a client whose configured defaults are ``defaults``.

        The checks pass ``temperature`` and ``max_tokens``; an omitted key means the implementation's own
        default, or none. Clients built from one harness share its upstream queue and request log.
        """
        ...

    def queue_upstream(self, *responses: UpstreamStub) -> None:
        """Append answers the upstream gives, one per request, in order.

        Once the queue is empty the upstream must fail every further request, so an implementation that
        retries past the script ends in an error rather than a hang.
        """
        ...

    def sent_requests(self) -> list[EffectiveRequest]:
        """Return every request sent so far, oldest first.

        Each one must be normalised from the request captured as close to the wire as possible, never copied from
        the arguments ``achat`` received, or the checks turn into tautologies.
        """
        ...


ConformanceCheck = Callable[[LLMClientTestHarness], Awaitable[None]]
"""A check: a coroutine function that drives the harness and raises ``AssertionError`` on a violation."""

_CALL_TIMEOUT_S: Final = 10.0
_POLL_INTERVAL_S: Final = 0.01
_PNG_URI: Final = "data:image/png;base64,iVBORw0KGgo="
_PDF_URI: Final = "data:application/pdf;base64,JVBERi0xLjQK"
_VALID_REPLY: Final = '{"text": "complete"}'
# Closed and valid on purpose: the clause bites exactly where parsing cannot tell the reply was cut short.
_CLOSED_TRUNCATED_REPLY: Final = '{"text": "partial"}'
_REFUSAL: Final = "I cannot help with that request."


class _Probe(StrictOutputModel):
    text: str


def _prompt() -> list[LLMMessage]:
    return [
        LLMMessage(
            role="user",
            content='Reply with one JSON object: {"text": "<your answer>"}.',
        )
    ]


def _fail_unless(condition: bool, message: str) -> None:
    """Raise ``AssertionError`` with ``message`` unless ``condition`` holds; unlike ``assert``, survives ``-O``."""
    if not condition:
        raise AssertionError(message)


def _parses(text: str) -> bool:
    try:
        _Probe.model_validate_json(text)
    except ValidationError:
        return False
    return True


async def _attempt(
    client: LLMClient,
    messages: list[LLMMessage] | None = None,
    **params: Any,
) -> str | Exception:
    """Run one ``achat`` call under a deadline; return the reply text, or the exception the client raised.

    Raises:
        AssertionError: If the call does not finish before the deadline.
    """
    deadline = asyncio.timeout(_CALL_TIMEOUT_S)
    try:
        async with deadline:
            return await client.achat(
                messages if messages is not None else _prompt(),
                response_format=_Probe,
                **params,
            )
    except Exception as exc:
        if deadline.expired():
            raise AssertionError(
                f"achat() did not finish within {_CALL_TIMEOUT_S}s"
            ) from exc
        return exc


def _requests_since(
    harness: LLMClientTestHarness, start: int
) -> list[EffectiveRequest]:
    requests = harness.sent_requests()[start:]
    _fail_unless(bool(requests), "no request reached the upstream")
    return requests


async def check_explicit_values_win_over_defaults(
    harness: LLMClientTestHarness,
) -> None:
    """Explicit ``temperature`` / ``max_tokens`` reach the upstream unchanged, outranking defaults (§3.1).

    A zero temperature is included because ``value or default`` is the usual way this clause breaks.
    """
    client = harness.build(temperature=0.9, max_tokens=4096)
    for temperature, max_tokens in ((0.2, 123), (0.0, 64)):
        start = len(harness.sent_requests())
        harness.queue_upstream(UpstreamStub(content=_VALID_REPLY))
        outcome = await _attempt(client, temperature=temperature, max_tokens=max_tokens)
        _fail_unless(
            isinstance(outcome, str), f"a valid reply was not returned: {outcome!r}"
        )
        for request in _requests_since(harness, start):
            _fail_unless(
                request.temperature == temperature,
                f"explicit temperature={temperature} was sent as {request.temperature!r}",
            )
            _fail_unless(
                request.max_tokens == max_tokens,
                f"explicit max_tokens={max_tokens} was sent as {request.max_tokens!r}",
            )


async def check_retries_never_raise_max_tokens(harness: LLMClientTestHarness) -> None:
    """Every request of one call, internal retries included, carries the caller's ``max_tokens`` (§3.1).

    The upstream answers with truncations and garbage first, which is what tempts a retry, and the client's default
    is larger than the caller's value, so falling back to it on a retry counts as raising it.
    """
    client = harness.build(max_tokens=4096)
    start = len(harness.sent_requests())
    harness.queue_upstream(
        UpstreamStub(content='{"text": "cut', finish_reason="length"),
        UpstreamStub(content="not json at all"),
        UpstreamStub(content=_CLOSED_TRUNCATED_REPLY, finish_reason="length"),
        UpstreamStub(content=_VALID_REPLY),
    )
    await _attempt(client, max_tokens=100)
    requests = _requests_since(harness, start)
    for index, request in enumerate(requests, start=1):
        _fail_unless(
            request.max_tokens == 100,
            f"request {index} of {len(requests)} carried max_tokens={request.max_tokens!r}, not the caller's 100",
        )


async def check_none_means_default_or_omitted(harness: LLMClientTestHarness) -> None:
    """An unset ``temperature`` / ``max_tokens`` is sent as the configured default or omitted, never as null (§3.1)."""
    configured = harness.build(temperature=0.3, max_tokens=77)
    start = len(harness.sent_requests())
    harness.queue_upstream(UpstreamStub(content=_VALID_REPLY))
    await _attempt(configured)
    for request in _requests_since(harness, start):
        _fail_unless(
            request.temperature in (0.3, NOT_SENT),
            f"with default temperature=0.3 an unset temperature was sent as {request.temperature!r}",
        )
        _fail_unless(
            request.max_tokens in (77, NOT_SENT),
            f"with default max_tokens=77 an unset max_tokens was sent as {request.max_tokens!r}",
        )

    unconfigured = harness.build()
    start = len(harness.sent_requests())
    harness.queue_upstream(UpstreamStub(content=_VALID_REPLY))
    await _attempt(unconfigured)
    for request in _requests_since(harness, start):
        _fail_unless(
            request.temperature is not None,
            "an unset temperature was sent as a literal null",
        )
        _fail_unless(
            request.max_tokens is not None,
            "an unset max_tokens was sent as a literal null",
        )


async def check_non_text_parts_reach_upstream(harness: LLMClientTestHarness) -> None:
    """Every non-text content part reaches the upstream; raising instead passes, dropping one does not (§3.2)."""
    instruction = "Transcribe both attachments."
    message = LLMMessage(
        role="user",
        content=[
            TextPart(text=instruction),
            ImageUrlPart(image_url=ImageUrlInner(url=_PNG_URI)),
            ImageUrlPart(image_url=ImageUrlInner(url=_PDF_URI)),
        ],
    )
    client = harness.build()
    start = len(harness.sent_requests())
    harness.queue_upstream(UpstreamStub(content=_VALID_REPLY))
    outcome = await _attempt(client, [message])
    requests = harness.sent_requests()[start:]
    _fail_unless(
        bool(requests) or isinstance(outcome, Exception),
        "a reply was returned but no request was sent",
    )
    for request in requests:
        parts = [
            part
            for sent in request.messages
            if isinstance(sent.content, list)
            for part in sent.content
        ]
        sent_uris = {
            part.image_url.url for part in parts if isinstance(part, ImageUrlPart)
        }
        missing = sorted({_PNG_URI, _PDF_URI} - sent_uris)
        _fail_unless(
            not missing, f"non-text parts were dropped from the request: {missing}"
        )
        _fail_unless(
            any(
                isinstance(part, TextPart) and part.text == instruction
                for part in parts
            ),
            "the text part next to the attachments was dropped or rewritten",
        )


async def check_cancellation_propagates(harness: LLMClientTestHarness) -> None:
    """Cancelling an in-flight call ends it with ``asyncio.CancelledError``, not a result or another error (§3.3)."""
    client = harness.build()
    start = len(harness.sent_requests())
    harness.queue_upstream(UpstreamStub(hang=True))
    task = asyncio.create_task(client.achat(_prompt(), response_format=_Probe))
    await _wait_for_request(harness, start, task)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=_CALL_TIMEOUT_S)
    if not done:
        task.cancel()
        raise AssertionError(
            f"the call was still running {_CALL_TIMEOUT_S}s after being cancelled"
        )
    if not task.cancelled():
        outcome = task.exception() or task.result()
        raise AssertionError(
            f"a cancelled call ended with {outcome!r} instead of CancelledError"
        )


async def _wait_for_request(
    harness: LLMClientTestHarness, start: int, task: asyncio.Task[str]
) -> None:
    """Wait until the call has reached the upstream, so the cancellation lands while it is in flight.

    Raises:
        AssertionError: If the call finished, or sent nothing, before the deadline.
    """
    try:
        async with asyncio.timeout(_CALL_TIMEOUT_S):
            while len(harness.sent_requests()) <= start:
                if task.done():
                    outcome = task.exception() or task.result()
                    raise AssertionError(
                        f"the call ended with {outcome!r} before reaching a hanging upstream"
                    )
                await asyncio.sleep(_POLL_INTERVAL_S)
    except TimeoutError:
        task.cancel()
        raise AssertionError(
            f"the call sent no request within {_CALL_TIMEOUT_S}s"
        ) from None


async def check_refusal_raises(harness: LLMClientTestHarness) -> None:
    """An upstream-flagged refusal raises, even when its content is valid JSON (§3.4)."""
    client = harness.build()
    cases = {
        "refusal with valid JSON content": UpstreamStub(
            content=json.dumps({"text": _REFUSAL}), refusal=_REFUSAL
        ),
        "refusal without content": UpstreamStub(content=None, refusal=_REFUSAL),
    }
    for label, stub in cases.items():
        harness.queue_upstream(stub)
        outcome = await _attempt(client)
        _fail_unless(
            isinstance(outcome, Exception),
            f"{label}: the call returned {outcome!r} instead of raising",
        )


async def check_truncation_is_never_returned(harness: LLMClientTestHarness) -> None:
    """A reported truncation or abnormal stop is never returned, even as a closed valid object (§3.5).

    Only the truncated answer is scripted and the upstream fails once the script runs out, so a client that retries
    ends in an error as well; returning anything means the truncated reply was handed over.
    """
    client = harness.build()
    reasons: tuple[FinishReason, ...] = ("length", "content_filter", "abort", "error")
    for reason in reasons:
        start = len(harness.sent_requests())
        harness.queue_upstream(
            UpstreamStub(content=_CLOSED_TRUNCATED_REPLY, finish_reason=reason)
        )
        outcome = await _attempt(client, max_tokens=50)
        _fail_unless(
            isinstance(outcome, Exception),
            f"finish_reason={reason!r}: the call returned {outcome!r} instead of retrying or raising",
        )
        for request in harness.sent_requests()[start:]:
            _fail_unless(
                request.max_tokens == 50,
                f"finish_reason={reason!r}: a request carried max_tokens={request.max_tokens!r}, not the caller's 50",
            )


async def check_invalid_replies_are_not_returned(harness: LLMClientTestHarness) -> None:
    """A reply that does not validate against ``response_format`` is not handed to the operator (§7.1)."""
    client = harness.build()
    cases: dict[str, UpstreamStub] = {
        "leading prose": UpstreamStub(content=f"Sure, here it is: {_VALID_REPLY}"),
        "markdown fence": UpstreamStub(content=f"```json\n{_VALID_REPLY}\n```"),
        "missing required field": UpstreamStub(content="{}"),
        "unclosed object": UpstreamStub(content='{"text": "cut', finish_reason=None),
    }
    for label, stub in cases.items():
        harness.queue_upstream(stub)
        outcome = await _attempt(client)
        if isinstance(outcome, str):
            _fail_unless(
                _parses(outcome),
                f"{label}: returned {outcome!r}, which does not validate",
            )


async def check_missing_finish_reason_is_accepted(
    harness: LLMClientTestHarness,
) -> None:
    """A valid reply with no reported finish reason is returned normally (§7.1)."""
    client = harness.build()
    harness.queue_upstream(UpstreamStub(content=_VALID_REPLY, finish_reason=None))
    outcome = await _attempt(client)
    _fail_unless(
        isinstance(outcome, str)
        and _parses(outcome)
        and _Probe.model_validate_json(outcome).text == "complete",
        f"a valid reply without a finish reason came back as {outcome!r}",
    )


async def check_retries_are_bounded(harness: LLMClientTestHarness) -> None:
    """A call facing nothing but invalid replies still finishes before the deadline; not retrying passes (§7.3)."""
    client = harness.build()
    harness.queue_upstream(*(UpstreamStub(content="not json at all") for _ in range(8)))
    await _attempt(client)


REQUIRED_CHECKS: Final[tuple[ConformanceCheck, ...]] = (
    check_explicit_values_win_over_defaults,
    check_retries_never_raise_max_tokens,
    check_none_means_default_or_omitted,
    check_non_text_parts_reach_upstream,
    check_cancellation_propagates,
    check_refusal_raises,
    check_truncation_is_never_returned,
)
"""The clauses of the contract an operator cannot check for itself; every implementation must pass all of them."""

OPTIONAL_CHECKS: Final[tuple[ConformanceCheck, ...]] = (
    check_invalid_replies_are_not_returned,
    check_missing_finish_reason_is_accepted,
    check_retries_are_bounded,
)
"""Recommendations that lower the host's retry cost; failing them is not a breach of the contract."""
