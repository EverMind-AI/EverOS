"""``OpenAIProvider.achat`` -- failure reasons, observer reports, request shape.

The conformance suite (``test_conformance.py``) pins the contract clauses;
this file pins EverOS's own choices on top: which ``LLMCallError.reason`` each
upstream outcome maps to, what the observer is told per attempt, how
defaults, ``None`` and ``extra`` shape the request body.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from everalgo.llm import LLMMessage, StrictOutputModel
from pydantic import ValidationError

from everos.component.llm import LLMCallError, OpenAIProvider, TokenUsage

_VALID = '{"text": "ok"}'


class _Reply(StrictOutputModel):
    text: str


class _Upstream:
    """Scripted Chat Completions endpoint that records every request body."""

    def __init__(self, *responses: httpx.Response | Callable[[], Any]) -> None:
        self._responses = list(responses)
        self.bodies: list[dict[str, Any]] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        response = self._responses.pop(0)
        if callable(response):
            return await response()
        return response


def _completion(
    content: str | None = _VALID,
    *,
    finish_reason: str | None = "stop",
    refusal: str | None = None,
    usage: dict[str, int] | None = None,
    choices: bool = True,
) -> httpx.Response:
    message = {"role": "assistant", "content": content, "refusal": refusal}
    body: dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "served-model",
        "choices": (
            [{"index": 0, "message": message, "finish_reason": finish_reason}]
            if choices
            else []
        ),
    }
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


class _RecordingObserver:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def on_completion(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _provider(upstream: _Upstream, **overrides: Any) -> OpenAIProvider:
    kwargs: dict[str, Any] = {
        "model": "configured-model",
        "api_key": "k",
        "base_url": "http://test.local/v1",
        "max_retries": 0,
        "http_client": httpx.AsyncClient(
            transport=httpx.MockTransport(upstream.handle)
        ),
    }
    return OpenAIProvider(**(kwargs | overrides))


def _prompt() -> list[LLMMessage]:
    return [LLMMessage(role="user", content="reply as JSON")]


# ---------------------------------------------------------------------------
# LLMCallError.reason
# ---------------------------------------------------------------------------


async def test_refusal_raises_refusal() -> None:
    upstream = _Upstream(_completion(_VALID, refusal="no"))
    with pytest.raises(LLMCallError) as info:
        await _provider(upstream).achat(_prompt(), response_format=_Reply)
    assert info.value.reason == "refusal"


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
async def test_cut_short_raises_without_retry(finish_reason: str) -> None:
    upstream = _Upstream(_completion(_VALID, finish_reason=finish_reason))
    with pytest.raises(LLMCallError) as info:
        await _provider(upstream, invalid_reply_retries=3).achat(
            _prompt(), response_format=_Reply
        )
    assert info.value.reason == "cut_short"
    assert len(upstream.bodies) == 1


async def test_absent_content_raises_no_content() -> None:
    upstream = _Upstream(_completion(None))
    with pytest.raises(LLMCallError) as info:
        await _provider(upstream).achat(_prompt(), response_format=_Reply)
    assert info.value.reason == "no_content"


async def test_exhausted_validation_raises_invalid_reply_chained() -> None:
    upstream = _Upstream(_completion("nope"), _completion("{}"))
    with pytest.raises(LLMCallError) as info:
        await _provider(upstream, invalid_reply_retries=1).achat(
            _prompt(), response_format=_Reply
        )
    assert info.value.reason == "invalid_reply"
    assert isinstance(info.value.__cause__, ValidationError)
    assert len(upstream.bodies) == 2


async def test_sdk_failure_raises_transport() -> None:
    upstream = _Upstream(httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(LLMCallError) as info:
        await _provider(upstream).achat(_prompt(), response_format=_Reply)
    assert info.value.reason == "transport"


@pytest.mark.parametrize(
    "unusable",
    [
        lambda: _completion(_VALID, finish_reason="abort"),
        lambda: _completion(_VALID, finish_reason="error"),
        lambda: _completion(choices=False),
    ],
    ids=["abort", "error", "no-choices"],
)
async def test_resendable_upstream_failures(
    unusable: Callable[[], httpx.Response],
) -> None:
    """``abort`` / ``error`` / no choices are resent, then raise ``transport``."""
    recovered = _Upstream(unusable(), _completion())
    assert await _provider(recovered).achat(_prompt(), response_format=_Reply) == _VALID

    exhausted = _Upstream(unusable(), unusable())
    with pytest.raises(LLMCallError) as info:
        await _provider(exhausted).achat(_prompt(), response_format=_Reply)
    assert info.value.reason == "transport"


def test_negative_invalid_reply_retries_rejected() -> None:
    with pytest.raises(ValueError, match="invalid_reply_retries"):
        OpenAIProvider(model="m", api_key="k", invalid_reply_retries=-1)


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


async def test_observer_sees_successful_completion() -> None:
    observer = _RecordingObserver()
    upstream = _Upstream(
        _completion(
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        )
    )
    await _provider(upstream, observer=observer).achat(
        _prompt(), response_format=_Reply
    )
    assert observer.calls == [
        {
            "model": "served-model",
            "usage": TokenUsage(input_tokens=7, output_tokens=3),
            "finish_reason": "stop",
            "attempt": 1,
        }
    ]


async def test_observer_sees_every_attempt_of_a_retry() -> None:
    observer = _RecordingObserver()
    upstream = _Upstream(_completion("not json"), _completion())
    await _provider(upstream, observer=observer).achat(
        _prompt(), response_format=_Reply
    )
    assert [call["attempt"] for call in observer.calls] == [1, 2]
    assert observer.calls[0]["usage"] is None


async def test_observer_sees_a_refused_completion() -> None:
    observer = _RecordingObserver()
    upstream = _Upstream(_completion(_VALID, refusal="no"))
    with pytest.raises(LLMCallError):
        await _provider(upstream, observer=observer).achat(
            _prompt(), response_format=_Reply
        )
    assert [call["attempt"] for call in observer.calls] == [1]


# ---------------------------------------------------------------------------
# Request shape
# ---------------------------------------------------------------------------


async def test_unset_temperature_is_not_sent() -> None:
    upstream = _Upstream(_completion())
    await _provider(upstream).achat(_prompt(), response_format=_Reply)
    assert "temperature" not in upstream.bodies[0]
    assert "max_tokens" not in upstream.bodies[0]


async def test_zero_temperature_is_sent() -> None:
    upstream = _Upstream(_completion())
    await _provider(upstream, temperature=0.7).achat(
        _prompt(), response_format=_Reply, temperature=0.0
    )
    assert upstream.bodies[0]["temperature"] == 0.0


# ---------------------------------------------------------------------------
# Retry temperature ([llm] retry_temperature)
# ---------------------------------------------------------------------------


async def test_a_retry_is_sampled_at_the_retry_temperature() -> None:
    """At 0.0 the identical request tends to reproduce the identical bad reply."""
    upstream = _Upstream(_completion("not json"), _completion("{}"), _completion())
    await _provider(
        upstream, temperature=0.0, retry_temperature=0.3, invalid_reply_retries=2
    ).achat(_prompt(), response_format=_Reply)
    assert [body["temperature"] for body in upstream.bodies] == [0.0, 0.3, 0.3]
    # Nothing else in the request changes.
    first, *retries = upstream.bodies
    for body in retries:
        assert {k: v for k, v in body.items() if k != "temperature"} == {
            k: v for k, v in first.items() if k != "temperature"
        }


async def test_a_first_attempt_that_succeeds_sends_only_the_temperature() -> None:
    upstream = _Upstream(_completion())
    await _provider(upstream, temperature=0.0, retry_temperature=0.3).achat(
        _prompt(), response_format=_Reply
    )
    assert [body["temperature"] for body in upstream.bodies] == [0.0]


async def test_a_call_temperature_wins_and_a_retry_never_lowers_it() -> None:
    upstream = _Upstream(_completion("not json"), _completion())
    await _provider(upstream, temperature=0.0, retry_temperature=0.3).achat(
        _prompt(), response_format=_Reply, temperature=0.8
    )
    assert [body["temperature"] for body in upstream.bodies] == [0.8, 0.8]


@pytest.mark.parametrize(
    ("temperature", "retry_temperature"), [(None, 0.3), (0.0, None), (None, None)]
)
async def test_without_both_temperatures_a_retry_resends_the_request_unchanged(
    temperature: float | None, retry_temperature: float | None
) -> None:
    upstream = _Upstream(_completion("not json"), _completion())
    await _provider(
        upstream, temperature=temperature, retry_temperature=retry_temperature
    ).achat(_prompt(), response_format=_Reply)
    assert upstream.bodies[0] == upstream.bodies[1]
    assert upstream.bodies[0].get("temperature") == temperature


async def test_call_extra_overrides_configured_extra() -> None:
    configured = {"extra_body": {"a": 1}, "top_p": 0.5}
    upstream = _Upstream(_completion())
    await _provider(upstream, extra=configured).achat(
        _prompt(), response_format=_Reply, extra_body={"a": 2}
    )
    body = upstream.bodies[0]
    assert body["a"] == 2  # the SDK merges extra_body into the JSON body
    assert body["top_p"] == 0.5


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [("json_schema", "json_schema"), ("json_object", "json_object")],
)
async def test_response_format_follows_mode(mode: str, expected_type: str) -> None:
    upstream = _Upstream(_completion())
    await _provider(upstream, response_format_mode=mode).achat(
        _prompt(), response_format=_Reply
    )
    assert upstream.bodies[0]["response_format"]["type"] == expected_type


async def test_prompt_only_sends_no_response_format() -> None:
    upstream = _Upstream(_completion())
    await _provider(upstream, response_format_mode="prompt_only").achat(
        _prompt(), response_format=_Reply
    )
    assert "response_format" not in upstream.bodies[0]


async def test_cancellation_propagates() -> None:
    in_flight = asyncio.Event()

    async def hang() -> httpx.Response:
        in_flight.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    upstream = _Upstream(hang)
    task = asyncio.create_task(
        _provider(upstream).achat(_prompt(), response_format=_Reply)
    )
    await in_flight.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# achat_raw -- the decider's unchecked call (parity check 2026-11-10, R1 + R2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "finish_reason", "text"),
    [
        ("not json at all", "stop", "not json at all"),
        ('{"text": "ok"}\nand then more', "length", '{"text": "ok"}\nand then more'),
        (_VALID, "content_filter", _VALID),
        (None, "stop", ""),
    ],
    ids=["unparseable", "length-after-json", "content-filter", "no-content"],
)
async def test_raw_reply_comes_back_unchecked_from_one_request(
    content: str | None, finish_reason: str, text: str
) -> None:
    """No validation, no finish-reason check, no retry: the caller reads it."""
    upstream = _Upstream(_completion(content, finish_reason=finish_reason))

    reply = await _provider(upstream, invalid_reply_retries=3).achat_raw(
        _prompt(), response_format=_Reply
    )

    assert (reply.text, reply.finish_reason) == (text, finish_reason)
    assert len(upstream.bodies) == 1


async def test_raw_reply_carries_the_token_counts() -> None:
    observer = _RecordingObserver()
    upstream = _Upstream(
        _completion(
            usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
        )
    )

    reply = await _provider(upstream, observer=observer).achat_raw(_prompt())

    assert reply.usage == TokenUsage(input_tokens=7, output_tokens=3)
    assert [call["attempt"] for call in observer.calls] == [1]


async def test_raw_reply_without_a_choice_is_a_transport_failure() -> None:
    upstream = _Upstream(_completion(choices=False))
    with pytest.raises(LLMCallError) as info:
        await _provider(upstream).achat_raw(_prompt())
    assert info.value.reason == "transport"
    assert len(upstream.bodies) == 1


async def test_raw_request_is_the_achat_request_without_a_format() -> None:
    """Same body as ``achat`` builds, minus ``response_format`` when none is asked."""
    configured = {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    raw_upstream, chat_upstream = _Upstream(_completion()), _Upstream(_completion())
    await _provider(
        raw_upstream, extra=configured, response_format_mode="prompt_only"
    ).achat_raw(_prompt(), response_format=_Reply, temperature=0.0, max_tokens=512)
    await _provider(
        chat_upstream, extra=configured, response_format_mode="prompt_only"
    ).achat(_prompt(), response_format=_Reply, temperature=0.0, max_tokens=512)

    assert raw_upstream.bodies == chat_upstream.bodies
    assert "response_format" not in raw_upstream.bodies[0]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("json_object", {"type": "json_object"}), ("json_schema", "json_schema")],
)
async def test_raw_request_asks_for_a_format_only_as_the_mode_says(
    mode: str, expected: object
) -> None:
    upstream = _Upstream(_completion())
    await _provider(upstream, response_format_mode=mode).achat_raw(
        _prompt(), response_format=_Reply
    )
    sent = upstream.bodies[0]["response_format"]
    assert sent == expected or sent["type"] == expected

    unasked = _Upstream(_completion())
    await _provider(unasked, response_format_mode=mode).achat_raw(_prompt())
    assert "response_format" not in unasked.bodies[0]
