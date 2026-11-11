"""``[llm] temperature`` / ``retry_temperature``: extraction is sampled at 0.0 again.

Until 2026-10-30 the provider defaulted to ``temperature=0.0``, so every extraction
request carried it. The everalgo 2026-10 line (962c63e) made the default "leave it
out", and DeepSeek then samples at 1.0: two runs of the 2.0 audio-cost measurement
(2026-11-11) over the same 10K-word transcript cut 5 and 9 memcells and cost 50%
apart, with no temperature on any of their 59 requests. The setting restores 0.0 for
every ``[llm]`` call; a retry after an unusable reply is sampled at
``retry_temperature``, since at 0.0 the identical request tends to reproduce the
identical bad reply.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import openai
import pytest
from everalgo.llm import LLMMessage
from pydantic import BaseModel

from everos.component.llm import (
    OpenAIProvider,
    build_llm_provider,
    resolve_decider_config,
    resolve_llm_config,
    resolve_multimodal_config,
)
from everos.config import load_settings
from everos.config.settings import LLMSettings


class _Reply(BaseModel):
    ok: bool


@pytest.fixture(autouse=True)
def _llm_env(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv("EVEROS_LLM__API_KEY", "k")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", "http://gw.invalid/v1")
    monkeypatch.setenv("EVEROS_MULTIMODAL__API_KEY", "k")
    monkeypatch.setenv("EVEROS_MULTIMODAL__BASE_URL", "http://mm.invalid/v1")
    load_settings.cache_clear()
    yield
    load_settings.cache_clear()


def _wired(provider: OpenAIProvider, *replies: str) -> list[dict[str, Any]]:
    """Point ``provider`` at a scripted endpoint; returns the bodies it receives."""
    bodies: list[dict[str, Any]] = []
    queue = list(replies)

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": queue.pop(0)},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    provider._client = openai.AsyncOpenAI(
        api_key="k",
        base_url="http://gw.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )
    return bodies


def _extraction_provider() -> OpenAIProvider:
    return build_llm_provider(resolve_llm_config(load_settings()))


async def _ask(provider: OpenAIProvider, **call: Any) -> None:
    await provider.achat(
        [LLMMessage(role="user", content="reply as JSON")],
        response_format=_Reply,
        **call,
    )


def test_the_defaults_are_greedy_with_a_warmer_retry() -> None:
    assert (LLMSettings().temperature, LLMSettings().retry_temperature) == (0.0, 0.3)
    shipped = load_settings().llm
    assert (shipped.temperature, shipped.retry_temperature) == (0.0, 0.3)


async def test_an_extraction_call_sends_zero_by_default() -> None:
    provider = _extraction_provider()
    bodies = _wired(provider, '{"ok": true}')
    await _ask(provider)
    assert bodies[0]["temperature"] == 0.0


async def test_a_per_call_temperature_wins() -> None:
    provider = _extraction_provider()
    bodies = _wired(provider, '{"ok": true}')
    await _ask(provider, temperature=0.6)
    assert bodies[0]["temperature"] == 0.6


async def test_a_retry_after_an_unusable_reply_uses_the_retry_temperature() -> None:
    provider = _extraction_provider()  # invalid_reply_retries = 1 by default
    bodies = _wired(provider, "not json", '{"ok": true}')
    await _ask(provider)
    assert [body["temperature"] for body in bodies] == [0.0, 0.3]


async def test_env_tunes_both(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVEROS_LLM__TEMPERATURE", "0.1")
    monkeypatch.setenv("EVEROS_LLM__RETRY_TEMPERATURE", "0.5")
    load_settings.cache_clear()
    provider = _extraction_provider()
    bodies = _wired(provider, "not json", '{"ok": true}')
    await _ask(provider)
    assert [body["temperature"] for body in bodies] == [0.1, 0.5]


async def test_switched_off_the_field_is_left_out_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``none`` restores the 10-30 behaviour, retries included: nothing is sent."""
    monkeypatch.setenv("EVEROS_LLM__TEMPERATURE", "none")
    load_settings.cache_clear()
    assert load_settings().llm.temperature is None
    provider = _extraction_provider()
    bodies = _wired(provider, "not json", '{"ok": true}')
    await _ask(provider)
    assert ["temperature" in body for body in bodies] == [False, False]
    assert bodies[0] == bodies[1]


def test_the_decider_and_multimodal_providers_are_unchanged() -> None:
    """The decider sends ``[decider] temperature`` per call; neither inherits these."""
    settings = load_settings()
    decider = build_llm_provider(resolve_decider_config(settings))
    multimodal = build_llm_provider(resolve_multimodal_config(settings))
    for provider in (decider, multimodal):
        assert (provider._temperature, provider._retry_temperature) == (None, None)
