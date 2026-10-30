"""everalgo ``LLMClient`` conformance suite run against :class:`OpenAIProvider`.

The harness captures requests at the HTTP transport (``httpx.MockTransport``
under the real ``openai.AsyncOpenAI``), so ``EffectiveRequest`` is read off the
wire body rather than copied from ``achat``'s arguments (``llm-contract.md``
§9.1). Every check gets a fresh harness, as the suite requires.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from everalgo.llm import LLMMessage
from everalgo.testing import (
    NOT_SENT,
    OPTIONAL_CHECKS,
    REQUIRED_CHECKS,
    EffectiveRequest,
    UpstreamStub,
)
from everalgo.testing.conformance import ConformanceCheck

from everos.component.llm import OpenAIProvider

_BASE_URL = "http://test.local/v1"


class OpenAIProviderHarness:
    """``LLMClientTestHarness`` for :class:`OpenAIProvider` over a mock transport."""

    def __init__(self) -> None:
        self._queue: list[UpstreamStub] = []
        self._sent: list[dict[str, Any]] = []

    def build(self, **defaults: Any) -> OpenAIProvider:
        # SDK transport retries off: an exhausted script answers 500, and
        # backing off between those only slows the suite down.
        return OpenAIProvider(
            model="test-model",
            api_key="test-key",
            base_url=_BASE_URL,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self._handle)),
            **defaults,
        )

    def queue_upstream(self, *responses: UpstreamStub) -> None:
        self._queue.extend(responses)

    def sent_requests(self) -> list[EffectiveRequest]:
        return [_effective_request(body) for body in self._sent]

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        self._sent.append(json.loads(request.content))
        if not self._queue:
            return httpx.Response(500, json={"error": {"message": "script exhausted"}})
        stub = self._queue.pop(0)
        if stub.hang:
            await asyncio.Event().wait()
        return httpx.Response(200, json=_completion_body(stub))


def _completion_body(stub: UpstreamStub) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": stub.content,
                    "refusal": stub.refusal,
                },
                "finish_reason": stub.finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _effective_request(body: dict[str, Any]) -> EffectiveRequest:
    return EffectiveRequest(
        messages=[LLMMessage.model_validate(m) for m in body["messages"]],
        temperature=body.get("temperature", NOT_SENT),
        max_tokens=body.get("max_tokens", NOT_SENT),
    )


def _check_id(check: ConformanceCheck) -> str:
    return check.__name__


@pytest.mark.parametrize("check", REQUIRED_CHECKS, ids=_check_id)
async def test_required_check(check: ConformanceCheck) -> None:
    await check(OpenAIProviderHarness())


@pytest.mark.parametrize("check", OPTIONAL_CHECKS, ids=_check_id)
async def test_optional_check(check: ConformanceCheck) -> None:
    await check(OpenAIProviderHarness())
