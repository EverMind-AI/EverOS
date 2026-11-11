"""The decider's health: a startup probe, and the rounds every search counts.

``GET /health`` named the embedder, the reranker and the parser but not the
model every search round asks. A decider the endpoint did not serve answered
404 on every round, and every search fell back without a trace in /health.
"""

from __future__ import annotations

import asyncio
from typing import Any

from everalgo.llm import LLMMessage
from everalgo.testing import FakeLLMClient

from everos.component.llm.openai_provider import RawReply
from everos.config.settings import DeciderSettings
from everos.memory.search import DeciderStatus
from everos.memory.search.decider_health import PROBE_TEXT, _public_url
from everos.memory.search.llm_multiround import DeciderFailure
from tests.fakes import static_tuning


class _Endpoint:
    """A decider endpoint reached through ``achat_raw``."""

    def __init__(self, error: Exception | None = None, delay: float = 0.0) -> None:
        self.error = error
        self.delay = delay
        self.calls: list[tuple[list[LLMMessage], dict[str, Any]]] = []

    async def achat_raw(self, messages: list[LLMMessage], **params: Any) -> RawReply:
        self.calls.append((messages, params))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return RawReply(text="O", finish_reason="length", usage=None)

    async def achat(self, messages: list[LLMMessage], **_: Any) -> str:
        raise AssertionError("the probe uses achat_raw")


def _status(llm: Any, **kwargs: Any) -> DeciderStatus:
    tuning = DeciderSettings(extra={"thinking": {"type": "disabled"}})
    return DeciderStatus(
        llm,
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com/v1",
        tuning_source=static_tuning(tuning),
        **kwargs,
    )


async def test_the_probe_asks_what_a_round_asks_with_one_output_token() -> None:
    endpoint = _Endpoint()
    status = _status(endpoint)
    assert status.snapshot().reachable is None  # not probed yet

    await status.probe()

    ((messages, params),) = endpoint.calls
    assert [m.content for m in messages] == [PROBE_TEXT]
    assert params == {
        "temperature": 0.0,
        "thinking": {"type": "disabled"},
        "max_tokens": 1,
    }
    health = status.snapshot()
    assert (health.reachable, health.probe_error) == (True, None)
    assert health.probed_at is not None
    assert (health.model, health.base_url) == (
        "deepseek-v4-flash",
        "https://api.deepseek.com/v1",
    )


async def test_a_failing_probe_says_why() -> None:
    status = _status(_Endpoint(error=RuntimeError("Error code: 404 - no such model")))
    await status.probe()
    health = status.snapshot()
    assert health.reachable is False
    assert health.probe_error == "RuntimeError: Error code: 404 - no such model"


async def test_a_probe_that_hangs_times_out() -> None:
    status = _status(_Endpoint(delay=5.0), timeout=0.05)
    await status.probe()
    health = status.snapshot()
    assert health.reachable is False
    assert health.probe_error == "no reply within 0.05s"


async def test_a_client_without_a_raw_call_is_not_probed() -> None:
    def refuse(*_a: Any, **_k: Any) -> str:
        raise AssertionError("not called")

    status = _status(FakeLLMClient(handler=refuse))
    await status.probe()
    health = status.snapshot()
    assert (health.reachable, health.probe_error, health.probed_at) == (
        None,
        None,
        None,
    )


def test_rounds_are_counted_and_the_last_failure_named() -> None:
    status = _status(_Endpoint())
    status.record(2, ())
    status.record(1, (DeciderFailure(0, "fallback", "NotFoundError: 404", 4),))
    health = status.snapshot()
    assert (health.rounds, health.failed_rounds) == (3, 1)
    assert health.last_failure == "NotFoundError: 404"
    assert health.last_failure_at is not None


def test_the_endpoint_is_reported_without_credentials() -> None:
    assert (
        _public_url("https://user:secret@gw.example.com:8443/v1?key=abc#x")
        == "https://gw.example.com:8443/v1"
    )
    assert _public_url("http://127.0.0.1:9330/v1") == "http://127.0.0.1:9330/v1"
    assert _public_url("") == ""
