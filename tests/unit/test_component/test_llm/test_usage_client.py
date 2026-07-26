"""``UsageRecordingClient`` — wraps an LLM client and records token usage
onto the active span, without altering the response or the call.

The extractor calls happen inside an ``everos.extract`` generation span; the
wrapper's job is to surface ``response.usage`` there (Langfuse computes cost
from model + tokens). Everything else delegates to the wrapped client.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from everalgo.llm import ChatMessage, ChatResponse, Usage
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from everos.component.llm._usage_client import UsageRecordingClient
from everos.config.settings import ObservabilitySettings
from everos.core.observability.tracing import (
    force_flush,
    init_tracing,
    memory_span,
    shutdown_tracing,
)


class _FakeInner:
    """Minimal stand-in LLM client that returns a preset response."""

    def __init__(self, response: ChatResponse | None) -> None:
        self._response = response
        self.calls: list[Any] = []
        self.some_attr = 42

    async def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        self.calls.append((messages, kwargs))
        assert self._response is not None
        return self._response


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    shutdown_tracing()
    yield
    shutdown_tracing()


@pytest.fixture
def captured() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    yield exporter


async def test_records_usage_to_active_span(captured: InMemorySpanExporter) -> None:
    resp = ChatResponse(
        content="x", model="gpt-x", usage=Usage(prompt_tokens=7, completion_tokens=3)
    )
    client = UsageRecordingClient(_FakeInner(resp))
    with memory_span("everos.extract", observation_type="generation"):
        out = await client.chat([ChatMessage(role="user", content="hi")], model="gpt-x")
    assert out is resp
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.model"] == "gpt-x"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 3


async def test_passes_through_when_usage_absent(
    captured: InMemorySpanExporter,
) -> None:
    resp = ChatResponse(content="x", model="gpt-x", usage=None)
    client = UsageRecordingClient(_FakeInner(resp))
    with memory_span("everos.extract", observation_type="generation"):
        out = await client.chat([ChatMessage(role="user", content="hi")])
    assert out is resp
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.model"] == "gpt-x"
    assert "gen_ai.usage.input_tokens" not in attrs


async def test_delegates_unknown_attributes() -> None:
    client = UsageRecordingClient(_FakeInner(None))
    assert client.some_attr == 42


async def test_chat_forwards_kwargs_to_inner() -> None:
    resp = ChatResponse(content="x", model="m")
    inner = _FakeInner(resp)
    client = UsageRecordingClient(inner)
    await client.chat([ChatMessage(role="user", content="hi")], temperature=0.5)
    assert inner.calls[0][1]["temperature"] == 0.5
