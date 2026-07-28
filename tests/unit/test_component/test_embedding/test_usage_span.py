"""``OpenAIEmbeddingProvider`` surfaces token usage onto the active span.

The OpenAI embeddings response carries ``usage``; the provider records it
via ``set_generation_usage`` (no contract change) so the
``everos.search.embed_query`` embedding span shows model + input tokens.
Captured via an in-memory exporter.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from everos.component.embedding.openai_provider import OpenAIEmbeddingProvider
from everos.config.settings import ObservabilitySettings
from everos.core.observability.tracing import (
    force_flush,
    init_tracing,
    memory_span,
    shutdown_tracing,
)


class _FakeEmbeddings:
    def __init__(self, response: object) -> None:
        self._response = response

    async def create(
        self, *, model: str, input: list[str], dimensions: object = None
    ) -> object:
        return self._response


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.embeddings = _FakeEmbeddings(response)


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


def _provider(response: object) -> OpenAIEmbeddingProvider:
    provider = OpenAIEmbeddingProvider(
        model="emb-m", api_key="k", base_url="http://x", dim=8
    )
    provider._client = _FakeClient(response)  # type: ignore[assignment]
    return provider


async def test_embed_records_usage_to_active_span(
    captured: InMemorySpanExporter,
) -> None:
    resp = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1] * 8)],
        usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
    )
    provider = _provider(resp)
    with memory_span("everos.search.embed_query", observation_type="embedding"):
        vec = await provider.embed("hello")
    assert len(vec) == 8
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.model"] == "emb-m"
    assert attrs["gen_ai.usage.input_tokens"] == 5


async def test_embed_without_usage_does_not_raise(
    captured: InMemorySpanExporter,
) -> None:
    resp = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 8)], usage=None)
    provider = _provider(resp)
    with memory_span("everos.search.embed_query", observation_type="embedding"):
        vec = await provider.embed("hello")
    assert len(vec) == 8
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert "gen_ai.usage.input_tokens" not in attrs
