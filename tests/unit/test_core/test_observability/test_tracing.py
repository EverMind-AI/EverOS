"""``core.observability.tracing`` — provider lifecycle + tracer facade.

Spans are captured via an in-memory exporter (a ``SimpleSpanProcessor``
injected into ``init_tracing``) so assertions run offline, with no OTLP
endpoint. The provider is kept off the OTel *global* on purpose — the
module holds its own reference — so tests can init / shutdown repeatedly
without hitting OTel's set-global-once guard.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from everos.config.settings import ObservabilitySettings
from everos.core.observability.tracing import (
    force_flush,
    get_tracer,
    init_tracing,
    shutdown_tracing,
)


@pytest.fixture(autouse=True)
def _reset_tracing() -> Iterator[None]:
    """Ensure each test starts and ends with no provider installed."""
    shutdown_tracing()
    yield
    shutdown_tracing()


@pytest.fixture
def captured_spans() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    settings = ObservabilitySettings(enabled=True, endpoint="http://collector.invalid")
    init_tracing(settings, span_processor=SimpleSpanProcessor(exporter))
    yield exporter


def test_disabled_tracer_is_noop_and_never_raises() -> None:
    # No init_tracing → no provider → get_tracer returns a no-op tracer.
    tracer = get_tracer("everos.test")
    with tracer.start_as_current_span("everos.noop") as span:
        span.set_attribute("k", "v")  # must not raise


def test_init_returns_false_when_disabled() -> None:
    assert init_tracing(ObservabilitySettings(enabled=False)) is False


def test_init_returns_true_when_enabled(captured_spans: InMemorySpanExporter) -> None:
    # captured_spans fixture already called init_tracing(enabled=True).
    tracer = get_tracer("x")
    with tracer.start_as_current_span("s"):
        pass
    force_flush()
    assert len(captured_spans.get_finished_spans()) == 1


def test_span_captured_when_enabled(captured_spans: InMemorySpanExporter) -> None:
    tracer = get_tracer("everos.test")
    with tracer.start_as_current_span("everos.memory.search"):
        pass
    force_flush()
    names = [s.name for s in captured_spans.get_finished_spans()]
    assert "everos.memory.search" in names


def test_child_span_nests_under_parent(
    captured_spans: InMemorySpanExporter,
) -> None:
    tracer = get_tracer("everos.test")
    with (
        tracer.start_as_current_span("parent"),
        tracer.start_as_current_span("child"),
    ):
        pass
    force_flush()
    spans = {s.name: s for s in captured_spans.get_finished_spans()}
    assert spans["child"].parent is not None
    assert spans["child"].parent.span_id == spans["parent"].context.span_id


def test_resource_carries_service_name(
    captured_spans: InMemorySpanExporter,
) -> None:
    tracer = get_tracer("x")
    with tracer.start_as_current_span("s"):
        pass
    force_flush()
    span = captured_spans.get_finished_spans()[0]
    assert span.resource.attributes["service.name"] == "everos"


async def test_child_spans_nest_across_asyncio_gather(
    captured_spans: InMemorySpanExporter,
) -> None:
    """OTel context must survive the asyncio.gather task boundary: child
    spans started inside gathered coroutines nest under the parent span,
    not as siblings/roots. This is the mechanism SearchManager.search
    relies on (plan cross-cutting note #1)."""
    import asyncio

    tracer = get_tracer("everos.test")

    async def _child(name: str) -> None:
        with tracer.start_as_current_span(name):
            await asyncio.sleep(0)

    with tracer.start_as_current_span("parent"):
        await asyncio.gather(_child("child_a"), _child("child_b"))
    force_flush()

    spans = {s.name: s for s in captured_spans.get_finished_spans()}
    parent_span_id = spans["parent"].context.span_id
    trace_id = spans["parent"].context.trace_id
    for child in ("child_a", "child_b"):
        assert spans[child].parent is not None
        assert spans[child].parent.span_id == parent_span_id
        assert spans[child].context.trace_id == trace_id


def test_init_tracing_tears_down_previous_provider() -> None:
    """Re-init without an intervening shutdown must not leak the previous
    provider (its export thread + OTLP socket): the old provider is shut
    down first, so its span processor receives ``shutdown()``."""
    from opentelemetry.sdk.trace import SpanProcessor

    class _SpyProcessor(SpanProcessor):
        def __init__(self) -> None:
            self.shutdown_called = False

        def on_start(self, span: object, parent_context: object = None) -> None:
            pass

        def on_end(self, span: object) -> None:
            pass

        def shutdown(self) -> None:
            self.shutdown_called = True

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return True

    settings = ObservabilitySettings(enabled=True, endpoint="http://collector.invalid")
    first, second = _SpyProcessor(), _SpyProcessor()
    init_tracing(settings, span_processor=first)
    init_tracing(settings, span_processor=second)
    assert first.shutdown_called is True
    assert second.shutdown_called is False


def test_resolve_otlp_target_derives_from_langfuse_creds() -> None:
    import base64

    from pydantic import SecretStr

    from everos.core.observability.tracing.provider import _resolve_otlp_target

    settings = ObservabilitySettings(
        enabled=True,
        langfuse_public_key="pk",
        langfuse_secret_key=SecretStr("sk"),
        langfuse_host="https://us.cloud.langfuse.com",
    )
    endpoint, headers = _resolve_otlp_target(settings)
    assert endpoint == "https://us.cloud.langfuse.com/api/public/otel/v1/traces"
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"pk:sk").decode()


def test_resolve_otlp_target_explicit_values_win() -> None:
    from pydantic import SecretStr

    from everos.core.observability.tracing.provider import _resolve_otlp_target

    settings = ObservabilitySettings(
        enabled=True,
        endpoint="http://explicit/v1/traces",
        headers={"Authorization": "Basic explicit"},
        langfuse_public_key="pk",
        langfuse_secret_key=SecretStr("sk"),
        langfuse_host="https://us.cloud.langfuse.com",
    )
    endpoint, headers = _resolve_otlp_target(settings)
    assert endpoint == "http://explicit/v1/traces"
    assert headers["Authorization"] == "Basic explicit"


def test_resolve_otlp_target_no_langfuse_returns_as_is() -> None:
    from everos.core.observability.tracing.provider import _resolve_otlp_target

    settings = ObservabilitySettings(enabled=True, endpoint="http://x/v1/traces")
    endpoint, headers = _resolve_otlp_target(settings)
    assert endpoint == "http://x/v1/traces"
    assert "Authorization" not in headers
