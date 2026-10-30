"""``TracingObserver`` -- completion telemetry onto the active span.

The extractor calls run inside an ``everos.extract`` generation span; the
observer's job is to put each completion's model and token usage there
(Langfuse computes cost from them) and to flag any finish reason other than
``stop``, which is how truncations and upstream aborts get triaged.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import structlog.testing
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from everos.component.llm import TokenUsage
from everos.component.llm._tracing_observer import TracingObserver
from everos.config.settings import ObservabilitySettings
from everos.core.observability.tracing import (
    force_flush,
    init_tracing,
    memory_span,
    shutdown_tracing,
)


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


def test_records_usage_on_active_span(captured: InMemorySpanExporter) -> None:
    with memory_span("everos.extract", observation_type="generation"):
        TracingObserver().on_completion(
            model="gpt-x",
            usage=TokenUsage(input_tokens=7, output_tokens=3),
            finish_reason="stop",
            attempt=1,
        )
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.model"] == "gpt-x"
    assert attrs["gen_ai.usage.input_tokens"] == 7
    assert attrs["gen_ai.usage.output_tokens"] == 3


def test_usage_accumulates_across_attempts(captured: InMemorySpanExporter) -> None:
    """A retried call bills twice; both attempts land on the span."""
    observer = TracingObserver()
    with memory_span("everos.extract", observation_type="generation"):
        for attempt in (1, 2):
            observer.on_completion(
                model="gpt-x",
                usage=TokenUsage(input_tokens=5, output_tokens=2),
                finish_reason="stop",
                attempt=attempt,
            )
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 4


def test_absent_usage_records_model_only(captured: InMemorySpanExporter) -> None:
    with memory_span("everos.extract", observation_type="generation"):
        TracingObserver().on_completion(
            model="gpt-x", usage=None, finish_reason="stop", attempt=1
        )
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.model"] == "gpt-x"
    assert "gen_ai.usage.input_tokens" not in attrs


@pytest.mark.parametrize("finish_reason", ["length", "abort", "content_filter"])
def test_warns_on_non_stop_finish(finish_reason: str) -> None:
    with structlog.testing.capture_logs() as logs:
        TracingObserver().on_completion(
            model="gpt-x", usage=None, finish_reason=finish_reason, attempt=2
        )
    (warning,) = [e for e in logs if e["event"] == "llm_non_stop_finish"]
    assert warning["log_level"] == "warning"
    assert warning["finish_reason"] == finish_reason
    assert warning["attempt"] == 2


@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_silent_on_stop_or_missing_finish(finish_reason: str | None) -> None:
    with structlog.testing.capture_logs() as logs:
        TracingObserver().on_completion(
            model="gpt-x", usage=None, finish_reason=finish_reason, attempt=1
        )
    assert not [e for e in logs if e["event"] == "llm_non_stop_finish"]
