"""``memory_span`` / ``set_generation_usage`` — the langfuse.* + gen_ai.*
attribute contract applied to spans.

Captured via an in-memory exporter so the emitted attribute keys/values
are asserted directly against §4 of the implementation plan.
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
    capture_input,
    capture_output,
    force_flush,
    init_tracing,
    memory_span,
    set_capture_content,
    set_generation_usage,
    set_redactor,
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


def test_memory_span_sets_langfuse_attributes(
    captured: InMemorySpanExporter,
) -> None:
    with memory_span(
        "everos.memory.search",
        observation_type="retriever",
        session_id="s1",
        user_id="u1",
        metadata={"app_id": "a", "project_id": "p", "agent_id": None},
    ):
        pass
    force_flush()
    span = captured.get_finished_spans()[0]
    attrs = span.attributes
    assert span.name == "everos.memory.search"
    assert attrs["langfuse.observation.type"] == "retriever"
    assert attrs["langfuse.session.id"] == "s1"
    assert attrs["langfuse.user.id"] == "u1"
    assert attrs["langfuse.trace.metadata.app_id"] == "a"
    assert attrs["langfuse.trace.metadata.project_id"] == "p"
    # None-valued metadata is dropped, not emitted as "None".
    assert "langfuse.trace.metadata.agent_id" not in attrs
    assert list(attrs["langfuse.trace.tags"]) == ["everos", "memory"]


def test_set_generation_usage_annotates_current_span(
    captured: InMemorySpanExporter,
) -> None:
    with memory_span("everos.extract", observation_type="generation"):
        set_generation_usage(model="gpt-x", input_tokens=11, output_tokens=22)
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert attrs["gen_ai.request.model"] == "gpt-x"
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 22


def test_set_generation_usage_accumulates_across_calls(
    captured: InMemorySpanExporter,
) -> None:
    # A multi-call operation (e.g. agentic search issuing several chats)
    # inside one span must SUM token usage, not overwrite with the last call.
    with memory_span("everos.search.rank", observation_type="generation"):
        set_generation_usage(model="gpt-x", input_tokens=10, output_tokens=5)
        set_generation_usage(model="gpt-x", input_tokens=3, output_tokens=7)
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert attrs["gen_ai.usage.input_tokens"] == 13
    assert attrs["gen_ai.usage.output_tokens"] == 12


def test_nested_only_span_skips_when_no_active_parent(
    captured: InMemorySpanExporter,
) -> None:
    # A nested_only span with no active parent must NOT start a new root trace
    # (this is what prevented cascade-time embeddings from exploding into one
    # orphan trace per chunk).
    with memory_span(
        "everos.embedding", observation_type="embedding", nested_only=True
    ):
        pass
    force_flush()
    assert captured.get_finished_spans() == ()


def test_nested_only_span_opens_under_active_parent(
    captured: InMemorySpanExporter,
) -> None:
    with (
        memory_span("everos.memory.search", observation_type="retriever"),
        memory_span("everos.embedding", observation_type="embedding", nested_only=True),
    ):
        pass
    force_flush()
    names = {s.name for s in captured.get_finished_spans()}
    assert "everos.embedding" in names
    assert "everos.memory.search" in names


def test_set_generation_usage_outside_span_is_noop() -> None:
    # No active span → must not raise (and nothing to record).
    set_generation_usage(model="x", input_tokens=1, output_tokens=2)


def test_content_dropped_when_capture_off(captured: InMemorySpanExporter) -> None:
    # Default: capture_content off → no observation.input/output emitted.
    with memory_span("everos.extract", observation_type="generation") as span:
        capture_input(span, {"query": "sensitive query"})
        capture_output(span, "secret memory text")
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    assert "langfuse.observation.input" not in attrs
    assert "langfuse.observation.output" not in attrs


def test_content_emitted_when_capture_on(captured: InMemorySpanExporter) -> None:
    set_capture_content(True)
    try:
        with memory_span("everos.memory.search", observation_type="retriever") as span:
            capture_input(span, {"query": "hello"})
            capture_output(span, "world")
    finally:
        set_capture_content(False)
    force_flush()
    attrs = captured.get_finished_spans()[0].attributes
    import json

    assert json.loads(attrs["langfuse.observation.input"]) == {"query": "hello"}
    assert attrs["langfuse.observation.output"] == "world"


def test_content_redacted_and_truncated(captured: InMemorySpanExporter) -> None:
    calls: list[str] = []

    def redact(text: str) -> str:
        calls.append(text)
        return text.replace("SECRET", "***")

    set_redactor(redact)
    set_capture_content(True)
    try:
        with memory_span("everos.extract", observation_type="generation") as span:
            capture_output(span, "SECRET " + "a" * 6000)
    finally:
        set_capture_content(False)
        set_redactor(None)
    force_flush()
    val = captured.get_finished_spans()[0].attributes["langfuse.observation.output"]
    assert "SECRET" not in val
    assert "***" in val
    assert len(val) <= 4096  # truncated
    assert calls  # redaction hook was invoked
