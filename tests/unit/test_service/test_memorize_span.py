"""``memorize`` opens an everos.memory.add / everos.memory.flush span.

The inner critical section is mocked out — this asserts only the span
wrapping + name selection (add vs flush) driven by ``is_final``.
"""

from __future__ import annotations

import importlib
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from everos.config import Settings
from everos.core.observability.tracing import (
    force_flush,
    init_tracing,
    shutdown_tracing,
)

mm = importlib.import_module("everos.service.memorize")


@pytest.fixture(autouse=True)
def _patch(monkeypatch: pytest.MonkeyPatch) -> Iterator[InMemorySpanExporter]:
    monkeypatch.setattr(mm, "load_settings", lambda: Settings())
    monkeypatch.setattr(
        mm,
        "_memorize_locked",
        AsyncMock(return_value=mm.MemorizeResult(message_count=0, status="extracted")),
    )

    @asynccontextmanager
    async def _fake_lock(session_id: str) -> AsyncIterator[None]:
        yield

    monkeypatch.setattr(mm, "get_session_lock", _fake_lock)

    exporter = InMemorySpanExporter()
    shutdown_tracing()
    init_tracing(
        Settings().observability.model_copy(update={"enabled": True}),
        span_processor=SimpleSpanProcessor(exporter),
    )
    yield exporter
    shutdown_tracing()


async def test_add_emits_memory_add_span(_patch: InMemorySpanExporter) -> None:
    await mm.memorize({"session_id": "s1", "messages": []}, is_final=False)
    force_flush()
    spans = {s.name: s for s in _patch.get_finished_spans()}
    assert "everos.memory.add" in spans
    assert spans["everos.memory.add"].attributes["langfuse.session.id"] == "s1"


async def test_flush_emits_memory_flush_span(_patch: InMemorySpanExporter) -> None:
    await mm.memorize({"session_id": "s2", "messages": []}, is_final=True)
    force_flush()
    names = {s.name for s in _patch.get_finished_spans()}
    assert "everos.memory.flush" in names
