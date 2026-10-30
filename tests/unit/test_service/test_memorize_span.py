"""``memorize`` opens an add, deferred-buffer, or flush span.

The boundary stage cuts nothing — this asserts only the span wrapping +
name selection (add vs buffer vs flush) driven by ``is_final`` /
``defer_extraction``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

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
from everos.memory import IngestResult
from everos.memory.extract.boundary import BoundaryOutcome
from everos.service import MemoryService
from tests.fakes import make_memory_service


class _EmptyBoundary:
    """A boundary stage that buffers everything and cuts nothing."""

    async def buffer(self, ingested: IngestResult, **_: Any) -> BoundaryOutcome:
        return _nothing_cut()

    async def detect(self, ingested: IngestResult, **_: Any) -> BoundaryOutcome:
        return _nothing_cut()


def _nothing_cut() -> BoundaryOutcome:
    return BoundaryOutcome(
        cells=[],
        memcell_ids=[],
        per_cell_message_ids=[],
        per_cell_all_senders=[],
        status="accumulated",
        message_count=0,
    )


@pytest.fixture
def mm() -> MemoryService:
    return make_memory_service(boundary=_EmptyBoundary())


@pytest.fixture(autouse=True)
def _patch() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    shutdown_tracing()
    init_tracing(
        Settings().observability.model_copy(update={"enabled": True}),
        span_processor=SimpleSpanProcessor(exporter),
    )
    yield exporter
    shutdown_tracing()


async def test_add_emits_memory_add_span(
    mm: MemoryService, _patch: InMemorySpanExporter
) -> None:
    await mm._memorize({"session_id": "s1", "messages": []}, is_final=False)
    force_flush()
    spans = {s.name: s for s in _patch.get_finished_spans()}
    assert "everos.memory.add" in spans
    assert spans["everos.memory.add"].attributes["langfuse.session.id"] == "s1"


async def test_flush_emits_memory_flush_span(
    mm: MemoryService, _patch: InMemorySpanExporter
) -> None:
    await mm._memorize({"session_id": "s2", "messages": []}, is_final=True)
    force_flush()
    names = {s.name for s in _patch.get_finished_spans()}
    assert "everos.memory.flush" in names


async def test_deferred_add_emits_memory_buffer_span(
    mm: MemoryService, _patch: InMemorySpanExporter
) -> None:
    await mm._memorize({"session_id": "s3", "messages": []}, defer_extraction=True)
    force_flush()
    names = {s.name for s in _patch.get_finished_spans()}
    assert "everos.memory.buffer" in names
