"""The memorize() lock is keyed by (app_id, project_id, session_id).

Buffers and memcells are scoped by app / project, so two sessions that
merely share a ``session_id`` across projects must not wait on each other;
two calls on the very same scope still serialise.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from everos.memory import IngestResult
from everos.memory.extract.boundary import BoundaryOutcome
from everos.service import MemoryService
from tests.fakes import make_memory_service


class _Gate:
    """A boundary stage that holds each call (under the lock) until released."""

    def __init__(self) -> None:
        self.inside: list[str] = []
        self.release = asyncio.Event()
        self.entered = asyncio.Event()

    async def detect(self, ingested: IngestResult, **_: Any) -> BoundaryOutcome:
        key = f"{ingested.project_id}/{ingested.session_id}"
        self.inside.append(key)
        self.entered.set()
        await self.release.wait()
        self.inside.remove(key)
        return BoundaryOutcome(
            cells=[],
            memcell_ids=[],
            per_cell_message_ids=[],
            per_cell_all_senders=[],
            status="accumulated",
            message_count=0,
        )


@pytest.fixture
def gate() -> _Gate:
    return _Gate()


@pytest.fixture
def mm(gate: _Gate) -> MemoryService:
    return make_memory_service(boundary=gate)


async def test_same_session_in_different_projects_runs_side_by_side(
    mm: MemoryService, gate: _Gate
) -> None:
    a = asyncio.create_task(
        mm._memorize({"session_id": "s", "project_id": "p1", "messages": []})
    )
    b = asyncio.create_task(
        mm._memorize({"session_id": "s", "project_id": "p2", "messages": []})
    )
    await asyncio.sleep(0.05)
    assert sorted(gate.inside) == ["p1/s", "p2/s"]  # both inside at once
    gate.release.set()
    await asyncio.gather(a, b)


async def test_same_scope_still_serialises(mm: MemoryService, gate: _Gate) -> None:
    a = asyncio.create_task(
        mm._memorize({"session_id": "s", "project_id": "p1", "messages": []})
    )
    b = asyncio.create_task(
        mm._memorize({"session_id": "s", "project_id": "p1", "messages": []})
    )
    await asyncio.sleep(0.05)
    assert gate.inside == ["p1/s"]  # the second waits on the lock
    gate.release.set()
    await asyncio.gather(a, b)
