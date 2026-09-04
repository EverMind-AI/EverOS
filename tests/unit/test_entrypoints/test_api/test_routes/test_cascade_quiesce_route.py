"""``POST /cascade/quiesce`` — freeze the projection before the read stages.

Cascade rewrites files to keep LanceDB in step with the markdown that is the real
source of truth: ``merge_insert`` lands each write in a fresh fragment,
``optimize()`` merges the accumulation, and ``prune()`` reclaims the superseded
copies after a 60s retention window (short on purpose -- the storage soak measured
a 300s window retaining ~24 full-table copies).

Prune's safety argument is that 60s outlives an in-flight read, which it documents
as "sub-second to a few seconds". A multi-round retrieval whose decider reads full
episode text spends 56-72s in one search call (measured), so it does not: the
reader resolves a version, works, and comes back to reclaimed files --

    LanceError(IO): Object at location .../_indices/<uuid>/tokens.lance not found

The first full LoCoMo run lost 225 of 493 questions to that: HTTP 500 per search,
an empty context handed to the answer model, a scored zero. These tests pin the
endpoint that removes the race -- drain first (else the frozen index is an
incomplete projection), then stop (else prune keeps running).
"""

from __future__ import annotations

import unittest.mock as mock
from dataclasses import dataclass

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.routes.cascade import router as cascade_router
from everos.memory.cascade import CascadeOrchestrator


@dataclass
class _Summary:
    """Shape of ``md_change_state_repo.queue_summary()``."""

    pending: int = 0
    done: int = 0
    failed_retryable: int = 0
    failed_permanent: int = 0


def _orch(*, before: int, after: int, drained: int) -> mock.MagicMock:
    """An autospec orchestrator — ``isinstance(_, CascadeOrchestrator)`` holds."""
    orch = mock.create_autospec(CascadeOrchestrator, instance=True)
    orch.queue_summary.side_effect = [
        _Summary(pending=before),
        _Summary(pending=after, failed_permanent=2),
    ]
    orch.sync_once.return_value = drained
    return orch


def _client(orch: object | None) -> AsyncClient:
    app = FastAPI()
    app.include_router(cascade_router, prefix="/api/v1")
    app.state.lifespan_data = {"cascade": orch} if orch is not None else {}
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_drains_before_stopping() -> None:
    """The order IS the contract: stopping first freezes a partial index, which
    then reads as a healthy store that quietly under-recalls."""
    orch = _orch(before=7, after=0, drained=7)
    order: list[str] = []
    orch.sync_once.side_effect = lambda **_: order.append("drain") or 7
    orch.stop.side_effect = lambda: order.append("stop")

    async with _client(orch) as c:
        resp = await c.post("/api/v1/cascade/quiesce")

    assert resp.status_code == 200, resp.text
    assert order == ["drain", "stop"], "stopped before draining"


async def test_reports_the_queue_on_both_sides() -> None:
    """``pending_before`` is how a caller discovers the projection was behind --
    which the benchmark's ``add.done`` marker does not tell it, because that
    marker tracks the OME extraction queue, not this one."""
    async with _client(_orch(before=42, after=0, drained=42)) as c:
        resp = await c.post("/api/v1/cascade/quiesce")

    body = resp.json()
    assert body["quiesced"] is True
    assert body["pending_before"] == 42
    assert body["pending_after"] == 0
    assert body["drained"] == 42
    assert body["failed_permanent"] == 2


async def test_leftover_pending_is_surfaced_not_swallowed() -> None:
    """A drain that could not finish means the index is NOT a full projection of
    the markdown; a caller that reads it anyway scores the gap as a miss."""
    async with _client(_orch(before=10, after=3, drained=7)) as c:
        resp = await c.post("/api/v1/cascade/quiesce")

    assert resp.status_code == 200
    assert resp.json()["pending_after"] == 3


async def test_no_cascade_is_503_not_a_silent_success() -> None:
    """A read-only server has nothing to quiesce. Answering 200 would let the
    caller believe it froze a projection that was never running."""
    async with _client(None) as c:
        resp = await c.post("/api/v1/cascade/quiesce")

    assert resp.status_code == 503
    assert "quiesce" in resp.text.lower()


async def test_stop_is_called_exactly_once() -> None:
    """Quiesce is one-way; a restart is what brings the projection back."""
    orch = _orch(before=1, after=0, drained=1)
    async with _client(orch) as c:
        await c.post("/api/v1/cascade/quiesce")
    assert orch.stop.await_count == 1
