"""``cascade sync`` / ``rebuild`` drain until a pass processes nothing.

The worker bounds one ``drain_until_empty`` (``max_passes`` × batch size).
A ledger the one-time reconcile queues after an upgrade can exceed that
bound, and the command must not report success with rows still pending.
"""

from __future__ import annotations

from everos.entrypoints.cli.commands.cascade import _drain_fully


class _BoundedOrchestrator:
    """Stands in for the real one: a drain returns at most ``bound`` rows."""

    def __init__(self, queued: int, bound: int) -> None:
        self.queued = queued
        self.bound = bound
        self.drains = 0

    async def sync_once(self) -> int:
        return await self.drain_once()

    async def drain_once(self) -> int:
        self.drains += 1
        took = min(self.queued, self.bound)
        self.queued -= took
        return took


async def test_drain_fully_keeps_draining_past_the_worker_bound() -> None:
    orchestrator = _BoundedOrchestrator(queued=5003, bound=5000)
    assert await _drain_fully(orchestrator) == 5003  # type: ignore[arg-type]
    assert orchestrator.queued == 0
    assert orchestrator.drains == 3  # 5000, 3, then the empty pass that stops it


async def test_drain_fully_stops_on_an_empty_queue() -> None:
    orchestrator = _BoundedOrchestrator(queued=0, bound=5000)
    assert await _drain_fully(orchestrator) == 0  # type: ignore[arg-type]
    assert orchestrator.drains == 2  # the scan's drain and one confirming pass
