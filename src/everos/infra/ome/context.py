"""StrategyContext Protocol — injected as second arg to every strategy.

Strategies access run-local state through `run_id` and `logger`, and
chain-emit follow-up events via `emit(event)`. Business IO is NOT mediated
by this Protocol — strategies directly import their persistence adapters
(memory → infra is allowed under the project's DDD layering).
"""

from __future__ import annotations

from typing import Protocol

from structlog.types import FilteringBoundLogger

from everos.infra.ome.events import BaseEvent
from everos.infra.ome.records import RunRecord


class StrategyContext(Protocol):
    """Per-run context handed to a strategy function.

    Attributes:
        run_id: The current RunRecord id.
        logger: Structlog logger with ``strategy_name`` / ``run_id`` /
            ``attempt`` auto-bound.
        emit: Chain-emit a follow-up event (must be in decorator's
            ``emits=[...]``, else EmitNotDeclaredError).
        wait_for_event: Poll until all runs triggered by an event_id
            reach a terminal status.
        list_runs_by_event_id: Return all run records triggered by an
            event_id.
    """

    run_id: str
    logger: FilteringBoundLogger

    async def emit(self, event: BaseEvent) -> None: ...

    async def wait_for_event(
        self,
        event_id: str,
        *,
        timeout: float = 120.0,  # noqa: ASYNC109
    ) -> list[RunRecord]: ...

    async def list_runs_by_event_id(self, event_id: str) -> list[RunRecord]: ...
