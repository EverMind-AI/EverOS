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


class StrategyContext(Protocol):
    """Per-run context handed to a strategy function.

    - run_id: the current RunRecord id (string).
    - logger: structlog logger; ``strategy_name`` / ``run_id`` /
      ``attempt`` are auto-injected into every log record in this call
      — strategies don't have to use this specific logger to get those
      fields.
    - emit(event): chain-emit a follow-up event (must be in decorator's
      ``emits=[...]``, else EmitNotDeclaredError).
    """

    run_id: str
    logger: FilteringBoundLogger

    async def emit(self, event: BaseEvent) -> None: ...
