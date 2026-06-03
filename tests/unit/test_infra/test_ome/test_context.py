from __future__ import annotations

from typing import Protocol

import structlog

from everos.infra.ome.context import StrategyContext


def test_strategy_context_is_protocol() -> None:
    assert issubclass(StrategyContext, Protocol)  # type: ignore[arg-type]


def test_strategy_context_runtime_attributes() -> None:
    class _Impl:
        run_id = "r1"
        logger = structlog.get_logger("test")

        async def emit(self, event: object) -> None:
            return None

    ctx: StrategyContext = _Impl()
    assert ctx.run_id == "r1"
    assert callable(ctx.emit)
