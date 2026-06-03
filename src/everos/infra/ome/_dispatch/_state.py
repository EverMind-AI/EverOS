"""ContextVar shared between Runner and OfflineEngine.

Python copies ContextVar values into child tasks at
``asyncio.create_task`` (by design, for trace propagation), so
``@_refuse_inside_strategy`` reliably catches only *same-task* calls.
Never attach it to APS callback methods (``dispatch_run`` /
``run_idle_scan``) — cascade emits would misfire.
``test_engine_chain_emit_through_ctx`` is the regression.

TODO: ``sys._getframe`` walk for a ``Runner.run`` frame is leak-proof.
"""

from __future__ import annotations

from contextvars import ContextVar

from everos.infra.ome.decorator import StrategyMeta

_CURRENT_STRATEGY: ContextVar[StrategyMeta | None] = ContextVar(
    "current_strategy", default=None
)
"""Set by ``Runner.run`` around ``meta.func(event, ctx)``; read by
``@_refuse_inside_strategy``. ``None`` = not inside a strategy frame."""
