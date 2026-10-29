"""The in-strategy ContextVar and the guard that reads it.

``Runner.run`` sets :data:`_CURRENT_STRATEGY` around a strategy call;
:func:`refuse_inside_strategy` wraps the engine methods a strategy must not
call directly (they return their follow-up events instead).

Python copies ContextVar values into child tasks at
``asyncio.create_task`` (by design, for trace propagation), so
``@refuse_inside_strategy`` reliably catches only *same-task* calls.
Never attach it to APS callback methods (``dispatch_run`` /
``run_idle_scan``) — cascade emits would misfire.
``test_engine_chain_emit_through_ctx`` is the regression.

TODO: ``sys._getframe`` walk for a ``Runner.run`` frame is leak-proof.
"""

from __future__ import annotations

import functools
import inspect
from contextvars import ContextVar
from typing import Any

from everos.infra.ome._dispatch.strategy import StrategyMeta
from everos.infra.ome.exceptions import EngineCallFromStrategyError

_CURRENT_STRATEGY: ContextVar[StrategyMeta | None] = ContextVar(
    "current_strategy", default=None
)
"""Set by ``Runner.run`` around ``meta.func(event, ctx)``; read by
``@refuse_inside_strategy``. ``None`` = not inside a strategy frame."""


def refuse_inside_strategy(method: Any) -> Any:
    """Raise :class:`EngineCallFromStrategyError` when called from a strategy.

    Strategies return their follow-up events and OME publishes them through
    the run context, which enforces the declared ``emits``; a direct engine
    call would bypass that contract. Wraps sync and async methods alike.
    """
    if inspect.iscoroutinefunction(method):

        @functools.wraps(method)
        async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            current = _CURRENT_STRATEGY.get()
            if current is not None:
                raise EngineCallFromStrategyError(
                    strategy=current.name, method=method.__name__
                )
            return await method(self, *args, **kwargs)

        return async_wrapper

    @functools.wraps(method)
    def sync_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        current = _CURRENT_STRATEGY.get()
        if current is not None:
            raise EngineCallFromStrategyError(
                strategy=current.name, method=method.__name__
            )
        return method(self, *args, **kwargs)

    return sync_wrapper
