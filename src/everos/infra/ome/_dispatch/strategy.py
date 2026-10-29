"""The engine's record of one registered strategy, and its run context.

:class:`StrategyMeta` is what the registry, dispatcher and runner work
with; :func:`everos.infra.ome.adapter.to_ome_strategy` builds it from a
domain ``StrategySpec``. ``func`` is the adapted function: it receives the
event and a :class:`RunContext` and publishes follow-up events only through
``RunContext.emit``, which enforces the declared ``emits``.
"""

from __future__ import annotations

__all__ = ["RunContext", "StrategyFunc", "StrategyMeta"]

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from everos.infra.ome.exceptions import EmitNotDeclaredError
from everos.memory.ports import Counter, Event, Trigger

type StrategyFunc = Callable[[Event, RunContext], Awaitable[None]]
type AppliesToPredicate = str | Callable[[Event], bool] | None


class RunContext:
    """Per-attempt channel through which a strategy publishes follow-ups.

    Args:
        strategy_name: Name of the running strategy (for errors).
        declared_emits: Event classes the strategy may publish.
        emit_hook: Dispatches one event through the engine.
    """

    def __init__(
        self,
        *,
        strategy_name: str,
        declared_emits: frozenset[type[Event]],
        emit_hook: Callable[[Event], Awaitable[None]],
    ) -> None:
        self._strategy_name = strategy_name
        self._declared = declared_emits
        self._emit_hook = emit_hook

    async def emit(self, event: Event) -> None:
        """Dispatch ``event``.

        Raises:
            EmitNotDeclaredError: ``event``'s class is not declared.
        """
        if type(event) not in self._declared:
            raise EmitNotDeclaredError(strategy=self._strategy_name, event=event)
        await self._emit_hook(event)


@dataclass(frozen=True)
class StrategyMeta:
    """One registered strategy as the engine sees it."""

    name: str
    trigger: Trigger
    emits: frozenset[type[Event]]
    applies_to: AppliesToPredicate
    gate: Counter | None
    max_retries: int | None
    enabled: bool
    func: StrategyFunc
