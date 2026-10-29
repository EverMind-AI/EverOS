"""Turn a domain :class:`~everos.memory.ports.StrategySpec` into an OME job.

A spec's function is pure in the scheduler's sense: it takes ``(event,
deps)`` and returns the events it produced. The adapter is the one place
that binds the dependencies, checks the returned events against the
spec's ``emits`` and publishes them through the run context — so the
emit contract is enforced exactly as before, but by OME rather than by
code the strategy calls.

The whole return value is checked before any of it is published: a
strategy that returns one undeclared event publishes nothing and is
dead-lettered, instead of leaving its declared siblings half-delivered.
"""

from __future__ import annotations

__all__ = ["to_ome_strategy"]

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from everos.infra.ome._dispatch.strategy import RunContext, StrategyMeta
from everos.infra.ome.exceptions import EmitNotDeclaredError
from everos.memory.ports import AppliesTo, Event, StrategySpec

if TYPE_CHECKING:
    from everos.memory.deps import MemoryDeps


def to_ome_strategy(spec: StrategySpec, deps: MemoryDeps) -> StrategyMeta:
    """Bind ``deps`` into ``spec`` and wrap it as a registrable strategy.

    Args:
        spec: The domain's strategy declaration.
        deps: What the strategy function receives on every run.

    Returns:
        A strategy for :meth:`OfflineEngine.register`.
    """

    async def run(event: Event, ctx: RunContext) -> None:
        produced = await spec.fn(event, deps)
        undeclared = next((e for e in produced if type(e) not in spec.emits), None)
        if undeclared is not None:
            raise EmitNotDeclaredError(strategy=spec.name, event=undeclared)
        for follow_up in produced:
            await ctx.emit(follow_up)

    return StrategyMeta(
        name=spec.name,
        trigger=spec.trigger,
        emits=spec.emits,
        applies_to=_bind_applies_to(spec.applies_to, deps),
        gate=spec.gate,
        max_retries=spec.max_retries,
        enabled=spec.enabled,
        func=run,
    )


def _bind_applies_to(
    applies_to: AppliesTo, deps: MemoryDeps
) -> str | Callable[[Event], bool] | None:
    """The dispatcher evaluates ``applies_to(event)``; a callable spec also
    needs the dependencies, which are fixed for the process, so bind them."""
    if applies_to is None or isinstance(applies_to, str):
        return applies_to
    predicate = applies_to

    def applies(event: Any) -> bool:
        return predicate(event, deps)

    return applies
