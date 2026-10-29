"""to_ome_strategy: binding deps, enforcing emits, publishing returned events."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from everos.infra.ome.adapter import to_ome_strategy
from everos.infra.ome.exceptions import EmitNotDeclaredError
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Counter, Event, Immediate, StrategySpec
from tests.fakes import make_deps


class _In(Event):
    flag: str = ""


class _Out(Event):
    n: int


class _Other(Event):
    pass


class _Ctx:
    """Collects what the adapter publishes through the run context."""

    def __init__(self) -> None:
        self.emitted: list[Event] = []

    async def emit(self, event: Event) -> None:
        self.emitted.append(event)


def _spec(fn: Callable, **options: object) -> StrategySpec:
    return StrategySpec(
        name="s", fn=fn, trigger=Immediate(on=[_In]), emits=frozenset({_Out}), **options
    )


async def test_fn_gets_bound_deps_and_its_events_are_published() -> None:
    deps = make_deps()
    seen: list[MemoryDeps] = []

    async def fn(event: _In, d: MemoryDeps) -> list[Event]:
        seen.append(d)
        return [_Out(n=1), _Out(n=2)]

    ctx = _Ctx()
    await to_ome_strategy(_spec(fn), deps).func(_In(), ctx)

    assert seen == [deps]
    assert [e.n for e in ctx.emitted] == [1, 2]


async def test_an_undeclared_event_publishes_nothing() -> None:
    async def fn(event: _In, d: MemoryDeps) -> list[Event]:
        return [_Out(n=1), _Other()]

    ctx = _Ctx()
    with pytest.raises(EmitNotDeclaredError):
        await to_ome_strategy(_spec(fn), make_deps()).func(_In(), ctx)
    assert ctx.emitted == []


def test_spec_fields_carry_over() -> None:
    async def fn(event: _In, d: MemoryDeps) -> list[Event]:
        return []

    gate = Counter(threshold=2)
    meta = to_ome_strategy(
        _spec(fn, applies_to="flag", gate=gate, max_retries=4, enabled=False),
        make_deps(),
    )

    assert (meta.name, meta.emits, meta.applies_to, meta.gate) == (
        "s",
        frozenset({_Out}),
        "flag",
        gate,
    )
    assert (meta.max_retries, meta.enabled) == (4, False)


def test_callable_applies_to_sees_the_deps() -> None:
    deps = make_deps(embedding=None)

    def no_embedding(event: _In, d: MemoryDeps) -> bool:
        return d.embedding is None

    async def fn(event: _In, d: MemoryDeps) -> list[Event]:
        return []

    meta = to_ome_strategy(_spec(fn, applies_to=no_embedding), deps)

    assert callable(meta.applies_to)
    assert meta.applies_to(_In()) is True
