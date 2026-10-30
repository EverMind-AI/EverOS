"""StrategySpec: the domain's declaration of one strategy."""

from __future__ import annotations

import pytest

from everos.memory.deps import MemoryDeps
from everos.memory.ports import Counter, Event, Immediate, StrategySpec


class _E(Event):
    user_id: str


async def _noop(event: _E, deps: MemoryDeps) -> list[Event]:
    return []


def test_defaults() -> None:
    spec = StrategySpec(name="x", fn=_noop, trigger=Immediate(on=[_E]))
    assert spec.emits == frozenset()
    assert spec.applies_to is None
    assert spec.gate is None
    assert spec.max_retries is None
    assert spec.enabled is True


def test_full_declaration_is_kept_verbatim() -> None:
    gate = Counter(threshold=5)
    spec = StrategySpec(
        name="cluster",
        fn=_noop,
        trigger=Immediate(on=[_E]),
        emits=frozenset({_E}),
        applies_to="user_id",
        gate=gate,
        max_retries=3,
        enabled=False,
    )
    assert spec.emits == frozenset({_E})
    assert spec.applies_to == "user_id"
    assert spec.gate is gate
    assert spec.max_retries == 3
    assert spec.enabled is False


@pytest.mark.parametrize("name", ["", "   "])
def test_rejects_blank_name(name: str) -> None:
    with pytest.raises(ValueError):
        StrategySpec(name=name, fn=_noop, trigger=Immediate(on=[_E]))


def test_rejects_non_async_function() -> None:
    def sync_fn(event: _E, deps: MemoryDeps) -> list[Event]:
        return []

    with pytest.raises(TypeError):
        StrategySpec(name="x", fn=sync_fn, trigger=Immediate(on=[_E]))  # type: ignore[arg-type]  # the rejection is the point
