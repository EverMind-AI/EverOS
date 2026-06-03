from __future__ import annotations

from typing import Any

import pytest

from everos.infra.ome._dispatch.registry import StrategyRegistry
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.events import BaseEvent, CronTick, IdleTick, ManualTick
from everos.infra.ome.exceptions import StartupValidationError
from everos.infra.ome.triggers import Cron, Idle, Immediate


class _A(BaseEvent):
    pass


class _B(BaseEvent):
    pass


def _make(
    name: str,
    on: list[type[BaseEvent]],
    emits: list[type[BaseEvent]],
):
    @offline_strategy(name=name, trigger=Immediate(on=on), emits=emits)
    async def _f(event: Any, ctx: StrategyContext) -> None:
        return None

    return _f


def test_register_extracts_meta() -> None:
    reg = StrategyRegistry()
    reg.register(_make("s1", [_A], [_B]))
    assert reg.get("s1").name == "s1"


def test_register_duplicate_raises() -> None:
    reg = StrategyRegistry()
    reg.register(_make("s1", [_A], [_B]))
    with pytest.raises(StartupValidationError):
        reg.register(_make("s1", [_A], [_B]))


def test_register_non_decorated_raises() -> None:
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg = StrategyRegistry()
    with pytest.raises(StartupValidationError):
        reg.register(f)


def test_replace_swaps_meta_in_place() -> None:
    from dataclasses import replace

    reg = StrategyRegistry()
    reg.register(_make("s1", [_A], [_B]))
    original = reg.get("s1")
    new_meta = replace(original, max_retries=99)

    reg.replace("s1", new_meta)

    assert reg.get("s1").max_retries == 99
    assert reg.get("s1") is new_meta


def test_replace_unknown_strategy_raises() -> None:
    reg = StrategyRegistry()
    reg.register(_make("s1", [_A], [_B]))
    existing = reg.get("s1")
    with pytest.raises(KeyError):
        reg.replace("missing", existing)


def test_lookup_by_event() -> None:
    reg = StrategyRegistry()
    reg.register(_make("s_a", [_A], []))
    reg.register(_make("s_b", [_B], []))
    metas = reg.lookup_by_event(_A)
    assert {m.name for m in metas} == {"s_a"}


def test_validate_detects_cycle() -> None:
    # s1 emits _B, listens _A;  s2 emits _A, listens _B  -> cycle
    reg = StrategyRegistry()
    reg.register(_make("s1", [_A], [_B]))
    reg.register(_make("s2", [_B], [_A]))
    with pytest.raises(StartupValidationError, match=r"(?i)cycle"):
        reg.validate()


def test_validate_passes_dag() -> None:
    reg = StrategyRegistry()
    reg.register(_make("s1", [_A], [_B]))
    reg.register(_make("s2", [_B], []))
    reg.validate()  # must not raise


def test_lookup_by_event_finds_cron_strategy_for_cron_tick() -> None:
    reg = StrategyRegistry()

    @offline_strategy(name="cron_s", trigger=Cron(expr="0 * * * *"), emits=[])
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg.register(f)
    metas = reg.lookup_by_event(CronTick)
    assert [m.name for m in metas] == ["cron_s"]


def test_lookup_by_event_finds_idle_strategy_for_idle_tick() -> None:
    reg = StrategyRegistry()

    @offline_strategy(
        name="idle_s",
        trigger=Idle(on=[_A], event_field="event_id", idle_seconds=900),
        emits=[],
    )
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg.register(f)
    metas = reg.lookup_by_event(IdleTick)
    assert [m.name for m in metas] == ["idle_s"]


def test_lookup_by_event_returns_empty_for_manual_tick() -> None:
    reg = StrategyRegistry()

    @offline_strategy(name="manual_s", trigger=Immediate(on=[_A]), emits=[])
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg.register(f)
    metas = reg.lookup_by_event(ManualTick)
    assert metas == []


class _EventWithUid(BaseEvent):
    user_id: str


class _EventWithoutUid(BaseEvent):
    other: str


def test_validate_passes_when_gate_event_field_present() -> None:
    from everos.infra.ome.gates import Counter

    reg = StrategyRegistry()

    @offline_strategy(
        name="s",
        trigger=Immediate(on=[_EventWithUid]),
        emits=[],
        gate=Counter(threshold=3, event_field="user_id"),
    )
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg.register(f)
    reg.validate()  # must not raise


def test_validate_raises_when_gate_event_field_missing_on_immediate() -> None:
    from everos.infra.ome.gates import Counter

    reg = StrategyRegistry()

    @offline_strategy(
        name="s",
        trigger=Immediate(on=[_EventWithoutUid]),
        emits=[],
        gate=Counter(threshold=3, event_field="user_id"),
    )
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg.register(f)
    with pytest.raises(StartupValidationError) as exc:
        reg.validate()
    msg = str(exc.value)
    assert "user_id" in msg
    assert "_EventWithoutUid" in msg
    assert "s" in msg


def test_validate_raises_when_gate_event_field_missing_in_one_of_multiple() -> None:
    from everos.infra.ome.gates import Counter

    reg = StrategyRegistry()

    @offline_strategy(
        name="s",
        trigger=Immediate(on=[_EventWithUid, _EventWithoutUid]),
        emits=[],
        gate=Counter(threshold=3, event_field="user_id"),
    )
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg.register(f)
    with pytest.raises(StartupValidationError):
        reg.validate()


def test_validate_passes_when_gate_event_field_is_none() -> None:
    """gate.event_field=None means global bucket; no field-existence check."""
    from everos.infra.ome.gates import Counter

    reg = StrategyRegistry()

    @offline_strategy(
        name="s",
        trigger=Immediate(on=[_EventWithoutUid]),
        emits=[],
        gate=Counter(threshold=3),  # event_field defaults to None
    )
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg.register(f)
    reg.validate()  # must not raise


def test_validate_passes_when_no_gate() -> None:
    reg = StrategyRegistry()

    @offline_strategy(
        name="s",
        trigger=Immediate(on=[_EventWithoutUid]),
        emits=[],
    )
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg.register(f)
    reg.validate()  # must not raise


def test_validate_raises_when_gate_event_field_missing_on_cron_tick() -> None:
    """Cron strategy: gate.event_field must exist on CronTick."""
    from everos.infra.ome.gates import Counter

    reg = StrategyRegistry()

    @offline_strategy(
        name="cron_s",
        trigger=Cron(expr="0 3 * * *"),
        emits=[],
        gate=Counter(threshold=3, event_field="user_id"),  # not in CronTick
    )
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    reg.register(f)
    with pytest.raises(StartupValidationError):
        reg.validate()
