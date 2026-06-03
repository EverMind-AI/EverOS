from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from everos.infra.ome._dispatch.dispatcher import EventDispatcher
from everos.infra.ome._dispatch.registry import StrategyRegistry
from everos.infra.ome._stores.counter import CounterStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.events import BaseEvent, CronTick
from everos.infra.ome.gates import Counter
from everos.infra.ome.triggers import Cron, Immediate


class _E(BaseEvent):
    user_id: str


def _make_strategy(name: str, **kw):
    @offline_strategy(name=name, trigger=Immediate(on=[_E]), emits=[], **kw)
    async def _f(event: Any, ctx: StrategyContext) -> None:
        return None

    return _f


@pytest.fixture
async def dispatcher(tmp_path: Path) -> EventDispatcher:
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    registry = StrategyRegistry()
    counter = CounterStore(storage=storage)
    return EventDispatcher(registry=registry, counter_store=counter)


@pytest.mark.asyncio
async def test_dispatch_passes_when_no_gate(
    dispatcher: EventDispatcher,
) -> None:
    dispatcher._registry.register(_make_strategy("s_pass"))
    routes = await dispatcher.dispatch(_E(user_id="u1"))
    assert [m.name for m, _ in routes] == ["s_pass"]


@pytest.mark.asyncio
async def test_dispatch_skips_disabled(dispatcher: EventDispatcher) -> None:
    dispatcher._registry.register(_make_strategy("s_off", enabled=False))
    routes = await dispatcher.dispatch(_E(user_id="u1"))
    assert routes == []


@pytest.mark.asyncio
async def test_dispatch_applies_to_string(
    dispatcher: EventDispatcher,
) -> None:
    dispatcher._registry.register(
        _make_strategy("s", applies_to="user_id"),
    )
    routes_empty = await dispatcher.dispatch(_E(user_id=""))
    routes_set = await dispatcher.dispatch(_E(user_id="u1"))
    assert routes_empty == []
    assert len(routes_set) == 1


@pytest.mark.asyncio
async def test_dispatch_applies_to_callable(
    dispatcher: EventDispatcher,
) -> None:
    def is_paid(e: _E) -> bool:
        return e.user_id.startswith("paid_")

    dispatcher._registry.register(_make_strategy("s", applies_to=is_paid))
    assert await dispatcher.dispatch(_E(user_id="free_a")) == []
    assert len(await dispatcher.dispatch(_E(user_id="paid_a"))) == 1


@pytest.mark.asyncio
async def test_dispatch_counter_gate(dispatcher: EventDispatcher) -> None:
    dispatcher._registry.register(
        _make_strategy("s", gate=Counter(threshold=3, event_field="user_id"))
    )
    for _ in range(2):
        routes = await dispatcher.dispatch(_E(user_id="u1"))
        assert routes == []
    routes = await dispatcher.dispatch(_E(user_id="u1"))
    assert len(routes) == 1


@pytest.mark.asyncio
async def test_inspect_returns_route_info(
    dispatcher: EventDispatcher,
) -> None:
    dispatcher._registry.register(
        _make_strategy("s", gate=Counter(threshold=3, event_field="user_id"))
    )
    infos = await dispatcher.inspect(_E(user_id="u1"))
    assert len(infos) == 1
    assert infos[0].counter_progress == (1, 3)
    assert infos[0].will_run is False


def _make_cron_strategy(name: str):
    @offline_strategy(name=name, trigger=Cron(expr="0 * * * *"), emits=[])
    async def _f(event: Any, ctx: StrategyContext) -> None:
        return None

    return _f


@pytest.mark.asyncio
async def test_dispatch_routes_engine_tick_to_named_strategy_only(
    dispatcher: EventDispatcher,
) -> None:
    dispatcher._registry.register(_make_cron_strategy("cron_a"))
    dispatcher._registry.register(_make_cron_strategy("cron_b"))
    routes = await dispatcher.dispatch(CronTick(strategy_name="cron_a"))
    assert [m.name for m, _ in routes] == ["cron_a"]


@pytest.mark.asyncio
async def test_inspect_engine_tick_skips_non_target_strategy(
    dispatcher: EventDispatcher,
) -> None:
    dispatcher._registry.register(_make_cron_strategy("cron_a"))
    dispatcher._registry.register(_make_cron_strategy("cron_b"))
    infos = await dispatcher.inspect(CronTick(strategy_name="cron_b"))
    assert [i.strategy_name for i in infos] == ["cron_b"]


@pytest.mark.asyncio
async def test_dispatch_force_enabled_bypasses_enabled_gate(
    dispatcher: EventDispatcher,
) -> None:
    dispatcher._registry.register(_make_strategy("s_off", enabled=False))
    assert await dispatcher.dispatch(_E(user_id="u1")) == []
    routes = await dispatcher.dispatch(_E(user_id="u1"), force_enabled=True)
    assert [m.name for m, _ in routes] == ["s_off"]


@pytest.mark.asyncio
async def test_dispatch_force_enabled_still_applies_applies_to_and_counter(
    dispatcher: EventDispatcher,
) -> None:
    dispatcher._registry.register(
        _make_strategy(
            "s",
            enabled=False,
            applies_to="user_id",
            gate=Counter(threshold=2, event_field="user_id"),
        ),
    )
    assert await dispatcher.dispatch(_E(user_id=""), force_enabled=True) == []
    assert await dispatcher.dispatch(_E(user_id="u1"), force_enabled=True) == []
    routes = await dispatcher.dispatch(_E(user_id="u1"), force_enabled=True)
    assert len(routes) == 1


@pytest.mark.asyncio
async def test_dispatch_strategy_filter_scopes_to_single_strategy(
    dispatcher: EventDispatcher,
) -> None:
    dispatcher._registry.register(_make_strategy("s_a"))
    dispatcher._registry.register(_make_strategy("s_b"))
    routes = await dispatcher.dispatch(_E(user_id="u1"), strategy_filter="s_a")
    assert [m.name for m, _ in routes] == ["s_a"]


@pytest.mark.asyncio
async def test_dispatch_strategy_filter_unknown_raises(
    dispatcher: EventDispatcher,
) -> None:
    dispatcher._registry.register(_make_strategy("s_a"))
    with pytest.raises(KeyError):
        await dispatcher.dispatch(_E(user_id="u1"), strategy_filter="missing")


@pytest.mark.asyncio
async def test_dispatch_isolates_faulty_applies_to_callable(
    dispatcher: EventDispatcher,
) -> None:
    """A single strategy's buggy ``applies_to`` callable must not tank
    the fan-out for siblings subscribed to the same event class.
    """

    def _boom(_e: _E) -> bool:
        raise RuntimeError("applies_to is buggy")

    dispatcher._registry.register(_make_strategy("s_buggy", applies_to=_boom))
    dispatcher._registry.register(_make_strategy("s_healthy"))

    routes = await dispatcher.dispatch(_E(user_id="u1"))

    # s_buggy is treated as not-applies; s_healthy still routes.
    assert [m.name for m, _ in routes] == ["s_healthy"]


@pytest.mark.asyncio
async def test_inspect_isolates_faulty_applies_to_callable(
    dispatcher: EventDispatcher,
) -> None:
    def _boom(_e: _E) -> bool:
        raise RuntimeError("applies_to is buggy")

    dispatcher._registry.register(_make_strategy("s_buggy", applies_to=_boom))
    dispatcher._registry.register(_make_strategy("s_healthy"))

    infos = await dispatcher.inspect(_E(user_id="u1"))

    by_name = {i.strategy_name: i for i in infos}
    assert by_name["s_buggy"].applies_to_pass is False
    assert by_name["s_healthy"].applies_to_pass is True
