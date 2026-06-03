"""Tests for ConfigReloader."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from everos.infra.ome._background.config_reloader import (
    ConfigReloader,
    apply_overrides,
)
from everos.infra.ome._dispatch.registry import StrategyRegistry
from everos.infra.ome.config import CounterOverride, StrategyOverride, TomlRoot
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.engine import OfflineEngine
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.gates import Counter
from everos.infra.ome.triggers import Cron, Idle, Immediate


class _E(BaseEvent):
    pass


class _EventUid(BaseEvent):
    user_id: str


def _make(name: str, **kw: Any) -> Any:
    @offline_strategy(name=name, trigger=Immediate(on=[_E]), emits=[], **kw)
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    return f


def _make_cron(name: str, expr: str = "0 3 * * *", **kw: Any) -> Any:
    @offline_strategy(name=name, trigger=Cron(expr=expr), emits=[], **kw)
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    return f


def _make_idle(name: str, **kw: Any) -> Any:
    @offline_strategy(
        name=name,
        trigger=Idle(
            on=[_EventUid],
            event_field="user_id",
            idle_seconds=30,
            scan_interval_seconds=10,
        ),
        emits=[],
        **kw,
    )
    async def f(event: Any, ctx: StrategyContext) -> None:
        return None

    return f


@pytest.fixture
def fake_engine() -> MagicMock:
    """Mock OfflineEngine; spec catches typos in mocked method names."""
    return MagicMock(spec=OfflineEngine)


def test_apply_overrides_replaces_enabled(fake_engine: MagicMock) -> None:
    reg = StrategyRegistry()
    reg.register(_make("s", enabled=True))
    root = TomlRoot(strategies={"s": StrategyOverride(enabled=False)})
    apply_overrides(reg, root, fake_engine)
    assert reg.get("s").enabled is False


def test_apply_overrides_max_retries(fake_engine: MagicMock) -> None:
    reg = StrategyRegistry()
    reg.register(_make("s", max_retries=1))
    root = TomlRoot(strategies={"s": StrategyOverride(max_retries=5)})
    apply_overrides(reg, root, fake_engine)
    assert reg.get("s").max_retries == 5


def test_apply_overrides_counter_partial(fake_engine: MagicMock) -> None:
    reg = StrategyRegistry()
    reg.register(_make("s", gate=Counter(threshold=3, event_field="user_id")))
    root = TomlRoot(
        strategies={"s": StrategyOverride(gate=CounterOverride(threshold=10))}
    )
    apply_overrides(reg, root, fake_engine)
    g = reg.get("s").gate
    assert g.threshold == 10
    assert g.event_field == "user_id"  # untouched


def test_apply_overrides_unknown_strategy_ignored(fake_engine: MagicMock) -> None:
    reg = StrategyRegistry()
    reg.register(_make("s"))
    root = TomlRoot(strategies={"unknown": StrategyOverride(enabled=False)})
    apply_overrides(reg, root, fake_engine)  # must not raise


def test_apply_overrides_updates_cron_expr(fake_engine: MagicMock) -> None:
    reg = StrategyRegistry()
    reg.register(_make_cron("s", "0 3 * * *"))
    root = TomlRoot(strategies={"s": StrategyOverride(cron="*/5 * * * *")})

    apply_overrides(reg, root, fake_engine)

    assert isinstance(reg.get("s").trigger, Cron)
    assert reg.get("s").trigger.expr == "*/5 * * * *"
    fake_engine.reschedule_cron_job.assert_called_once_with("s", "*/5 * * * *")


def test_apply_overrides_skips_atomic_group_on_reschedule_failure(
    fake_engine: MagicMock,
) -> None:
    """Even though StrategyOverride.cron is now syntactically validated at
    parse time, reschedule_cron_job can still fail at runtime (APS internal
    error, scheduler stopped, etc.). The atomic-group rollback must hold
    against those failures too.
    """
    reg = StrategyRegistry()
    reg.register(_make_cron("s", "0 3 * * *", enabled=True, max_retries=1))
    fake_engine.reschedule_cron_job.side_effect = RuntimeError("APS error")
    root = TomlRoot(
        strategies={
            "s": StrategyOverride(enabled=False, cron="*/5 * * * *", max_retries=99)
        }
    )

    apply_overrides(reg, root, fake_engine)

    # enabled applied independently
    assert reg.get("s").enabled is False
    # atomic group rolled back: cron unchanged, max_retries unchanged
    assert reg.get("s").trigger.expr == "0 3 * * *"
    assert reg.get("s").max_retries == 1
    fake_engine.reschedule_cron_job.assert_called_once_with("s", "*/5 * * * *")


def test_apply_overrides_skips_atomic_group_on_cron_type_mismatch(
    fake_engine: MagicMock,
) -> None:
    reg = StrategyRegistry()
    reg.register(_make("s", enabled=True))  # Immediate strategy
    root = TomlRoot(strategies={"s": StrategyOverride(enabled=False, cron="0 3 * * *")})

    apply_overrides(reg, root, fake_engine)

    assert reg.get("s").enabled is False
    assert isinstance(reg.get("s").trigger, Immediate)
    fake_engine.reschedule_cron_job.assert_not_called()


def test_apply_overrides_updates_idle_seconds_and_scan_interval(
    fake_engine: MagicMock,
) -> None:
    reg = StrategyRegistry()
    reg.register(_make_idle("s"))
    root = TomlRoot(
        strategies={"s": StrategyOverride(idle_seconds=120, scan_interval_seconds=15)}
    )

    apply_overrides(reg, root, fake_engine)

    t = reg.get("s").trigger
    assert t.idle_seconds == 120
    assert t.scan_interval_seconds == 15
    fake_engine.reschedule_idle_job.assert_called_once_with(
        "s", scan_interval_seconds=15
    )


def test_apply_overrides_updates_only_idle_seconds_does_not_reschedule_aps(
    fake_engine: MagicMock,
) -> None:
    """idle_seconds is consumed by dispatcher / engine on each scan,
    not by APS IntervalTrigger, so changing only it must NOT trigger
    an APS reschedule (which would reset the pending tick).
    """
    reg = StrategyRegistry()
    reg.register(_make_idle("s"))
    root = TomlRoot(strategies={"s": StrategyOverride(idle_seconds=120)})

    apply_overrides(reg, root, fake_engine)

    assert reg.get("s").trigger.idle_seconds == 120
    fake_engine.reschedule_idle_job.assert_not_called()


def test_apply_overrides_skips_atomic_group_on_idle_type_mismatch(
    fake_engine: MagicMock,
) -> None:
    reg = StrategyRegistry()
    reg.register(_make_cron("s"))  # Cron strategy
    root = TomlRoot(strategies={"s": StrategyOverride(idle_seconds=60)})

    apply_overrides(reg, root, fake_engine)

    assert isinstance(reg.get("s").trigger, Cron)
    fake_engine.reschedule_cron_job.assert_not_called()
    fake_engine.reschedule_idle_job.assert_not_called()


def test_apply_overrides_rollback_on_aps_reschedule_failure(
    fake_engine: MagicMock,
) -> None:
    fake_engine.reschedule_cron_job.side_effect = RuntimeError("APS exploded")

    reg = StrategyRegistry()
    reg.register(_make_cron("s", "0 3 * * *", enabled=True, max_retries=1))
    root = TomlRoot(
        strategies={
            "s": StrategyOverride(enabled=False, cron="*/5 * * * *", max_retries=99)
        }
    )

    apply_overrides(reg, root, fake_engine)

    # enabled applied (Step 1, before atomic group)
    assert reg.get("s").enabled is False
    # atomic group rolled back: cron + max_retries unchanged
    assert reg.get("s").trigger.expr == "0 3 * * *"
    assert reg.get("s").max_retries == 1


def test_apply_overrides_enabled_survives_reschedule_failure(
    fake_engine: MagicMock,
) -> None:
    """enabled=false is emergency-stop semantics; must apply even when the
    paired cron update fails at reschedule time.
    """
    reg = StrategyRegistry()
    reg.register(_make_cron("s", "0 3 * * *", enabled=True))
    fake_engine.reschedule_cron_job.side_effect = RuntimeError("APS error")
    root = TomlRoot(
        strategies={"s": StrategyOverride(enabled=False, cron="*/5 * * * *")}
    )

    apply_overrides(reg, root, fake_engine)

    assert reg.get("s").enabled is False
    assert reg.get("s").trigger.expr == "0 3 * * *"


def test_apply_overrides_strategy_isolation(fake_engine: MagicMock) -> None:
    """One strategy's atomic-group failure must not affect another."""
    reg = StrategyRegistry()
    reg.register(_make_cron("a", "0 3 * * *"))
    reg.register(_make_cron("b", "0 4 * * *"))

    def _reschedule(name: str, expr: str) -> None:
        if name == "b":
            raise RuntimeError("simulated APS failure for b")

    fake_engine.reschedule_cron_job.side_effect = _reschedule
    root = TomlRoot(
        strategies={
            "a": StrategyOverride(cron="*/5 * * * *"),
            "b": StrategyOverride(cron="*/7 * * * *"),
        }
    )

    apply_overrides(reg, root, fake_engine)

    assert reg.get("a").trigger.expr == "*/5 * * * *"
    assert reg.get("b").trigger.expr == "0 4 * * *"


def test_apply_overrides_atomic_group_no_partial_application(
    fake_engine: MagicMock,
) -> None:
    """A failure in the atomic group must roll back max_retries / gate too."""
    reg = StrategyRegistry()
    reg.register(
        _make_cron(
            "s",
            "0 3 * * *",
            max_retries=1,
            gate=Counter(threshold=3, event_field="user_id"),
        )
    )
    fake_engine.reschedule_cron_job.side_effect = RuntimeError("APS error")
    root = TomlRoot(
        strategies={
            "s": StrategyOverride(
                cron="*/5 * * * *",
                max_retries=99,
                gate=CounterOverride(threshold=100),
            )
        }
    )

    apply_overrides(reg, root, fake_engine)

    assert reg.get("s").trigger.expr == "0 3 * * *"
    assert reg.get("s").max_retries == 1
    assert reg.get("s").gate.threshold == 3


def test_apply_overrides_succeeds_on_combined_enabled_and_trigger(
    fake_engine: MagicMock,
) -> None:
    reg = StrategyRegistry()
    reg.register(_make_cron("s", "0 3 * * *", enabled=True))
    root = TomlRoot(
        strategies={"s": StrategyOverride(enabled=False, cron="*/5 * * * *")}
    )

    apply_overrides(reg, root, fake_engine)

    assert reg.get("s").enabled is False
    assert reg.get("s").trigger.expr == "*/5 * * * *"
    fake_engine.reschedule_cron_job.assert_called_once_with("s", "*/5 * * * *")


def test_atomic_group_skipped_when_introducing_gate_without_threshold(
    fake_engine: MagicMock,
) -> None:
    """N5: TOML that introduces a gate via cooldown alone (no threshold)
    must be rejected, not silently defaulted to threshold=1 ('fire every event').
    """
    reg = StrategyRegistry()
    reg.register(_make("s"))  # no gate
    assert reg.get("s").gate is None

    root = TomlRoot(
        strategies={
            "s": StrategyOverride(gate=CounterOverride(cooldown_seconds=60)),
        }
    )

    apply_overrides(reg, root, fake_engine)

    # Atomic group rolled back: still no gate.
    assert reg.get("s").gate is None


def test_atomic_group_accepts_introducing_gate_with_explicit_threshold(
    fake_engine: MagicMock,
) -> None:
    """N5 happy path: explicit threshold on a previously-gateless strategy
    is the user opt-in we require.
    """
    reg = StrategyRegistry()
    reg.register(_make("s"))
    assert reg.get("s").gate is None

    root = TomlRoot(
        strategies={
            "s": StrategyOverride(
                gate=CounterOverride(threshold=5, cooldown_seconds=60)
            ),
        }
    )

    apply_overrides(reg, root, fake_engine)

    g = reg.get("s").gate
    assert g is not None
    assert g.threshold == 5
    assert g.cooldown_seconds == 60


@pytest.mark.asyncio
async def test_start_twice_raises(tmp_path: Path) -> None:
    """N7: calling start() twice surfaces the caller bug instead of
    silently dropping the original task reference and racing two watchers.
    """
    config_path = tmp_path / "ome.toml"
    config_path.write_text("")
    reloader = ConfigReloader(
        config_path=config_path,
        registry=StrategyRegistry(),
        engine=MagicMock(spec=OfflineEngine),
    )
    reloader.start()
    try:
        with pytest.raises(RuntimeError, match=r"already started"):
            reloader.start()
    finally:
        await reloader.stop()


@pytest.mark.asyncio
async def test_start_after_stop_is_allowed(tmp_path: Path) -> None:
    """N7: idempotency check only fires while a task is live; once stopped,
    start() must work again so callers can restart the reloader.
    """
    config_path = tmp_path / "ome.toml"
    config_path.write_text("")
    reloader = ConfigReloader(
        config_path=config_path,
        registry=StrategyRegistry(),
        engine=MagicMock(spec=OfflineEngine),
    )
    reloader.start()
    await reloader.stop()
    # Must not raise.
    reloader.start()
    await reloader.stop()
