from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from everos.infra.ome.config import OMEConfig
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.engine import OfflineEngine
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.exceptions import (
    EngineLockHeldError,
    OMEError,
    StartupValidationError,
)
from everos.infra.ome.records import RunStatus
from everos.infra.ome.triggers import Cron, Idle, Immediate


class _E(BaseEvent):
    pass


class _A(BaseEvent):
    pass


class _B(BaseEvent):
    pass


@pytest.fixture
def cfg(tmp_path: Path) -> OMEConfig:
    return OMEConfig(jobstore_path=tmp_path / "ome.db", config_watch=False)


@pytest.mark.asyncio
async def test_engine_register_and_start(cfg: OMEConfig) -> None:
    @offline_strategy(name="s", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    await engine.start()
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_register_after_start_raises(cfg: OMEConfig) -> None:
    engine = OfflineEngine(config=cfg)
    await engine.start()
    try:

        @offline_strategy(name="s", trigger=Immediate(on=[_E]), emits=[])
        async def s(event: _E, ctx: StrategyContext) -> None:
            return None

        with pytest.raises(OMEError):
            engine.register(s)
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_lock_prevents_double_open(cfg: OMEConfig) -> None:
    engine1 = OfflineEngine(config=cfg)
    await engine1.start()
    try:
        engine2 = OfflineEngine(config=cfg)
        with pytest.raises(EngineLockHeldError):
            await engine2.start()
    finally:
        await engine1.stop()


@pytest.mark.asyncio
async def test_engine_validates_dag_at_start(tmp_path: Path) -> None:
    cfg = OMEConfig(jobstore_path=tmp_path / "ome.db", config_watch=False)

    @offline_strategy(name="s1", trigger=Immediate(on=[_A]), emits=[_B])
    async def _s1(e: Any, ctx: StrategyContext) -> None:
        return None

    @offline_strategy(name="s2", trigger=Immediate(on=[_B]), emits=[_A])
    async def _s2(e: Any, ctx: StrategyContext) -> None:
        return None

    engine = OfflineEngine(config=cfg)
    engine.register(_s1)
    engine.register(_s2)
    with pytest.raises(StartupValidationError, match=r"(?i)cycle"):
        await engine.start()


@pytest.mark.asyncio
async def test_engine_emit_drives_strategy(cfg: OMEConfig) -> None:
    seen: list[_E] = []

    @offline_strategy(name="collector", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        seen.append(event)

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    await engine.start()
    try:
        await engine.emit(_E())
        # Poll because APScheduler offers no completion signal; retry up to ~2.5s.
        for _ in range(50):
            if seen:
                break
            await asyncio.sleep(0.05)
    finally:
        await engine.stop()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_engine_chain_emit_through_ctx(cfg: OMEConfig) -> None:
    seen_b: list = []

    @offline_strategy(name="a_to_b", trigger=Immediate(on=[_A]), emits=[_B])
    async def s_a(event: _A, ctx: StrategyContext) -> None:
        await ctx.emit(_B())

    @offline_strategy(name="b_collector", trigger=Immediate(on=[_B]), emits=[])
    async def s_b(event: _B, ctx: StrategyContext) -> None:
        seen_b.append(event)

    engine = OfflineEngine(config=cfg)
    engine.register(s_a)
    engine.register(s_b)
    await engine.start()
    try:
        await engine.emit(_A())
        for _ in range(50):
            if seen_b:
                break
            await asyncio.sleep(0.05)
    finally:
        await engine.stop()
    assert len(seen_b) == 1


@pytest.mark.asyncio
async def test_strategy_calling_engine_emit_directly_is_rejected(
    cfg: OMEConfig,
) -> None:
    """Strategy code must emit follow-up events through ctx.emit.

    Calling engine.emit from inside a strategy raises
    EngineCallFromStrategyError (a StrategyContractError) so Runner
    short-circuits the retry budget and dead-letters on the very first
    attempt — re-running the same buggy code can't fix a programming bug.
    """
    engine = OfflineEngine(config=cfg)

    @offline_strategy(name="bad", trigger=Immediate(on=[_A]), emits=[_B])
    async def bad_strategy(event: _A, ctx: StrategyContext) -> None:
        # Captured engine reference is the common, intended pattern for
        # external triggers; using it from INSIDE a strategy is the
        # convention violation we want to catch.
        await engine.emit(_B())

    engine.register(bad_strategy)
    await engine.start()
    try:
        await engine.emit(_A())
        for _ in range(50):
            runs = await engine.list_runs("bad")
            if runs and runs[0].status == RunStatus.DEAD_LETTER:
                break
            await asyncio.sleep(0.05)
        runs = await engine.list_runs("bad")
    finally:
        await engine.stop()

    assert runs, "expected at least one run record"
    # Permanent error → exactly one attempt, no retry.
    assert len(runs) == 1
    final = runs[0]
    assert final.status == RunStatus.DEAD_LETTER
    assert "EngineCallFromStrategyError" in (final.error or "")
    assert "emit" in (final.error or "")


# Module-level singleton — proxies the "strategy reads engine via
# globals/DI/import" pattern. Guard is contextvars-based so it catches
# this path identically to the closure case.
_MODULE_ENGINE: OfflineEngine | None = None


@pytest.mark.asyncio
async def test_strategy_reaching_engine_via_module_global_is_rejected(
    cfg: OMEConfig,
) -> None:
    """The guard is contextvars-based: it doesn't matter how the strategy
    got the engine reference (closure, module singleton, DI container).
    """
    global _MODULE_ENGINE
    _MODULE_ENGINE = OfflineEngine(config=cfg)

    @offline_strategy(name="bad_global", trigger=Immediate(on=[_A]), emits=[_B])
    async def bad_strategy(event: _A, ctx: StrategyContext) -> None:
        assert _MODULE_ENGINE is not None
        await _MODULE_ENGINE.emit(_B())

    _MODULE_ENGINE.register(bad_strategy)
    await _MODULE_ENGINE.start()
    try:
        await _MODULE_ENGINE.emit(_A())
        for _ in range(50):
            runs = await _MODULE_ENGINE.list_runs("bad_global")
            if runs and runs[0].status == RunStatus.DEAD_LETTER:
                break
            await asyncio.sleep(0.05)
        runs = await _MODULE_ENGINE.list_runs("bad_global")
    finally:
        await _MODULE_ENGINE.stop()
        _MODULE_ENGINE = None

    assert len(runs) == 1
    assert runs[0].status == RunStatus.DEAD_LETTER
    assert "EngineCallFromStrategyError" in (runs[0].error or "")


@pytest.mark.asyncio
async def test_strategy_calling_other_engine_methods_is_rejected(
    cfg: OMEConfig,
) -> None:
    """The guard covers every public engine method, not just emit —
    strategies must interact with the engine only via (event, ctx).
    """
    engine = OfflineEngine(config=cfg)

    @offline_strategy(name="bad_lookup", trigger=Immediate(on=[_A]), emits=[])
    async def bad_strategy(event: _A, ctx: StrategyContext) -> None:
        # trigger_manual is another public engine method that strategies
        # must not call directly.
        await engine.trigger_manual("bad_lookup")

    engine.register(bad_strategy)
    await engine.start()
    try:
        await engine.emit(_A())
        for _ in range(50):
            runs = await engine.list_runs("bad_lookup")
            if runs and runs[0].status == RunStatus.DEAD_LETTER:
                break
            await asyncio.sleep(0.05)
        runs = await engine.list_runs("bad_lookup")
    finally:
        await engine.stop()

    assert len(runs) == 1
    assert runs[0].status == RunStatus.DEAD_LETTER
    assert "EngineCallFromStrategyError" in (runs[0].error or "")
    assert "trigger_manual" in (runs[0].error or "")


@pytest.mark.asyncio
async def test_trigger_manual_with_default_event_uses_manual_tick(
    cfg: OMEConfig,
) -> None:
    seen: list = []

    from everos.infra.ome.events import ManualTick

    @offline_strategy(
        name="manual_only",
        trigger=Immediate(on=[ManualTick]),
        emits=[],
    )
    async def s(event: ManualTick, ctx: StrategyContext) -> None:
        seen.append(event)

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    await engine.start()
    try:
        await engine.trigger_manual("manual_only")
        for _ in range(50):
            if seen:
                break
            await asyncio.sleep(0.05)
    finally:
        await engine.stop()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_trigger_manual_force_bypasses_enabled(
    cfg: OMEConfig,
) -> None:
    seen: list = []
    from everos.infra.ome.events import ManualTick

    @offline_strategy(
        name="off",
        trigger=Immediate(on=[ManualTick]),
        emits=[],
        enabled=False,
    )
    async def s(event: ManualTick, ctx: StrategyContext) -> None:
        seen.append(event)

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    await engine.start()
    try:
        await engine.trigger_manual("off", force=True)
        for _ in range(50):
            if seen:
                break
            await asyncio.sleep(0.05)
    finally:
        await engine.stop()
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_on_dead_letter_callback_invoked(cfg: OMEConfig) -> None:
    calls: list = []

    @offline_strategy(
        name="bad_dl", trigger=Immediate(on=[_E]), emits=[], max_retries=0
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        raise RuntimeError("always-fail")

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    engine.on_dead_letter(lambda rec: calls.append(rec.run_id))
    await engine.start()
    try:
        await engine.emit(_E())
        for _ in range(50):
            if calls:
                break
            await asyncio.sleep(0.05)
    finally:
        await engine.stop()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_inspect_dispatch_returns_routes(cfg: OMEConfig) -> None:
    @offline_strategy(name="s_t24a", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    await engine.start()
    try:
        infos = await engine.inspect_dispatch(_E())
        assert len(infos) == 1
        assert infos[0].will_run is True
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_get_run_status_and_list(cfg: OMEConfig) -> None:
    @offline_strategy(name="s_t24b", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    await engine.start()
    try:
        await engine.emit(_E())
        # Poll because APScheduler offers no completion signal; up to ~2.5s.
        for _ in range(50):
            runs = await engine.list_runs("s_t24b")
            if runs and runs[0].status.value == "success":
                break
            await asyncio.sleep(0.05)
        runs = await engine.list_runs("s_t24b")
        assert len(runs) == 1
        rec = await engine.get_run_status(runs[0].run_id)
        assert rec is not None
        assert rec.status.value == "success"
    finally:
        await engine.stop()


class _EventWithUid(BaseEvent):
    user_id: str


@pytest.mark.asyncio
async def test_engine_reschedule_cron_job_updates_aps(cfg: OMEConfig) -> None:
    @offline_strategy(name="cron_s", trigger=Cron(expr="0 3 * * *"), emits=[])
    async def s(event: Any, ctx: StrategyContext) -> None:
        return None

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    await engine.start()
    try:
        from apscheduler.triggers.cron import CronTrigger

        engine.reschedule_cron_job("cron_s", "*/5 * * * *")

        job = engine._scheduler.get_job("cron::cron_s")
        assert isinstance(job.trigger, CronTrigger)
        # CronTrigger stores parsed crontab fields; minute step=5 means "*/5".
        minute_field = next(f for f in job.trigger.fields if f.name == "minute")
        assert str(minute_field) == "*/5"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_engine_reschedule_idle_job_updates_interval(cfg: OMEConfig) -> None:
    @offline_strategy(
        name="idle_s",
        trigger=Idle(
            on=[_EventWithUid],
            event_field="user_id",
            idle_seconds=60,
            scan_interval_seconds=30,
        ),
        emits=[],
    )
    async def s(event: Any, ctx: StrategyContext) -> None:
        return None

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    await engine.start()
    try:
        engine.reschedule_idle_job("idle_s", scan_interval_seconds=10)
        job = engine._scheduler.get_job("idle::idle_s")
        # IntervalTrigger.interval is a timedelta.
        assert job.trigger.interval.total_seconds() == 10
    finally:
        await engine.stop()


def test_reschedule_cron_job_before_start_raises(cfg: OMEConfig) -> None:
    engine = OfflineEngine(config=cfg)
    with pytest.raises(OMEError, match="engine not started"):
        engine.reschedule_cron_job("x", "* * * * *")


def test_reschedule_idle_job_before_start_raises(cfg: OMEConfig) -> None:
    engine = OfflineEngine(config=cfg)
    with pytest.raises(OMEError, match="engine not started"):
        engine.reschedule_idle_job("x", scan_interval_seconds=30)


@pytest.mark.asyncio
async def test_start_failure_cleans_up_engines_and_scheduler(
    cfg: OMEConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure between scheduler start and ``_started = True`` must roll
    back: pop from the module-level ``_ENGINES`` registry, shut the
    scheduler thread down, and release the lock so a fresh ``OfflineEngine``
    can start on the same jobstore.
    """
    from everos.infra.ome import engine as engine_mod

    async def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("crash recovery exploded")

    monkeypatch.setattr(engine_mod, "scan_and_resume", _boom)

    engine = OfflineEngine(config=cfg)
    with pytest.raises(RuntimeError, match="crash recovery exploded"):
        await engine.start()

    assert engine._engine_id not in engine_mod._ENGINES
    assert engine._scheduler is None
    assert engine._started is False
    assert engine._lock_handle is None

    monkeypatch.undo()
    engine2 = OfflineEngine(config=cfg)
    await engine2.start()
    await engine2.stop()


# ── active_runs / wait_idle ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_wait_idle_returns_true_when_no_runs(cfg: OMEConfig) -> None:
    """Pre-emit idle: counter starts at 0, idle_event starts set."""
    engine = OfflineEngine(config=cfg)
    await engine.start()
    try:
        assert engine._active_runs == 0
        assert await engine.wait_idle(timeout=0.5) is True
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_wait_idle_blocks_until_strategy_finishes(cfg: OMEConfig) -> None:
    """A strategy mid-flight keeps active_runs > 0 and idle_event clear
    until it completes."""
    release = asyncio.Event()
    entered = asyncio.Event()

    @offline_strategy(name="slow", trigger=Immediate(on=[_E]), emits=[])
    async def slow(event: _E, ctx: StrategyContext) -> None:
        entered.set()
        await release.wait()

    engine = OfflineEngine(config=cfg)
    engine.register(slow)
    await engine.start()
    try:
        await engine.emit(_E())
        await asyncio.wait_for(entered.wait(), timeout=2.0)
        # Strategy is now mid-flight.
        assert engine._active_runs >= 1
        assert await engine.wait_idle(timeout=0.2) is False
        # Release the strategy and verify wait_idle resolves.
        release.set()
        assert await engine.wait_idle(timeout=2.0) is True
        assert engine._active_runs == 0
    finally:
        release.set()
        await engine.stop()


@pytest.mark.asyncio
async def test_stop_waits_for_in_flight_run_to_complete(cfg: OMEConfig) -> None:
    """stop() must not cancel in-flight strategies. Pre-fix this used
    scheduler.shutdown(wait=True) which APS 3.x AsyncIOExecutor cancels
    silently; post-fix stop() drains through wait_idle first.
    """
    completed: list[str] = []
    started = asyncio.Event()
    release = asyncio.Event()

    @offline_strategy(name="slow_to_finish", trigger=Immediate(on=[_E]), emits=[])
    async def slow(event: _E, ctx: StrategyContext) -> None:
        started.set()
        await release.wait()
        completed.append("done")

    engine = OfflineEngine(config=cfg)
    engine.register(slow)
    await engine.start()
    await engine.emit(_E())
    await asyncio.wait_for(started.wait(), timeout=2.0)

    # Stop concurrently with the in-flight strategy; release it after a
    # tick so stop() has to actually wait.
    stop_task = asyncio.create_task(engine.stop())
    await asyncio.sleep(0.05)
    assert not stop_task.done()
    release.set()
    await asyncio.wait_for(stop_task, timeout=5.0)
    assert completed == ["done"]


@pytest.mark.asyncio
async def test_active_runs_decrements_on_strategy_exception(cfg: OMEConfig) -> None:
    """A strategy that raises (and exhausts retries → DEAD_LETTER) must
    still release its counter — dispatch_run's finally guarantees -1.
    """

    @offline_strategy(name="boom", trigger=Immediate(on=[_E]), emits=[])
    async def boom(event: _E, ctx: StrategyContext) -> None:
        raise RuntimeError("strategy boom")

    cfg2 = OMEConfig(
        jobstore_path=cfg.jobstore_path,
        config_watch=False,
        max_retries=0,
    )
    engine = OfflineEngine(config=cfg2)
    engine.register(boom)
    await engine.start()
    try:
        await engine.emit(_E())
        assert await engine.wait_idle(timeout=2.0) is True
        runs = await engine.list_runs("boom")
        assert len(runs) == 1
        assert runs[0].status == RunStatus.DEAD_LETTER
        assert engine._active_runs == 0
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_enqueue_run_rolls_back_counter_on_add_job_failure(
    cfg: OMEConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If APScheduler ``add_job`` raises, the matching dispatch_run never
    runs — _enqueue_run must roll back the pre-emptive +1 itself.
    """

    @offline_strategy(name="s", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    engine = OfflineEngine(config=cfg)
    engine.register(s)
    await engine.start()
    try:

        def _boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("add_job exploded")

        monkeypatch.setattr(engine._scheduler, "add_job", _boom)
        with pytest.raises(RuntimeError, match="add_job exploded"):
            await engine.emit(_E())
        assert engine._active_runs == 0
        assert await engine.wait_idle(timeout=0.5) is True
    finally:
        monkeypatch.undo()
        await engine.stop()
