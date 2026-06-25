from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from everos.infra.ome._dispatch.runner import Runner
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import offline_strategy
from everos.infra.ome.events import BaseEvent
from everos.infra.ome.records import RunStatus
from everos.infra.ome.triggers import Immediate


class _E(BaseEvent):
    user_id: str = "u1"


@pytest.fixture
async def setup(tmp_path: Path):
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    rec_store = RunRecordStore(storage=storage, max_records_per_strategy=1000)
    sem = asyncio.Semaphore(20)
    return rec_store, sem


@pytest.mark.asyncio
async def test_runner_success_marks_record(setup) -> None:
    rec_store, sem = setup

    @offline_strategy(name="ok", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        engine=MagicMock(),
    )
    await runner.run(
        s.meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=1,
    )

    rec = await rec_store.get("r1")
    assert rec.status == RunStatus.SUCCESS


@pytest.mark.asyncio
async def test_runner_retries_on_failure(setup) -> None:
    rec_store, sem = setup
    calls = {"n": 0}

    @offline_strategy(
        name="flaky",
        trigger=Immediate(on=[_E]),
        emits=[],
        max_retries=2,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        engine=MagicMock(),
    )
    await runner.run(
        s.meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=2,
    )
    assert calls["n"] == 3
    # Final successful attempt 2 has a new run_id (not "r1");
    # find by status=SUCCESS, strategy_name=flaky
    success_runs = await rec_store.list_runs(
        strategy_name="flaky",
        status=RunStatus.SUCCESS,
    )
    assert len(success_runs) == 1
    assert success_runs[0].attempt == 2


@pytest.mark.asyncio
async def test_runner_dead_letter_after_exhaust(setup) -> None:
    rec_store, sem = setup

    @offline_strategy(
        name="bad",
        trigger=Immediate(on=[_E]),
        emits=[],
        max_retries=1,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        raise RuntimeError("always-fail")

    dl_calls: list = []

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        on_dead_letter=lambda r: dl_calls.append(r),
        engine=MagicMock(),
    )
    await runner.run(
        s.meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=1,
    )
    dead_runs = await rec_store.list_runs(
        strategy_name="bad",
        status=RunStatus.DEAD_LETTER,
    )
    assert len(dead_runs) == 1
    assert len(dl_calls) == 1


@pytest.mark.asyncio
async def test_runner_emit_must_be_declared(setup) -> None:
    rec_store, sem = setup

    class _Other(BaseEvent):
        pass

    @offline_strategy(
        name="emit_undeclared",
        trigger=Immediate(on=[_E]),
        emits=[],
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        await ctx.emit(_Other())  # not declared

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        engine=MagicMock(),
    )
    await runner.run(
        s.meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=0,
    )
    rec = await rec_store.get("r1")
    assert rec.status == RunStatus.DEAD_LETTER
    assert "EmitNotDeclaredError" in (rec.error or "")


@pytest.mark.asyncio
async def test_runner_negative_max_retries_raises(setup) -> None:
    """``max_retries_snapshot < 0`` is an internal-bug condition (Pydantic
    constrains the user-supplied source to ``>= 0``), so the framework
    fails fast rather than silently no-op the run.
    """
    rec_store, sem = setup

    @offline_strategy(name="ok", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        engine=MagicMock(),
    )
    with pytest.raises(ValueError, match=r"max_retries_snapshot must be >= 0"):
        await runner.run(
            s.meta,
            _E(),
            run_id="r1",
            max_retries_snapshot=-1,
        )


@pytest.mark.asyncio
async def test_runner_aborts_silently_when_mark_running_fails(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When persistence itself fails before the strategy is invoked,
    the run must exit cleanly (no exception escaping the framework) and
    the strategy body must NOT execute — no RUNNING row exists for
    crash recovery to pick up, so re-execution via recovery is
    impossible. The emergency log is the only audit trail.
    """
    rec_store, sem = setup
    called = {"n": 0}

    @offline_strategy(name="ok", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        called["n"] += 1

    async def _boom(**_: object) -> None:
        raise RuntimeError("disk_full")

    monkeypatch.setattr(rec_store, "mark_running", _boom)

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        engine=MagicMock(),
    )
    # Must NOT raise; the framework swallows + logs.
    await runner.run(
        s.meta,
        _E(),
        run_id="r1",
        max_retries_snapshot=1,
    )
    assert called["n"] == 0


async def _no_emit(event: BaseEvent) -> None:
    return None
