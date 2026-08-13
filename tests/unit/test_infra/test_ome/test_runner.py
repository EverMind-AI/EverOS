from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from everos.infra.ome._dispatch.runner import Runner
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.context import StrategyContext
from everos.infra.ome.decorator import StrategyMeta, offline_strategy
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
    config = OMEConfig(jobstore_path=tmp_path / "ome.db")
    return rec_store, sem, config


@pytest.mark.asyncio
async def test_runner_success_marks_record(setup) -> None:
    rec_store, sem, config = setup

    @offline_strategy(name="ok", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        config=config,
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
async def test_runner_retries_on_failure(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Retry loop now sleeps between attempts (real backoff config from
    # ``setup``); fake the sleep so this test stays fast — it asserts
    # retry *counting*, not backoff timing (see test_runner_applies_
    # exponential_backoff_between_attempts for that).
    monkeypatch.setattr(
        "everos.infra.ome._dispatch.runner.asyncio.sleep",
        _instant_sleep,
    )
    rec_store, sem, config = setup
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
        config=config,
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
async def test_runner_dead_letter_after_exhaust(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "everos.infra.ome._dispatch.runner.asyncio.sleep",
        _instant_sleep,
    )
    rec_store, sem, config = setup

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
        config=config,
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
    rec_store, sem, config = setup

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
        config=config,
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
    rec_store, sem, config = setup

    @offline_strategy(name="ok", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        config=config,
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
    rec_store, sem, config = setup
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
        config=config,
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


async def _instant_sleep(seconds: float) -> None:
    """Drop-in ``asyncio.sleep`` replacement for tests that exercise the
    retry loop but don't care about backoff timing."""
    return None


@pytest.mark.asyncio
async def test_runner_emits_ome_agent_span(setup) -> None:
    """Runner wraps the strategy body in an everos.ome.<name> agent span
    (its own trace — runs in an APScheduler task, no request context)."""
    rec_store, sem, config = setup

    @offline_strategy(name="traced_strat", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import (
        force_flush,
        init_tracing,
        shutdown_tracing,
    )

    exporter = InMemorySpanExporter()
    shutdown_tracing()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    try:
        runner = Runner(
            run_record_store=rec_store,
            engine_sem=sem,
            emit_hook=_no_emit,
            config=config,
            engine=MagicMock(),
        )
        await runner.run(s.meta, _E(), run_id="r_trace", max_retries_snapshot=1)
        force_flush()
    finally:
        shutdown_tracing()

    spans = {sp.name: sp for sp in exporter.get_finished_spans()}
    assert "everos.ome.traced_strat" in spans
    assert (
        spans["everos.ome.traced_strat"].attributes["langfuse.observation.type"]
        == "agent"
    )


@pytest.mark.asyncio
async def test_runner_ome_span_links_to_upstream_traceparent(setup) -> None:
    """Given a traceparent (captured where a request span was active), the
    everos.ome.<name> span nests under that upstream trace, not a new root."""
    rec_store, sem, config = setup

    @offline_strategy(name="linked_strat", trigger=Immediate(on=[_E]), emits=[])
    async def s(event: _E, ctx: StrategyContext) -> None:
        return None

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import (
        current_traceparent,
        force_flush,
        init_tracing,
        memory_span,
        shutdown_tracing,
    )

    exporter = InMemorySpanExporter()
    shutdown_tracing()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    try:
        # Simulate the triggering request: capture its traceparent, then close.
        with memory_span("everos.memory.flush", observation_type="span") as parent:
            tp = current_traceparent()
            parent_tid = parent.get_span_context().trace_id

        runner = Runner(
            run_record_store=rec_store,
            engine_sem=sem,
            emit_hook=_no_emit,
            config=config,
            engine=MagicMock(),
        )
        await runner.run(
            s.meta, _E(), run_id="r_link", max_retries_snapshot=1, traceparent=tp
        )
        force_flush()
    finally:
        shutdown_tracing()

    ome = {sp.name: sp for sp in exporter.get_finished_spans()}[
        "everos.ome.linked_strat"
    ]
    assert ome.context.trace_id == parent_tid  # same trace as the request
    assert ome.parent is not None  # child, not a fresh root


async def _make_runner_with_transient_failing_strategy(
    tmp_path: Path,
    *,
    max_retries: int,
    backoff_base: float = 1.0,
    backoff_cap: float = 10.0,
    jitter: float = 0.5,
) -> tuple[Runner, StrategyMeta]:
    """Build a ``Runner`` wired to a strategy that raises on every attempt.

    Drives the retry loop to exhaustion so the backoff sleep fires between
    each of the ``max_retries`` retries, for tests asserting on
    ``asyncio.sleep`` call arguments.
    """
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    rec_store = RunRecordStore(storage=storage, max_records_per_strategy=1000)
    sem = asyncio.Semaphore(20)
    config = OMEConfig(
        jobstore_path=tmp_path / "ome.db",
        retry_backoff_base_seconds=backoff_base,
        retry_backoff_cap_seconds=backoff_cap,
        retry_jitter_seconds=jitter,
    )

    @offline_strategy(
        name="transient_failing",
        trigger=Immediate(on=[_E]),
        emits=[],
        max_retries=max_retries,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        raise RuntimeError("transient")

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        config=config,
        engine=MagicMock(),
    )
    return runner, s.meta


@pytest.mark.asyncio
async def test_runner_applies_exponential_backoff_between_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each attempt after the first sleeps ~ base * 2**(attempt-1), capped at
    cap_seconds, with up to jitter_seconds added. Verifies that a strategy
    raising a transient exception gets real wall-clock breathing room."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("everos.infra.ome._dispatch.runner.asyncio.sleep", fake_sleep)

    runner, meta = await _make_runner_with_transient_failing_strategy(
        tmp_path, max_retries=3
    )
    await runner.run(meta, _E(), run_id="r1", max_retries_snapshot=3)

    # attempts 1, 2, 3 (0 has no preceding sleep); base=1, cap=10, jitter=0.5
    assert len(sleeps) == 3
    assert 1.0 <= sleeps[0] <= 1.5  # ~1s + jitter
    assert 2.0 <= sleeps[1] <= 2.5  # ~2s + jitter
    assert 4.0 <= sleeps[2] <= 4.5  # ~4s + jitter


@pytest.mark.asyncio
async def test_runner_releases_engine_sem_across_backoff_sleep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The concurrency slot must be free while a run waits to retry.

    ``engine_sem`` bounds concurrent strategy *work* — LLM calls,
    embeddings, storage IO — and a coroutine sleeping between attempts
    consumes none of it. Holding the slot across the sleep turns a
    partial outage into a total stall: with N slots and a 1s/2s/4s
    backoff, N simultaneously-failing runs park every slot in
    ``asyncio.sleep`` and starve strategies that would have succeeded.

    Uses a single-permit semaphore so ``locked()`` is unambiguous, and
    asserts a second waiter actually acquires — ``locked()`` alone would
    pass on an implementation that released the slot but left a waiter
    unable to take it.
    """
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    rec_store = RunRecordStore(storage=storage, max_records_per_strategy=1000)
    sem = asyncio.Semaphore(1)
    config = OMEConfig(
        jobstore_path=tmp_path / "ome.db",
        retry_backoff_base_seconds=1.0,
        retry_backoff_cap_seconds=10.0,
        retry_jitter_seconds=0.0,
    )

    held_during_sleep: list[bool] = []
    acquired_during_sleep: list[bool] = []

    async def fake_sleep(seconds: float) -> None:
        held_during_sleep.append(sem.locked())
        try:
            async with asyncio.timeout(0.5):
                await sem.acquire()
        except TimeoutError:
            acquired_during_sleep.append(False)
        else:
            acquired_during_sleep.append(True)
            sem.release()

    monkeypatch.setattr("everos.infra.ome._dispatch.runner.asyncio.sleep", fake_sleep)

    @offline_strategy(
        name="sem_probe_failing",
        trigger=Immediate(on=[_E]),
        emits=[],
        max_retries=2,
    )
    async def s(event: _E, ctx: StrategyContext) -> None:
        raise RuntimeError("transient")

    runner = Runner(
        run_record_store=rec_store,
        engine_sem=sem,
        emit_hook=_no_emit,
        config=config,
        engine=MagicMock(),
    )
    await runner.run(s.meta, _E(), run_id="r1", max_retries_snapshot=2)

    assert held_during_sleep == [False, False]
    assert acquired_during_sleep == [True, True]


@pytest.mark.asyncio
async def test_runner_backoff_caps_at_configured_maximum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backoff cap prevents unbounded growth on high max_retries."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr("everos.infra.ome._dispatch.runner.asyncio.sleep", fake_sleep)

    runner, meta = await _make_runner_with_transient_failing_strategy(
        tmp_path,
        max_retries=6,
        backoff_base=1.0,
        backoff_cap=3.0,
        jitter=0.0,
    )
    await runner.run(meta, _E(), run_id="r1", max_retries_snapshot=6)

    # 1, 2, 3, 3, 3, 3 -- all capped after attempt 3
    assert sleeps == [1.0, 2.0, 3.0, 3.0, 3.0, 3.0]
