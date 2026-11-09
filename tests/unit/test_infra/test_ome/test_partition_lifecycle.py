"""OME persistence and retry semantics while partition waiters yield capacity."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from everos.core.locks import LOCKS
from everos.infra.ome import OfflineEngine
from everos.infra.ome._dispatch.runner import Runner
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.records import RunStatus
from everos.memory._partition_locks import get_partition_lock
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, Immediate
from tests.helpers.ome import ome_strategy


class PartitionEvent(Event):
    key: str = "owner"


async def _no_emit(event: Event) -> None:
    pass


@pytest.mark.parametrize("phase", ["lock", "capacity", "body"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_interruption_preserves_records_identity_and_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, cancel: bool
) -> None:
    storage = OMEStorage(tmp_path / "ome.db")
    await storage.init()
    records = RunRecordStore(storage, max_records_per_strategy=100)
    sem = asyncio.BoundedSemaphore(1)
    runner = Runner(
        run_record_store=records,
        engine_sem=sem,
        emit_hook=_no_emit,
        config=OMEConfig(jobstore_path=storage.db_path, run_timeout_seconds=0.1),
    )
    arrived = asyncio.Event()
    body = asyncio.Event()
    finish = asyncio.Event()
    backoff = asyncio.Event()
    retry = asyncio.Event()
    failure_write = asyncio.Event()
    original_failed = records.mark_failed

    async def observed_failed(**kwargs) -> None:
        failure_write.set()
        await original_failed(**kwargs)

    async def controlled_backoff(attempt: int) -> None:
        backoff.set()
        await retry.wait()

    monkeypatch.setattr(records, "mark_failed", observed_failed)
    monkeypatch.setattr(runner, "_sleep_backoff", controlled_backoff)
    LOCKS.set_acquire_hook(lambda *_: arrived.set())

    @ome_strategy(name="partition", trigger=Immediate(on=[PartitionEvent]))
    async def work(event: PartitionEvent, deps: MemoryDeps) -> list[Event]:
        async with get_partition_lock("partition", event.key):
            body.set()
            await finish.wait()
        return []

    external_lock = get_partition_lock("partition", "owner")
    await external_lock.__aenter__()
    owns_lock = True
    owns_capacity = False
    arrived.clear()
    event = PartitionEvent()
    task = asyncio.create_task(
        runner.run(work, event, run_id="original", max_retries_snapshot=1)
    )
    try:
        async with asyncio.timeout(5):
            await arrived.wait()
            if phase != "body":
                # Independent work consumes the returned slot. A timeout must
                # not open failure-state connections outside the global cap.
                await sem.acquire()
                owns_capacity = True
            if phase != "lock":
                await external_lock.__aexit__(None, None, None)
                owns_lock = False
            if phase == "capacity":
                # The released lock schedules its waiter before this loop turn.
                await asyncio.sleep(0)
                assert ("partition:partition", "owner") in LOCKS.held(task)
                assert not body.is_set()
            if phase == "body":
                await body.wait()
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert (await records.get("original")).status == RunStatus.RUNNING
                assert not failure_write.is_set()
                if owns_lock:
                    await external_lock.__aexit__(None, None, None)
                    owns_lock = False
                if owns_capacity:
                    assert sem.locked()
                    sem.release()
                    owns_capacity = False
                finish.set()
                for key in ("owner", "other"):
                    await runner.run(
                        work,
                        PartitionEvent(key=key),
                        run_id=f"after_{key}",
                        max_retries_snapshot=0,
                    )
                    assert (
                        await records.get(f"after_{key}")
                    ).status == RunStatus.SUCCESS
                await sem.acquire()
                assert sem.locked()
                sem.release()
                return
            if phase != "body":
                with pytest.raises(TimeoutError):
                    await asyncio.wait_for(failure_write.wait(), 0.2)
                assert (await records.get("original")).status == RunStatus.RUNNING
                sem.release()
                owns_capacity = False
            await backoff.wait()
            failed = await records.get("original")
            assert failed.status == RunStatus.FAILED
            assert "TimeoutError" in failed.error
            if owns_lock:
                await external_lock.__aexit__(None, None, None)
                owns_lock = False
            # Backoff leaves room for unrelated work even with capacity=1.
            finish.set()
            await runner.run(
                work,
                PartitionEvent(key="other"),
                run_id="unrelated",
                max_retries_snapshot=0,
            )
            retry.set()
            await task
    finally:
        LOCKS.set_acquire_hook(None)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if owns_lock:
            await external_lock.__aexit__(None, None, None)
        if owns_capacity:
            sem.release()

    runs = sorted(
        await records.list_by_event_id(event.event_id), key=lambda r: r.attempt
    )
    assert [r.status for r in runs] == [RunStatus.FAILED, RunStatus.SUCCESS]
    assert [r.attempt for r in runs] == [0, 1]
    assert len({r.run_id for r in runs}) == 2
    assert (await records.get("unrelated")).status == RunStatus.SUCCESS
    async with asyncio.timeout(1):
        await sem.acquire()
    assert sem.locked()
    sem.release()


async def test_real_engine_recovers_more_waiters_than_history_limit(
    tmp_path: Path,
) -> None:
    """Emit real jobs, cancel their tasks, then reopen the same SQLite store."""
    config = OMEConfig(
        jobstore_path=tmp_path / "ome.db",
        config_watch=False,
        max_concurrent_runs=2,
        max_records_per_strategy=3,
        retry_backoff_base_seconds=0,
    )
    release = asyncio.Event()
    all_arrived = asyncio.Event()
    attempts: set[asyncio.Task] = set()
    completed: list[str] = []
    events = [PartitionEvent() for _ in range(6)]

    def observe(task, wanted, held) -> None:
        if wanted == ("partition:recoverable", "owner"):
            attempts.add(task)
            if len(attempts) == len(events):
                all_arrived.set()

    @ome_strategy(name="recoverable", trigger=Immediate(on=[PartitionEvent]))
    async def work(event: PartitionEvent, deps: MemoryDeps) -> list[Event]:
        async with get_partition_lock("recoverable", event.key):
            await release.wait()
            completed.append(event.event_id)
        return []

    engine = OfflineEngine(config=config)
    engine.register(work)
    await engine.start()
    LOCKS.set_acquire_hook(observe)
    original_ids: set[str] = set()
    try:
        async with asyncio.timeout(10):
            for event in events:
                await engine.emit(event)
            await all_arrived.wait()
            running = await engine.list_runs("recoverable", status=RunStatus.RUNNING)
            assert {r.event_id for r in running} == {e.event_id for e in events}
            original_ids = {r.run_id for r in running}
            assert engine._scheduler.get_jobs() == []
            assert not completed
            # These are the exact tasks announced by the real partition lock,
            # not arbitrary tasks from the process/event loop.
            for task in attempts:
                task.cancel()
            await asyncio.gather(*attempts, return_exceptions=True)
            assert await engine.wait_idle(timeout=2)
            assert (
                len(await engine.list_runs("recoverable", status=RunStatus.RUNNING))
                == 6
            )
    finally:
        LOCKS.set_acquire_hook(None)
        for task in attempts:
            task.cancel()
        await asyncio.gather(*attempts, return_exceptions=True)
        await engine.stop()

    # Keep the new attempts blocked to inspect identity before history rolls.
    resumed = OfflineEngine(config=config)
    resumed.register(work)
    attempts.clear()
    all_arrived.clear()
    LOCKS.set_acquire_hook(observe)
    try:
        async with asyncio.timeout(10):
            await resumed.start()
            await all_arrived.wait()
            running = await resumed.list_runs("recoverable", status=RunStatus.RUNNING)
            assert {r.event_id for r in running} == {e.event_id for e in events}
            assert not original_ids.intersection(r.run_id for r in running)
            release.set()
            assert await resumed.wait_idle(timeout=5)
            assert sorted(completed) == sorted(e.event_id for e in events)
            history = await resumed.list_runs("recoverable")
            assert len(history) == config.max_records_per_strategy
            assert all(r.status == RunStatus.SUCCESS for r in history)
    finally:
        LOCKS.set_acquire_hook(None)
        release.set()
        await resumed.stop()
