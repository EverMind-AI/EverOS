"""Partition waiters must leave OME capacity for unrelated strategy work."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from everos.core.locks import LOCKS
from everos.infra.ome._dispatch.runner import Runner
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.config import OMEConfig
from everos.infra.ome.records import RunStatus
from everos.memory._partition_locks import get_partition_lock
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event, Immediate
from tests.helpers.ome import ome_strategy


class ProfileTask(Event):
    owner: str


async def _no_emit(event: Event) -> None:
    pass


@pytest.mark.parametrize("capacity", [2, 20])
async def test_one_owners_waiters_do_not_block_another_owner(
    tmp_path: Path, capacity: int
) -> None:
    storage = OMEStorage(tmp_path / "ome.db")
    await storage.init()
    await storage.hold_open()
    records = RunRecordStore(storage, max_records_per_strategy=1000)
    runner = Runner(
        run_record_store=records,
        engine_sem=asyncio.BoundedSemaphore(capacity),
        emit_hook=_no_emit,
        config=OMEConfig(jobstore_path=storage.db_path),
    )
    release_alice = asyncio.Event()
    alice_waiters_arrived = asyncio.Event()
    bob_started = asyncio.Event()
    arrived = 0
    alice_active = 0
    peak_alice = 0
    writes: list[str] = []

    def on_acquire(task, wanted, held) -> None:
        nonlocal arrived
        if wanted == ("partition:profile", "alice"):
            arrived += 1
            if arrived == capacity:
                alice_waiters_arrived.set()

    @ome_strategy(name="profile", trigger=Immediate(on=[ProfileTask]))
    async def profile(event: ProfileTask, deps: MemoryDeps) -> list[Event]:
        nonlocal alice_active, peak_alice
        async with get_partition_lock("profile", event.owner):
            if event.owner == "alice":
                alice_active += 1
                peak_alice = max(peak_alice, alice_active)
                try:
                    await release_alice.wait()
                finally:
                    alice_active -= 1
            else:
                bob_started.set()
            writes.append(event.owner)
        return []

    LOCKS.set_acquire_hook(on_acquire)
    tasks = [
        asyncio.create_task(
            runner.run(
                profile,
                ProfileTask(owner="alice"),
                run_id=f"alice_{i}",
                max_retries_snapshot=0,
            )
        )
        for i in range(capacity)
    ]
    try:
        async with asyncio.timeout(5):
            await alice_waiters_arrived.wait()
            # Waiting attempts retain recoverable records, not just coroutines.
            assert len(await records.find_running()) == capacity
            tasks.append(
                asyncio.create_task(
                    runner.run(
                        profile,
                        ProfileTask(owner="bob"),
                        run_id="bob",
                        max_retries_snapshot=0,
                    )
                )
            )
            # Alice's model remains blocked throughout this observation.
            await asyncio.wait_for(bob_started.wait(), timeout=0.5)
            assert not release_alice.is_set()
            assert writes == ["bob"]
    finally:
        release_alice.set()
        LOCKS.set_acquire_hook(None)
        try:
            async with asyncio.timeout(5):
                await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await storage.release()

    assert peak_alice == 1
    assert writes.count("alice") == capacity
    assert writes.count("bob") == 1
    completed = await records.list_runs(strategy_name="profile")
    assert len(completed) == capacity + 1
    assert all(record.status == RunStatus.SUCCESS for record in completed)
