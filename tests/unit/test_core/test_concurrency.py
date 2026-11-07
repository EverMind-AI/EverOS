"""Task ownership, cancellation and capacity at cooperative lock boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from everos.core.concurrency import ConcurrencySlot, acquire_without_slot
from everos.memory._partition_locks import get_partition_lock, user_cluster_lock


async def _assert_capacity(sem: asyncio.Semaphore, capacity: int) -> None:
    async with asyncio.timeout(1):
        for _ in range(capacity):
            await sem.acquire()
    assert sem.locked(), "A cancelled waiter released someone else's permit"
    for _ in range(capacity):
        sem.release()


@pytest.mark.parametrize("phase", ["lock", "capacity", "body"])
async def test_cancellation_returns_only_owned_capacity(phase: str) -> None:
    sem = asyncio.BoundedSemaphore(1)
    lock = asyncio.Lock()
    waiting = asyncio.Event()
    acquired_lock = asyncio.Event()
    body = asyncio.Event()
    never = asyncio.Event()
    await lock.acquire()

    @asynccontextmanager
    async def observed_lock() -> AsyncIterator[None]:
        waiting.set()
        async with lock:
            acquired_lock.set()
            yield

    async def worker() -> None:
        async with ConcurrencySlot(sem), acquire_without_slot(observed_lock()):
            body.set()
            await never.wait()

    task = asyncio.create_task(worker())
    owns_lock = True
    owns_capacity = False
    try:
        async with asyncio.timeout(2):
            await waiting.wait()
            if phase == "capacity":
                await sem.acquire()
                owns_capacity = True
            if phase != "lock":
                lock.release()
                owns_lock = False
                await acquired_lock.wait()
            if phase == "body":
                await body.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            if owns_capacity:
                assert sem.locked()
                assert not body.is_set()
            if not owns_lock:
                assert not lock.locked()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if owns_lock:
            lock.release()
        if owns_capacity:
            sem.release()
    await _assert_capacity(sem, 1)
    async with ConcurrencySlot(sem), acquire_without_slot(lock):
        assert sem.locked()


async def test_child_does_not_return_parent_capacity() -> None:
    sem = asyncio.BoundedSemaphore(1)
    waiting = asyncio.Event()
    lock = asyncio.Lock()
    await lock.acquire()

    @asynccontextmanager
    async def observed_lock() -> AsyncIterator[None]:
        waiting.set()
        async with lock:
            yield

    async def child() -> None:
        async with acquire_without_slot(observed_lock()):
            assert sem.locked()

    async with ConcurrencySlot(sem):
        task = asyncio.create_task(child())
        try:
            async with asyncio.timeout(2):
                await waiting.wait()
                assert sem.locked()
                lock.release()
                await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    await _assert_capacity(sem, 1)


async def test_caught_lock_error_restores_capacity() -> None:
    sem = asyncio.BoundedSemaphore(1)

    @asynccontextmanager
    async def broken_lock() -> AsyncIterator[None]:
        assert not sem.locked()
        raise ValueError("lock failed")
        yield  # pragma: no cover

    async with ConcurrencySlot(sem):
        with pytest.raises(ValueError, match="lock failed"):
            async with acquire_without_slot(broken_lock()):
                pytest.fail("Must not enter body")
        assert sem.locked()
    await _assert_capacity(sem, 1)


async def test_nested_partition_wait_releases_capacity_but_not_outer_lock() -> None:
    """Reflection's deprecate -> user_cluster order with a single OME slot."""
    sem = asyncio.BoundedSemaphore(1)
    outer_entered = asyncio.Event()
    inner_entered = asyncio.Event()

    async def reflection() -> None:
        async with (
            ConcurrencySlot(sem),
            get_partition_lock("reflection_deprecate", "cluster"),
        ):
            outer_entered.set()
            async with user_cluster_lock("app", "project", "owner"):
                inner_entered.set()

    async with user_cluster_lock("app", "project", "owner"):
        task = asyncio.create_task(reflection())
        try:
            async with asyncio.timeout(2):
                await outer_entered.wait()
                # Waiting for inner lock gives other OME work the only slot.
                async with ConcurrencySlot(sem):
                    assert not inner_entered.is_set()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    async with asyncio.timeout(2):
        # Both locks must be reusable after cancellation, with the same order.
        await reflection()
    assert inner_entered.is_set()
    await _assert_capacity(sem, 1)


async def test_resuming_many_partitions_keeps_capacity_exact() -> None:
    capacity = 3
    sem = asyncio.BoundedSemaphore(capacity)
    locks = [asyncio.Lock() for _ in range(12)]
    for lock in locks:
        await lock.acquire()
    all_waiting = asyncio.Event()
    full = asyncio.Event()
    finish = asyncio.Event()
    waiting = 0
    active = 0
    peak = 0

    @asynccontextmanager
    async def observed(lock: asyncio.Lock) -> AsyncIterator[None]:
        nonlocal waiting
        waiting += 1
        if waiting == len(locks):
            all_waiting.set()
        async with lock:
            yield

    async def worker(lock: asyncio.Lock) -> None:
        nonlocal active, peak
        async with ConcurrencySlot(sem), acquire_without_slot(observed(lock)):
            active += 1
            peak = max(peak, active)
            try:
                if active == capacity:
                    full.set()
                await finish.wait()
            finally:
                active -= 1

    tasks = [asyncio.create_task(worker(lock)) for lock in locks]
    try:
        async with asyncio.timeout(3):
            await all_waiting.wait()
            for lock in locks:
                lock.release()
            await full.wait()
            # A loop turn lets every already-awakened contender try capacity.
            await asyncio.sleep(0)
            assert active == capacity
            finish.set()
            await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    assert peak == capacity
    await _assert_capacity(sem, capacity)
