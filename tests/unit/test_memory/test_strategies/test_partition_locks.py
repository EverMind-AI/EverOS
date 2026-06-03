"""Tests for :mod:`everos.memory.strategies._partition_locks`.

The helper is the foundation under every strategy that performs a
read → modify → write on shared state; its own behaviour (lock reuse,
strategy isolation, FIFO serialisation, parallel keys) is exercised
here, in isolation from any business strategy.
"""

from __future__ import annotations

import asyncio

import pytest

from everos.memory.strategies._partition_locks import (
    _reset_for_tests,
    get_partition_lock,
)


@pytest.fixture(autouse=True)
def _isolate_locks() -> None:
    """Each test gets a clean registry — no inherited holders / waiters."""
    _reset_for_tests()


def test_same_strategy_same_key_returns_identical_lock() -> None:
    """Repeat lookups must reuse the lock (otherwise serialisation breaks)."""
    a = get_partition_lock("strategy_x", "k1")
    b = get_partition_lock("strategy_x", "k1")
    assert a is b


def test_same_strategy_different_keys_return_distinct_locks() -> None:
    """Different partition keys must not block each other."""
    assert get_partition_lock("strategy_x", "k1") is not get_partition_lock(
        "strategy_x", "k2"
    )


def test_different_strategies_share_no_locks_for_identical_key() -> None:
    """Strategy namespaces are independent — same key string is two locks."""
    assert get_partition_lock("strategy_x", "k1") is not get_partition_lock(
        "strategy_y", "k1"
    )


def test_reset_for_tests_drops_every_lock() -> None:
    """After reset the registry is empty; the next lookup returns a fresh lock."""
    before = get_partition_lock("strategy_x", "k1")
    _reset_for_tests()
    after = get_partition_lock("strategy_x", "k1")
    assert before is not after


async def test_same_key_serialises_concurrent_acquirers() -> None:
    """Two tasks contending the same key must not overlap critical sections."""
    log: list[str] = []

    async def worker(tag: str) -> None:
        async with get_partition_lock("strategy_x", "k1"):
            log.append(f"enter:{tag}")
            await asyncio.sleep(0.01)
            log.append(f"leave:{tag}")

    await asyncio.gather(worker("a"), worker("b"))

    # The two critical sections must run one after the other (either order
    # is fine — asyncio scheduling decides who acquires first).
    assert log in (
        ["enter:a", "leave:a", "enter:b", "leave:b"],
        ["enter:b", "leave:b", "enter:a", "leave:a"],
    )


async def test_different_keys_run_in_parallel() -> None:
    """Two tasks on distinct keys must overlap (no false serialisation)."""
    log: list[str] = []

    async def worker(key: str, tag: str) -> None:
        async with get_partition_lock("strategy_x", key):
            log.append(f"enter:{tag}")
            await asyncio.sleep(0.01)
            log.append(f"leave:{tag}")

    await asyncio.gather(worker("k1", "a"), worker("k2", "b"))

    # Both must enter before either leaves — proves no cross-key blocking.
    assert log.index("enter:a") < log.index("leave:b")
    assert log.index("enter:b") < log.index("leave:a")


async def test_concurrent_acquirers_fifo_fairness() -> None:
    """asyncio.Lock is FIFO — queued waiters acquire in arrival order."""
    log: list[str] = []
    holder_in = asyncio.Event()
    holder_release = asyncio.Event()

    async def holder() -> None:
        async with get_partition_lock("strategy_x", "k1"):
            holder_in.set()
            await holder_release.wait()
            log.append("leave:holder")

    async def waiter(tag: str, arrived: asyncio.Event) -> None:
        arrived.set()
        async with get_partition_lock("strategy_x", "k1"):
            log.append(f"enter:{tag}")

    arrived_a = asyncio.Event()
    arrived_b = asyncio.Event()
    task_holder = asyncio.create_task(holder())
    await holder_in.wait()  # holder owns the lock

    # Enqueue A first, then B — Lock's deque preserves this order.
    task_a = asyncio.create_task(waiter("a", arrived_a))
    await arrived_a.wait()
    await asyncio.sleep(0)  # let A actually park on the lock
    task_b = asyncio.create_task(waiter("b", arrived_b))
    await arrived_b.wait()
    await asyncio.sleep(0)  # let B park on the lock

    holder_release.set()
    await asyncio.gather(task_holder, task_a, task_b)

    assert log == ["leave:holder", "enter:a", "enter:b"]
