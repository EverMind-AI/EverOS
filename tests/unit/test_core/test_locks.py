"""Tests for :mod:`everos.core.locks`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from prometheus_client import REGISTRY

from everos.core.locks import LOCKS, owner_key
from everos.core.observability.metrics import generate_metrics_response


def _table_entries(kind: str) -> float | None:
    return REGISTRY.get_sample_value("everos_lock_table_entries", {"kind": kind})


async def test_hold_serialises_same_key_and_records_holder() -> None:
    log: list[str] = []
    first_in = asyncio.Event()
    release = asyncio.Event()

    async def first() -> None:
        async with LOCKS.hold("owner", "k"):
            log.append("first-in")
            first_in.set()
            await release.wait()
            log.append("first-out")

    async def second() -> None:
        async with LOCKS.hold("owner", "k"):
            log.append("second-in")

    t1 = asyncio.create_task(first())
    await first_in.wait()
    t2 = asyncio.create_task(second())
    await asyncio.sleep(0.01)
    assert log == ["first-in"]
    assert LOCKS.held(t1) == {("owner", "k")}
    assert LOCKS.held(t2) == frozenset()
    release.set()
    await asyncio.gather(t1, t2)
    assert log == ["first-in", "first-out", "second-in"]
    assert LOCKS.held(t1) == frozenset()


async def test_different_keys_do_not_block() -> None:
    async with LOCKS.hold("owner", "a"), LOCKS.hold("owner", "b"):
        assert LOCKS.held() == {("owner", "a"), ("owner", "b")}


async def test_timeout_raises_and_holds_nothing() -> None:
    release = asyncio.Event()
    holding = asyncio.Event()

    async def holder() -> None:
        async with LOCKS.hold("session", "s"):
            holding.set()
            await release.wait()

    t = asyncio.create_task(holder())
    await holding.wait()
    with pytest.raises(TimeoutError):
        async with LOCKS.hold("session", "s", max_wait=0.05):
            pytest.fail("acquired a held lock")
    assert LOCKS.held() == frozenset()
    release.set()
    await t
    # The timed-out waiter left nothing behind: the lock is free again.
    async with LOCKS.hold("session", "s", max_wait=0.05):
        pass


async def test_timeout_bounds_only_the_wait() -> None:
    async with LOCKS.hold("session", "s", max_wait=0.01):
        await asyncio.sleep(0.05)


async def test_child_task_does_not_inherit_held_locks() -> None:
    async with LOCKS.hold("owner", "k"):
        child = asyncio.create_task(asyncio.sleep(0))
        assert LOCKS.held(child) == frozenset()
        assert LOCKS.held() == {("owner", "k")}
        await child


async def test_acquire_hook_sees_already_held() -> None:
    seen: list[tuple[Any, tuple[str, str], frozenset[tuple[str, str]]]] = []
    LOCKS.set_acquire_hook(lambda task, wanted, held: seen.append((task, wanted, held)))
    async with LOCKS.hold("owner", "k"):
        with LOCKS.observe("md_path", "p"):
            assert LOCKS.held() == {("owner", "k"), ("md_path", "p")}
    me = asyncio.current_task()
    assert seen == [
        (me, ("owner", "k"), frozenset()),
        (me, ("md_path", "p"), frozenset({("owner", "k")})),
    ]
    assert LOCKS.held() == frozenset()


async def test_observe_nests_same_key() -> None:
    with LOCKS.observe("sqlite_session", "checkout"):
        with LOCKS.observe("sqlite_session", "checkout"):
            pass
        assert LOCKS.held() == {("sqlite_session", "checkout")}
    assert LOCKS.held() == frozenset()


async def test_hold_histogram_has_20ms_edge() -> None:
    async with LOCKS.hold("owner", "k"):
        pass
    text = generate_metrics_response().decode()
    assert 'everos_lock_hold_seconds_bucket{kind="owner",le="0.02"}' in text
    assert 'everos_lock_wait_seconds_bucket{kind="owner",le="0.02"}' in text


async def test_lock_table_gauge_counts_distinct_keys() -> None:
    for key in ("a", "b", "c", "a"):
        async with LOCKS.hold("owner", key):
            pass
    assert _table_entries("owner") == 3


def test_table_follows_the_running_loop() -> None:
    """Contention binds a lock to its loop; a second loop gets fresh locks."""

    async def contend() -> None:
        async def holder() -> None:
            async with LOCKS.hold("session", "s"):
                await asyncio.sleep(0.01)

        await asyncio.gather(holder(), holder())

    asyncio.run(contend())
    asyncio.run(contend())  # RuntimeError "bound to a different event loop"


def test_delete_count_per_owner() -> None:
    k1 = owner_key("app", "proj", "users", "u1")
    k2 = owner_key("app", "proj", "users", "u2")
    assert LOCKS.delete_count(k1) == 0
    LOCKS.bump_delete_count(k1)
    LOCKS.bump_delete_count(k1)
    assert (LOCKS.delete_count(k1), LOCKS.delete_count(k2)) == (2, 0)
    assert _table_entries("delete_count") == 1


def test_owner_key_tells_users_from_agents() -> None:
    assert owner_key("a", "p", "users", "u") != owner_key("a", "p", "agents", "u")


# ── Cross-test isolation ────────────────────────────────────────────────
# ``tests/conftest.py`` resets the table around every test: pytest-asyncio
# gives each test its own loop, and a lock carried over may be bound to a
# closed one. These two pin that, in order.


async def test_dirty_the_lock_table() -> None:
    async with LOCKS.hold("isolation_probe", "k"):
        pass
    assert _table_entries("isolation_probe") == 1


def test_lock_table_starts_empty_in_every_test() -> None:
    assert not _table_entries("isolation_probe"), (
        "a previous test's locks survived into this one -- the table is not "
        "reset between tests"
    )
