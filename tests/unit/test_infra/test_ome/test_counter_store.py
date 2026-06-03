from __future__ import annotations

from pathlib import Path

import pytest

from everos.infra.ome._stores.counter import CounterStore
from everos.infra.ome._stores.storage import OMEStorage


@pytest.fixture
async def store(tmp_path: Path) -> CounterStore:
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    return CounterStore(storage=storage)


@pytest.mark.asyncio
async def test_increments_until_threshold(store: CounterStore) -> None:
    for i in range(1, 5):
        passed, cur = await store.incr_and_check(
            "s",
            "u1",
            threshold=5,
            cooldown_seconds=0,
        )
        assert passed is False
        assert cur == i

    passed, cur = await store.incr_and_check(
        "s",
        "u1",
        threshold=5,
        cooldown_seconds=0,
    )
    assert passed is True
    assert cur == 5


@pytest.mark.asyncio
async def test_resets_after_pass(store: CounterStore) -> None:
    for _ in range(5):
        await store.incr_and_check("s", "u1", threshold=5, cooldown_seconds=0)
    passed, cur = await store.incr_and_check(
        "s",
        "u1",
        threshold=5,
        cooldown_seconds=0,
    )
    assert passed is False
    assert cur == 1


@pytest.mark.asyncio
async def test_cooldown_blocks_pass(store: CounterStore) -> None:
    # First pass
    for _ in range(5):
        await store.incr_and_check("s", "u1", threshold=5, cooldown_seconds=10)
    # Threshold met again immediately, but cooldown blocks
    for _ in range(5):
        passed, _ = await store.incr_and_check(
            "s",
            "u1",
            threshold=5,
            cooldown_seconds=10,
        )
    assert passed is False


@pytest.mark.asyncio
async def test_buckets_are_isolated(store: CounterStore) -> None:
    for _ in range(5):
        await store.incr_and_check("s", "u1", threshold=5, cooldown_seconds=0)
    passed, cur = await store.incr_and_check(
        "s",
        "u2",
        threshold=5,
        cooldown_seconds=0,
    )
    assert cur == 1
    assert passed is False


@pytest.mark.asyncio
async def test_progress_query(store: CounterStore) -> None:
    await store.incr_and_check("s", "u1", threshold=5, cooldown_seconds=0)
    await store.incr_and_check("s", "u1", threshold=5, cooldown_seconds=0)
    cur = await store.get_progress("s", "u1")
    assert cur == 2


@pytest.mark.asyncio
async def test_returned_counter_reflects_actual_value_when_threshold_lowered(
    store: CounterStore,
) -> None:
    """When threshold drops via hot-reload after counter accumulation,
    the returned counter must reflect the *actual* count at trigger
    moment, not the (lower) threshold. Diagnostics rely on this.
    """
    # Accumulate 7 hits under a high threshold; none pass.
    for _ in range(7):
        passed, _ = await store.incr_and_check(
            "s", "u1", threshold=10, cooldown_seconds=0
        )
        assert passed is False

    # Threshold is "lowered" to 5 (config hot-reload semantics).
    # Counter goes 7 -> 8, which is past the new threshold.
    passed, cur = await store.incr_and_check("s", "u1", threshold=5, cooldown_seconds=0)
    assert passed is True
    assert cur == 8  # actual count, not threshold (=5)
