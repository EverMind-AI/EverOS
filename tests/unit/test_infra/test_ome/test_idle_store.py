from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from everos.component.utils.datetime import get_now_with_timezone
from everos.infra.ome._stores.idle import IdleStore
from everos.infra.ome._stores.storage import OMEStorage


@pytest.fixture
async def store(tmp_path: Path) -> IdleStore:
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    return IdleStore(storage=storage)


@pytest.mark.asyncio
async def test_touch_records_activity(store: IdleStore) -> None:
    now = get_now_with_timezone()
    await store.touch("s", "u1", at=now)
    last = await store.get_last_activity("s", "u1")
    assert last == now


@pytest.mark.asyncio
async def test_scan_idle_returns_overdue(store: IdleStore) -> None:
    now = get_now_with_timezone()
    old = now - timedelta(seconds=1000)
    fresh = now - timedelta(seconds=100)

    await store.touch("s", "u_old", at=old)
    await store.touch("s", "u_fresh", at=fresh)

    overdue = await store.scan_idle("s", idle_seconds=900, now=now)
    assert overdue == ["u_old"]


@pytest.mark.asyncio
async def test_scan_idle_empty_when_none_overdue(store: IdleStore) -> None:
    now = get_now_with_timezone()
    await store.touch("s", "u1", at=now)
    overdue = await store.scan_idle("s", idle_seconds=900, now=now)
    assert overdue == []


@pytest.mark.asyncio
async def test_touch_updates_existing_row(store: IdleStore) -> None:
    early = get_now_with_timezone() - timedelta(seconds=500)
    late = get_now_with_timezone()
    await store.touch("s", "u1", at=early)
    await store.touch("s", "u1", at=late)
    assert await store.get_last_activity("s", "u1") == late


@pytest.mark.asyncio
async def test_scan_idle_returns_buckets_oldest_first(store: IdleStore) -> None:
    """``scan_idle`` must return buckets in ascending ``last_activity_ts``
    order so IdleTick emission order is reproducible across SQLite versions
    and query plans.
    """
    now = get_now_with_timezone()
    await store.touch("s", "u_mid", at=now - timedelta(seconds=1500))
    await store.touch("s", "u_oldest", at=now - timedelta(seconds=2000))
    await store.touch("s", "u_newest", at=now - timedelta(seconds=1100))

    overdue = await store.scan_idle("s", idle_seconds=900, now=now)
    assert overdue == ["u_oldest", "u_mid", "u_newest"]


@pytest.mark.asyncio
async def test_scan_idle_uses_composite_index(tmp_path: Path) -> None:
    """``scan_idle``'s SQL must keep ``last_activity_ts`` un-wrapped so
    the ``(strategy_name, last_activity_ts)`` index is honoured. Verify
    via EXPLAIN QUERY PLAN — if a future refactor wraps the column in a
    function/CAST again, this test fails immediately instead of waiting
    for a perf regression in production.
    """
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()

    cutoff_iso = "2026-05-13T00:00:00+00:00"
    async with storage.connect() as conn:
        cur = await conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT bucket_key FROM idle_store "
            "WHERE strategy_name = ? AND last_activity_ts <= ?",
            ("s", cutoff_iso),
        )
        rows = await cur.fetchall()

    plan = " ".join(str(r) for r in rows)
    assert "idx_idle_scan" in plan, f"expected idx_idle_scan in plan, got: {plan}"
