"""Tests for RunRecordStore persistence layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.utils.datetime import get_now_with_timezone
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.records import RunStatus


@pytest.fixture
async def store(tmp_path: Path) -> RunRecordStore:
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    return RunRecordStore(storage=storage, max_records_per_strategy=3)


@pytest.mark.asyncio
async def test_mark_running_inserts_row(store: RunRecordStore) -> None:
    await store.mark_running(
        run_id="r1",
        strategy_name="s",
        attempt=0,
        event_topic="x:Y",
        event_payload="{}",
        max_retries_snapshot=1,
    )
    rec = await store.get("r1")
    assert rec is not None
    assert rec.status == RunStatus.RUNNING


@pytest.mark.asyncio
async def test_mark_success_updates_row(store: RunRecordStore) -> None:
    await store.mark_running(
        run_id="r1",
        strategy_name="s",
        attempt=0,
        event_topic="x:Y",
        event_payload="{}",
        max_retries_snapshot=1,
    )
    await store.mark_success(run_id="r1", finished_at=get_now_with_timezone())
    rec = await store.get("r1")
    assert rec.status == RunStatus.SUCCESS
    assert rec.finished_at is not None


@pytest.mark.asyncio
async def test_mark_failed_records_error(store: RunRecordStore) -> None:
    await store.mark_running(
        run_id="r1",
        strategy_name="s",
        attempt=0,
        event_topic="x:Y",
        event_payload="{}",
        max_retries_snapshot=1,
    )
    await store.mark_failed(
        run_id="r1", finished_at=get_now_with_timezone(), error="boom"
    )
    rec = await store.get("r1")
    assert rec.status == RunStatus.FAILED
    assert rec.error == "boom"


@pytest.mark.asyncio
async def test_mark_dead_letter(store: RunRecordStore) -> None:
    await store.mark_running(
        run_id="r1",
        strategy_name="s",
        attempt=2,
        event_topic="x:Y",
        event_payload="{}",
        max_retries_snapshot=2,
    )
    await store.mark_dead_letter(
        run_id="r1", finished_at=get_now_with_timezone(), error="exhausted"
    )
    rec = await store.get("r1")
    assert rec.status == RunStatus.DEAD_LETTER


@pytest.mark.asyncio
async def test_ring_buffer_caps_strategy_records(store: RunRecordStore) -> None:
    """Trim runs inside the same transaction as each ``mark_running``
    insert; the per-strategy row count never exceeds the cap.
    """
    for i in range(5):
        await store.mark_running(
            run_id=f"r{i}",
            strategy_name="s",
            attempt=0,
            event_topic="x:Y",
            event_payload="{}",
            max_retries_snapshot=1,
        )
        listed = await store.list_runs(strategy_name="s")
        assert len(listed) <= 3  # never transiently above cap

    listed = await store.list_runs(strategy_name="s")
    assert [r.run_id for r in listed] == ["r4", "r3", "r2"]  # newest 3


@pytest.mark.asyncio
async def test_list_runs_filters_by_status(store: RunRecordStore) -> None:
    await store.mark_running(
        run_id="r1",
        strategy_name="s",
        attempt=0,
        event_topic="x:Y",
        event_payload="{}",
        max_retries_snapshot=1,
    )
    await store.mark_success(run_id="r1", finished_at=get_now_with_timezone())
    await store.mark_running(
        run_id="r2",
        strategy_name="s",
        attempt=0,
        event_topic="x:Y",
        event_payload="{}",
        max_retries_snapshot=1,
    )
    success_runs = await store.list_runs(strategy_name="s", status=RunStatus.SUCCESS)
    assert [r.run_id for r in success_runs] == ["r1"]


@pytest.mark.asyncio
async def test_find_running_for_crash_recovery(store: RunRecordStore) -> None:
    await store.mark_running(
        run_id="r1",
        strategy_name="s",
        attempt=0,
        event_topic="x:Y",
        event_payload="{}",
        max_retries_snapshot=1,
    )
    running = await store.find_running()
    assert len(running) == 1
    assert running[0].run_id == "r1"
