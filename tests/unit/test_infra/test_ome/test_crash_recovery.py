from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from everos.core.utils.datetime import get_now_with_timezone, to_iso_format
from everos.infra.ome._background.crash_recovery import scan_and_resume
from everos.infra.ome._stores.run_record import RunRecordStore
from everos.infra.ome._stores.storage import OMEStorage
from everos.infra.ome.records import RunStatus


@pytest.fixture
async def rec_store(tmp_path: Path) -> RunRecordStore:
    storage = OMEStorage(db_path=tmp_path / "ome.db")
    await storage.init()
    return RunRecordStore(storage=storage, max_records_per_strategy=1000)


@pytest.mark.asyncio
async def test_marks_running_as_crashed_and_resumes(rec_store: RunRecordStore) -> None:
    await rec_store.mark_running(
        run_id="r_old",
        strategy_name="s",
        attempt=0,
        event_topic="x:E",
        event_payload="{}",
        max_retries_snapshot=1,
        event_id="evt_test",
    )
    async with rec_store._storage.connect() as conn:
        rewind = to_iso_format(get_now_with_timezone() - timedelta(hours=2))
        await conn.execute(
            "UPDATE run_record SET started_at = ? WHERE run_id = ?",
            (rewind, "r_old"),
        )
        await conn.commit()

    resumed: list = []

    async def add_job_hook(name, run_id, event_topic, event_payload, max_retries):
        resumed.append((name, run_id, event_topic, event_payload, max_retries))

    await scan_and_resume(
        run_record_store=rec_store,
        add_job=add_job_hook,
    )

    rec = await rec_store.get("r_old")
    assert rec.status == RunStatus.CRASHED
    assert len(resumed) == 1
    new_name, new_run_id, ec, ep, mr = resumed[0]
    assert new_name == "s"
    assert new_run_id != "r_old"
    assert ec == "x:E"
    assert ep == "{}"
    assert mr == 1


@pytest.mark.asyncio
async def test_fresh_running_row_is_recovered_too(rec_store: RunRecordStore) -> None:
    """A row written seconds before the crash is as dead as an old one (#540).

    ``start`` holds the jobstore lock, so no run of a previous process can
    still be alive; an age filter here only lost the runs cut short by a
    restart within the window.
    """
    await rec_store.mark_running(
        run_id="r_fresh",
        strategy_name="s",
        attempt=0,
        event_topic="x:E",
        event_payload="{}",
        max_retries_snapshot=1,
        event_id="evt_test",
    )
    resumed: list = []

    async def add_job_hook(*args, **kw):
        resumed.append(args)

    await scan_and_resume(run_record_store=rec_store, add_job=add_job_hook)
    rec = await rec_store.get("r_fresh")
    assert rec.status == RunStatus.CRASHED
    assert len(resumed) == 1


@pytest.mark.asyncio
async def test_add_job_failure_does_not_abort_loop(
    rec_store: RunRecordStore,
) -> None:
    """add_job raising on one row must not block sibling stale rows.

    mark_crashed runs before add_job, so both rows end up CRASHED even
    when add_job fails for one. This pins the at-most-once contract
    documented in the module docstring.
    """
    for run_id in ("r_old_1", "r_old_2"):
        await rec_store.mark_running(
            run_id=run_id,
            strategy_name="s",
            attempt=0,
            event_topic="x:E",
            event_payload="{}",
            max_retries_snapshot=1,
            event_id="evt_test",
        )
    async with rec_store._storage.connect() as conn:
        rewind = to_iso_format(get_now_with_timezone() - timedelta(hours=2))
        await conn.execute(
            "UPDATE run_record SET started_at = ? WHERE run_id IN (?, ?)",
            (rewind, "r_old_1", "r_old_2"),
        )
        await conn.commit()

    calls: list[tuple] = []

    async def flaky_add_job(name, run_id, event_topic, event_payload, max_retries):
        calls.append((name, run_id, event_topic, event_payload, max_retries))
        if len(calls) == 1:
            raise RuntimeError("APS jobstore unavailable")

    await scan_and_resume(
        run_record_store=rec_store,
        add_job=flaky_add_job,
    )

    rec1 = await rec_store.get("r_old_1")
    rec2 = await rec_store.get("r_old_2")
    assert rec1.status == RunStatus.CRASHED
    assert rec2.status == RunStatus.CRASHED
    assert len(calls) == 2
