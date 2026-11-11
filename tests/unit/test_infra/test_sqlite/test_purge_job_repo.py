"""``purge_job``: one active row per root; the confirm time reads back aware."""

from __future__ import annotations

from datetime import UTC, datetime

from everos.core.utils.datetime import get_utc_now
from tests.fakes.storage import Storage

T = datetime(2026, 11, 11, 10, 0, 0, 123456, tzinfo=UTC)


def _job(job_id: str) -> dict[str, object]:
    return {
        "job_id": job_id,
        "scope": "project",
        "app_id": "a1",
        "project_id": "p1",
        "created_at": T,
        "steps_json": "[]",
    }


async def test_one_active_row_per_root(storage: Storage) -> None:
    await storage.create_schema()
    repo = storage.sqlite.purge_job
    assert await repo.insert(**_job("j1")) is not None
    assert await repo.insert(**_job("j2")) is None
    assert (await repo.active()).job_id == "j1"  # type: ignore[union-attr]
    await repo.finish(
        "j1", status="succeeded", finished_at=T, steps_json="[]", last_error=None
    )
    assert await repo.active() is None
    assert await repo.insert(**_job("j3")) is not None
    assert [r.job_id for r in await repo.all_rows()] == ["j1", "j3"]
    done = await repo.get("j1")
    assert done is not None and (done.status, done.active) == ("succeeded", None)


async def test_created_at_round_trips_as_aware_utc(storage: Storage) -> None:
    await storage.create_schema()
    repo = storage.sqlite.purge_job
    await repo.insert(**_job("j1"))
    row = await repo.get("j1")
    assert row is not None
    assert row.created_at == T  # comparing naive with aware would raise
    assert row.created_at.tzinfo is not None
    event_ts = datetime(2026, 11, 11, 9, 59, tzinfo=UTC)
    assert row.created_at > event_ts


async def test_update_progress_moves_the_heartbeat(storage: Storage) -> None:
    await storage.create_schema()
    repo = storage.sqlite.purge_job
    confirmed_at = get_utc_now()  # T is the confirm time: now, not later
    await repo.insert(**(_job("j1") | {"created_at": confirmed_at}))
    before = await repo.get("j1")
    await repo.update_progress("j1", status="running", attempts=1, run_seconds=1.5)
    after = await repo.get("j1")
    assert before is not None and after is not None
    assert (after.status, after.attempts, after.run_seconds) == ("running", 1, 1.5)
    assert after.updated_at >= before.updated_at
    assert after.created_at == confirmed_at
