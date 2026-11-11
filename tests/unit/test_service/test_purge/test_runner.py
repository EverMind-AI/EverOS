"""The purge runner: ten steps, retries, budget, terminal state, self-healing."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from everos.core import failpoints
from everos.core.purge_marks import MARKS
from everos.core.utils.datetime import get_utc_now
from everos.service.purge import _runner
from tests.fakes.storage import Storage

from ._harness import (
    FakeRun,
    Purge,
    arm,
    disarm,
    make_purge,
    payload,
    release,
    seed_scope,
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    MARKS.reset()
    yield
    MARKS.reset()
    disarm(monkeypatch)


async def _confirmed(
    storage: Storage,
    *,
    scope: tuple[str, str | None, str | None] = ("project", "a1", "p1"),
    **settings: Any,
) -> tuple[Purge, str]:
    await storage.create_schema()
    p = make_purge(storage, **settings)
    await seed_scope(storage, "a1", "p1")
    await seed_scope(storage, "a1", "p2")
    p.engine.failed_payloads = {"s": payload("a1", "p1"), "o": payload("a1", "p2")}
    job = await p.service.confirm((await p.service.request(*scope)).nonce)
    return p, job.job_id


def _record(monkeypatch: pytest.MonkeyPatch, p: Purge) -> list[str]:
    """Every failpoint the runner passes, plus ``finish`` and ``lift``."""
    log: list[str] = []
    real = failpoints.afailpoint

    async def recording(name: str) -> bool:
        log.append(name)
        return await real(name)

    monkeypatch.setattr(_runner, "afailpoint", recording)
    finish, lift = p.service.jobs.finish, p.marks.lift_mark

    async def finishing(*a: Any, **k: Any) -> None:
        log.append("finish")
        await finish(*a, **k)

    def lifting() -> None:
        log.append("lift")
        lift()

    monkeypatch.setattr(p.service.jobs, "finish", finishing)
    monkeypatch.setattr(p.marks, "lift_mark", lifting)
    return log


async def _row(p: Purge, job_id: str):  # type: ignore[no-untyped-def]
    row = await p.service.jobs.get(job_id)
    assert row is not None
    return row


async def test_happy_path_steps_and_order(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, job_id = await _confirmed(storage)
    log = _record(monkeypatch, p)
    assert await p.run_to_end(job_id) == "succeeded"
    steps = [n.removeprefix("purge.") for n in log if n not in ("finish", "lift")]
    first = [f"before_step{n}" for n in range(1, 7)]
    assert [s for s in steps if s.startswith("before_step")] == [
        *first,
        "before_step7",
        "before_step8",
        *first,
        "before_step9",
        "before_step10",
    ]
    assert log.index("finish") < log.index("lift")
    view = await p.service.get(job_id)
    assert {s.store: (s.state, s.deleted) for s in view.steps} == {
        "inbox": ("done", 0),
        "markdown": ("done", 2),
        "lancedb": ("done", 2),
        "sqlite": ("done", 2),
        "ome": ("done", 1),
        "md_change_state": ("done", 0),
    }
    assert p.engine.failed_payloads == {
        "s": '{"purged_by":"' + job_id + '"}',
        "o": payload("a1", "p2"),
    }
    s_left = (await p.service.request("project", "a1", "p1")).summary
    assert (s_left.markdown_total, s_left.sqlite_total, s_left.index_total) == (0, 0, 0)
    other = (await p.service.request("project", "a1", "p2")).summary
    assert (other.markdown_total, other.sqlite_total, other.index_total) == (2, 2, 2)
    assert p.woken == ["queue"] and p.marks.active_job_id is None


async def test_retryable_error_reruns_from_step1(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.before_step3=raise:infra:1")
    assert await p.run_to_end(job_id) == "succeeded"
    row = await _row(p, job_id)
    assert (row.attempts, row.retries) == (2, 1)


async def test_fast_failing_attempts_reach_failed(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.before_step3=raise:infra:99")
    assert await p.run_to_end(job_id) == "failed"
    row = await _row(p, job_id)
    assert (row.retries, row.active) == (_runner.MAX_RETRIES + 1, None)
    assert "failpoint purge.before_step3" in (row.last_error or "")


async def test_busy_retries_in_step_without_attempt(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.step3.delete=raise:busy:2")
    assert await p.run_to_end(job_id) == "succeeded"
    row = await _row(p, job_id)
    assert (row.attempts, row.retries) == (1, 0)


async def test_non_retryable_fails_and_lifts(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.before_step4=raise:domain:1")
    assert await p.run_to_end(job_id) == "failed"
    row = await _row(p, job_id)
    assert (row.attempts, row.retries) == (1, 0)
    view = await p.service.get(job_id)
    assert {s.store: s.state for s in view.steps}["sqlite"] == "pending"
    row_s = type("R", (), {"app_id": "a1", "project_id": "p1", "id": "x"})()
    assert not p.fence.blocks(row_s) and p.woken == ["queue"]


async def test_budget_timeout_fails_hung_step(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage, run_budget_seconds=0.5)
    arm(monkeypatch, tmp_path, "purge.before_step3=pause")  # never released
    assert await p.run_to_end(job_id) == "failed"
    assert "budget" in ((await _row(p, job_id)).last_error or "")


async def test_finalize_failure_keeps_mark(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.finalize=raise:sqlite_locked:2")
    marks_at_finalize: list[str | None] = []
    real = failpoints.afailpoint

    async def watching(name: str) -> bool:
        if name == "purge.finalize":
            marks_at_finalize.append(p.marks.active_job_id)
        return await real(name)

    monkeypatch.setattr(_runner, "afailpoint", watching)
    assert await p.run_to_end(job_id) == "succeeded"
    assert marks_at_finalize == [job_id, job_id, job_id]


async def test_cancelled_attempt_stays_running(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.before_step3=pause")
    runner = p.runner()
    runner.start()
    for _ in range(100):
        if (await _row(p, job_id)).status == "running":
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.1)
    await runner.stop()  # a normal shutdown mid-step (SIGTERM)
    row = await _row(p, job_id)
    assert (row.status, row.finished_at, row.active) == ("running", None, 1)
    assert p.marks.active_job_id == job_id
    release(tmp_path, "purge.before_step3")
    assert await p.run_to_end(job_id) == "succeeded"  # the next start resumes


async def test_loop_self_heals(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.runner_loop=raise:runtime:1")
    assert await p.run_to_end(job_id) == "succeeded"


async def test_poll_picks_up_without_wake(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    await storage.create_schema()
    p = make_purge(storage)
    runner = p.runner(poll_seconds=0.2)
    runner.start()
    try:
        arm(monkeypatch, tmp_path, "purge.confirm_wake=skip")
        await asyncio.sleep(0.1)  # the loop is idle, waiting
        job = await p.service.confirm(
            (await p.service.request("space", None, None)).nonce
        )
        async with asyncio.timeout(5):
            while True:
                if (await _row(p, job.job_id)).status == "succeeded":
                    break
                await asyncio.sleep(0.05)
    finally:
        await runner.stop()


async def test_settle_wait_only_snapshot(storage: Storage) -> None:
    await storage.create_schema()
    p = make_purge(storage, settle_wait_seconds=5)
    future = get_utc_now() + timedelta(hours=1)
    p.engine.running = [FakeRun("late", "e1", future, payload("a1", "p1"))]
    job = await p.service.confirm(
        (await p.service.request("project", "a1", "p1")).nonce
    )
    started = asyncio.get_running_loop().time()
    assert await p.run_to_end(job.job_id) == "succeeded"
    assert asyncio.get_running_loop().time() - started < 2  # never waited on it


async def test_settle_wait_waits_for_snapshot_run(storage: Storage) -> None:
    await storage.create_schema()
    p = make_purge(storage, settle_wait_seconds=5)
    old = FakeRun(
        "old", "e1", get_utc_now() - timedelta(seconds=1), payload("a1", "p1")
    )
    p.engine.running = [old]
    job = await p.service.confirm(
        (await p.service.request("project", "a1", "p1")).nonce
    )
    loop = asyncio.get_running_loop()
    loop.call_later(0.5, p.engine.running.clear)
    started = loop.time()
    assert await p.run_to_end(job.job_id) == "succeeded"
    assert 0.5 <= loop.time() - started < 3


async def test_settle_wait_skips_gated_runs(storage: Storage) -> None:
    await storage.create_schema()
    p = make_purge(storage, settle_wait_seconds=5)
    old = FakeRun(
        "old", "e1", get_utc_now() - timedelta(seconds=1), payload("a1", "p1")
    )
    p.engine.running = [old]
    job = await p.service.confirm(
        (await p.service.request("project", "a1", "p1")).nonce
    )
    p.marks.waiting_events.add("e1")  # it waits at the gate for this purge
    started = asyncio.get_running_loop().time()
    assert await p.run_to_end(job.job_id) == "succeeded"
    assert asyncio.get_running_loop().time() - started < 2


async def test_steps_deleted_accumulate_across_attempts(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.before_step4=raise:infra:1")
    assert await p.run_to_end(job_id) == "succeeded"
    view = await p.service.get(job_id)
    deleted = {s.store: s.deleted for s in view.steps}
    assert (deleted["markdown"], deleted["lancedb"], deleted["sqlite"]) == (2, 2, 2)
