"""The purge runner: nine steps, the wait, retries, budget, terminal state."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import delete

from everos.core import failpoints
from everos.core.purge_marks import MARKS
from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.sqlite import session_scope
from everos.infra.persistence.sqlite._tables import Memcell
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
    assert p.marks.active_job_id is None


async def test_retryable_error_reruns_from_step1(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.before_step3=raise:infra:1")
    assert await p.run_to_end(job_id) == "succeeded"
    row = await _row(p, job_id)
    assert (row.attempts, row.retries) == (2, 1)


@pytest.mark.parametrize("method", ["count_where", "delete"])
async def test_index_backend_error_reruns_the_attempt(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    p, job_id = await _confirmed(storage)

    class BackendDownError(Exception):  # a backend's own type, as pymilvus raises
        pass

    repo = next(iter(storage.repos.all()))
    real, calls = getattr(repo, method), []

    async def flaky(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        if len(calls) == 1:
            raise BackendDownError("server unavailable")
        return await real(*args, **kwargs)

    monkeypatch.setattr(repo, method, flaky)
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
    assert not p.fence.blocks(row_s) and p.marks.active_job_id is None


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


async def _confirm(p: Purge, *scope: Any) -> str:
    job = await p.service.confirm(
        (await p.service.request(*(scope or ("project", "a1", "p1")))).nonce
    )
    return job.job_id


async def _waited(p: Purge, job_id: str, release_after: float, undo: Any) -> float:
    """Seconds the job took to end, ``undo`` run after ``release_after``."""
    loop = asyncio.get_running_loop()
    loop.call_later(release_after, undo)
    started = loop.time()
    assert await p.run_to_end(job_id) == "succeeded"
    return loop.time() - started


async def _until_succeeded(p: Purge, job_id: str) -> None:
    async with asyncio.timeout(5):
        while True:
            if (await _row(p, job_id)).status == "succeeded":
                return
            await asyncio.sleep(0.05)


def _old() -> Any:
    return get_utc_now() - timedelta(seconds=1)


async def test_drain_ignores_runs_on_newer_input(storage: Storage) -> None:
    await storage.create_schema()
    p = make_purge(storage)
    future = get_utc_now() + timedelta(hours=1)
    p.engine.running = [FakeRun("new", "e1", future, payload("a1", "p1", future))]
    job_id = await _confirm(p)
    assert await _waited(p, job_id, 30, lambda: None) < 2


async def test_drain_waits_for_an_older_run_of_the_scope(storage: Storage) -> None:
    await storage.create_schema()
    p = make_purge(storage)
    p.engine.running = [FakeRun("old", "e1", _old(), payload("a1", "p1", _old()))]
    job_id = await _confirm(p)
    assert 0.5 <= await _waited(p, job_id, 0.5, p.engine.running.clear) < 3


async def test_drain_waits_for_an_older_unscoped_run(storage: Storage) -> None:
    """A tick (no scope) may write any scope: waited for like the scope's own."""
    await storage.create_schema()
    p = make_purge(storage)
    tick = json.dumps({"strategy_name": "reflect", "ts": _old().isoformat()})
    p.engine.running = [FakeRun("tick", "e1", _old(), tick)]
    job_id = await _confirm(p)
    assert 0.5 <= await _waited(p, job_id, 0.5, p.engine.running.clear) < 3


async def test_drain_ignores_a_tick_that_started_after_the_confirm(
    storage: Storage,
) -> None:
    """Fired before the confirm, started after it: it reads what is there
    then and is not old work, whatever its event time says."""
    await storage.create_schema()
    p = make_purge(storage)
    later = get_utc_now() + timedelta(hours=1)
    tick = json.dumps({"strategy_name": "reflect", "ts": _old().isoformat()})
    p.engine.running = [FakeRun("tick", "e1", later, tick)]
    job_id = await _confirm(p)
    assert await _waited(p, job_id, 30, lambda: None) < 2


async def test_drain_ignores_older_runs_of_other_scopes(storage: Storage) -> None:
    await storage.create_schema()
    p = make_purge(storage)
    p.engine.running = [FakeRun("o", "e1", _old(), payload("a2", "p1", _old()))]
    job_id = await _confirm(p)
    assert await _waited(p, job_id, 30, lambda: None) < 2


async def test_drain_waits_for_a_tracked_request(storage: Storage) -> None:
    await storage.create_schema()
    p = make_purge(storage)
    tracked = p.marks.track(_old())
    tracked.__enter__()
    job_id = await _confirm(p)
    with p.marks.track(get_utc_now() + timedelta(hours=1)):  # newer: not waited
        waited = await _waited(
            p, job_id, 0.5, lambda: tracked.__exit__(None, None, None)
        )
    assert 0.5 <= waited < 3


async def test_drain_waits_for_an_extraction_started_after_pass_one(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A request in flight at T may hand off its extraction only later."""
    await storage.create_schema()
    p = make_purge(storage)
    job_id = await _confirm(p)
    arm(monkeypatch, tmp_path, "purge.before_step7=pause")
    runner = p.runner()
    runner.start()
    try:
        await asyncio.sleep(0.3)  # pass one done, held before the wait
        p.extractions.append(_old())
        release(tmp_path, "purge.before_step7")
        await asyncio.sleep(0.5)
        assert (await _row(p, job_id)).status == "running"
        p.extractions.clear()
        await _until_succeeded(p, job_id)
    finally:
        await runner.stop()


async def test_drain_time_counts_toward_the_budget(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Persisted while it waits, so a restart does not reset it; at the
    budget the job fails and says why on the store the second pass missed."""
    monkeypatch.setattr(_runner, "_DRAIN_SAVE_S", 0.05)
    p, job_id = await _confirmed(storage, run_budget_seconds=1)
    p.extractions.append(_old())  # never ends
    assert await p.run_to_end(job_id) == "failed"
    row = await _row(p, job_id)
    assert "budget" in (row.last_error or "")
    assert row.run_seconds >= 0.8
    steps = {s.store: s for s in (await p.service.get(job_id)).steps}
    assert (steps["inbox"].state, "budget" in (steps["inbox"].error or "")) == (
        "failed",
        True,
    )


async def test_second_pass_waits_for_projections_reading_the_scope(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    await storage.create_schema()
    p = make_purge(storage)
    job_id = await _confirm(p)
    arm(monkeypatch, tmp_path, "purge.before_step8=pause")
    runner = p.runner()
    runner.start()
    try:
        await asyncio.sleep(0.3)  # waiting for in-flight work is over
        token = p.fence.register_run()  # a projection started meanwhile
        release(tmp_path, "purge.before_step8")
        await asyncio.sleep(0.5)
        assert (await _row(p, job_id)).status == "running"
        p.fence.end_run(token)
        await _until_succeeded(p, job_id)
    finally:
        await runner.stop()


async def test_rows_the_cascade_drops_after_step2_count_as_deleted(
    storage: Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    p, job_id = await _confirmed(storage)
    move = _runner.PurgeRunner._step_markdown

    async def move_then_cascade(self: Any, run: Any) -> int:
        moved = await move(self, run)
        # The cascade sees the moved files and drops their index rows (and
        # the knowledge row of a document) before step 3 gets there.
        await storage.repos.episode.delete(_runner.scope_filter(run.scope))
        async with session_scope(storage.database.session_factory) as s:
            await s.execute(delete(Memcell).where(Memcell.memcell_id == "mc_a1_p1_0"))
            await s.commit()
        return moved

    monkeypatch.setattr(_runner.PurgeRunner, "_step_markdown", move_then_cascade)
    assert await p.run_to_end(job_id) == "succeeded"
    deleted = {s.store: s.deleted for s in (await p.service.get(job_id)).steps}
    assert (deleted["lancedb"], deleted["sqlite"]) == (2, 2)


async def test_steps_deleted_accumulate_across_attempts(
    storage: Storage, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p, job_id = await _confirmed(storage)
    arm(monkeypatch, tmp_path, "purge.before_step4=raise:infra:1")
    assert await p.run_to_end(job_id) == "succeeded"
    view = await p.service.get(job_id)
    deleted = {s.store: s.deleted for s in view.steps}
    assert (deleted["markdown"], deleted["lancedb"], deleted["sqlite"]) == (2, 2, 2)


async def test_rerun_after_a_kill_mid_step2_succeeds(storage: Storage) -> None:
    """An attempt killed after moving S's files out (step 2b) left them in the
    job's trash; the rerun restores today's counters into place first, and
    must then still be able to move the scope out."""
    p, job_id = await _confirmed(storage)
    scope = p.service.scope_of("project", "a1", "p1")
    await p.service.markdown.move_out(scope, job_id)  # the killed attempt
    assert await p.run_to_end(job_id) == "succeeded"
    trash = storage.root.tmp_dir / f"purge-{job_id}"
    assert not trash.exists()
    s_left = (await p.service.request("project", "a1", "p1")).summary
    assert s_left.markdown_total == 0
