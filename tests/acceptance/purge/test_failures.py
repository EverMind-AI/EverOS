"""T18.1, T19.1, T25.1, T29.1, T30.1: failures, budget, finalize, self-healing."""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.acceptance.memory_delete.conftest import chat

from .journey import finish, purge
from .observe import rows, system_db
from .seed import scope_token, talk

pytestmark = pytest.mark.acceptance

S = {"scope": "project", "app_id": "a1", "project_id": "p1"}


def _one_session(srv: Any) -> None:
    talk(srv, "u1", "s1", scope_token("a1", "p1"), {"app_id": "a1", "project_id": "p1"})


def _write(srv: Any, app: str = "a2", project: str = "p1") -> int:
    return srv.post(
        "/memory/add",
        {
            "session_id": f"w-{time.monotonic_ns()}",
            "app_id": app,
            "project_id": project,
            "messages": chat("u1", scope_token(app, project)),
            "defer_extraction": True,
        },
    ).status_code


def _job_row(root: Any) -> tuple[Any, ...]:
    return rows(system_db(root), "SELECT attempts, status, retries FROM purge_job")[0]


@pytest.mark.parametrize(
    ("spec", "status", "attempts"),
    [
        ("purge.step3.delete=raise:infra:1", "succeeded", 2),
        ("purge.step3.delete=raise:domain:1", "failed", 1),
        ("purge.step3.delete=raise:busy:2", "succeeded", 1),
    ],
    ids=["retryable", "not-retryable", "busy"],
)
def test_t18_1_delete_failures(
    make_world: Any, spec: str, status: str, attempts: int
) -> None:
    world = make_world(
        f"T18.1-{spec.rsplit(':', 2)[-2]}", seed=False, EVEROS_TEST_FAILPOINTS=spec
    )
    srv = world.srv
    _one_session(srv)
    job_id, _ = purge(srv, **S)
    job = finish(srv, job_id, timeout=120)
    world.ev.json("job", job)
    row = _job_row(srv.root)
    world.ev.json(
        "purge_job_row", dict(zip(("attempts", "status", "retries"), row, strict=True))
    )
    assert job["status"] == status, job
    assert row[:2] == (attempts, status)
    if status == "failed":
        index_step = next(s for s in job["steps"] if s["store"] == "lancedb")
        assert index_step["state"] == "failed" and index_step["error"], job
        assert _write(srv) == 200  # the mark is down


def test_t19_1_budget_ends_a_stuck_purge(make_world: Any) -> None:
    world = make_world(
        "T19.1",
        seed=False,
        EVEROS_PURGE__RUN_BUDGET_SECONDS="5",
        EVEROS_TEST_FAILPOINTS="purge.before_step3=pause",
    )
    srv = world.srv
    _one_session(srv)
    job_id, _ = purge(srv, **S)
    confirmed = time.monotonic()
    timeline = []
    while time.monotonic() - confirmed < 30:
        code = _write(srv)
        timeline.append((round(time.monotonic() - confirmed, 2), code))
        if code == 200:
            break
        time.sleep(0.5)
    world.ev.json("write_timeline", timeline)
    held_for = timeline[-1][0]
    assert timeline[-1][1] == 200 and timeline[0][1] == 409
    assert held_for <= 5 + 2 + 1, held_for  # budget + finalize + one poll
    job = finish(srv, job_id)
    assert job["status"] == "failed", job


def test_t25_1_mark_stays_until_the_terminal_state_is_written(
    make_world: Any,
) -> None:
    world = make_world(
        "T25.1",
        seed=False,
        EVEROS_TEST_FAILPOINTS="purge.finalize=raise:sqlite_locked:2",
    )
    srv, root = world.srv, world.srv.root
    _one_session(srv)
    job_id, _ = purge(srv, **S)
    timeline = []
    started = time.monotonic()
    while time.monotonic() - started < 60:
        data = srv.job(job_id).json()["data"]
        code = _write(srv)
        timeline.append(
            {
                "t": round(time.monotonic() - started, 2),
                "status": data["status"],
                "steps_done": all(s["state"] == "done" for s in data["steps"]),
                "write": code,
            }
        )
        if code == 200:
            break
        time.sleep(0.5)
    world.ev.json("timeline", timeline)
    for point in timeline:
        if point["write"] == 409:
            assert point["status"] in ("queued", "running"), point
    assert any(p["steps_done"] and p["write"] == 409 for p in timeline), timeline
    assert finish(srv, job_id)["status"] == "succeeded"
    talk(
        srv,
        "u1",
        "after",
        scope_token("a1", "p1"),
        {"app_id": "a1", "project_id": "p1"},
    )
    time.sleep(10)
    memcells = rows(
        system_db(root),
        "SELECT COUNT(*) FROM memcell WHERE app_id='a1' AND project_id='p1'",
    )[0][0]
    assert memcells == 1  # the write after the lift was not purged again


def test_t29_1_runner_heals_itself(make_world: Any) -> None:
    world = make_world(
        "T29.1-heal",
        seed=False,
        EVEROS_TEST_FAILPOINTS="purge.runner_loop=raise:runtime:1",
    )
    srv = world.srv
    _one_session(srv)
    job_id, _ = purge(srv, **S)
    assert finish(srv, job_id, timeout=30)["status"] == "succeeded"


def test_t29_1_lost_wake_is_caught_by_the_poll(make_world: Any) -> None:
    world = make_world(
        "T29.1-poll", seed=False, EVEROS_TEST_FAILPOINTS="purge.confirm_wake=skip"
    )
    srv = world.srv
    _one_session(srv)
    job_id, _ = purge(srv, **S)
    job = finish(srv, job_id, timeout=30)
    world.ev.json("job", job)
    assert job["status"] == "succeeded"
    from everos.core.utils.datetime import from_iso_format

    waited = from_iso_format(job["started_at"]) - from_iso_format(job["created_at"])
    assert waited.total_seconds() <= 10, waited


def test_t30_1_a_slow_count_reads_null(make_world: Any) -> None:
    world = make_world(
        "T30.1-count",
        seed=False,
        EVEROS_PURGE__COUNT_TIMEOUT_SECONDS="1",
        EVEROS_TEST_FAILPOINTS="purge.count.index=pause",
    )
    srv = world.srv
    _one_session(srv)
    resp = srv.purge(**S)
    assert resp.status_code == 200
    summary = resp.json()["data"]["summary"]
    world.ev.json("summary", summary)
    assert summary["lancedb_total"] is None and summary["complete"] is False
    assert summary["markdown_total"] > 0 and summary["sqlite_total"] > 0


def test_t30_1_failed_job_write_leaves_nothing(make_world: Any) -> None:
    world = make_world(
        "T30.1-insert",
        seed=False,
        EVEROS_TEST_FAILPOINTS="purge.confirm_insert=raise:sqlite_locked:1",
    )
    srv = world.srv
    _one_session(srv)
    nonce = srv.purge(**S).json()["data"]["nonce"]
    first = srv.confirm(nonce)
    assert first.status_code == 503, first.text
    assert _write(srv, "a1", "p1") == 200  # no mark was raised
    second = srv.confirm(nonce)
    assert second.status_code == 202, second.text
    world.ev.json("responses", {"first": first.json(), "second": second.json()})
    finish(srv, second.json()["data"]["job_id"])
