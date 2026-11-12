"""T18.1, T19.1-T19.3, T25.1, T29.1, T30.1: failures, budget, finalize, healing."""

from __future__ import annotations

import time
from typing import Any

import pytest

from everos.core.utils.datetime import from_iso_format
from tests.acceptance.memory_delete.conftest import chat
from tests.acceptance.memory_delete.proxy import Rule

from .journey import finish, purge
from .observe import rows, s_leftovers, system_db, wait_idle
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
        assert _write(srv) == 200


def _writes_until_terminal(srv: Any, job_id: str, limit: float) -> list[Any]:
    """Write outside S every 0.5 s while the job runs; ``(t, code, status)``."""
    timeline = []
    started = time.monotonic()
    while time.monotonic() - started < limit:
        status = srv.job(job_id).json()["data"]["status"]
        timeline.append((round(time.monotonic() - started, 2), _write(srv), status))
        if status in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    return timeline


def _assert_budget_failure(world: Any, job_id: str, timeline: list[Any]) -> Any:
    srv = world.srv
    world.ev.json("write_timeline", timeline)
    job = finish(srv, job_id)
    world.ev.json("job", job)
    assert job["status"] == "failed", job
    row = rows(system_db(srv.root), "SELECT last_error FROM purge_job")[0][0]
    assert "budget" in (row or ""), row
    assert {code for _, code, _ in timeline} == {200}, timeline  # never blocked
    return job


def test_t19_1_budget_ends_a_stuck_step(make_world: Any) -> None:
    world = make_world(
        "T19.1",
        seed=False,
        EVEROS_PURGE__RUN_BUDGET_SECONDS="5",
        EVEROS_TEST_FAILPOINTS="purge.before_step3=pause",
    )
    srv = world.srv
    _one_session(srv)
    job_id, _ = purge(srv, **S)
    job = _assert_budget_failure(world, job_id, _writes_until_terminal(srv, job_id, 30))
    index_step = next(s for s in job["steps"] if s["store"] == "lancedb")
    assert index_step["state"] == "failed" and "budget" in index_step["error"], job


def _held_s_run(world: Any, proxy: Any) -> None:
    """A run of S on input taken in before the purge, that never ends."""
    token = scope_token("a1", "p1")
    proxy.add(
        Rule(
            target="llm", pattern=f'(?=.*"_AtomicFactsReply")(?=.*{token})', mode="hold"
        )
    )
    _one_session(world.srv)
    assert proxy.wait_held("llm", timeout=60)


def test_t19_2_budget_ends_a_wait_that_never_ends(
    make_world: Any, purge_proxy: Any
) -> None:
    world = make_world("T19.2", seed=False, EVEROS_PURGE__RUN_BUDGET_SECONDS="10")
    srv = world.srv
    _held_s_run(world, purge_proxy)
    job_id, _ = purge(srv, **S)
    job = _assert_budget_failure(world, job_id, _writes_until_terminal(srv, job_id, 60))
    inbox = next(s for s in job["steps"] if s["store"] == "inbox")
    assert inbox["state"] == "failed" and "budget" in inbox["error"], job


def test_t19_3_purging_again_after_a_restart_succeeds(
    make_world: Any, purge_proxy: Any
) -> None:
    """The documented way out of T19.2: restart the server, purge again."""
    world = make_world("T19.3", seed=False, EVEROS_PURGE__RUN_BUDGET_SECONDS="10")
    srv, root = world.srv, world.srv.root
    _held_s_run(world, purge_proxy)
    job_id, _ = purge(srv, **S)
    assert finish(srv, job_id, timeout=60)["status"] == "failed"
    srv.stop()
    purge_proxy.clear()
    srv.start()
    again, _ = purge(srv, **S)
    job = finish(srv, again, timeout=120)
    world.ev.json("second_job", job)
    assert job["status"] == "succeeded", job
    wait_idle(srv.base, root, timeout=300)
    left = s_leftovers(root, "a1", "p1", before=from_iso_format(job["created_at"]))
    world.ev.json("s_left", left)
    assert left == {}, left


def test_t25_1_job_stays_running_until_the_terminal_state_is_written(
    make_world: Any,
) -> None:
    world = make_world(
        "T25.1",
        seed=False,
        EVEROS_TEST_FAILPOINTS="purge.finalize=raise:sqlite_locked:2",
    )
    srv = world.srv
    _one_session(srv)
    job_id, _ = purge(srv, **S)
    timeline = []
    started = time.monotonic()
    while time.monotonic() - started < 60:
        data = srv.job(job_id).json()["data"]
        timeline.append(
            {
                "t": round(time.monotonic() - started, 2),
                "status": data["status"],
                "steps_done": all(s["state"] == "done" for s in data["steps"]),
                "write": _write(srv),
            }
        )
        if data["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.5)
    world.ev.json("timeline", timeline)
    assert {p["write"] for p in timeline} == {200}, timeline  # never blocked
    # The finalize retries took a few seconds with every step done: the job
    # read running all along, then succeeded.
    assert any(p["steps_done"] and p["status"] == "running" for p in timeline)
    assert finish(srv, job_id)["status"] == "succeeded"


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
    assert _write(srv, "a1", "p1") == 200  # nothing started
    second = srv.confirm(nonce)
    assert second.status_code == 202, second.text
    world.ev.json("responses", {"first": first.json(), "second": second.json()})
    finish(srv, second.json()["data"]["job_id"])
