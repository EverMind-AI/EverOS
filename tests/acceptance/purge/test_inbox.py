"""T8.1: the purged scope's async adds, queued before and sent during its purge."""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.acceptance.memory_delete.conftest import chat
from tests.acceptance.memory_delete.proxy import Rule

from .journey import finish, purge
from .observe import md_counts, release, rows, system_db
from .seed import scope_token

pytestmark = pytest.mark.acceptance


def _send(srv: Any, session: str, app: str, project: str, words: str) -> Any:
    return srv.post(
        "/memory/add",
        {
            "session_id": session,
            "app_id": app,
            "project_id": project,
            "messages": chat("u1", words),
            "async_mode": True,
        },
    )


def _async_add(srv: Any, session: str, app: str, project: str, words: str) -> str:
    resp = _send(srv, session, app, project, words)
    assert resp.status_code == 202, resp.text
    return resp.json()["request_id"]


def _task(srv: Any, request_id: str) -> dict[str, Any]:
    return srv.call("GET", f"/tasks/{request_id}").json()["data"]


def test_t8_1_queued_adds_split_by_when_they_came_in(
    make_world: Any, purge_proxy: Any
) -> None:
    """A8: the scope's adds queued before the purge fail as purge_in_progress;
    one in delivery is waited for (its data then purged with the rest); one
    sent while the purge runs is refused and never queued (D37); other
    scopes complete."""
    world = make_world("T8.1", EVEROS_TEST_FAILPOINTS="purge.before_step8=pause")
    srv = world.srv
    s_token, o_token = scope_token("a1", "p1"), scope_token("a2", "p1")
    hold = purge_proxy.add(Rule(target="llm", pattern=f"{s_token}hold", mode="hold"))
    held_id = _async_add(srv, "queued", "a1", "p1", f"{s_token}hold")
    assert purge_proxy.wait_held("llm", timeout=60)
    s_ids = [_async_add(srv, "queued", "a1", "p1", s_token) for _ in range(3)]
    o_ids = [_async_add(srv, f"other{i}", "a2", "p1", o_token) for i in range(2)]
    before_other = md_counts(srv.root, "a2", "p1")
    job_id, _ = purge(srv, scope="project", app_id="a1", project_id="p1")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:  # the first pass is over
        steps = srv.job(job_id).json()["data"]["steps"]
        if all(s["state"] == "done" for s in steps):
            break
        time.sleep(0.2)
    time.sleep(1)
    waited = srv.job(job_id).json()["data"]["status"]
    hold.release.set()  # the delivery in flight ends: the wait is over
    time.sleep(2)
    refused = _send(srv, "new", "a1", "p1", s_token)  # while it runs
    release(srv.fp_dir, "purge.before_step8")
    job = finish(srv, job_id, timeout=180)
    assert job["status"] == "succeeded", job
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if all(_task(srv, r)["status"] == "success" for r in o_ids):
            break
        time.sleep(0.5)
    held = srv.call("GET", f"/tasks/{held_id}")
    tasks = {r: _task(srv, r) for r in [*s_ids, *o_ids]}
    inbox = {
        r[0]: r[1:]
        for r in rows(
            system_db(srv.root),
            "SELECT request_id, status, payload_json FROM memorize_inbox",
        )
    }
    world.ev.json(
        "tasks",
        {
            "held": [held.status_code, held.text[:300]],
            "while_waiting": waited,
            "refused_while_running": [refused.status_code, refused.text[:300]],
            **tasks,
        },
    )
    world.ev.json("inbox", inbox)
    assert waited == "running"  # the purge waited for the held delivery
    # Delivered, then cleared with the scope: its history row goes too.
    assert held.status_code == 404, held.text
    for rid in s_ids:
        assert (tasks[rid]["status"], tasks[rid]["error_code"]) == (
            "failed",
            "purge_in_progress",
        ), tasks[rid]
        assert inbox[rid] == ("failed", "")
    assert refused.status_code == 409, refused.text
    assert refused.json()["error"]["code"] == "PURGE_IN_PROGRESS"
    assert refused.json()["request_id"] not in inbox  # never queued
    for rid in o_ids:
        assert tasks[rid]["status"] == "success", tasks[rid]
    after_other = md_counts(srv.root, "a2", "p1")
    assert after_other["md.episode"] > before_other["md.episode"]
