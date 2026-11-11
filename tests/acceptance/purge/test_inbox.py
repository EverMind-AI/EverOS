"""T8.1: the purged scope's queued async adds fail as purge_in_progress."""

from __future__ import annotations

import time
from typing import Any

import pytest

from tests.acceptance.memory_delete.conftest import chat
from tests.acceptance.memory_delete.proxy import Rule

from .journey import finish, purge
from .observe import md_counts, rows, system_db
from .seed import scope_token

pytestmark = pytest.mark.acceptance


def _async_add(srv: Any, session: str, app: str, project: str, words: str) -> str:
    resp = srv.post(
        "/memory/add",
        {
            "session_id": session,
            "app_id": app,
            "project_id": project,
            "messages": chat("u1", words),
            "async_mode": True,
        },
    )
    assert resp.status_code == 202, resp.text
    return resp.json()["request_id"]


def _task(srv: Any, request_id: str) -> dict[str, Any]:
    return srv.call("GET", f"/tasks/{request_id}").json()["data"]


def test_t8_1_queued_adds_of_the_scope_fail_others_complete(
    make_world: Any, purge_proxy: Any
) -> None:
    world = make_world("T8.1")
    srv = world.srv
    s_token, o_token = scope_token("a1", "p1"), scope_token("a2", "p1")
    hold = purge_proxy.add(Rule(target="llm", pattern=f"{s_token}hold", mode="hold"))
    held_id = _async_add(srv, "queued", "a1", "p1", f"{s_token}hold")
    assert purge_proxy.wait_held("llm", timeout=60)
    s_ids = [_async_add(srv, "queued", "a1", "p1", s_token) for _ in range(3)]
    o_ids = [_async_add(srv, f"other{i}", "a2", "p1", o_token) for i in range(2)]
    before_other = md_counts(srv.root, "a2", "p1")
    job_id, _ = purge(srv, scope="project", app_id="a1", project_id="p1")
    hold.release.set()
    job = finish(srv, job_id, timeout=180)
    assert job["status"] == "succeeded", job
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if all(_task(srv, r)["status"] == "success" for r in o_ids):
            break
        time.sleep(0.5)
    tasks = {r: _task(srv, r) for r in [held_id, *s_ids, *o_ids]}
    inbox = {
        r[0]: r[1:]
        for r in rows(
            system_db(srv.root),
            "SELECT request_id, status, payload_json FROM memorize_inbox",
        )
    }
    world.ev.json("tasks", tasks)
    world.ev.json("inbox", inbox)
    for rid in [held_id, *s_ids]:
        assert (tasks[rid]["status"], tasks[rid]["error_code"]) == (
            "failed",
            "purge_in_progress",
        ), tasks[rid]
        assert inbox[rid] == ("failed", "")
    for rid in o_ids:
        assert tasks[rid]["status"] == "success", tasks[rid]
    after_other = md_counts(srv.root, "a2", "p1")
    assert after_other["md.episode"] > before_other["md.episode"]
