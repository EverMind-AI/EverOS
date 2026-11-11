"""T28.2: an async add of S killed mid-delivery is dropped when replayed."""

from __future__ import annotations

from typing import Any

import pytest

from tests.acceptance.memory_delete.conftest import chat
from tests.acceptance.memory_delete.proxy import Rule

from .journey import finish, purge
from .observe import md_counts, rows, system_db, wait_idle
from .seed import scope_token

pytestmark = pytest.mark.acceptance


def test_t28_2_row_killed_mid_delivery_is_dropped_on_replay(
    make_world: Any, purge_proxy: Any
) -> None:
    world = make_world("T28.2", seed=False, EVEROS_PURGE__SETTLE_WAIT_SECONDS="5")
    srv, root = world.srv, world.srv.root
    word = f"{scope_token('a1', 'p1')}replay"
    purge_proxy.add(Rule(target="llm", pattern=word, mode="hold"))
    resp = srv.post(
        "/memory/add",
        {
            "session_id": "replay",
            "app_id": "a1",
            "project_id": "p1",
            "messages": chat("u1", word),
            "async_mode": True,
        },
    )
    request_id = resp.json()["request_id"]
    assert purge_proxy.wait_held("llm", timeout=60)
    status = rows(
        system_db(root),
        "SELECT status FROM memorize_inbox WHERE request_id=?",
        request_id,
    )
    assert status == [("processing",)]
    job_id, _ = purge(srv, scope="project", app_id="a1", project_id="p1")
    assert finish(srv, job_id, timeout=60)["status"] == "succeeded"
    srv.kill()
    purge_proxy.clear()
    srv.start()
    wait_idle(srv.base, root, timeout=120)
    task = srv.call("GET", f"/tasks/{request_id}").json()["data"]
    row = rows(
        system_db(root),
        "SELECT status, payload_json FROM memorize_inbox WHERE request_id=?",
        request_id,
    )
    world.ev.json("task", task)
    assert (task["status"], task["error_code"]) == ("failed", "purge_in_progress")
    assert row == [("failed", "")]
    assert md_counts(root, "a1", "p1")["md.episode"] == 0
