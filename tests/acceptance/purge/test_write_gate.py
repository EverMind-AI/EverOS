"""T7.1: while a purge runs every write route answers 409, every read passes."""

from __future__ import annotations

import re
import time
from typing import Any

import pytest

from everos.entrypoints.api.middleware.purge_gate import READ_ROUTES, WRITE_ROUTES
from tests.acceptance.memory_delete.conftest import chat

from .journey import finish, purge
from .observe import release
from .seed import scope_token

pytestmark = pytest.mark.acceptance

_IDS = {"doc_id": "d_abcdef123456", "topic_id": "d_abcdef123456_1", "task_id": "t1"}


def _job_running_before_step3(srv: Any, job_id: str) -> None:
    job = srv.job(job_id).json()["data"]
    steps = {s["store"]: s["state"] for s in job["steps"]}
    assert job["status"] == "running" and steps["lancedb"] == "pending", job


def test_t7_1_writes_409_reads_pass(make_world: Any) -> None:
    world = make_world(
        "T7.1", ENV="DEV", EVEROS_TEST_FAILPOINTS="purge.before_step3=pause"
    )
    srv = world.srv
    job_id, _ = purge(srv, scope="project", app_id="a1", project_id="p1")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:  # step 2 done, step 3 held
        steps = {
            s["store"]: s["state"] for s in srv.job(job_id).json()["data"]["steps"]
        }
        if steps["markdown"] == "done":
            break
        time.sleep(0.2)
    _job_running_before_step3(srv, job_id)
    spec = srv.call("GET", "/openapi.json", version="").json()
    routes = [
        (method.upper(), path) for path, ops in spec["paths"].items() for method in ops
    ] + [("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json")]
    outcomes: list[dict[str, Any]] = []
    for method, path in routes:
        bare = re.sub(r"^/api/v[12]", "", path)
        is_read = (method, bare) in READ_ROUTES
        assert is_read or (method, bare) in WRITE_ROUTES, (
            f"unclassified {method} {path}"
        )
        concrete = path.replace("{job_id}", job_id)
        for name, value in _IDS.items():
            concrete = concrete.replace("{" + name + "}", value)
        resp = srv.call(method, concrete, {}, version="")
        outcomes.append(
            {
                "method": method,
                "path": concrete,
                "read": is_read,
                "status": resp.status_code,
                "body": resp.text[:300],
            }
        )
        if is_read:
            assert resp.status_code != 409, (method, concrete, resp.text)
        else:
            assert resp.status_code == 409, (method, concrete, resp.text)
            error = resp.json()["error"]
            assert error["code"] == "PURGE_IN_PROGRESS" and job_id in error["message"]
    world.ev.json("routes", outcomes)
    _job_running_before_step3(srv, job_id)  # the window was open throughout
    assert {o["method"] for o in outcomes if not o["read"]} >= {
        "POST",
        "PUT",
        "PATCH",
        "DELETE",
    }
    release(srv.fp_dir, "purge.before_step3")
    assert finish(srv, job_id)["status"] == "succeeded"
    after = srv.post(
        "/memory/add",
        {
            "session_id": "after",
            "app_id": "a1",
            "project_id": "p1",
            "messages": chat("u1", scope_token("a1", "p1")),
        },
    )
    assert after.status_code == 200, after.text
