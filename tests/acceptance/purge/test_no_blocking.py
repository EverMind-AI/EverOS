"""T7.1: while a purge runs every route answers as it does without one (A7).

Nothing is refused: reads, writes outside the purged range and writes with
no range answer the same status code as before the purge; writes to the
purged range itself are accepted too (spec v3 C14, D3).
"""

from __future__ import annotations

import re
import time
from typing import Any

import pytest

from tests.acceptance.memory_delete.conftest import chat

from .journey import finish, purge
from .observe import release
from .seed import scope_token

pytestmark = pytest.mark.acceptance

S = {"app_id": "a1", "project_id": "p1"}
OTHER = {"app_id": "a2", "project_id": "p1"}
_READ_BODIES: dict[str, dict[str, Any]] = {
    "/memory/search": {"query": "museum", "user_id": "u1", "method": "keyword"},
    "/memory/get": {"user_id": "u1", "memory_type": "episode"},
    "/knowledge/search": {"query": "hiking"},
}
"""Valid bodies for the reads sent as POST: a read must run, not just pass."""
_MISSING = "zz_missing_id"
"""Path ids for writes: a write must not change the seed between the two runs."""


def _first(data: Any, key: str) -> str:
    """The first value of ``key`` anywhere in a JSON document."""
    if isinstance(data, dict):
        if isinstance(data.get(key), str):
            return data[key]
        values = data.values()
    elif isinstance(data, list):
        values = data
    else:
        values = []
    for value in values:
        if found := _first(value, key):
            return found
    return ""


def _read_ids(srv: Any, where: dict[str, str]) -> dict[str, str]:
    """A document, topic and task of the seed in ``where``."""
    docs = srv.call("GET", "/knowledge/documents", params=where).json()["data"]
    doc_id = docs["documents"][0]["doc_id"]
    detail = srv.call("GET", f"/knowledge/documents/{doc_id}", params=where).json()
    tasks = srv.call("GET", "/tasks").json()
    ids = {
        "doc_id": doc_id,
        "topic_id": _first(detail, "topic_id"),
        "task_id": tasks["data"]["items"][0]["id"],
    }
    assert all(ids.values()), ids
    return ids


def _routes(srv: Any) -> list[tuple[str, str]]:
    spec = srv.call("GET", "/openapi.json", version="").json()
    return [
        (method.upper(), path) for path, ops in spec["paths"].items() for method in ops
    ] + [("GET", "/docs"), ("GET", "/redoc"), ("GET", "/openapi.json")]


def _sweep(
    srv: Any,
    routes: list[tuple[str, str]],
    where: dict[str, str],
    ids: dict[str, str],
) -> dict[tuple[str, str], int]:
    """Status of one request per route against ``where`` (purge routes skipped)."""
    out: dict[tuple[str, str], int] = {}
    for method, path in routes:
        bare = re.sub(r"^/api/v[12]", "", path)
        if bare.startswith("/memory/purge") or bare == "/cascade/quiesce":
            continue  # purge routes: below; quiesce stops the cascade for good
        is_read = method == "GET" or bare in _READ_BODIES
        concrete = path
        for name, value in ids.items():
            concrete = concrete.replace(
                "{" + name + "}", value if is_read else _MISSING
            )
        body = {**_READ_BODIES[bare], **where} if bare in _READ_BODIES else {}
        params = where if method == "GET" and "/knowledge/" in bare else None
        resp = srv.call(method, concrete, body or None, version="", params=params)
        out[(method, path)] = resp.status_code
    return out


def _add(srv: Any, where: dict[str, str], session: str) -> int:
    token = scope_token(where["app_id"], where["project_id"])
    return srv.post(
        "/memory/add",
        {"session_id": session, "messages": chat("u1", token), **where},
    ).status_code


def test_t7_1_routes_answer_as_without_a_purge(make_world: Any) -> None:
    world = make_world(
        "T7.1", ENV="DEV", EVEROS_TEST_FAILPOINTS="purge.before_step8=pause"
    )
    srv = world.srv
    routes = _routes(srv)
    o_ids, s_ids = _read_ids(srv, OTHER), _read_ids(srv, S)
    baseline = {"other": _sweep(srv, routes, OTHER, o_ids)}
    job_id, _ = purge(srv, scope="project", **S)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:  # the first pass done, held before step 8
        job = srv.job(job_id).json()["data"]
        if all(s["state"] == "done" for s in job["steps"]):
            break
        time.sleep(0.2)
    assert job["status"] == "running", job
    during = {
        "other": _sweep(srv, routes, OTHER, o_ids),
        "s": _sweep(srv, routes, S, s_ids),
    }
    # Confirming another range answers the running job (D25), not 409.
    nonce = srv.purge(scope="app", app_id="a2").json()["data"]["nonce"]
    confirm = srv.call("POST", "/memory/purge/confirm", {"nonce": nonce})
    adds = {"other": _add(srv, OTHER, "during-o"), "s": _add(srv, S, "during-s")}
    assert srv.job(job_id).json()["data"]["status"] == "running"  # window open
    assert confirm.status_code == 202 and confirm.json()["data"]["job_id"] == job_id
    assert adds == {"other": 200, "s": 200}
    release(srv.fp_dir, "purge.before_step8")
    assert finish(srv, job_id)["status"] == "succeeded"
    # (b): the purged project answers as the cleared project does.
    cleared = _sweep(srv, routes, S, s_ids)

    def show(codes: dict[tuple[str, str], int]) -> dict[str, int]:
        return {f"{m} {p}": c for (m, p), c in codes.items()}

    world.ev.json(
        "routes",
        {
            "baseline_other": show(baseline["other"]),
            "during_other": show(during["other"]),
            "during_s": show(during["s"]),
            "cleared_s": show(cleared),
            "confirm_other": confirm.status_code,
            "adds": adds,
        },
    )
    assert during["other"] == baseline["other"]
    assert during["s"] == cleared
    for codes in during.values():
        assert 409 not in codes.values()
        assert all(c < 500 for c in codes.values()), codes
    assert {m for m, _ in routes} >= {"POST", "PUT", "PATCH", "DELETE"}
