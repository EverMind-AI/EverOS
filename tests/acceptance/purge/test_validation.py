"""T5.1, T5.2, T6.1, T11.1: nonces, validation, ``default``, one active job."""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
import pytest

from .journey import finish, purge
from .observe import (
    nothing_deleted_snapshot,
    release,
    rows,
    scope_counts,
    scope_dump,
    system_db,
    wait_idle,
)
from .seed import WORLD

pytestmark = pytest.mark.acceptance

S = {"scope": "project", "app_id": "a1", "project_id": "p1"}
EMPTY = {"scope": "project", "app_id": "a9", "project_id": "p9"}


def _jobs(root: Any) -> list[tuple[Any, ...]]:
    return rows(system_db(root), "SELECT job_id, status, active FROM purge_job")


def test_t5_1_invalid_nonces_delete_nothing(make_world: Any) -> None:
    world = make_world("T5.1", EVEROS_PURGE__NONCE_TTL_SECONDS="2")
    srv, root = world.srv, world.srv.root
    before = nothing_deleted_snapshot(root)
    dumps = {s: scope_dump(root, *s) for s in WORLD}
    # 1: a random string
    one = srv.confirm("xxxxxxxxxxxxxxxxxxxxxxxx")
    # 2: S's nonce, expired
    stale = srv.purge(**S).json()["data"]["nonce"]
    time.sleep(3)
    two = srv.confirm(stale)
    # 4: S's fresh nonce with an extra field
    fresh = srv.purge(**S).json()["data"]["nonce"]
    four = srv.post("/memory/purge/confirm", {"nonce": fresh, "scope": "space"})
    world.ev.json(
        "responses",
        {
            k: (r.status_code, r.json())
            for k, r in (("1", one), ("2", two), ("4", four))
        },
    )
    for resp in (one, two):
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "PURGE_NONCE_INVALID"
    assert four.status_code == 422
    assert nothing_deleted_snapshot(root) == before
    assert {s: scope_dump(root, *s) for s in WORLD} == dumps
    assert _jobs(root) == []
    # 3: a used nonce (on an empty range, so the snapshots above stay valid)
    nonce = srv.purge(**EMPTY).json()["data"]["nonce"]
    first = srv.confirm(nonce)
    assert first.status_code == 202
    assert finish(srv, first.json()["data"]["job_id"])["status"] == "succeeded"
    three = srv.confirm(nonce)
    assert three.status_code == 404
    assert three.json()["error"]["code"] == "PURGE_NONCE_INVALID"
    assert len(_jobs(root)) == 1


def test_t5_2_one_nonce_confirmed_twice_at_once(make_world: Any) -> None:
    world = make_world(
        "T5.2", seed=False, EVEROS_TEST_FAILPOINTS="purge.confirm_insert=pause"
    )
    srv = world.srv
    nonce = srv.purge(**EMPTY).json()["data"]["nonce"]
    first: dict[str, httpx.Response] = {}
    worker = threading.Thread(target=lambda: first.update(r=srv.confirm(nonce)))
    worker.start()
    time.sleep(1.0)
    assert worker.is_alive() and _jobs(srv.root) == []  # held in the window
    second = srv.confirm(nonce)
    release(srv.fp_dir, "purge.confirm_insert")
    worker.join(timeout=30)
    world.ev.json(
        "responses",
        {"first": first["r"].json(), "second": (second.status_code, second.json())},
    )
    assert second.status_code == 404
    assert second.json()["error"]["code"] == "PURGE_NONCE_INVALID"
    assert first["r"].status_code == 202
    finish(srv, first["r"].json()["data"]["job_id"])
    assert len(_jobs(srv.root)) == 1


@pytest.mark.parametrize(
    "body",
    [
        {"scope": "space", "app_id": "a1"},
        {"scope": "project", "app_id": "a1"},
        {"scope": "app"},
        {"scope": "app", "app_id": "default_app"},
        {"scope": "project", "app_id": "a1", "project_id": "p1", "x": 1},
        {"scope": "tenant"},
    ],
)
def test_t6_1_invalid_requests_get_no_nonce(make_purge_server: Any, body: Any) -> None:
    srv = make_purge_server()
    resp = srv.post("/memory/purge", body)
    assert resp.status_code == 422, resp.text
    assert "nonce" not in resp.text


def test_t6_1_default_maps_to_reserved_dirs(make_world: Any) -> None:
    world = make_world("T6.1")
    srv, root = world.srv, world.srv.root
    controls = [s for s in WORLD if s != ("default", "default")]
    dumps = {s: scope_dump(root, *s) for s in controls}
    job_id, _ = purge(srv, scope="project", app_id="default", project_id="default")
    assert finish(srv, job_id)["status"] == "succeeded"
    wait_idle(srv.base, root, timeout=300)
    after = scope_counts(root, "default", "default")
    world.ev.json("after_default", after)
    assert {k: n for k, n in after.items() if n and k != "sqlite.md_change_state"} == {}
    stored_default = rows(
        system_db(root),
        "SELECT COUNT(*) FROM memcell WHERE app_id='default' AND project_id='default'",
    )
    assert stored_default == [(0,)]
    assert {s: scope_dump(root, *s) for s in controls} == dumps


def test_t11_1_second_scope_gets_the_active_job(make_world: Any) -> None:
    world = make_world("T11.1", EVEROS_TEST_FAILPOINTS="purge.before_step3=pause")
    srv, root = world.srv, world.srv.root
    other_before = scope_dump(root, "a2", "p1")
    job_a, _ = purge(srv, **S)
    second = srv.purge(scope="project", app_id="a2", project_id="p1")
    nonce_b = second.json()["data"]["nonce"]
    again = srv.confirm(nonce_b)
    assert again.status_code == 202
    assert again.json()["data"]["job_id"] == job_a
    shown = srv.job(job_a).json()["data"]
    assert (shown["scope"], shown["app_id"], shown["project_id"]) == (
        "project",
        "a1",
        "p1",
    )
    live = [r for r in _jobs(root) if r[1] in ("queued", "running")]
    assert len(live) == 1
    release(srv.fp_dir, "purge.before_step3")
    assert finish(srv, job_a)["status"] == "succeeded"
    wait_idle(srv.base, root, timeout=300)
    assert scope_dump(root, "a2", "p1") == other_before
    spent = srv.confirm(nonce_b)
    assert spent.status_code == 404
    world.ev.json("jobs", _jobs(root))
