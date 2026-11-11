"""T12.1, T16.1-T16.3: a purge resumes after a kill; ids are never reused."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.acceptance.memory_delete.conftest import chat
from tests.acceptance.memory_delete.observe import md_entries

from .journey import finish, purge
from .observe import release, rows, scope_counts, scope_dump, system_db, wait_idle
from .seed import WORLD, scope_token, talk
from .test_ome import _backlog, _runs

pytestmark = pytest.mark.acceptance

S = {"scope": "project", "app_id": "a1", "project_id": "p1"}
WHERE = {"app_id": "a1", "project_id": "p1"}


def _zero(counts: dict[str, int]) -> dict[str, int]:
    return {k: n for k, n in counts.items() if n and k != "sqlite.md_change_state"}


def test_t12_1_killed_at_step3_resumes_on_restart(
    make_world: Any, purge_proxy: Any
) -> None:
    world = make_world(
        "T12.1",
        EVEROS_OME__MAX_CONCURRENT_RUNS="1",
        EVEROS_TEST_FAILPOINTS="purge.before_step3",
    )
    srv, root = world.srv, world.srv.root
    controls = [s for s in WORLD if s != ("a1", "p1")]
    dumps = {s: scope_dump(root, *s) for s in controls}
    e0 = _backlog(world, purge_proxy, "a1", "p1", 3)
    job_id, _ = purge(srv, **S)
    assert srv.wait_exit(timeout=60) == 137
    purge_proxy.clear()
    srv.env_extra["EVEROS_TEST_FAILPOINTS"] = "purge.before_step4=pause"
    srv.start()
    first_write = srv.post(
        "/memory/add",
        {
            "session_id": "x",
            "app_id": "a2",
            "project_id": "p1",
            "messages": chat("u1", scope_token("a2", "p1")),
        },
    )
    world.ev.json("first_write_after_restart", first_write.json())
    assert first_write.status_code == 409
    release(srv.fp_dir, "purge.before_step4")
    job = finish(srv, job_id, timeout=180)
    wait_idle(srv.base, root, timeout=300)
    world.ev.json("job", job)
    row = rows(system_db(root), "SELECT attempts, status FROM purge_job")
    assert row == [(2, "succeeded")]
    assert _zero(scope_counts(root, "a1", "p1")) == {}
    e0_rows = [r for r in _runs(root) if (r["event_id"], r["strategy_name"]) == e0]
    world.ev.json("e0_rows", e0_rows)
    statuses = {r["status"] for r in e0_rows}
    assert "crashed" in statuses and "success" in statuses, e0_rows
    assert {s: scope_dump(root, *s) for s in controls} == dumps
    assert not list((root / ".tmp").glob("purge-*"))


def _s_episode_ids(root: Path) -> set[str]:
    return {
        e["entry_id"]
        for e in md_entries(root)
        if e["path"].startswith("a1/p1/users/u1/") and e["kind"] == "episodes"
    }


def _talks(srv: Any, n: int, tag: str) -> None:
    for i in range(n):
        talk(srv, "u1", f"{tag}-{i}", scope_token("a1", "p1"), WHERE)


def _today_shell(root: Path) -> list[Path]:
    return sorted((root / "a1" / "p1" / "users" / "u1" / "episodes").glob("*.md"))


def test_t16_1_ids_are_not_reused_after_a_purge(make_world: Any) -> None:
    world = make_world("T16.1", seed=False)
    srv, root = world.srv, world.srv.root
    _talks(srv, 5, "before")
    old = _s_episode_ids(root)
    assert len(old) == 5
    job_id, _ = purge(srv, **S)
    assert finish(srv, job_id)["status"] == "succeeded"
    assert _s_episode_ids(root) == set() and _today_shell(root)  # shell kept
    _talks(srv, 3, "after")
    new = _s_episode_ids(root)
    world.ev.json("ids", {"old": sorted(old), "new": sorted(new)})
    assert len(new) == 3 and not new & old


def test_t16_2_killed_after_step2b_keeps_counters(make_world: Any) -> None:
    world = make_world("T16.2", seed=False, EVEROS_TEST_FAILPOINTS="purge.after_step2b")
    srv, root = world.srv, world.srv.root
    _talks(srv, 5, "before")
    old = _s_episode_ids(root)
    job_id, _ = purge(srv, **S)
    assert srv.wait_exit(timeout=60) == 137
    assert list((root / ".tmp").glob("purge-*"))  # the moved files sit there
    srv.env_extra.pop("EVEROS_TEST_FAILPOINTS")
    srv.start()
    assert finish(srv, job_id)["status"] == "succeeded"
    assert not list((root / ".tmp").glob("purge-*"))
    _talks(srv, 3, "after")
    new = _s_episode_ids(root)
    world.ev.json("ids", {"old": sorted(old), "new": sorted(new)})
    assert len(new) == 3 and not new & old


def test_t16_3_delete_stopped_mid_way_keeps_counters(make_world: Any) -> None:
    world = make_world(
        "T16.3", seed=False, EVEROS_TEST_FAILPOINTS="delete_owner.after_rename=pause"
    )
    srv, root = world.srv, world.srv.root
    _talks(srv, 5, "before")
    old = _s_episode_ids(root)
    deleted: dict[str, httpx.Response] = {}
    deleting = threading.Thread(
        target=lambda: deleted.update(
            r=srv.post("/memory/delete", {"user_id": "u1", **WHERE}, timeout=120)
        )
    )
    deleting.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not list((root / ".tmp").glob("delete-*")):
        time.sleep(0.1)
    assert list((root / ".tmp").glob("delete-*")) and deleting.is_alive()
    job_id, _ = purge(srv, **S)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:  # the purge got past its markdown step
        steps = {
            s["store"]: s["state"] for s in srv.job(job_id).json()["data"]["steps"]
        }
        if steps["md_change_state"] == "done":
            break
        time.sleep(0.2)
    release(srv.fp_dir, "delete_owner.after_rename")
    deleting.join(timeout=60)
    assert finish(srv, job_id)["status"] == "succeeded"
    world.ev.json(
        "delete_response",
        {"status": deleted["r"].status_code, "body": deleted["r"].text[:500]},
    )
    assert not list((root / ".tmp").glob("delete-*"))
    _talks(srv, 3, "after")
    new = _s_episode_ids(root)
    world.ev.json("ids", {"old": sorted(old), "new": sorted(new)})
    assert len(new) == 3 and not new & old
