"""T12.1, T12.2, T16.1-T16.3: a purge resumes after a kill; ids are never reused."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from everos.core.utils.datetime import from_iso_format
from tests.acceptance.memory_delete.observe import md_entries

from .journey import finish, purge
from .observe import (
    release,
    rows,
    s_leftovers,
    scope_dump,
    system_db,
    wait_idle,
)
from .seed import WORLD, scope_token, talk
from .test_ome import _hold_facts, _runs

pytestmark = pytest.mark.acceptance

S = {"scope": "project", "app_id": "a1", "project_id": "p1"}
WHERE = {"app_id": "a1", "project_id": "p1"}


def _hold_s_backlog(world: Any, proxy: Any) -> tuple[str, str]:
    """S's first fact extraction held in the only OME slot (``E0``, a RUNNING
    row), two more sessions of S queued behind it."""
    srv = world.srv
    token = scope_token("a1", "p1")
    _hold_facts(proxy, token)
    for i in range(3):
        talk(srv, "u1", f"backlog-{i}", token, WHERE)
    assert proxy.wait_held("llm", timeout=60)
    running = [r for r in _runs(srv.root) if r["status"] == "running"]
    assert len(running) == 1, running
    return running[0]["event_id"], running[0]["strategy_name"]


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
    e0 = _hold_s_backlog(world, purge_proxy)
    job_id, _ = purge(srv, **S)
    assert srv.wait_exit(timeout=60) == 137
    purge_proxy.clear()
    srv.env_extra.pop("EVEROS_TEST_FAILPOINTS")
    srv.start()
    job = finish(srv, job_id, timeout=180)
    wait_idle(srv.base, root, timeout=300)
    world.ev.json("job", job)
    row = rows(system_db(root), "SELECT attempts, status FROM purge_job")
    world.ev.json("purge_job_rows", row)
    assert row == [(2, "succeeded")]
    at = from_iso_format(job["created_at"])
    left = s_leftovers(root, "a1", "p1", before=at)
    world.ev.json("s_left", left)
    assert left == {}, left
    e0_rows = [r for r in _runs(root) if (r["event_id"], r["strategy_name"]) == e0]
    world.ev.json("e0_rows", e0_rows)
    statuses = {r["status"] for r in e0_rows}
    assert "crashed" in statuses and "success" in statuses, e0_rows
    assert {s: scope_dump(root, *s) for s in controls} == dumps
    assert not list((root / ".tmp").glob("purge-*"))


def test_t12_2_an_old_run_replayed_after_the_purge_stays_out(
    make_world: Any, purge_proxy: Any
) -> None:
    """A41: the tombstones are loaded before crash recovery replays a run."""
    world = make_world(
        "T12.2",
        EVEROS_OME__MAX_CONCURRENT_RUNS="1",
        EVEROS_PURGE__RUN_BUDGET_SECONDS="15",
    )
    srv, root = world.srv, world.srv.root
    e0 = _hold_s_backlog(world, purge_proxy)
    job_id, _ = purge(srv, **S)
    job = finish(srv, job_id, timeout=120)  # the held run never ends
    assert job["status"] == "failed", job
    srv.kill()
    purge_proxy.clear()
    srv.start()
    wait_idle(srv.base, root, timeout=300)
    e0_rows = [r for r in _runs(root) if (r["event_id"], r["strategy_name"]) == e0]
    world.ev.json("e0_rows", e0_rows)
    replayed = [r for r in e0_rows if r["status"] == "success"]
    assert replayed and all(r["event_payload"] == "" for r in replayed), e0_rows
    at = from_iso_format(job["created_at"])
    left = s_leftovers(root, "a1", "p1", before=at)
    world.ev.json("s_left_after_failed_purge", left)
    # The job failed, so its second pass never ran: the run's own crashed
    # row keeps the event until the range is purged again -- nothing else.
    assert left == {"ome.run_record_payload": [(e0[1], "crashed")]}, left
    again, _ = purge(srv, **S)
    second = finish(srv, again, timeout=120)
    assert second["status"] == "succeeded", second
    left = s_leftovers(root, "a1", "p1", before=from_iso_format(second["created_at"]))
    world.ev.json("s_left_after_second_purge", left)
    assert left == {}, left


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
    wait_idle(srv.base, root, timeout=300)
    after_purge = s_leftovers(root, "a1", "p1")
    world.ev.json("after_purge_left", after_purge)
    assert after_purge == {}
    world.ev.json(
        "delete_response",
        {"status": deleted["r"].status_code, "body": deleted["r"].text[:500]},
    )
    # A39 / D31: its trash went with the purge -- nothing left to delete.
    assert deleted["r"].status_code == 200, deleted["r"].text
    assert deleted["r"].json()["data"]["count"] == 0, deleted["r"].text
    assert not list((root / ".tmp").glob("delete-*"))
    _talks(srv, 3, "after")
    new = _s_episode_ids(root)
    world.ev.json("ids", {"old": sorted(old), "new": sorted(new)})
    assert len(new) == 3 and not new & old
