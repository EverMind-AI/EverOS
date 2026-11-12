"""T31.1-T34.1, T38.1, T40.1, T42.1: the purge waits for old work in flight,
drops what that work derives, and holds nothing else back (spec v3 §5.6)."""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
import pytest

from everos.core.utils.datetime import from_iso_format
from tests.acceptance.memory_delete.conftest import chat
from tests.acceptance.memory_delete.proxy import Rule

from .journey import finish, purge
from .observe import md_counts, release, s_leftovers, wait_idle
from .seed import scope_token, talk
from .test_ome import _discarded, _runs

pytestmark = pytest.mark.acceptance

S = {"scope": "project", "app_id": "a1", "project_id": "p1"}
WHERE = {"app_id": "a1", "project_id": "p1"}
OTHER = {"app_id": "a2", "project_id": "p1"}
TOKEN = scope_token("a1", "p1")


def _hold(proxy: Any, reply: str, word: str = TOKEN) -> Rule:
    return proxy.add(
        Rule(target="llm", pattern=f'(?=.*"{reply}")(?=.*{word})', mode="hold")
    )


def _first_pass_done(srv: Any, job_id: str, timeout: float = 60) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = srv.job(job_id).json()["data"]
        if all(s["state"] == "done" for s in data["steps"]):
            return data
        time.sleep(0.2)
    raise TimeoutError("the first pass never ended")


def _cleared(world: Any, job: dict[str, Any]) -> None:
    srv = world.srv
    wait_idle(srv.base, srv.root, timeout=300)
    left = s_leftovers(srv.root, "a1", "p1", before=from_iso_format(job["created_at"]))
    world.ev.json("s_left", left)
    assert left == {}, left


def test_t31_1_an_old_run_busy_past_a_minute_is_waited_for(
    make_world: Any, purge_proxy: Any
) -> None:
    """A31: 60 s are no limit any more; the job reads running, its heartbeat
    moving, until the run ends -- then the second pass clears what it wrote."""
    world = make_world("T31.1")
    srv = world.srv
    _hold(purge_proxy, "_SkillExtractReply", word="")
    talk(srv, "u1", "slow-skill", TOKEN, WHERE)
    assert purge_proxy.wait_held("llm", timeout=120)
    job_id, _ = purge(srv, **S)
    time.sleep(70)
    first = srv.job(job_id).json()["data"]
    time.sleep(6)
    second = srv.job(job_id).json()["data"]
    world.ev.json("after_70s", {"first": first, "second": second})
    assert first["status"] == second["status"] == "running"
    assert second["heartbeat_at"] > first["heartbeat_at"]
    purge_proxy.clear()
    job = finish(srv, job_id, timeout=180)
    assert job["status"] == "succeeded", job
    _cleared(world, job)


def _case_in_flight(world: Any, proxy: Any, *, hold_skill: bool) -> tuple[Any, Any]:
    """S's agent-case extraction held mid-call (a run on old input); the
    skill extraction it would lead to optionally held too."""
    srv = world.srv
    case = _hold(proxy, "_CaseCompressReply")
    skill = _hold(proxy, "_SkillExtractReply", word="") if hold_skill else None
    talk(srv, "u1", "case-in-flight", TOKEN, WHERE)
    assert proxy.wait_held("llm", timeout=120)
    return case, skill


def test_t32_1_what_an_old_run_derives_is_dropped(
    make_world: Any, purge_proxy: Any
) -> None:
    """A32: the case extraction ends during the wait; the clustering it
    triggers inherits its old time and is dropped before it runs, so the
    skill extraction downstream is never even called."""
    world = make_world("T32.1")
    srv = world.srv
    case, skill = _case_in_flight(world, purge_proxy, hold_skill=True)
    job_id, _ = purge(srv, **S)
    at = from_iso_format(srv.job(job_id).json()["data"]["created_at"])
    _first_pass_done(srv, job_id)
    case.release.set()
    job = finish(srv, job_id, timeout=180)
    assert job["status"] == "succeeded", job
    calls = skill.hits  # type: ignore[union-attr]
    purge_proxy.clear()
    clustering = [
        r
        for r in _runs(srv.root)
        if r["strategy_name"] == "trigger_skill_clustering"
        and from_iso_format(r["started_at"]) > at
    ]
    world.ev.json("after_t", {"skill_calls": calls, "clustering": clustering})
    assert clustering and {r["run_id"] for r in clustering} <= _discarded(srv)
    assert calls == 0
    _cleared(world, job)


def test_t33_1_an_old_run_ending_in_the_wait_leaves_no_content(
    make_world: Any, purge_proxy: Any
) -> None:
    """A33: nothing of S's events stays in ome.db (no dead letter left by a
    derived run that found its cluster gone)."""
    world = make_world("T33.1")
    srv = world.srv
    case, _ = _case_in_flight(world, purge_proxy, hold_skill=False)
    job_id, _ = purge(srv, **S)
    _first_pass_done(srv, job_id)
    case.release.set()
    job = finish(srv, job_id, timeout=180)
    assert job["status"] == "succeeded", job
    purge_proxy.clear()
    _cleared(world, job)


def test_t34_1_a_knowledge_upload_in_flight_is_waited_for(
    make_world: Any, purge_proxy: Any
) -> None:
    """A34 (deviations D10): an upload is a request like any other."""
    world = make_world("T34.1")
    srv, root = world.srv, world.srv.root
    word = f"{TOKEN}upload"
    hold = purge_proxy.add(Rule(target="llm", pattern=word, mode="hold"))
    sent: dict[str, httpx.Response] = {}

    def upload() -> None:
        sent["r"] = srv.call(
            "POST",
            "/knowledge/documents",
            data={"title": f"Late {word}", **WHERE},
            files={"file": ("late.txt", f"Rope guide {word}.".encode(), "text/plain")},
        )

    client = threading.Thread(target=upload)
    client.start()
    assert purge_proxy.wait_held("llm", timeout=60)
    job_id, _ = purge(srv, **S)
    _first_pass_done(srv, job_id)
    time.sleep(2)
    waited = srv.job(job_id).json()["data"]["status"]
    hold.release.set()
    client.join(timeout=120)
    job = finish(srv, job_id, timeout=180)
    world.ev.json(
        "upload", {"status": sent["r"].status_code, "while_held": waited, "job": job}
    )
    assert waited == "running", waited
    assert job["status"] == "succeeded", job
    _cleared(world, job)
    assert md_counts(root, "a1", "p1")["md.knowledge_document"] == 0


def test_t38_1_reads_answer_between_the_move_and_the_index(make_world: Any) -> None:
    """A38: index rows still point at markdown the purge moved away."""
    world = make_world("T38.1", EVEROS_TEST_FAILPOINTS="purge.before_step3=pause")
    srv = world.srv
    job_id, _ = purge(srv, **S)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        steps = {
            s["store"]: s["state"] for s in srv.job(job_id).json()["data"]["steps"]
        }
        if steps["markdown"] == "done":
            break
        time.sleep(0.2)
    answers = {
        method: srv.post(
            "/memory/search",
            {"query": "museum", "user_id": "u1", "method": method, **WHERE},
        ).status_code
        for method in ("keyword", "vector", "hybrid")
    }
    answers["get"] = srv.post(
        "/memory/get", {"user_id": "u1", "memory_type": "episode", **WHERE}
    ).status_code
    world.ev.json("answers", answers)
    release(srv.fp_dir, "purge.before_step3")
    assert finish(srv, job_id)["status"] == "succeeded"
    assert set(answers.values()) == {200}, answers


def test_t40_1_new_input_after_a_purge_is_not_dropped(
    make_world: Any,
) -> None:
    """A40: what is written after the purge -- and what a tick derives from
    it -- runs; none of it is dropped as stale."""
    world = make_world("T40.1")
    srv, root = world.srv, world.srv.root
    job_id, _ = purge(srv, **S)
    job = finish(srv, job_id)
    assert job["status"] == "succeeded", job
    wait_idle(srv.base, root, timeout=300)
    at = from_iso_format(job["created_at"])
    for i in range(2):
        talk(srv, "u1", f"after-{i}", TOKEN, WHERE)
    trig = srv.post(
        "/ome/trigger",
        {"name": "reflect_episodes", "force": True, "timeout": 120},
        timeout=150,
    )
    assert trig.status_code == 200, trig.text
    wait_idle(srv.base, root, timeout=300)
    after = [r for r in _runs(root) if from_iso_format(r["started_at"]) > at]
    dropped = {r["run_id"] for r in after} & _discarded(srv)
    counts = md_counts(root, "a1", "p1")
    world.ev.json("after", {"counts": counts, "dropped": sorted(dropped)})
    assert counts["md.episode"] > 0 and counts["md.atomic_fact"] > 0
    assert dropped == set()


def test_t42_1_another_projects_backlog_holds_no_purge_back(
    make_world: Any, purge_proxy: Any
) -> None:
    """A42: queued deliveries of another project, older than the purge,
    are not waited for -- only work that can write the purged range is."""
    world = make_world("T42.1", seed=False)
    srv = world.srv
    talk(srv, "u1", "s-seed", TOKEN, WHERE)
    wait_idle(srv.base, srv.root, timeout=300)
    other = scope_token("a2", "p1")
    purge_proxy.add(Rule(target="llm", pattern=f"{other}hold", mode="hold"))
    ids = []
    for i in range(3):
        resp = srv.post(
            "/memory/add",
            {
                "session_id": f"bulk-{i}",
                "messages": chat("u1", f"{other}hold"),
                "async_mode": True,
                **OTHER,
            },
        )
        assert resp.status_code == 202, resp.text
        ids.append(resp.json()["request_id"])
    assert purge_proxy.wait_held("llm", timeout=60)  # a delivery is in flight
    job_id, _ = purge(srv, **S)
    started = time.monotonic()
    job = finish(srv, job_id, timeout=60)
    took = time.monotonic() - started
    world.ev.json("job", {**job, "seconds": took})
    assert job["status"] == "succeeded", job
    assert took < 30, took
    purge_proxy.clear()
