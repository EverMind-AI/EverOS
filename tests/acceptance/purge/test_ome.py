"""T9.1, T10.1, T15.1, T24.1, T24.2: background (OME) work and purges."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from everos.core.utils.datetime import from_iso_format, get_utc_now
from tests.acceptance.memory_delete.proxy import Rule
from tests.fakes.llm import UNUSABLE_REPLY

from .fake_upstream import Override
from .journey import finish, purge
from .observe import md_counts, rows, s_leftovers, wait_idle
from .seed import scope_token, talk

pytestmark = pytest.mark.acceptance

S = {"scope": "project", "app_id": "a1", "project_id": "p1"}


def _ome(root: Path) -> Path:
    return root / ".index" / "sqlite" / "ome.db"


def _runs(root: Path) -> list[dict[str, Any]]:
    cols = [
        "run_id",
        "strategy_name",
        "status",
        "attempt",
        "started_at",
        "finished_at",
        "error",
        "event_topic",
        "event_payload",
        "event_id",
    ]
    return [
        dict(zip(cols, r, strict=True))
        for r in rows(_ome(root), f"SELECT {', '.join(cols)} FROM run_record")
    ]


def _scope_of(payload: str) -> tuple[str, str] | None:
    try:
        data = json.loads(payload)
    except ValueError:
        return None
    if isinstance(data, dict) and "app_id" in data:
        return data["app_id"], data["project_id"]
    return None


def _after(runs: list[dict[str, Any]], at: datetime) -> list[dict[str, Any]]:
    return [r for r in runs if from_iso_format(r["started_at"]) > at]


def _hold_facts(proxy: Any, token: str) -> Rule:
    """Hold the atomic-fact extraction of the scope ``token`` names."""
    return proxy.add(
        Rule(
            target="llm",
            pattern=f'(?=.*"_AtomicFactsReply")(?=.*{token})',
            mode="hold",
        )
    )


def _backlog(world: Any, proxy: Any, app: str, project: str) -> None:
    """A backlog of (app, project) the purge must drop, behind a held run of
    (a2, p1) in the only OME slot (the purge does not wait for another
    range, so it ends with the backlog still queued).

    The backlog is the part no source check covers: (app, project)'s agent
    case is extracted first (its memcell still there), and the event it
    emits -- which carries the case's content to the skill clustering --
    lands in the queue behind the held (a2, p1) run. First-stage events
    alone would prove nothing: their writes are skipped once the memcell
    is gone, drop rule or not."""
    srv = world.srv
    token = scope_token(app, project)
    began = get_utc_now()
    cases = md_counts(srv.root, app, project)["md.agent_case"]
    case = proxy.add(
        Rule(
            target="llm",
            pattern=f'(?=.*"_CaseCompressReply")(?=.*{token})',
            mode="hold",
        )
    )
    talk(srv, "u1", "backlog", token, {"app_id": app, "project_id": project})
    assert proxy.wait_held("llm", timeout=60)  # the case extraction, in the slot
    blocker = scope_token("a2", "p1")
    _hold_facts(proxy, blocker)
    talk(srv, "u1", "blocker", blocker, {"app_id": "a2", "project_id": "p1"})
    case.release.set()  # the case lands; its follow-up queues behind a2's runs
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        running = [r for r in _runs(srv.root) if r["status"] == "running"]
        if [_scope_of(r["event_payload"]) for r in running] == [("a2", "p1")]:
            break
        time.sleep(0.2)
    assert [_scope_of(r["event_payload"]) for r in running] == [("a2", "p1")]
    assert md_counts(srv.root, app, project)["md.agent_case"] > cases  # it landed
    clustering = [
        r
        for r in _after(_runs(srv.root), began)
        if r["strategy_name"] == "trigger_skill_clustering"
    ]
    assert clustering == [], clustering  # its follow-up is still queued


def _discarded(srv: Any) -> set[str]:
    """Run ids the OME gate dropped, from the server log."""
    log = re.sub(r"\x1b\[[0-9;]*m", "", srv.log_text())
    return set(re.findall(r"ome_run_discarded_by_gate .*?run_id=(\w+)", log))


def _drain_and_check(world: Any, proxy: Any, job: dict[str, Any]) -> None:
    srv = world.srv
    assert job["status"] == "succeeded", job
    at = from_iso_format(job["created_at"])
    proxy.clear()  # release the blocker: the backlog runs now
    wait_idle(srv.base, srv.root, timeout=300)
    world.ev.json("runs_after_t", _after(_runs(srv.root), at))
    left = s_leftovers(srv.root, "a1", "p1", before=at)
    world.ev.json("s_left", left)
    assert left == {}, left
    assert md_counts(srv.root, "a2", "p1")["md.atomic_fact"] > 0  # I1: it ran


def test_t9_1_backlog_never_writes_back(make_world: Any, purge_proxy: Any) -> None:
    world = make_world("T9.1", EVEROS_OME__MAX_CONCURRENT_RUNS="1")
    srv = world.srv
    _backlog(world, purge_proxy, "a1", "p1")
    job_id, _ = purge(srv, **S)
    _drain_and_check(world, purge_proxy, finish(srv, job_id, timeout=180))


def test_t24_1_app_tombstone_covers_its_projects(
    make_world: Any, purge_proxy: Any
) -> None:
    world = make_world("T24.1", EVEROS_OME__MAX_CONCURRENT_RUNS="1")
    srv = world.srv
    _backlog(world, purge_proxy, "a1", "p1")
    job_id, _ = purge(srv, scope="app", app_id="a1")
    _drain_and_check(world, purge_proxy, finish(srv, job_id, timeout=180))


def test_t24_2_latest_purge_of_a_scope_wins(make_world: Any, purge_proxy: Any) -> None:
    world = make_world("T24.2", EVEROS_OME__MAX_CONCURRENT_RUNS="1")
    srv = world.srv
    first, _ = purge(srv, **S)
    assert finish(srv, first)["status"] == "succeeded"
    wait_idle(srv.base, srv.root, timeout=180)
    _backlog(world, purge_proxy, "a1", "p1")  # events after T1
    second, _ = purge(srv, **S)
    _drain_and_check(world, purge_proxy, finish(srv, second, timeout=180))


def test_t15_1_failed_runs_keep_rows_lose_content(make_world: Any, fake: Any) -> None:
    world = make_world("T15.1")
    srv, root = world.srv, world.srv.root
    poisoned = []
    for app, project in (("a1", "p1"), ("a2", "p1")):
        token = scope_token(app, project)
        poisoned.append(
            fake.override(
                Override(
                    pattern=f"{token}poison",
                    reply=UNUSABLE_REPLY,
                    name="_ForesightsReply",
                )
            )
        )
        talk(
            srv,
            "u2",
            "poisoned",
            f"{token}poison",
            {"app_id": app, "project_id": project},
        )
    deadline = time.monotonic() + 180
    dead: dict[tuple[str, str], list[dict[str, Any]]] = {}
    while time.monotonic() < deadline:
        dead = {}
        for run in _runs(root):
            scope = _scope_of(run["event_payload"])
            if (
                run["status"] == "dead_letter"
                and scope
                and "poison" in run["event_payload"]
            ):
                dead.setdefault(scope, []).append(run)
        if set(dead) == {("a1", "p1"), ("a2", "p1")}:
            break
        time.sleep(1)
    assert set(dead) == {("a1", "p1"), ("a2", "p1")}, dead
    for item in poisoned:
        item.times = 0
    wait_idle(srv.base, root, timeout=180)
    ids = {scope: {r["run_id"] for r in runs} for scope, runs in dead.items()}
    before = {r["run_id"]: r["event_payload"] for r in _runs(root)}
    job_id, _ = purge(srv, **S)
    assert finish(srv, job_id)["status"] == "succeeded"
    after = {r["run_id"]: r for r in _runs(root)}
    world.ev.json("dead_rows", {f"{a}/{p}": sorted(v) for (a, p), v in ids.items()})
    world.ev.json(
        "payloads",
        {
            run_id: {"before": before[run_id], "after": after[run_id]["event_payload"]}
            for v in ids.values()
            for run_id in v
        },
    )
    marker = json.dumps({"purged_by": job_id}, separators=(",", ":"))
    for run_id in ids[("a1", "p1")]:
        assert after[run_id]["status"] == "dead_letter"
        assert after[run_id]["event_payload"] == marker
        assert "poison" not in after[run_id]["event_payload"]
    for run_id in ids[("a2", "p1")]:
        assert after[run_id]["event_payload"] == before[run_id]
    _read_back_runs(root, [*ids[("a1", "p1")], *ids[("a2", "p1")]])


def _read_back_runs(root: Path, run_ids: list[str]) -> None:
    """The rewritten rows still read back through the run-record store."""
    import asyncio

    from everos.infra.ome._stores.run_record import RunRecordStore
    from everos.infra.ome._stores.storage import OMEStorage

    async def read() -> None:
        storage = OMEStorage(db_path=_ome(root))
        await storage.init()
        store = RunRecordStore(storage=storage, max_records_per_strategy=10_000)
        for run_id in run_ids:
            record = await store.get(run_id)
            assert record is not None
            await store.list_by_event_id(record.event_id)

    asyncio.run(read())


def test_t10_1_other_scope_and_ticks_run_meanwhile(
    make_world: Any, purge_proxy: Any
) -> None:
    """While the purge of S waits for its own work in flight, another range's
    events and a tick run and land: nothing outside S waits for it."""
    world = make_world("T10.1")
    srv, root = world.srv, world.srv.root
    token = scope_token("a1", "p1")
    hold = _hold_facts(purge_proxy, token)
    talk(srv, "u1", "inflight", token, {"app_id": "a1", "project_id": "p1"})
    assert purge_proxy.wait_held("llm", timeout=60)
    job_id, _ = purge(srv, **S)
    at = from_iso_format(srv.job(job_id).json()["data"]["created_at"])
    facts = md_counts(root, "a2", "p1")["md.atomic_fact"]
    other = scope_token("a2", "p1")
    for i in range(2):
        talk(srv, "u1", f"other-{i}", other, {"app_id": "a2", "project_id": "p1"})
    trig = srv.post(
        "/ome/trigger",
        {"name": "reflect_episodes", "force": True, "timeout": 60},
        timeout=90,
    )
    assert trig.status_code == 200, trig.text
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if md_counts(root, "a2", "p1")["md.atomic_fact"] > facts:
            break
        time.sleep(0.5)
    reflect = [
        r
        for r in _after(_runs(root), at)
        if r["strategy_name"] == "reflect_episodes" and r["status"] == "success"
    ]
    status = srv.job(job_id).json()["data"]["status"]
    world.ev.json(
        "during_wait",
        {
            "job_status": status,
            "a2_facts": [facts, md_counts(root, "a2", "p1")["md.atomic_fact"]],
            "reflect_runs": reflect,
        },
    )
    assert status == "running"  # the purge still waits for S's run
    assert md_counts(root, "a2", "p1")["md.atomic_fact"] > facts
    assert reflect and not {r["run_id"] for r in reflect} & _discarded(srv)
    hold.release.set()
    purge_proxy.clear()
    assert finish(srv, job_id, timeout=180)["status"] == "succeeded"
    wait_idle(srv.base, root, timeout=300)
    left = s_leftovers(root, "a1", "p1", before=at)
    world.ev.json("s_left", left)
    assert left == {}, left
