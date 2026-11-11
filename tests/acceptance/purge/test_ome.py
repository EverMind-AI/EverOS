"""T9.1, T10.1, T13.1, T15.1, T24.1, T24.2: background (OME) work and purges."""

from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from everos.core.utils.datetime import from_iso_format
from tests.acceptance.memory_delete.proxy import Rule
from tests.fakes.llm import UNUSABLE_REPLY

from .fake_upstream import Override
from .journey import finish, purge
from .observe import md_counts, release, rows, scope_counts, system_db, wait_idle
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
    """Hold the scope's atomic-fact extraction: with one OME slot, the
    first such run blocks every other background run (the backlog)."""
    return proxy.add(
        Rule(
            target="llm",
            pattern=f'(?=.*"_AtomicFactsReply")(?=.*{token})',
            mode="hold",
        )
    )


def _backlog(world: Any, proxy: Any, app: str, project: str, n: int) -> tuple[str, str]:
    """``n`` sessions of (app, project) behind a held fact extraction; E0
    as ``(event_id, strategy_name)``."""
    srv = world.srv
    token = scope_token(app, project)
    _hold_facts(proxy, token)
    where = {"app_id": app, "project_id": project}
    for i in range(n):
        talk(srv, "u1", f"backlog-{i}", token, where)
    assert proxy.wait_held("llm", timeout=60)
    running = [r for r in _runs(srv.root) if r["status"] == "running"]
    assert len(running) == 1, running  # one slot, held; the rest queue
    assert _scope_of(running[0]["event_payload"]) == (app, project)
    return running[0]["event_id"], running[0]["strategy_name"]


def _discarded(srv: Any) -> set[str]:
    """Run ids the OME gate dropped, from the server log: the decision leaves
    no other trace (a dropped run and a run that found its source deleted
    both end SUCCESS with an empty payload)."""
    log = re.sub(r"\x1b\[[0-9;]*m", "", srv.log_text())
    return set(re.findall(r"ome_run_discarded_by_gate .*?run_id=(\w+)", log))


def _assert_backlog_dropped(
    srv: Any, at: datetime, e0: tuple[str, str], app: str, project: str
) -> dict[str, Any]:
    root = srv.root
    counts = scope_counts(root, app, project)
    for key in (
        "md.atomic_fact",
        "md.foresight",
        "md.user_profile",
        "sqlite.cluster",
        "sqlite.cluster_member",
    ):
        assert counts[key] == 0, (key, counts)
    for key in [k for k in counts if k.startswith("index.")]:
        assert counts[key] == 0, (key, counts)
    # One event feeds several strategies; each (event, strategy) is one run.
    by_run: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in _after(_runs(root), at):
        by_run.setdefault((run["event_id"], run["strategy_name"]), []).append(run)
    for (event_id, strategy), runs in by_run.items():
        statuses = [r["status"] for r in runs]
        if (event_id, strategy) == e0:  # its first try fails after T (gap)
            assert statuses[-1] == "success", runs
            continue
        assert statuses == ["success"], runs
        assert runs[0]["event_payload"] == "", runs
    after = {r["run_id"] for r in _after(_runs(root), at)}
    assert after and after <= _discarded(srv), after - _discarded(srv)
    return counts


def test_t9_1_backlog_never_writes_back(make_world: Any, purge_proxy: Any) -> None:
    world = make_world(
        "T9.1",
        EVEROS_OME__MAX_CONCURRENT_RUNS="1",
        EVEROS_PURGE__SETTLE_WAIT_SECONDS="5",
    )
    srv = world.srv
    e0 = _backlog(world, purge_proxy, "a1", "p1", 10)
    assert md_counts(srv.root, "a1", "p1")["md.atomic_fact"] > 0  # the seed's
    job_id, _ = purge(srv, **S)
    job = finish(srv, job_id, timeout=180)
    assert job["status"] == "succeeded", job
    at = from_iso_format(job["created_at"])
    purge_proxy.clear()  # release the held extraction
    wait_idle(srv.base, srv.root, timeout=180)
    world.ev.json("runs_after_t", _after(_runs(srv.root), at))
    counts = _assert_backlog_dropped(srv, at, e0, "a1", "p1")
    world.ev.json("s_after", counts)


def test_t24_1_app_tombstone_covers_its_projects(
    make_world: Any, purge_proxy: Any
) -> None:
    world = make_world(
        "T24.1",
        EVEROS_OME__MAX_CONCURRENT_RUNS="1",
        EVEROS_PURGE__SETTLE_WAIT_SECONDS="5",
    )
    srv = world.srv
    e0 = _backlog(world, purge_proxy, "a1", "p1", 10)
    job_id, _ = purge(srv, scope="app", app_id="a1")
    job = finish(srv, job_id, timeout=180)
    assert job["status"] == "succeeded", job
    purge_proxy.clear()
    wait_idle(srv.base, srv.root, timeout=180)
    at = from_iso_format(job["created_at"])
    world.ev.json("runs_after_t", _after(_runs(srv.root), at))
    world.ev.json("s_after", _assert_backlog_dropped(srv, at, e0, "a1", "p1"))
    world.ev.json("runs_after_t", _after(_runs(srv.root), at))


def test_t24_2_latest_purge_of_a_scope_wins(make_world: Any, purge_proxy: Any) -> None:
    world = make_world(
        "T24.2",
        EVEROS_OME__MAX_CONCURRENT_RUNS="1",
        EVEROS_PURGE__SETTLE_WAIT_SECONDS="5",
    )
    srv = world.srv
    first, _ = purge(srv, **S)
    assert finish(srv, first)["status"] == "succeeded"
    wait_idle(srv.base, srv.root, timeout=180)
    e0 = _backlog(world, purge_proxy, "a1", "p1", 6)  # events after T1
    second, _ = purge(srv, **S)
    job = finish(srv, second, timeout=180)
    assert job["status"] == "succeeded", job
    purge_proxy.clear()
    wait_idle(srv.base, srv.root, timeout=180)
    at = from_iso_format(job["created_at"])
    world.ev.json("s_after", _assert_backlog_dropped(srv, at, e0, "a1", "p1"))
    world.ev.json("runs_after_t", _after(_runs(srv.root), at))


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


def _ome_rows_waiting(root: Path, at: datetime, app: str) -> list[dict[str, Any]]:
    return [
        r
        for r in _after(_runs(root), at)
        if r["status"] == "running"
        and (_scope_of(r["event_payload"]) or (None, None))[0] in (app, None)
    ]


def _start_held_other_scope(world: Any, proxy: Any) -> tuple[threading.Thread, int]:
    """(a2, p1) work held in the only slot, a Reflection run queued behind it."""
    srv = world.srv
    token = scope_token("a2", "p1")
    _hold_facts(proxy, token)
    for i in range(2):
        talk(srv, "u1", f"other-{i}", token, {"app_id": "a2", "project_id": "p1"})
    assert proxy.wait_held("llm", timeout=60)
    trigger = threading.Thread(
        target=lambda: srv.post(
            "/ome/trigger",
            {"name": "reflect_episodes", "force": True, "timeout": 1},
            timeout=30,
        )
    )
    trigger.start()
    return trigger, _reports_outside_s(srv.root)


def _reports_outside_s(root: Path) -> int:
    """Reflection reports the purge of S does not touch."""
    return rows(
        system_db(root),
        "SELECT COUNT(*) FROM reflection_report"
        " WHERE NOT (app_id = 'a1' AND project_id = 'p1')",
    )[0][0]


def test_t10_1_other_scope_and_ticks_run_after(
    make_world: Any, purge_proxy: Any
) -> None:
    world = make_world(
        "T10.1",
        EVEROS_OME__MAX_CONCURRENT_RUNS="1",
        EVEROS_PURGE__SETTLE_WAIT_SECONDS="600",
        EVEROS_TEST_FAILPOINTS="purge.before_step3=pause",
    )
    srv, root = world.srv, world.srv.root
    trigger, reports = _start_held_other_scope(world, purge_proxy)
    job_id, _ = purge(srv, **S)
    confirmed = time.monotonic()
    at = from_iso_format(srv.job(job_id).json()["data"]["created_at"])
    purge_proxy.clear()
    deadline = time.monotonic() + 60
    waiting: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        waiting = _ome_rows_waiting(root, at, "a2")
        if any(r["strategy_name"] == "reflect_episodes" for r in waiting) and any(
            _scope_of(r["event_payload"]) == ("a2", "p1") for r in waiting
        ):
            break
        time.sleep(0.5)
    world.ev.json("waiting", waiting)
    assert any(r["strategy_name"] == "reflect_episodes" for r in waiting), waiting
    facts = md_counts(root, "a2", "p1")["md.atomic_fact"]
    time.sleep(2)  # they wait on the mark: nothing of theirs lands meanwhile
    assert md_counts(root, "a2", "p1")["md.atomic_fact"] == facts
    paused = time.monotonic()
    release(srv.fp_dir, "purge.before_step3")
    resumed = time.monotonic()
    job = finish(srv, job_id, timeout=120)
    took = time.monotonic() - confirmed - (resumed - paused)
    assert job["status"] == "succeeded", job
    trigger.join(timeout=30)
    wait_idle(srv.base, root, timeout=300)
    final = {r["run_id"]: r for r in _runs(root)}
    for run in waiting:
        later = [r for r in final.values() if r["event_id"] == run["event_id"]]
        assert any(r["status"] == "success" for r in later), later
    # They ran, not dropped: a dropped run also ends SUCCESS, so ask the gate.
    assert {r["run_id"] for r in waiting} & _discarded(srv) == set()
    world.ev.json(
        "reflection_reports_outside_s",
        {"before": reports, "after": _reports_outside_s(root)},
    )
    assert md_counts(root, "a2", "p1")["md.atomic_fact"] > facts
    s_left = scope_counts(root, "a1", "p1")
    assert all(n == 0 for k, n in s_left.items() if k.startswith("index."))
    world.ev.json("timing", {"confirm_to_terminal_minus_pause_s": took})
    assert took < 30, took


def test_t13_1_waiting_runs_survive_a_kill(make_world: Any, purge_proxy: Any) -> None:
    world = make_world(
        "T13.1",
        EVEROS_OME__MAX_CONCURRENT_RUNS="1",
        EVEROS_PURGE__SETTLE_WAIT_SECONDS="600",
        EVEROS_TEST_FAILPOINTS="purge.before_step3=pause",
    )
    srv, root = world.srv, world.srv.root
    token = scope_token("a2", "p1")
    _hold_facts(purge_proxy, token)
    for i in range(2):
        talk(srv, "u1", f"other-{i}", token, {"app_id": "a2", "project_id": "p1"})
    assert purge_proxy.wait_held("llm", timeout=60)
    job_id, _ = purge(srv, **S)
    at = from_iso_format(srv.job(job_id).json()["data"]["created_at"])
    purge_proxy.clear()
    deadline = time.monotonic() + 60
    waiting: list[dict[str, Any]] = []
    while time.monotonic() < deadline:
        waiting = [
            r
            for r in _ome_rows_waiting(root, at, "a2")
            if _scope_of(r["event_payload"]) == ("a2", "p1")
        ]
        if waiting:
            break
        time.sleep(0.5)
    assert waiting, "no (a2, p1) run waiting on the mark"
    facts = md_counts(root, "a2", "p1")["md.atomic_fact"]
    world.ev.json("waiting", waiting)
    srv.kill()
    srv.env_extra.pop("EVEROS_TEST_FAILPOINTS")
    srv.start()
    assert finish(srv, job_id, timeout=120)["status"] == "succeeded"
    wait_idle(srv.base, root, timeout=300)
    final = _runs(root)
    for run in waiting:
        same = [r for r in final if r["event_id"] == run["event_id"]]
        old = [r for r in same if r["run_id"] == run["run_id"]]
        assert old and old[0]["status"] == "crashed", same
        assert any(
            r["status"] == "success" and r["run_id"] != run["run_id"] for r in same
        ), same
    assert md_counts(root, "a2", "p1")["md.atomic_fact"] > facts
    s_left = scope_counts(root, "a1", "p1")
    ids = {w["event_id"] for w in waiting}
    world.ev.json("s_after", s_left)
    world.ev.json("final_rows_of_waiting", [r for r in final if r["event_id"] in ids])
    assert all(n == 0 for n in s_left.values()), s_left
