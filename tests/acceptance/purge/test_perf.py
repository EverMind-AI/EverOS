"""T35.1, T36.1: a purge does not slow other callers down (A35, A36).

Run alone (``-k "t35 or t36"``): the timings are wall clock and other
servers on the machine skew them.
"""

from __future__ import annotations

import datetime as dt
import math
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from tests.acceptance.memory_delete.conftest import chat
from tests.acceptance.memory_delete.proxy import Rule
from tests.acceptance.memory_delete.seed import write_episode

from .journey import Evidence, evidence_base, finish, purge
from .observe import system_db, wait_idle
from .seed import AGENT, scope_token

pytestmark = pytest.mark.acceptance

S = {"scope": "project", "app_id": "a1", "project_id": "p1"}
WHERE = {"app_id": "a1", "project_id": "p1"}
OTHER = {"app_id": "a2", "project_id": "p1"}
PER_SCOPE = 5000
PER_DAY = 50
PROBES = 20


def _write_many(root: Any, app: str, project: str) -> None:
    start = dt.date.today() - dt.timedelta(days=PER_SCOPE // PER_DAY + 1)
    for i in range(PER_SCOPE):
        day = (start + dt.timedelta(days=i // PER_DAY)).isoformat()
        write_episode(
            root,
            f"u{i % 5}",
            day,
            app=app,
            project=project,
            session=f"s{i}",
            text=f"memory {i} of {app}/{project}",
        )


def _p95(seconds: list[float]) -> float:
    ordered = sorted(seconds)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _probes(srv: Any) -> dict[str, Callable[[], httpx.Response]]:
    def search(where: dict[str, str]) -> Callable[[], httpx.Response]:
        return lambda: srv.post(
            "/memory/search",
            {"query": "memory", "user_id": "u1", "method": "keyword", **where},
        )

    def add_other() -> httpx.Response:
        return srv.post(
            "/memory/add",
            {
                "session_id": f"probe-{time.monotonic_ns()}",
                "messages": chat("u9", "probe"),
                "defer_extraction": True,
                **OTHER,
            },
        )

    return {
        "search_s": search(WHERE),
        "search_other": search(OTHER),
        "add_other": add_other,
    }


def _measure(srv: Any, until: Callable[[], bool] | None = None) -> dict[str, Any]:
    """``PROBES`` calls of each probe (or as many as fit until ``until``)."""
    out: dict[str, Any] = {}
    for name, call in _probes(srv).items():
        codes, seconds = [], []
        for _ in range(PROBES):
            if until is not None and until():
                break
            t0 = time.monotonic()
            codes.append(call().status_code)
            seconds.append(time.monotonic() - t0)
        out[name] = {"codes": sorted(set(codes)), "n": len(codes), "seconds": seconds}
    return out


def _compare(base: dict[str, Any], during: dict[str, Any]) -> dict[str, Any]:
    verdict = {}
    for name, b in base.items():
        d = during[name]
        verdict[name] = {
            "codes": [b["codes"], d["codes"]],
            "p95": [round(_p95(b["seconds"]), 3), round(_p95(d["seconds"]), 3)]
            if d["seconds"]
            else [round(_p95(b["seconds"]), 3), None],
            "n": d["n"],
        }
    return verdict


def _assert_not_slowed(verdict: dict[str, Any]) -> None:
    for name, v in verdict.items():
        assert v["n"] > 0, (name, v)
        assert v["codes"][0] == v["codes"][1], (name, v)
        base, during = v["p95"]
        assert during <= base * 2 + 0.2, (name, v)


def test_t35_1a_callers_are_not_slowed_while_the_purge_waits(
    make_world: Any, purge_proxy: Any
) -> None:
    world = make_world("T35.1a")
    srv = world.srv
    base = _measure(srv)
    word = f"{scope_token('a1', 'p1')}slow"
    hold = purge_proxy.add(
        Rule(target="llm", pattern=f'(?=.*"_BatchReply")(?=.*{word})', mode="hold")
    )
    sent: dict[str, httpx.Response] = {}
    adder = threading.Thread(
        target=lambda: sent.update(
            r=srv.post(
                "/memory/add",
                {
                    "session_id": "slow",
                    **WHERE,
                    "messages": chat("u1", word, assistant=AGENT),
                },
            )
        )
    )
    adder.start()
    assert purge_proxy.wait_held("llm", timeout=60)
    job_id, _ = purge(srv, **S)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        data = srv.job(job_id).json()["data"]
        if all(s["state"] == "done" for s in data["steps"]):
            break
        time.sleep(0.2)
    during = _measure(srv)
    status = srv.job(job_id).json()["data"]["status"]
    hold.release.set()
    adder.join(timeout=60)
    assert finish(srv, job_id, timeout=180)["status"] == "succeeded"
    verdict = _compare(base, during)
    world.ev.json("probes", {"job_status_after": status, "verdict": verdict})
    assert status == "running"  # measured while the purge waited
    _assert_not_slowed(verdict)


def test_t35_1b_callers_are_not_slowed_while_the_purge_deletes(
    make_purge_server: Any, tmp_path: Any
) -> None:
    srv = make_purge_server(start=False)
    ev = Evidence("T35.1b", evidence_base(tmp_path), [srv])
    try:
        _write_many(srv.root, "a1", "p1")
        write_episode(
            srv.root,
            "u1",
            dt.date.today().isoformat(),
            **{"app": "a2", "project": "p1"},
        )
        srv.start()
        wait_idle(srv.base, srv.root, timeout=1800)
        base = _measure(srv)
        job_id, _ = purge(srv, **S)

        def ended() -> bool:
            return srv.job(job_id).json()["data"]["status"] in ("succeeded", "failed")

        during = _measure(srv, until=ended)
        job = finish(srv, job_id, timeout=300)
        verdict = _compare(base, during)
        ev.json("probes", {"job": job, "verdict": verdict})
        assert job["status"] == "succeeded", job
        _assert_not_slowed(verdict)
    finally:
        ev.close()


def _bulk_rows(db: Any, n: int) -> None:
    """``n`` memcell rows of S, written straight into system.db."""
    con = sqlite3.connect(db)
    try:
        cols = con.execute("PRAGMA table_info(memcell)").fetchall()
        names, values = [], []
        for _, name, kind, notnull, default, _pk in cols:
            if name == "memcell_id":
                names.append(name)
                values.append("'mc_bulk_' || n")
            elif name in ("app_id", "project_id"):
                names.append(name)
                values.append(f"'{WHERE[name]}'")
            elif notnull and default is None:
                names.append(name)
                upper = (kind or "").upper()
                if "INT" in upper:
                    values.append("0")
                elif "DATE" in upper or "TIME" in upper:
                    values.append("'2026-01-01 00:00:00.000000'")
                elif name.endswith("_json"):
                    values.append("'[]'")
                else:
                    values.append("'x'")
        step = f"SELECT n + 1 FROM c WHERE n < {n}"
        counter = f"WITH RECURSIVE c(n) AS (SELECT 1 UNION ALL {step})"
        columns, picked = ", ".join(names), ", ".join(values)
        con.execute(f"{counter} INSERT INTO memcell ({columns}) SELECT {picked} FROM c")
        con.commit()
    finally:
        con.close()


def test_t36_1_a_big_delete_does_not_lock_other_writers(
    make_purge_server: Any, tmp_path: Any
) -> None:
    srv = make_purge_server(start=False)
    ev = Evidence("T36.1", evidence_base(tmp_path), [srv])
    try:
        srv.start()  # creates the schema
        srv.stop()
        _bulk_rows(system_db(srv.root), 100_000)
        srv.start()
        add = _probes(srv)["add_other"]
        base_seconds: list[float] = []
        started = time.monotonic()
        while time.monotonic() - started < 10:
            t0 = time.monotonic()
            assert add().status_code == 200
            base_seconds.append(time.monotonic() - t0)
        job_id, _ = purge(srv, **S)
        codes, seconds = [], []
        while srv.job(job_id).json()["data"]["status"] not in ("succeeded", "failed"):
            t0 = time.monotonic()
            codes.append(add().status_code)
            seconds.append(time.monotonic() - t0)
        job = finish(srv, job_id)
        log = srv.log_text()
        ev.json(
            "writes",
            {
                "baseline_max": max(base_seconds),
                "during_max": max(seconds) if seconds else None,
                "during_n": len(seconds),
                "codes": sorted(set(codes)),
                "job": job,
            },
        )
        assert job["status"] == "succeeded", job
        assert set(codes) <= {200}, codes
        assert "database is locked" not in log
        if seconds:
            assert max(seconds) <= max(base_seconds) + 1, (max(seconds), base_seconds)
    finally:
        ev.close()
