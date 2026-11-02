"""T4.1, T5.1, T11.1 (a), T13.1, T23.1: deletions timed into windows."""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
import pytest

from .conftest import chat
from .observe import md_entries, nothing_deleted_snapshot, rows, wait_idle
from .proxy import BOUNDARY, EPISODE, Rule, llm_call

pytestmark = pytest.mark.acceptance


def _in_background(fn: Any) -> threading.Thread:
    t = threading.Thread(target=fn, daemon=True)
    t.start()
    return t


def _session_entries(srv: Any, session: str) -> int:
    return sum(1 for e in md_entries(srv.root) if e["session_id"] == session)


def test_t4_1_a_queued_work_is_503_and_nothing_goes(server: Any) -> None:
    """A4 (a): async /add batches of the session still queued or processing."""
    srv = server
    assert srv.add("s0", chat("u1", "lynx0")).status_code == 200
    assert srv.flush("s0").status_code == 200
    srv.proxy.add(Rule("llm", llm_call(BOUNDARY, "lynx1"), "hold"))
    for i in (1, 2):
        resp = srv.add("s1", chat("u1", f"lynx{i}"), async_mode=True)
        assert resp.status_code == 202, resp.text
    assert srv.proxy.wait_held("llm")
    before = nothing_deleted_snapshot(srv.root)

    resp = srv.delete(session_id="s1")

    assert resp.status_code == 503
    error = resp.json()["error"]
    assert error["code"] == "SESSION_BUSY"
    assert "nothing was deleted, retry later" in error["message"]
    assert nothing_deleted_snapshot(srv.root) == before
    srv.proxy.clear()


def test_t4_1_b_queued_while_waiting_for_the_lock(server: Any) -> None:
    """A4 (b): a sync /add holds the session lock (its cut is delayed 8 s);
    the deletion waits on it; at 3 s an async /add queues a batch; when the
    lock frees the deletion sees it and answers 503. Deterministic."""
    srv = server
    assert srv.add("s1", chat("u1", "puma0")).status_code == 200
    assert srv.flush("s1").status_code == 200
    srv.proxy.add(Rule("llm", llm_call(BOUNDARY, "puma1"), "delay", delay=8))
    before = nothing_deleted_snapshot(srv.root)
    adder = _in_background(lambda: srv.add("s1", chat("u1", "puma1")))
    time.sleep(1)  # the add holds the session lock
    result: dict[str, Any] = {}
    deleter = _in_background(
        lambda: result.setdefault("resp", srv.delete(session_id="s1"))
    )
    time.sleep(2)
    queued = srv.add("s1", chat("u1", "puma2"), async_mode=True)
    assert queued.status_code == 202
    deleter.join(60)
    adder.join(60)

    assert result["resp"].status_code == 503, result["resp"].text
    assert result["resp"].json()["error"]["code"] == "SESSION_BUSY"
    # Nothing of s1 was deleted (the delayed add may have added to it).
    assert _session_entries(srv, "s1") >= 1
    assert before
    # The batch queued while the deletion waited is delivered afterwards.
    task = queued.json()["request_id"]
    deadline = time.monotonic() + 180
    status = None
    while time.monotonic() < deadline:
        got = httpx.get(f"{srv.base}/api/v2/tasks/{task}", timeout=10).json()
        status = got.get("data", {}).get("status")
        if status in ("success", "failed"):
            break
        time.sleep(1)
    assert status == "success", status


def test_t5_1_lock_wait_is_bounded(server: Any) -> None:
    """A5: an add holding the session lock for 25 s -> 503 within 10 s + 1 s."""
    srv = server
    assert srv.add("s1", chat("u1", "orca0")).status_code == 200
    assert srv.flush("s1").status_code == 200
    wait_idle(srv.base, srv.root)
    srv.proxy.add(Rule("llm", llm_call(BOUNDARY, "orca1"), "delay", delay=25))
    adder = _in_background(lambda: srv.add("s1", chat("u1", "orca1")))
    time.sleep(1)
    before = nothing_deleted_snapshot(srv.root)

    started = time.monotonic()
    resp = srv.delete(session_id="s1")
    took = time.monotonic() - started

    assert resp.status_code == 503 and resp.json()["error"]["code"] == "SESSION_BUSY"
    assert 9 <= took <= 11, took
    assert nothing_deleted_snapshot(srv.root) == before
    adder.join(60)


def test_t11_1_a_model_that_never_answers_does_not_hang_a_deletion(server: Any) -> None:
    """A11 (a): the cut's model call accepts and never answers; the session
    deletion answers 503 in 10 +- 1 s instead of hanging."""
    srv = server
    rule = srv.proxy.add(Rule("llm", llm_call(BOUNDARY, "yak1"), "hold"))
    adder = _in_background(lambda: srv.add("s1", chat("u1", "yak1"), timeout=600))
    assert srv.proxy.wait_held("llm")

    started = time.monotonic()
    resp = srv.delete(session_id="s1")
    took = time.monotonic() - started

    assert resp.status_code == 503
    assert 9 <= took <= 11, took
    rule.release.set()
    adder.join(120)


def test_t13_1_other_session_of_the_same_owner_loses_nothing(server: Any) -> None:
    """A13: S2's entries share u1's day file with S's; S is deleted while a
    stream of S2 adds runs; every S2 entry is there afterwards."""
    srv = server
    for i in range(3):
        assert srv.add("S", chat("u1", f"wren{i}", n=1)).status_code == 200
        assert srv.flush("S").status_code == 200
        assert srv.add("S2", chat("u1", f"kite{i}", n=1)).status_code == 200
        assert srv.flush("S2").status_code == 200
    seeded = {e["entry_id"] for e in md_entries(srv.root) if e["session_id"] == "S2"}
    assert seeded and _session_entries(srv, "S") > 0
    stop = threading.Event()

    def stream() -> None:
        i = 0
        while not stop.is_set():
            srv.add("S2", chat("u1", f"kite{10 + i}", n=1))
            srv.flush("S2")
            i += 1

    writer = _in_background(stream)
    time.sleep(2)
    resp = srv.delete(session_id="S")
    time.sleep(2)
    stop.set()
    writer.join(300)

    assert resp.status_code == 200
    after = {e["entry_id"] for e in md_entries(srv.root) if e["session_id"] == "S2"}
    assert seeded <= after
    assert _session_entries(srv, "S") == 0


def test_t23_1_add_in_flight_when_its_session_is_deleted(server: Any) -> None:
    """A23: an /add whose episode extraction is held; the session is deleted
    meanwhile; released, the /add answers 200, no episode is written and no
    run starts from one. 20 rounds; a round counts only if the call was held."""
    srv = server
    ome = srv.root / ".index" / "sqlite" / "ome.db"
    count_sql = (
        "SELECT COUNT(*) FROM run_record WHERE strategy_name IN"
        " ('extract_atomic_facts', 'trigger_profile_clustering')"
    )
    hits = 0
    for r in range(20):
        session = f"t23s{r}"
        token = f"tern{r}x"
        rule = srv.proxy.add(Rule("llm", llm_call(EPISODE, token), "hold", times=1))
        out: dict[str, Any] = {}
        adder = _in_background(
            lambda s=session, t=token, o=out: o.setdefault(
                "r", srv.add(s, chat("u1", t))
            )
        )
        held = srv.proxy.wait_held("llm", timeout=20)
        if not held:  # the add only buffered: the flush cuts and extracts
            adder.join(60)
            adder = _in_background(
                lambda s=session, o=out: o.__setitem__("r", srv.flush(s))
            )
            held = srv.proxy.wait_held("llm", timeout=40)
        if not held:
            rule.release.set()
            adder.join(120)
            continue
        hits += 1
        before = rows(ome, count_sql) if ome.exists() else [(0,)]
        assert srv.delete(session_id=session).status_code == 200
        rule.release.set()
        adder.join(120)
        assert out["r"].status_code == 200, out["r"].text
        time.sleep(4)
        assert _session_entries(srv, session) == 0
        after = rows(ome, count_sql) if ome.exists() else [(0,)]
        assert after == before, rows(
            ome,
            "SELECT strategy_name, status FROM run_record"
            " ORDER BY started_at DESC LIMIT 4",
        )
        srv.proxy.clear()
    assert hits >= 16, f"window hit {hits}/20: not manufactured"
    srv.stop()
    assert "memory_write_skipped_source_gone" in srv.log_text()
