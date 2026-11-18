"""T7.1, T16.1, T19.1, T24.1 (c) on a real server."""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from .conftest import chat
from .observe import lance_ids, metric_value, metrics, rows, wait_converged
from .proxy import Rule, cascade_embedding, single_embedding

pytestmark = pytest.mark.acceptance


def _episode_ids(srv: Any, user: str) -> set[str]:
    return {i for i in lance_ids(srv.root, "episode") if i.startswith(f"{user}_")}


def _search(srv: Any, user: str, query: str) -> list[str]:
    resp = srv.post(
        "/memory/search",
        {"user_id": user, "query": query, "method": "llm_multiround"},
    )
    assert resp.status_code == 200, resp.text
    return [e["id"] for e in resp.json()["data"].get("episodes", [])]


def test_t7_1_index_rows_go_with_the_response(server: Any) -> None:
    """A7: with the projection stopped, only the deletion itself can remove
    the index row; it is gone when the 200 arrives."""
    srv = server
    assert srv.add("s1", chat("u1", "okapi41")).status_code == 200
    assert srv.flush("s1").status_code == 200
    wait_converged(srv.base)
    ids = _episode_ids(srv, "u1")
    assert ids, "positive control: the episode is indexed"
    assert set(_search(srv, "u1", "okapi41")) & ids, "positive control: searchable"
    assert srv.post("/cascade/quiesce", {}).status_code == 200

    assert srv.delete(session_id="s1").status_code == 200

    assert not _episode_ids(srv, "u1") & ids
    assert not set(_search(srv, "u1", "okapi41")) & ids
    listed = srv.post("/memory/get", {"user_id": "u1", "memory_type": "episode"}).json()
    assert not {e["id"] for e in listed["data"]["episodes"]} & ids


def test_tb4_1_by_id_on_lancedb(server: Any) -> None:
    """B4 on LanceDB (the Milvus half is in test_milvus.py): with the
    projection stopped, a by-id deletion removes the episode's row before the
    200; /search and /get stop returning it."""
    srv = server
    assert srv.add("s1", chat("u1", "tapir52")).status_code == 200
    assert srv.flush("s1").status_code == 200
    wait_converged(srv.base)
    ids = _episode_ids(srv, "u1")
    assert len(ids) == 1, ids
    assert set(_search(srv, "u1", "tapir52")) & ids, "positive control: searchable"
    assert srv.post("/cascade/quiesce", {}).status_code == 200

    (target,) = ids
    assert srv.delete_by_ids([target]).status_code == 200

    assert target not in _episode_ids(srv, "u1")
    assert target not in set(_search(srv, "u1", "tapir52"))
    listed = srv.post("/memory/get", {"user_id": "u1", "memory_type": "episode"}).json()
    assert target not in {e["id"] for e in listed["data"]["episodes"]}


def test_t16_1_succeeded_runs_keep_no_conversation_text(server: Any) -> None:
    """A16: every success row of ome.db has an empty event_payload; a run
    killed while RUNNING is replayed after the restart and also ends empty.

    The run killed is the new episode's ``trigger_profile_clustering``, held
    in its embedding of the episode."""
    srv = server
    assert srv.add("s1", chat("u1", "heron77")).status_code == 200
    assert srv.flush("s1").status_code == 200
    wait_converged(srv.base)
    ome = srv.root / ".index" / "sqlite" / "ome.db"
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and rows(
        ome, "SELECT COUNT(*) FROM run_record WHERE status='running'"
    ) != [(0,)]:
        time.sleep(1)
    success = rows(ome, "SELECT event_payload FROM run_record WHERE status='success'")
    assert success and all(p == "" for (p,) in success), success[:3]

    hold = srv.proxy.add(Rule("emb", single_embedding("heron88"), "hold"))
    assert srv.add("s2", chat("u1", "heron88")).status_code == 200
    flush = threading.Thread(target=lambda: srv.flush("s2"), daemon=True)
    flush.start()
    assert srv.proxy.wait_held("emb", timeout=120)
    srv.kill()
    hold.release.set()
    srv.proxy.clear()
    srv.start()
    wait_converged(srv.base)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline and rows(
        ome, "SELECT COUNT(*) FROM run_record WHERE status='running'"
    ) != [(0,)]:
        time.sleep(1)
    success = rows(ome, "SELECT event_payload FROM run_record WHERE status='success'")
    assert all(p == "" for (p,) in success)
    assert rows(ome, "SELECT COUNT(*) FROM run_record WHERE status='running'") == [(0,)]
    # The killed run is marked crashed and replayed as a new run that succeeded.
    crashed = rows(
        ome,
        "SELECT strategy_name FROM run_record WHERE status='crashed'"
        " AND strategy_name='trigger_profile_clustering'",
    )
    assert crashed, "the run killed in flight is recorded as crashed"
    replayed = rows(
        ome,
        "SELECT COUNT(*) FROM run_record WHERE status='success'"
        " AND strategy_name='trigger_profile_clustering'",
    )
    assert replayed[0][0] >= 2  # the first add's and the replay


def test_t19_1_and_t24_1_c_fence_is_used_and_emptied(make_server: Any) -> None:
    """A19 / A24 (c): while a cascade run of the file is held in its embedding,
    a deletion fences rows (gauges > 0, warning above a threshold of 3); once
    the run ends both gauges are 0. With the cascade idle they stay 0."""
    srv = make_server()
    srv.start(EVEROS_CASCADE__DELETE_FENCE_WARN_ENTRIES="3")
    for i in range(5):
        assert srv.add("s1", chat("u1", f"ibis{i}x", n=1)).status_code == 200
        assert srv.flush("s1").status_code == 200
    assert srv.add("s0", chat("u1", "heron0x", n=1)).status_code == 200
    assert srv.flush("s0").status_code == 200
    wait_converged(srv.base)
    idle = metrics(srv.base)
    assert metric_value(idle, "everos_memory_delete_fence_entries") == 0
    # (b) The cascade is idle: a deletion that removes rows keeps none fenced.
    idle_delete = srv.delete(session_id="s0")
    assert idle_delete.status_code == 200
    assert idle_delete.json()["data"]["count"] > 0, "positive control: rows removed"
    after_idle = metrics(srv.base)
    assert metric_value(after_idle, "everos_memory_delete_fence_entries") == 0
    assert metric_value(after_idle, "everos_memory_delete_fence_runs") == 0

    hold = srv.proxy.add(Rule("emb", cascade_embedding("ibis9z"), "hold"))
    assert srv.add("s1", chat("u1", "ibis9z", n=1)).status_code == 200
    assert srv.flush("s1").status_code == 200
    assert srv.proxy.wait_held("emb", timeout=120), "the cascade's embedding is held"
    assert srv.delete(session_id="s1").status_code == 200
    held = metrics(srv.base)
    assert (metric_value(held, "everos_memory_delete_fence_entries") or 0) > 3
    assert (metric_value(held, "everos_memory_delete_fence_runs") or 0) > 0
    hold.release.set()
    wait_converged(srv.base)
    done = metrics(srv.base)
    assert metric_value(done, "everos_memory_delete_fence_entries") == 0
    assert metric_value(done, "everos_memory_delete_fence_runs") == 0
    srv.stop()
    assert "memory_delete_fence_large" in srv.log_text()
