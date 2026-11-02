"""T10.1: ten minutes of mixed traffic against a slow, jittery model.

Three users /add (sync and async), each pausing 3-8 s between requests the
way a client does (an async /add answers at once; without the pause it only
fills the per-session queue and collects 429s); a deleter mixes the four
deletion kinds;
Reflection is triggered every minute; the proxy delays every model and
embedding call by 0-5 s; /health is probed every second. Afterwards every
queued request and every OME run must reach an end.
"""

from __future__ import annotations

import random
import threading
import time
from typing import Any

import httpx
import pytest

from .conftest import chat
from .observe import bucket_edge, histogram, md_entries, metrics, rows, wait_idle
from .proxy import Rule

pytestmark = pytest.mark.acceptance

_SECONDS = 600
_USERS = ("m1", "m2", "m3")
_TABLES = 4  # episode, atomic_fact, foresight, agent_case
_REWRITE_S = 4.6
"""Ten times the slowest plain session deletion measured in T18.1."""
_ADD_CALLS = 2
"""Model calls inside one sync /add (boundary + episode); the test re-counts
them from the proxy log before the run."""
_BOUND = {
    "session": 10 + 15 * _TABLES + _REWRITE_S,
    "user": 15 * _TABLES + _REWRITE_S,
    "user+session": 15 * _TABLES + _REWRITE_S,
    "by_id": 15 * _TABLES + _REWRITE_S,
}
_ADD_BOUND = _ADD_CALLS * 5 * 4


def _llm_calls(srv: Any) -> int:
    return sum(1 for target, _, _ in list(srv.proxy.log) if target == "llm")


def test_t10_1_mixed_traffic(make_server: Any) -> None:
    srv = make_server()
    (srv.root / "ome.toml").write_text(
        "[strategies.reflect_episodes]\nenabled = true\n"
    )
    srv.start()
    before = _llm_calls(srv)
    assert srv.add("calibrate", chat("m0", "calib1q")).status_code == 200
    assert _llm_calls(srv) - before == _ADD_CALLS, "calls inside one sync /add"
    wait_idle(srv.base, srv.root)

    srv.proxy.add(Rule("llm", ".", "jitter", delay=5))
    srv.proxy.add(Rule("emb", ".", "jitter", delay=5))
    stop = threading.Event()
    adds: list[tuple[str, int, float]] = []
    deletes: list[tuple[str, int, float]] = []
    probes: list[tuple[int, float]] = []
    accepted: list[str] = []
    rng = random.Random(10)

    def adder(user: str, seed: int) -> None:
        r = random.Random(seed)
        n = 0
        while not stop.is_set():
            asynchronous = r.random() < 0.5
            started = time.monotonic()
            try:
                resp = srv.add(
                    f"{user}-s{r.randrange(4)}",
                    chat(user, f"otter{n}{user}"),
                    async_mode=asynchronous,
                )
            except httpx.HTTPError as exc:
                adds.append((f"error {exc!r}", 0, time.monotonic() - started))
            else:
                took = time.monotonic() - started
                kind = "async" if asynchronous else "sync"
                adds.append((kind, resp.status_code, took))
                if resp.status_code == 202:
                    accepted.append(resp.json()["request_id"])
                n += 1
            stop.wait(r.uniform(3, 8))

    def deleter() -> None:
        while not stop.wait(rng.uniform(5, 15)):
            user = rng.choice(_USERS)
            session = f"{user}-s{rng.randrange(4)}"
            kind = rng.choice(list(_BOUND))
            path, body = (
                "/memory/delete",
                {
                    "session": {"session_id": session},
                    "user": {"user_id": user},
                    "user+session": {"user_id": user, "session_id": session},
                    "by_id": {},
                }[kind],
            )
            if kind == "by_id":
                ids = [
                    f"{e['owner']}_{e['entry_id']}"
                    for e in md_entries(srv.root)
                    if e["owner"] == user and e["kind"] == "episodes"
                ]
                if not ids:
                    continue
                path = "/memory/delete_by_ids"
                body = {"memory_type": "episode", "memory_ids": [rng.choice(ids)]}
            started = time.monotonic()
            status = srv.post(path, body).status_code
            deletes.append((kind, status, time.monotonic() - started))

    def reflector() -> None:
        while not stop.wait(60):
            srv.post("/ome/trigger", {"name": "reflect_episodes", "force": True})

    def prober() -> None:
        while not stop.wait(1):
            started = time.monotonic()
            status = httpx.get(f"{srv.base}/health", timeout=10).status_code
            probes.append((status, time.monotonic() - started))

    workers = [
        threading.Thread(target=adder, args=(u, i), daemon=True)
        for i, u in enumerate(_USERS)
    ] + [threading.Thread(target=f, daemon=True) for f in (deleter, reflector, prober)]
    for w in workers:
        w.start()
    time.sleep(_SECONDS)
    stop.set()
    for w in workers:
        w.join(300)
    srv.proxy.clear()

    deadline = time.monotonic() + 600
    unfinished = list(accepted)
    while unfinished and time.monotonic() < deadline:
        unfinished = [
            rid
            for rid in unfinished
            if httpx.get(f"{srv.base}/api/v2/tasks/{rid}", timeout=10).json()["data"][
                "status"
            ]
            not in ("success", "failed")
        ]
        time.sleep(2)
    wait_idle(srv.base, srv.root, timeout=600)
    ome = srv.root / ".index" / "sqlite" / "ome.db"
    running = rows(ome, "SELECT COUNT(*) FROM run_record WHERE status='running'")
    text = metrics(srv.base)

    by_kind: dict[str, list[tuple[int, float]]] = {}
    for kind, status, took in deletes:
        by_kind.setdefault(kind, []).append((status, took))
    sync_max = max((t for k, _, t in adds if k == "sync"), default=0.0)
    print(
        f"adds: {len(adds)} (accepted async {len(accepted)},"
        f" 429 {sum(1 for _, status, _ in adds if status == 429)}),"
        f" sync max {sync_max:.1f}s"
    )
    for kind, got in sorted(by_kind.items()):
        codes, slowest = sorted({s for s, _ in got}), max(t for _, t in got)
        print(f"delete {kind}: n={len(got)} codes={codes} max={slowest:.2f}s")
    print(f"health: n={len(probes)} max={max(t for _, t in probes):.3f}s")
    for kind in ("session", "owner"):
        cum = histogram(text, "everos_lock_wait_seconds", kind=kind)
        n = cum.get(float("inf"), 0)
        print(f"lock wait {kind}: n={n:.0f} max<={bucket_edge(cum, 1.0)}")

    assert len(by_kind) == len(_BOUND), f"not every deletion kind ran: {by_kind}"
    for kind, status, took in deletes:
        assert status in (200, 503), (kind, status)
        assert took <= _BOUND[kind], (kind, took)
    assert not [a for a in adds if a[0].startswith("error")], adds
    for kind, status, took in adds:
        assert status in (200, 202, 429), (kind, status)  # 429: queue backpressure
        if kind == "sync":
            assert took <= _ADD_BOUND, took
    assert all(status == 200 and took <= 1.0 for status, took in probes)
    assert not unfinished, f"requests never finished: {unfinished}"
    assert running == [(0,)], running
