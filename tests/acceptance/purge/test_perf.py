"""T20.1: how long a purge holds writes, with about 10,000 memories (A20).

Run alone (``-k t20``): the timings are wall clock and other servers on the
machine skew them.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from typing import Any

import httpx
import pytest

from tests.acceptance.memory_delete.conftest import chat
from tests.acceptance.memory_delete.proxy import Rule
from tests.acceptance.memory_delete.seed import write_episode

from .journey import Evidence, evidence_base, finish, purge
from .observe import md_counts, wait_idle
from .seed import AGENT, scope_token

pytestmark = pytest.mark.acceptance

PER_SCOPE = 5000
PER_DAY = 50


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


def _held_for(srv: Any, job_id: str) -> tuple[float, list[tuple[float, int]]]:
    """Seconds from the first 409 to the first 200 of a write sent every 0.2 s."""
    started = time.monotonic()
    timeline: list[tuple[float, int]] = []
    while time.monotonic() - started < 300:
        code = srv.post(
            "/memory/add",
            {
                "session_id": f"probe-{time.monotonic_ns()}",
                "app_id": "a2",
                "project_id": "p1",
                "messages": chat("u9", "probe"),
                "defer_extraction": True,
            },
        ).status_code
        timeline.append((round(time.monotonic() - started, 2), code))
        if code == 200 and any(c == 409 for _, c in timeline):
            break
        time.sleep(0.2)
    first_409 = next(t for t, c in timeline if c == 409)
    first_200 = next(t for t, c in timeline if c == 200 and t > first_409)
    return first_200 - first_409, timeline


def test_t20_1_write_hold_is_bounded(
    make_purge_server: Any, purge_proxy: Any, tmp_path: Any
) -> None:
    srv = make_purge_server(start=False)
    ev = Evidence("T20.1", evidence_base(tmp_path), [srv])
    try:
        for app, project in (("a1", "p1"), ("a2", "p1")):
            _write_many(srv.root, app, project)
        srv.start()
        wait_idle(srv.base, srv.root, timeout=1800)
        assert md_counts(srv.root, "a1", "p1")["md.episode"] == PER_SCOPE
        # 1: nothing in flight
        job_id, _ = purge(srv, scope="project", app_id="a1", project_id="p1")
        d0, timeline0 = _held_for(srv, job_id)
        assert finish(srv, job_id)["status"] == "succeeded"
        # 2: a synchronous add of S in flight when the purge starts
        word = f"{scope_token('a1', 'p1')}inflight"
        hold = purge_proxy.add(
            Rule(
                target="llm", pattern=f'(?=.*"_EpisodeReply")(?=.*{word})', mode="hold"
            )
        )
        sent: dict[str, httpx.Response] = {}
        adder = threading.Thread(
            target=lambda: sent.update(
                r=srv.post(
                    "/memory/add",
                    {
                        "session_id": "inflight",
                        "app_id": "a1",
                        "project_id": "p1",
                        "messages": chat("u1", word, assistant=AGENT),
                    },
                )
            )
        )
        adder.start()
        assert purge_proxy.wait_held("llm", timeout=60)
        job_id, _ = purge(srv, scope="project", app_id="a1", project_id="p1")
        d1, timeline1 = _held_for(srv, job_id)
        hold.release.set()
        adder.join(timeout=60)
        finish(srv, job_id)
        ev.json(
            "timings",
            {
                "d0_s": d0,
                "d1_s": d1,
                "bound_d1_s": 2 * d0 + 60,
                "timeline0": timeline0,
                "timeline1": timeline1,
            },
        )
        assert d0 <= 30, d0
        assert d1 <= 2 * d0 + 60, (d1, d0)
    finally:
        ev.close()
