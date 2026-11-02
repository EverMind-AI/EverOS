"""T9.1 <- A9 (main journey): a session deleted while its memories are being
extracted never gets them back. Two groups of 25 rounds:

* E1 -- the /add's episode extraction is held when the deletion runs;
* E2 -- the /add returned; the fact extraction (an OME run) is held.

Each group's first round is the positive control: the same window, no
deletion, and the entries do appear. A round counts only if its call was
held; fewer than 20 of 25 is "not manufactured", not a pass.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from .conftest import chat
from .observe import md_entries, wait_idle
from .proxy import EPISODE, FACTS, Rule, llm_call

pytestmark = pytest.mark.acceptance

ROUNDS = 25


def _entries(srv: Any, session: str) -> list[dict[str, Any]]:
    return [e for e in md_entries(srv.root) if e["session_id"] == session]


def _watch_empty(srv: Any, session: str, seconds: int = 20) -> None:
    for _ in range(seconds):
        found = _entries(srv, session)
        assert not found, ("written back after the deletion", found)
        time.sleep(1)


def _held_add(srv: Any, session: str, token: str, out: dict[str, Any]) -> bool:
    """Send the /add (then the flush, if the add only buffered) in the
    background; True once its episode extraction is held."""
    t = threading.Thread(
        target=lambda: out.setdefault("r", srv.add(session, chat("u1", token))),
        daemon=True,
    )
    t.start()
    out["t"] = t
    if srv.proxy.wait_held("llm", timeout=20):
        return True
    t.join(60)
    t = threading.Thread(
        target=lambda: out.__setitem__("r", srv.flush(session)), daemon=True
    )
    t.start()
    out["t"] = t
    return srv.proxy.wait_held("llm", timeout=40)


def test_t9_1_e1_episode_extraction_in_flight(server: Any) -> None:
    srv = server
    hits = 0
    for r in range(ROUNDS):
        session, token = f"e1s{r}", f"gnu{r}q"
        rule = srv.proxy.add(Rule("llm", llm_call(EPISODE, token), "hold", times=1))
        out: dict[str, Any] = {}
        if not _held_add(srv, session, token, out):
            rule.release.set()
            out["t"].join(120)
            srv.proxy.clear()
            continue
        hits += 1
        if r == 0:  # positive control: no deletion, the episode appears
            rule.release.set()
            out["t"].join(120)
            assert _entries(srv, session), "control: the held extraction writes"
            srv.proxy.clear()
            continue
        assert srv.delete(session_id=session).status_code == 200
        rule.release.set()
        out["t"].join(120)
        assert out["r"].status_code == 200
        _watch_empty(srv, session)
        srv.proxy.clear()
    wait_idle(srv.base, srv.root)
    for r in range(1, ROUNDS):
        assert not _entries(srv, f"e1s{r}")
    assert hits >= 20, f"window hit {hits}/{ROUNDS}: not manufactured"


def test_t9_1_e2_fact_extraction_in_flight(server: Any) -> None:
    srv = server
    hits = 0
    for r in range(ROUNDS):
        session, token = f"e2s{r}", f"elk{r}q"
        rule = srv.proxy.add(Rule("llm", llm_call(FACTS, token), "hold", times=1))
        assert srv.add(session, chat("u1", token)).status_code == 200
        assert srv.flush(session).status_code == 200
        if not srv.proxy.wait_held("llm", timeout=60):
            rule.release.set()
            srv.proxy.clear()
            continue
        hits += 1
        if r == 0:  # positive control: the held fact extraction writes facts
            rule.release.set()
            wait_idle(srv.base, srv.root)
            kinds = {e["kind"] for e in _entries(srv, session)}
            assert {"episodes", ".atomic_facts"} <= kinds, kinds
            srv.proxy.clear()
            continue
        assert srv.delete(session_id=session).status_code == 200
        rule.release.set()
        _watch_empty(srv, session)
        srv.proxy.clear()
    wait_idle(srv.base, srv.root)
    for r in range(1, ROUNDS):
        assert not _entries(srv, f"e2s{r}")
    assert hits >= 20, f"window hit {hits}/{ROUNDS}: not manufactured"
