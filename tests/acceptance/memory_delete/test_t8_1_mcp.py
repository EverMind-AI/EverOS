"""T8.1 <- A8: everos-mcp's forget_session against this server."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from .conftest import chat
from .observe import md_entries, wait_idle
from .proxy import BOUNDARY, Rule, llm_call

pytestmark = pytest.mark.acceptance

_DRIVER = Path(__file__).with_name("mcp_driver.py")


def _driver(srv: Any, session: str, *args: str) -> subprocess.Popen[str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("EVEROS_") and k != "VIRTUAL_ENV"
    }
    env |= {
        "EVEROS_BASE_URL": srv.base,
        "EVEROS_SESSION_ID": session,
        "EVEROS_USER_ID": "u_mcp",
    }
    return subprocess.Popen(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "everos-mcp==0.1.1",
            "python",
            "-I",
            str(_DRIVER),
            *args,
        ],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _session_entries(srv: Any, session: str) -> int:
    return sum(1 for e in md_entries(srv.root) if e["session_id"] == session)


def test_t8_1_forget_session_deletes_what_it_says(server: Any) -> None:
    srv = server
    proc = _driver(srv, "mcp-s1", "full", "jackal81")
    assert proc.stdout is not None and proc.stdin is not None
    lines = []
    for line in proc.stdout:
        lines.append(line.strip())
        if line.strip() == "ADDED":
            break
    assert all(not ln.startswith("ADD: True") for ln in lines), lines
    wait_idle(srv.base, srv.root)
    before = _session_entries(srv, "mcp-s1")
    assert before > 0, "positive control: the adds wrote memories"
    proc.stdin.write("\n")
    proc.stdin.flush()
    rest = [ln.strip() for ln in proc.stdout]
    proc.wait(120)
    print("\n".join(lines + rest))

    found = next(ln for ln in rest if ln.startswith("SEARCH_BEFORE:"))
    assert "jackal81" in found.split(" ", 2)[2], ("positive control", found)
    forget = next(ln for ln in rest if ln.startswith("FORGET:"))
    assert "does not support deleting" not in forget, forget
    assert forget.startswith("FORGET: False"), forget
    assert f"Deleted {before} stored item(s)" in forget, (forget, before)
    assert _session_entries(srv, "mcp-s1") == 0
    search = next(ln for ln in rest if ln.startswith("SEARCH:"))
    assert "jackal81" not in search.split(" ", 2)[2], search


def test_t8_1_busy_session_is_reported_to_the_model(server: Any) -> None:
    srv = server
    srv.proxy.add(Rule("llm", llm_call(BOUNDARY, "jackal90"), "hold"))
    for _ in range(2):
        assert (
            srv.add("mcp-busy", chat("u_mcp", "jackal90"), async_mode=True).status_code
            == 202
        )
    assert srv.proxy.wait_held("llm")

    proc = _driver(srv, "mcp-busy", "busy")
    out, _ = proc.communicate(timeout=180)
    srv.proxy.clear()

    # What reaches the model: an error carrying the server's message (the
    # client does not forward error.code; the code is checked in T4.1).
    forget = next(ln for ln in out.splitlines() if ln.startswith("FORGET:"))
    assert forget.startswith("FORGET: True"), forget
    assert "nothing was deleted, retry later" in forget, forget
