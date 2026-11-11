"""T17.1 <- A17: an existing user journey has the same shape before and after.

The same script runs against the base commit's real checkout and against this
branch: sync and async /add, /flush, a multi-round search, /get, a
Reflection trigger. Compared per step: the status code and the set of key
paths of the response (values differ with a real model), and the frontmatter
keys of the episode files. The baseline is the merge base with
``origin/main``, unpacked with ``git archive`` and synced into its own venv;
``EVEROS_ACC_BASELINE`` points at an existing checkout instead.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from .conftest import chat
from .observe import frontmatter, wait_idle

pytestmark = pytest.mark.acceptance


def _paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        out = {prefix} if prefix else set()
        for k, v in value.items():
            out |= _paths(v, f"{prefix}.{k}" if prefix else k)
        return out
    if isinstance(value, list):
        if not value:  # a real model may extract nothing: compare no item keys
            return {prefix, f"{prefix}[]?"}
        out = {prefix}
        for item in value[:1]:
            out |= _paths(item, f"{prefix}[]")
        return out
    return {prefix}


def _comparable(a: frozenset[str], b: frozenset[str]) -> tuple[set[str], set[str]]:
    """Drop item keys under a list that is empty on either side."""
    empty = {k[: -len("[]?")] for k in a | b if k.endswith("[]?")}

    def keep(keys: frozenset[str]) -> set[str]:
        return {k for k in keys if not any(k.startswith(f"{e}[]") for e in empty)}

    return keep(a), keep(b)


def _journey(srv: Any) -> list[tuple[str, int, frozenset[str]]]:
    steps: list[tuple[str, int, frozenset[str]]] = []

    def record(name: str, resp: httpx.Response) -> dict[str, Any]:
        body = resp.json()
        body.pop("request_id", None)
        steps.append((name, resp.status_code, frozenset(_paths(body))))
        return body

    record("add", srv.add("j1", chat("u1", "ferret11", assistant="a1")))
    record("flush", srv.flush("j1"))
    queued = srv.add("j2", chat("u1", "ferret12"), async_mode=True)
    record("add_async", queued)
    rid = queued.json()["request_id"]
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        got = httpx.get(f"{srv.base}/api/v2/tasks/{rid}", timeout=10)
        if got.json().get("data", {}).get("status") in ("success", "failed"):
            break
        time.sleep(1)
    record("task", got)
    wait_idle(srv.base, srv.root)
    record(
        "search",
        srv.post(
            "/memory/search",
            {"user_id": "u1", "query": "ferret11", "method": "llm_multiround"},
        ),
    )
    for kind in ("episode", "profile"):
        record(
            f"get_{kind}",
            srv.post("/memory/get", {"user_id": "u1", "memory_type": kind}),
        )
    record(
        "get_agent_case",
        srv.post("/memory/get", {"agent_id": "a1", "memory_type": "agent_case"}),
    )
    record(
        "reflect", srv.post("/ome/trigger", {"name": "reflect_episodes", "force": True})
    )
    wait_idle(srv.base, srv.root)
    keys: set[str] = set()
    for md in srv.root.rglob("episodes/*.md"):
        keys |= set(frontmatter(srv.root, str(md.relative_to(srv.root))))
    steps.append(("episode_frontmatter", 0, frozenset(keys)))
    return steps


def _baseline(into: Path) -> Path:
    """The merge base's tree with its own venv; its ``everos``."""
    repo = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "HEAD", "origin/main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    into.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", commit],
        check=True,
        capture_output=True,
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(into)], input=archive, check=True)
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    subprocess.run(
        ["uv", "sync", "--quiet"], cwd=into, env=env, check=True, timeout=600
    )
    return into


def test_t17_1_same_journey_same_shape(make_server: Any, tmp_path: Path) -> None:
    given = os.environ.get("EVEROS_ACC_BASELINE")
    checkout = Path(given) if given else _baseline(tmp_path / "base_tree")
    base = make_server("baseline")
    base.everos = checkout / ".venv" / "bin" / "everos"
    (base.root / "everos.toml").unlink(missing_ok=True)
    base.start()
    before = _journey(base)
    gone = base.post("/memory/delete", {"session_id": "j1"}).status_code
    assert gone == 404, "positive control: the baseline is the old code"
    base.stop()

    branch = make_server("branch")
    branch.start()
    after = _journey(branch)

    for side, steps in (("baseline", before), ("branch", after)):
        print(f"T17.1 {side}: " + " ".join(f"{n}={c}" for n, c, _ in steps))
    # Code 0 marks the markdown step, which is no HTTP call.
    failed = [(n, c) for n, c, _ in before + after if c and not 200 <= c < 300]
    assert not failed, failed
    assert [s[0] for s in after] == [s[0] for s in before]
    for (name, code_a, keys_a), (_, code_b, keys_b) in zip(before, after, strict=True):
        assert code_b == code_a, (name, code_a, code_b)
        a, b = _comparable(keys_a, keys_b)
        assert b == a, (name, sorted(a ^ b))
    for name, _, keys in before + after:  # positive control on both sides
        if name in ("search", "get_episode"):
            assert "data.episodes[]?" not in keys, f"{name} found no episode"
