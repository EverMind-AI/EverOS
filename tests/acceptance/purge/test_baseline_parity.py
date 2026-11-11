"""T17.1: with no purge running, existing uses behave as on origin/main (A17).

One workload, one per row of spec §5.2's existing-uses table, runs against
a server of the baseline checkout and of this branch,
both over the same fake upstream; the normalised responses and stores of
the two runs must be equal.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from tests.acceptance.memory_delete.observe import md_entries
from tests.acceptance.memory_delete.proxy import Rule
from tests.fakes.llm import UNUSABLE_REPLY

from .conftest import PurgeServer
from .fake_upstream import Override
from .journey import Evidence, evidence_base
from .observe import INDEX_TABLES, index_ids, scope_counts, wait_converged, wait_idle

pytestmark = pytest.mark.acceptance

# The ``everos`` of an origin/main checkout, e.g. a worktree:
#   git worktree add ../everos-base origin/main && (cd ../everos-base && uv sync)
#   PURGE_BASELINE_EVEROS=../everos-base/.venv/bin/everos make acceptance-purge
BASELINE = os.environ.get("PURGE_BASELINE_EVEROS")
WHERE = {"app_id": "a1", "project_id": "p1"}
T0 = 1_762_041_600_000
_VOLATILE = {"request_id", "took_ms", "latency_ms", "elapsed_ms", "duration_ms"}
_PATTERNS = [
    (
        re.compile(
            r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?"
        ),
        "<ts>",
    ),
    (re.compile(r"\bmc_[0-9a-f]{6,}\b"), "mc_<id>"),
    (re.compile(r"d_[0-9a-f]{12,32}"), "d_<id>"),
    (re.compile(r"\bcl_[0-9a-f]{6,}\b"), "cl_<id>"),
    (re.compile(r"\b[0-9a-f]{32}\b"), "<hex>"),
    (re.compile(r"\b1[78]\d{11}\b"), "<ms>"),
]


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in _VOLATILE}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


def _norm(obj: Any, root: Path | None = None) -> str:
    text = json.dumps(_strip(obj), sort_keys=True, default=str)
    if root is not None:  # each run has its own root directory
        for form in {str(root), str(root.resolve())}:
            text = text.replace(form, "<root>")
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def _msgs(user: str, word: str, base: int) -> list[dict[str, Any]]:
    return [
        {
            "sender_id": user,
            "role": "user",
            "timestamp": base + i * 2000,
            "content": f"I want to remember {word}: the {word} museum, item {i}.",
        }
        for i in range(3)
    ] + [
        {
            "sender_id": "ag1",
            "role": "assistant",
            "timestamp": base + 7000,
            "content": f"Noted, the {word} museum.",
        }
    ]


def _body(resp: Any) -> Any:
    try:
        return resp.json()
    except ValueError:
        return resp.text


def _workload(srv: PurgeServer, proxy: Any, fake: Any) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []

    def rec(step: str, resp: Any) -> Any:
        out.append(
            (step, _norm({"status": resp.status_code, "body": _body(resp)}, srv.root))
        )
        return resp

    # Memory writes (sync).
    for i, user in enumerate(("u1", "u2", "u3")):
        session = f"{user}-s"
        rec(
            f"add {user}",
            srv.post(
                "/memory/add",
                {
                    "session_id": session,
                    "messages": _msgs(user, "zqparity", T0 + i * 60_000),
                    **WHERE,
                },
            ),
        )
        rec(
            f"flush {user}", srv.post("/memory/flush", {"session_id": session, **WHERE})
        )
    # Async add and /tasks, one of them failing.
    fake.override(Override(pattern="zqbadword", reply=UNUSABLE_REPLY))
    ids = []
    for name, word in (("ok", "zqparity"), ("bad", "zqbadword")):
        resp = rec(
            f"async {name}",
            srv.post(
                "/memory/add",
                {
                    "session_id": f"async-{name}",
                    "messages": _msgs("u4", word, T0 + 600_000),
                    "async_mode": True,
                    **WHERE,
                },
            ),
        )
        ids.append(resp.json()["request_id"])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        tasks = [srv.call("GET", f"/tasks/{r}").json()["data"] for r in ids]
        if all(t["status"] in ("success", "failed") for t in tasks):
            break
        time.sleep(0.5)
    for name, task in zip(("ok", "bad"), tasks, strict=True):
        out.append(
            (
                f"task {name}",
                _norm({"status": task["status"], "error_code": task["error_code"]}),
            )
        )
    wait_idle(srv.base, srv.root, timeout=300)
    # A second turn per user. Profile extraction reads the owner's episodes
    # from the index; on the first turn a fake upstream can reach it before
    # the index has them (it then skips the profile), on either server. By
    # now the index holds them, so every profile is extracted on both sides.
    for i, user in enumerate(("u1", "u2", "u3", "u4")):
        session = f"{user}-s2"
        rec(
            f"second turn {user}",
            srv.post(
                "/memory/add",
                {
                    "session_id": session,
                    "messages": _msgs(user, "zqparity", T0 + 1_200_000 + i * 60_000),
                    **WHERE,
                },
            ),
        )
        rec(
            f"second flush {user}",
            srv.post("/memory/flush", {"session_id": session, **WHERE}),
        )
    wait_idle(srv.base, srv.root, timeout=300)
    # Reads.
    rec(
        "search",
        srv.post(
            "/memory/search",
            {"query": "museum", "user_id": "u1", "method": "keyword", **WHERE},
        ),
    )
    got = rec(
        "get",
        srv.post("/memory/get", {"user_id": "u1", "memory_type": "episode", **WHERE}),
    )
    rec("tasks list", srv.call("GET", "/tasks"))
    # Knowledge writes and reads.
    created = rec(
        "knowledge create",
        srv.call(
            "POST",
            "/knowledge/documents",
            data={"title": "Hiking", **WHERE},
            files={
                "file": (
                    "g.txt",
                    b"Mountain hiking safety guide zqparity.",
                    "text/plain",
                )
            },
        ),
    )
    doc_id = created.json()["data"]["doc_id"]
    rec(
        "knowledge put",
        srv.call(
            "PUT",
            f"/knowledge/documents/{doc_id}",
            data={"title": "Hiking 2", **WHERE},
            files={
                "file": ("g.txt", b"Mountain hiking guide zqparity two.", "text/plain")
            },
        ),
    )
    rec(
        "knowledge patch",
        srv.call(
            "PATCH", f"/knowledge/documents/{doc_id}", {"title": "Hiking 3", **WHERE}
        ),
    )
    rec("knowledge list", srv.call("GET", "/knowledge/documents", params=WHERE))
    rec("categories", srv.call("GET", "/knowledge/categories", params=WHERE))
    wait_idle(srv.base, srv.root, timeout=300)
    rec("knowledge search", srv.post("/knowledge/search", {"query": "hiking", **WHERE}))
    rec(
        "knowledge delete",
        srv.call("DELETE", f"/knowledge/documents/{doc_id}", params=WHERE),
    )
    # Deletions.
    episodes = (got.json().get("data") or {}).get("episodes") or []
    if episodes:
        rec(
            "delete_by_ids",
            srv.post(
                "/memory/delete_by_ids",
                {
                    "memory_type": "episode",
                    "memory_ids": [episodes[0]["id"]],
                    "user_id": "u1",
                    **WHERE,
                },
            ),
        )
    rec("delete owner", srv.post("/memory/delete", {"user_id": "u2", **WHERE}))
    # Deletion racing a log projection: the fence keeps its rows out.
    hold = proxy.add(Rule(target="emb", pattern="zqfence", mode="hold"))
    rec(
        "add for fence",
        srv.post(
            "/memory/add",
            {
                "session_id": "fence",
                "messages": _msgs("u5", "zqfence", T0 + 900_000),
                **WHERE,
            },
        ),
    )
    rec("flush for fence", srv.post("/memory/flush", {"session_id": "fence", **WHERE}))
    proxy.wait_held("emb", timeout=60)
    rec(
        "delete while projecting",
        srv.post("/memory/delete", {"user_id": "u5", **WHERE}),
    )
    hold.release.set()
    wait_converged(srv.base, timeout=240)
    rec(
        "ome trigger",
        srv.post(
            "/ome/trigger", {"name": "reflect_episodes", "force": True, "timeout": 60}
        ),
    )
    wait_idle(srv.base, srv.root, timeout=300)
    # A deletion cut short, finished at startup; the CLI with the server down.
    srv.stop()
    trash = srv.root / ".tmp" / "delete-parity"
    (trash / "owner").mkdir(parents=True)
    (trash / "scope.json").write_text(
        json.dumps(
            {"app_id": "a1", "project_id": "p1", "owner_type": "user", "owner_id": "u9"}
        )
    )
    sync = subprocess.run(
        [str(srv.everos), "cascade", "sync", "--root", str(srv.root)],
        env=srv.env(),
        capture_output=True,
        text=True,
    )
    out.append(("cascade sync (server down)", _norm({"code": sync.returncode})))
    srv.start()
    out.append(("startup swept trash", _norm({"left": trash.exists()})))
    wait_idle(srv.base, srv.root, timeout=300)
    rec("quiesce", srv.post("/cascade/quiesce", {}))
    # The stores.
    out.append(
        (
            "md entries",
            _norm(
                sorted(
                    (e["kind"], e["owner"], e["session_id"] or "", e["entry_id"])
                    for e in md_entries(srv.root)
                )
            ),
        )
    )
    out.append(
        (
            "index ids",
            _norm(
                {t: sorted(index_ids(srv.root, t, "a1", "p1")) for t in INDEX_TABLES}
            ),
        )
    )
    out.append(("counts", _norm(scope_counts(srv.root, "a1", "p1"))))
    return out


def test_t17_1_existing_uses_match_the_baseline(
    make_purge_server: Any, purge_proxy: Any, fake: Any, tmp_path: Path
) -> None:
    if BASELINE is None:
        pytest.skip("set PURGE_BASELINE_EVEROS (see the comment at BASELINE)")
    if not Path(BASELINE).exists():
        pytest.fail(f"PURGE_BASELINE_EVEROS does not exist: {BASELINE}")
    base = make_purge_server("base", everos=Path(BASELINE))
    ours = make_purge_server("ours")
    ev = Evidence("T17.1", evidence_base(tmp_path), [base, ours])
    try:
        before = _workload(base, purge_proxy, fake)
        purge_proxy.clear()
        after = _workload(ours, purge_proxy, fake)
        ev.json("baseline", before)
        ev.json("branch", after)
        diff = [
            (b[0], b[1], a[1]) for b, a in zip(before, after, strict=True) if b != a
        ]
        ev.json("diff", diff)
        assert [s for s, _ in before] == [s for s, _ in after]
        assert diff == [], diff
    finally:
        ev.close()
