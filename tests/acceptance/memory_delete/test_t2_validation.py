"""T2.1 <- A2 / TB1.1 <- B1: invalid requests are 422 and delete nothing."""

from __future__ import annotations

import pytest

from .conftest import Server, chat
from .observe import md_entries, nothing_deleted_snapshot

pytestmark = pytest.mark.acceptance

_DELETE = [
    {},
    {"app_id": "default"},
    {"user_id": "u1", "agent_id": "a1"},
    {"user_id": "a/b"},
    {"user_id": ".."},
    {"session_id": "s1", "surprise": True},
]


def _by_ids(ok: str) -> list[dict[str, object]]:
    """Each invalid request carries ``ok``, the id of a stored episode."""
    return [
        {"memory_type": "agent_case", "memory_ids": [ok]},
        {"memory_type": "episode", "memory_ids": []},
        {"memory_type": "episode", "memory_ids": [ok] * 51},
        {
            "memory_type": "episode",
            "memory_ids": [ok, "_ep_20250101_0001"],
        },
        {
            "memory_type": "episode",
            "memory_ids": [ok, "u1_ep_20251399_0001"],
        },
        {
            "memory_type": "episode",
            "memory_ids": [ok, "a/b_ep_20250101_0001"],
        },
        {
            "memory_type": "episode",
            "memory_ids": [ok, ".._ep_20250101_0001"],
        },
        {
            "memory_type": "episode",
            "memory_ids": [ok],
            "reason": {"code": "nope"},
        },
        {
            "memory_type": "episode",
            "memory_ids": [ok],
            "reason": {"code": "other", "note": "x" * 257},
        },
        {"memory_type": "episode", "memory_ids": [ok], "extra": 1},
    ]


def test_t2_1_and_tb1_1_invalid_requests_are_422_and_delete_nothing(
    server: Server,
) -> None:
    assert server.add("s1", chat("u1", "zebra7")).status_code == 200
    assert server.flush("s1").status_code == 200
    before = nothing_deleted_snapshot(server.root)
    stored = [e for e in md_entries(server.root) if e["kind"] == "episodes"]
    assert stored, "precondition: something to (not) delete"
    by_ids = _by_ids(f"u1_{stored[0]['entry_id']}")

    codes = [server.post("/memory/delete", body).status_code for body in _DELETE]
    codes += [server.post("/memory/delete_by_ids", body).status_code for body in by_ids]

    assert codes == [422] * (len(_DELETE) + len(by_ids)), codes
    assert nothing_deleted_snapshot(server.root) == before
