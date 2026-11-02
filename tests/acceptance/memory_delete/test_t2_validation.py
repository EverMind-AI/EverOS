"""T2.1 <- A2 / TB1.1 <- B1: invalid requests are 422 and delete nothing."""

from __future__ import annotations

import pytest

from .conftest import Server, chat
from .observe import nothing_deleted_snapshot

pytestmark = pytest.mark.acceptance

_DELETE = [
    {},
    {"app_id": "default"},
    {"user_id": "u1", "agent_id": "a1"},
    {"user_id": "a/b"},
    {"user_id": ".."},
    {"session_id": "s1", "surprise": True},
]
_BY_IDS = [
    {"memory_type": "atomic_fact", "memory_ids": ["u1_ep_20250101_0001"]},
    {"memory_type": "episode", "memory_ids": []},
    {"memory_type": "episode", "memory_ids": ["u1_ep_20250101_0001"] * 51},
    {
        "memory_type": "episode",
        "memory_ids": ["u1_ep_20250101_0001", "_ep_20250101_0001"],
    },
    {
        "memory_type": "episode",
        "memory_ids": ["u1_ep_20250101_0001", "u1_ep_20251399_0001"],
    },
    {
        "memory_type": "episode",
        "memory_ids": ["u1_ep_20250101_0001", "a/b_ep_20250101_0001"],
    },
    {
        "memory_type": "episode",
        "memory_ids": ["u1_ep_20250101_0001", ".._ep_20250101_0001"],
    },
    {
        "memory_type": "episode",
        "memory_ids": ["u1_ep_20250101_0001"],
        "reason": {"code": "nope"},
    },
    {
        "memory_type": "episode",
        "memory_ids": ["u1_ep_20250101_0001"],
        "reason": {"code": "other", "note": "x" * 257},
    },
    {"memory_type": "episode", "memory_ids": ["u1_ep_20250101_0001"], "extra": 1},
]


def test_t2_1_and_tb1_1_invalid_requests_are_422_and_delete_nothing(
    server: Server,
) -> None:
    assert server.add("s1", chat("u1", "zebra7")).status_code == 200
    assert server.flush("s1").status_code == 200
    before = nothing_deleted_snapshot(server.root)
    entries = server.root.rglob("episodes/*.md")
    assert list(entries), "precondition: something to (not) delete"

    codes = [server.post("/memory/delete", body).status_code for body in _DELETE]
    codes += [
        server.post("/memory/delete_by_ids", body).status_code for body in _BY_IDS
    ]

    assert codes == [422] * (len(_DELETE) + len(_BY_IDS)), codes
    assert nothing_deleted_snapshot(server.root) == before
