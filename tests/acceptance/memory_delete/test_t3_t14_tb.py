"""T3.1, T14.1, TB2.1, TB3.1, TB10.1, TB12.1 on a real server."""

from __future__ import annotations

from typing import Any

import pytest

from .observe import (
    lance_ids,
    logical_dump,
    md_entries,
    nothing_deleted_snapshot,
    rows,
    system_db,
    wait_converged,
)
from .seed import seed_session_state, seed_world, write_episode

pytestmark = pytest.mark.acceptance

_U1 = "default_app/default_project/users/u1"


def _session_rows(srv: Any, session: str) -> dict[str, int]:
    db = system_db(srv.root)
    return {
        t: rows(db, f"SELECT COUNT(*) FROM {t} WHERE session_id = ?", session)[0][0]
        for t in ("memcell", "unprocessed_buffer", "conversation_status")
    }


def _inbox(srv: Any) -> dict[str, str]:
    db = system_db(srv.root)
    return dict(rows(db, "SELECT request_id, payload_json FROM memorize_inbox"))


_INBOX = {
    "failed_s1": '{"messages": ["the original words"]}',
    "queued_s9": '{"messages": ["other"]}',
}


def test_t3_1_session_delete_clears_the_conversation_state(make_server: Any) -> None:
    """A3: only session_id clears memcells, both buffer tracks, the status and
    the failed request's body; a scope with an owner leaves them."""
    srv = make_server()
    seed_world(srv.root)
    seed_session_state(srv.root, "s1")
    srv.start()
    assert _session_rows(srv, "s1") == {
        "memcell": 1,
        "unprocessed_buffer": 2,
        "conversation_status": 1,
    }
    assert _inbox(srv) == _INBOX

    assert srv.delete(user_id="u1", session_id="s1").status_code == 200
    assert _session_rows(srv, "s1") == {
        "memcell": 1,
        "unprocessed_buffer": 2,
        "conversation_status": 1,
    }
    assert _inbox(srv) == _INBOX
    assert srv.delete(session_id="s1").status_code == 200

    assert _session_rows(srv, "s1") == {
        "memcell": 0,
        "unprocessed_buffer": 0,
        "conversation_status": 0,
    }
    assert _inbox(srv) == {"failed_s1": "", "queued_s9": _INBOX["queued_s9"]}


def test_t14_1_whole_owner_and_escapes(make_server: Any) -> None:
    """A14 / A6: a whole user, then a whole agent; path-like ids refused."""
    srv = make_server()
    seeded = seed_world(srv.root)
    srv.start()
    wait_converged(srv.base)
    a1 = srv.root / "default_app/default_project/agents/a1"
    skills = [d for d in (a1 / "skills").iterdir() if d.name.startswith("skill_")]
    assert len(skills) == 2, skills  # counted here, independently of the server
    before = nothing_deleted_snapshot(srv.root)
    for bad in ({"user_id": ".."}, {"user_id": "a/../b"}, {"agent_id": "../users"}):
        assert srv.delete(**bad).status_code == 422
    assert nothing_deleted_snapshot(srv.root) == before

    user = srv.delete(user_id="u1").json()["data"]
    agent = srv.delete(agent_id="a1").json()["data"]

    assert user == {"filters": ["user_id"], "count": seeded.count("u1") + 1}
    assert agent == {"filters": ["agent_id"], "count": seeded.count("a1") + len(skills)}
    assert not [p for p in a1.rglob("*") if "skills" in p.parts or p.name == "SKILL.md"]
    left = list((srv.root / _U1).rglob("*"))
    assert all(p.is_dir() or p.suffix == ".md" for p in left)
    assert [e for e in md_entries(srv.root) if e["owner"] in ("u1", "a1")] == []
    assert not (srv.root / _U1 / "user.md").exists()
    db = system_db(srv.root)
    assert rows(db, "SELECT COUNT(*) FROM cluster WHERE owner_id IN ('u1','a1')") == [
        (0,)
    ]
    assert rows(db, "SELECT COUNT(*) FROM reflection_report WHERE owner_id='u1'") == [
        (0,)
    ]
    owned = {
        i
        for t in ("episode", "agent_case", "user_profile", "agent_skill")
        for i in lance_ids(srv.root, t)
        if i.startswith(("u1", "a1"))
    }
    assert owned == set()
    assert [e for e in md_entries(srv.root) if e["owner"] == "u2"]  # untouched


def test_tb2_1_unknown_and_other_project_ids(make_server: Any) -> None:
    """B2: three kinds of miss and duplicates answer success; the same owner
    and entry id in another project stays."""
    srv = make_server()
    here = write_episode(srv.root, "u1", "2025-10-01")
    there = write_episode(srv.root, "u1", "2025-10-01", project="other")
    assert here == there
    srv.start()

    resp = srv.delete_by_ids(
        [
            f"nobody_{here}",  # no such owner
            "u1_ep_20240101_0001",  # no such file
            "u1_ep_20251001_0099",  # file, no such entry
            f"u1_{here}",
            f"u1_{here}",
        ]
    )

    assert resp.status_code == 200
    assert resp.json()["data"] == {"memory_type": "episode", "success": True}
    entries = {(e["path"].split("/")[1], e["entry_id"]) for e in md_entries(srv.root)}
    assert entries == {("other", there)}
    # The whole request again: the same answer, nothing else moves.
    again = srv.delete_by_ids(
        [f"nobody_{here}", "u1_ep_20240101_0001", "u1_ep_20251001_0099", f"u1_{here}"]
    )
    assert again.status_code == 200
    assert again.json()["data"] == {"memory_type": "episode", "success": True}
    assert {(e["path"].split("/")[1], e["entry_id"]) for e in md_entries(srv.root)} == {
        ("other", there)
    }


def test_tb3_1_one_episode(make_server: Any) -> None:
    """B3: the episode leaves md and the index; nothing else."""
    srv = make_server()
    seed_world(srv.root)
    srv.start()
    wait_converged(srv.base)
    target = next(
        e
        for e in md_entries(srv.root)
        if e["owner"] == "u1" and e["kind"] == "episodes"
    )
    srv.post("/cascade/quiesce", {})
    before = md_entries(srv.root)
    index_before = lance_ids(srv.root, "episode")
    tables_before = logical_dump(srv.root)
    profile = srv.root / "default_app/default_project/users/u1/user.md"
    profile_before = profile.read_bytes()

    assert srv.delete_by_ids([f"u1_{target['entry_id']}"]).status_code == 200

    after = md_entries(srv.root)
    gone = [e for e in before if e not in after]
    assert [(e["kind"], e["entry_id"]) for e in gone] == [
        ("episodes", target["entry_id"])
    ]
    episode = f"u1_{target['entry_id']}"
    assert episode in index_before, "positive control: indexed"
    assert lance_ids(srv.root, "episode") == index_before - {episode}
    tables_after = logical_dump(srv.root)
    for table, before_rows in tables_before.items():
        if table not in ("cluster", "cluster_member"):  # B3 unclusters it
            assert tables_after[table] == before_rows, table
    assert profile.read_bytes() == profile_before


def test_tb10_1_merged_episode_lists_with_null_session(make_server: Any) -> None:
    """B10: /get lists a Reflection-merged episode (no session) as null."""
    srv = make_server()
    write_episode(srv.root, "u1", "2025-10-01", session="s1")
    write_episode(
        srv.root,
        "u1",
        "2025-10-02",
        session=None,
        parent_type="cluster",
        parent_id="cl_1",
    )
    srv.start()
    wait_converged(srv.base)

    resp = srv.post("/memory/get", {"user_id": "u1", "memory_type": "episode"})

    assert resp.status_code == 200, resp.text
    sessions = sorted(str(e["session_id"]) for e in resp.json()["data"]["episodes"])
    assert sessions == ["None", "s1"]


def test_tb12_1_log_names_hits_and_skips_never_the_note(make_server: Any) -> None:
    """B12: one log line with the reason, hits and skips; neither the note nor
    the deleted text appears anywhere in the server log."""
    srv = make_server()
    hit = write_episode(srv.root, "u1", "2025-10-01", text="secret-body-7f3a")
    srv.start()

    srv.delete_by_ids(
        [f"u1_{hit}", "u1_ep_20240101_0001"],
        reason={"code": "privacy", "note": "secret-note-9c2e"},
    )
    srv.stop()

    log = srv.log_text()
    lines = [line for line in log.splitlines() if "memory_delete_by_ids" in line]
    assert len(lines) == 1, lines
    assert f"u1_{hit}" in lines[0] and "u1_ep_20240101_0001" in lines[0]
    assert "privacy" in lines[0]
    assert "secret-note-9c2e" not in log and "secret-body-7f3a" not in log
