"""T1.1 <- A1, A6 (five scopes, five copies of one root) and T6.1 <- A6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from .observe import (
    lance_ids,
    logical_dump,
    md_entries,
    rows,
    system_db,
    wait_converged,
)
from .seed import seed_world

pytestmark = pytest.mark.acceptance

_SPACE = "default_app/default_project"
_DAILY = ("episode", "agent_case")


def _skill_dirs(root: Path) -> list[str]:
    skills = root / _SPACE / "agents" / "a1" / "skills"
    if not skills.is_dir():
        return []
    return sorted(d.name for d in skills.iterdir() if d.name.startswith("skill_"))


def _index(root: Path) -> dict[str, set[str]]:
    return {t: lance_ids(root, t) for t in (*_DAILY, "user_profile", "agent_skill")}


# scope -> (filters, owners/kinds removed, md files gone, table effects)
_SCOPES: dict[str, dict[str, Any]] = {
    "session": {
        "req": {"session_id": "s1"},
        "filters": ["session_id"],
        "gone": lambda e: e["session_id"] == "s1",
        "extra_count": 0,
    },
    "user": {
        "req": {"user_id": "u1"},
        "filters": ["user_id"],
        "gone": lambda e: e["owner"] == "u1",
        "extra_count": 1,  # user.md
        "profile_gone": True,
    },
    "user_session": {
        "req": {"user_id": "u1", "session_id": "s1"},
        "filters": ["user_id", "session_id"],
        "gone": lambda e: e["owner"] == "u1" and e["session_id"] == "s1",
        "extra_count": 0,
    },
    "agent": {
        "req": {"agent_id": "a1"},
        "filters": ["agent_id"],
        "gone": lambda e: e["owner"] == "a1",
        "extra_count": "skills",  # one per skill directory, counted below
        "skills_gone": True,
    },
    "agent_session": {
        "req": {"agent_id": "a1", "session_id": "s1"},
        "filters": ["agent_id", "session_id"],
        "gone": lambda e: e["owner"] == "a1" and e["session_id"] == "s1",
        "extra_count": 0,
    },
}


@pytest.mark.parametrize("scope", list(_SCOPES))
def test_t1_1_each_scope_deletes_exactly_its_row(
    make_server: Any, tmp_path: Path, scope: str
) -> None:
    spec = _SCOPES[scope]
    srv = make_server(scope)
    seed_world(srv.root)
    srv.start()
    wait_converged(srv.base)
    before = md_entries(srv.root)
    kinds = {e["kind"] for e in before}
    # Precondition: every kind exists before (a 0 would make the check vacuous).
    assert {"episodes", ".cases"} <= kinds, kinds
    assert (srv.root / "default_app/default_project/users/u1/user.md").exists()
    tables_before = logical_dump(srv.root)
    assert tables_before["cluster"] and tables_before["reflection_report"]
    skills_before = _skill_dirs(srv.root)
    assert len(skills_before) == 2, skills_before
    index_before = _index(srv.root)
    assert all(index_before.values()), {t: len(v) for t, v in index_before.items()}

    resp = srv.delete(**spec["req"])

    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    expected_gone = [e for e in before if spec["gone"](e)]
    after = md_entries(srv.root)
    assert sorted((e["owner"], e["entry_id"], e["kind"]) for e in after) == sorted(
        (e["owner"], e["entry_id"], e["kind"]) for e in before if not spec["gone"](e)
    )
    extra = spec["extra_count"]
    extra = len(skills_before) if extra == "skills" else extra
    assert data == {"filters": spec["filters"], "count": len(expected_gone) + extra}
    assert _skill_dirs(srv.root) == ([] if spec.get("skills_gone") else skills_before)
    tables = logical_dump(srv.root)
    db = system_db(srv.root)
    if scope == "session":
        assert rows(db, "SELECT memcell_id FROM memcell") == [("mc_seed_s2",)]
    else:
        assert tables["memcell"] == tables_before["memcell"]
    owners_left = {r[0] for r in rows(db, "SELECT owner_id FROM cluster")}
    assert owners_left == (
        {"a1"} if scope == "user" else {"u1"} if scope == "agent" else {"u1", "a1"}
    )
    assert bool(tables["reflection_report"]) == (scope != "user")
    gone_ids = {f"{e['owner']}_{e['entry_id']}" for e in expected_gone}
    index_after = _index(srv.root)
    for table in _DAILY:
        assert index_after[table] == index_before[table] - gone_ids, table
    assert index_after["user_profile"] == index_before["user_profile"] - (
        {"u1"} if spec.get("profile_gone") else set()
    )
    assert index_after["agent_skill"] == (
        set() if spec.get("skills_gone") else index_before["agent_skill"]
    )
    assert (srv.root / "default_app/default_project/users/u1/user.md").exists() == (
        scope != "user"
    )


def test_t6_1_count_then_zero_on_resend_and_unknown(make_server: Any) -> None:
    srv = make_server()
    seed_world(srv.root)
    srv.start()
    wait_converged(srv.base)
    expected = sum(1 for e in md_entries(srv.root) if e["session_id"] == "s2")

    first = srv.delete(session_id="s2").json()["data"]["count"]
    again = srv.delete(session_id="s2").json()["data"]["count"]
    unknown = srv.delete(session_id="never-was").json()["data"]["count"]

    assert (first, again, unknown) == (expected, 0, 0)
    assert expected > 0
