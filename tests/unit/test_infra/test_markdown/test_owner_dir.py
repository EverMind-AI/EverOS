"""Deleting one owner's whole directory, and finishing it after a crash."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from everos.core.errors import PathTraversalError
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    MarkdownAgentCaseStore,
    MarkdownAgentSkillStore,
    MarkdownAtomicFactStore,
    MarkdownEpisodeStore,
    MarkdownOwnerDirStore,
    MarkdownProfileStore,
    MarkdownReader,
)
from everos.memory.models import AgentCase, AgentSkill, AtomicFact, Episode, Profile
from everos.memory.ports import NO_SOURCE_CHECK, Scope

_TS = 1_747_216_800_000


@pytest.fixture
def root(tmp_path: Path) -> MemoryRoot:
    memory_root = MemoryRoot(tmp_path)
    memory_root.ensure()
    return memory_root


def _user(owner: str = "u1") -> Scope:
    return Scope(app_id="app", project_id="p", owner_id=owner, owner_type="user")


def _episode(text: str, session: str = "s1") -> Episode:
    return Episode(
        owner_id="u1",
        episode=text,
        timestamp=_TS,
        subject="s",
        summary="s",
        session_id=session,
        parent_id="mc_1",
    )


async def _seed_user(root: MemoryRoot, scope: Scope) -> list[str]:
    episodes = MarkdownEpisodeStore(root)
    refs = await episodes.append_many(
        scope, [_episode(f"e{i}") for i in range(3)], only_if=NO_SOURCE_CHECK
    )
    await MarkdownAtomicFactStore(root).append_many(
        scope,
        [
            AtomicFact(
                owner_id=scope.owner_id,
                content="f",
                timestamp=_TS,
                parent_id=refs[0].entry_id,
            )
        ],
        only_if=NO_SOURCE_CHECK,
    )
    await MarkdownProfileStore(root).write(
        scope, Profile(owner_id=scope.owner_id, summary="p", timestamp=_TS)
    )
    return [r.entry_id for r in refs]


async def test_owner_delete_counts_and_keeps_today_counter(root: MemoryRoot) -> None:
    """A12 / R26 / A6: refs per table, count = entries + user.md; today's file
    stays (empty) so the next id continues past the deleted ones."""
    old = await _seed_user(root, _user())
    owners = MarkdownOwnerDirStore(root)

    deletion = await owners.delete(_user())

    assert [r.entry_id for r in deletion.refs["episode"]] == old
    assert len(deletion.refs["atomic_fact"]) == 1
    assert deletion.refs["foresight"] == []
    assert deletion.count == 3 + 1 + 1
    owner_dir = root.users_dir("app", "p") / "u1"
    assert sorted(p.name for p in owner_dir.rglob("*") if p.is_file()) == [
        path.name for path in sorted(owner_dir.rglob("*.md"))
    ]
    assert not (owner_dir / "user.md").exists()
    for md in owner_dir.rglob("*.md"):
        assert (await MarkdownReader.read(md)).entries == []
    (new,) = await MarkdownEpisodeStore(root).append_many(
        _user(), [_episode("after")], only_if=NO_SOURCE_CHECK
    )
    assert new.entry_id not in old and new.entry_id.endswith("04")
    assert not list(root.tmp_dir.glob("delete-*"))


async def test_agent_owner_delete_counts_skill_dirs(root: MemoryRoot) -> None:
    agent = Scope(app_id="app", project_id="p", owner_id="a1", owner_type="agent")
    await MarkdownAgentCaseStore(root).append(
        agent,
        AgentCase(
            owner_id="a1",
            task_intent="t",
            approach="a",
            quality_score=0.5,
            timestamp=_TS,
            parent_id="mc_1",
            session_id="s1",
        ),
        only_if=NO_SOURCE_CHECK,
    )
    skills = MarkdownAgentSkillStore(root)
    for name in ("x", "y"):
        await skills.write(agent, AgentSkill(id=name, name=name, content="c"))

    deletion = await MarkdownOwnerDirStore(root).delete(agent)

    assert len(deletion.refs["agent_case"]) == 1
    assert deletion.count == 1 + 2
    assert not (root.agents_dir("app", "p") / "a1" / "skills").exists()


async def test_delete_of_missing_owner_is_empty(root: MemoryRoot) -> None:
    deletion = await MarkdownOwnerDirStore(root).delete(_user("nobody"))
    assert (deletion.count, dict(deletion.refs)) == (0, {})


async def test_sweep_finishes_a_deletion_cut_after_the_rename(root: MemoryRoot) -> None:
    """R31: the process died right after moving the directory to the trash --
    the restart puts today's counter back and removes the trash."""
    old = await _seed_user(root, _user())
    owner_dir = root.users_dir("app", "p") / "u1"
    trash = root.tmp_dir / "delete-crashed"
    trash.mkdir(parents=True)
    (trash / "scope.json").write_text(
        json.dumps(
            {"app_id": "app", "project_id": "p", "owner_type": "user", "owner_id": "u1"}
        )
    )
    owner_dir.rename(trash / "owner")
    (root.tmp_dir / "delete-before-manifest").mkdir()  # died even earlier

    assert await MarkdownOwnerDirStore(root).sweep_trash() == 2

    assert not list(root.tmp_dir.glob("delete-*"))
    (new,) = await MarkdownEpisodeStore(root).append_many(
        _user(), [_episode("after")], only_if=NO_SOURCE_CHECK
    )
    assert new.entry_id not in old


@pytest.mark.parametrize("owner", ["a∕b", "a%2Fb"])
async def test_owner_ids_with_lookalike_separators_stay_literal(
    root: MemoryRoot, owner: str
) -> None:
    """RF4: a Unicode slash or a percent-encoded one is part of a name."""
    await _seed_user(root, _user(owner))
    await _seed_user(root, _user("a"))

    deletion = await MarkdownOwnerDirStore(root).delete(_user(owner))

    assert deletion.count == 5
    assert (root.users_dir("app", "p") / "a" / "user.md").exists()


@pytest.mark.parametrize(
    "owner",
    ["..", "a/../b", ".", "", "../users", *(["a\\b"] if os.sep == "\\" else [])],
)
async def test_owner_delete_rejects_escape(root: MemoryRoot, owner: str) -> None:
    """C7: the second line behind the route's id validation."""
    with pytest.raises(PathTraversalError):
        await MarkdownOwnerDirStore(root).delete(_user(owner))


async def test_owners_with_session(root: MemoryRoot) -> None:
    episodes = MarkdownEpisodeStore(root)
    await episodes.append_many(
        _user("u1"), [_episode("x", "s1")], only_if=NO_SOURCE_CHECK
    )
    await episodes.append_many(
        _user("u2"), [_episode("x", "s10")], only_if=NO_SOURCE_CHECK
    )
    agent = Scope(app_id="app", project_id="p", owner_id="a1", owner_type="agent")
    await MarkdownAgentCaseStore(root).append(
        agent,
        AgentCase(
            owner_id="a1",
            task_intent="t",
            approach="a",
            quality_score=0.5,
            timestamp=_TS,
            parent_id="mc_1",
            session_id="s1",
        ),
        only_if=NO_SOURCE_CHECK,
    )

    found = await MarkdownOwnerDirStore(root).owners_with_session("app", "p", "s1")

    # u2 only mentions s10: the slow path may over-include (its caller
    # matches exactly), so u2 is allowed but u1 and a1 are required.
    assert {(s.owner_type, s.owner_id) for s in found} >= {
        ("user", "u1"),
        ("agent", "a1"),
    }
