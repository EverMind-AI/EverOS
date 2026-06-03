"""Unit tests for :class:`GetManager` with in-memory stub repos.

These tests exercise the dispatch / shape / sort-override logic without
LanceDB. Each repo is replaced by a minimal stub that records the call
and returns canned rows; the manager's job is to:

* dispatch on ``memory_type`` to the matching repo,
* compile filters once and pass the same ``where`` to the repo,
* shape rows into the correct ``GetItem`` (lossless except score),
* silently override ``sort_by`` to ``updated_at`` for ``agent_skill``
  (the table has no ``timestamp`` column),
* fetch the owner's single profile row (KV-by-owner) and shape it into
  ``GetProfileItem``, or return ``[]`` on a cold-start miss.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import pytest

from everos.infra.persistence.lancedb import (
    AgentCase,
    AgentSkill,
    Episode,
    UserProfile,
)
from everos.memory.get import (
    GetManager,
    GetMemoryType,
    GetRequest,
)
from everos.memory.search import FilterNode

# ── Stub repos ──────────────────────────────────────────────────────────


@dataclass
class _CallRecord:
    where: str = ""
    sort_by: str = ""
    descending: bool = True
    page: int = 0
    page_size: int = 0


@dataclass
class _StubRepo:
    """Records the call and returns ``(rows, total)`` verbatim."""

    rows: list[Any] = field(default_factory=list)
    total: int = 0
    last: _CallRecord = field(default_factory=_CallRecord)

    async def find_where_paginated(
        self,
        where: str,
        *,
        sort_by: str,
        descending: bool = True,
        page: int = 1,
        page_size: int = 20,
        max_fetch: int = 20000,
    ) -> tuple[list[Any], int]:
        self.last = _CallRecord(
            where=where,
            sort_by=sort_by,
            descending=descending,
            page=page,
            page_size=page_size,
        )
        return list(self.rows), self.total


@dataclass
class _ProfileStubRepo:
    """Stub ``user_profile_repo`` — returns its configured row by id."""

    row: Any = None
    last_id: str | None = None

    async def get_by_id(self, id_: str) -> Any:
        self.last_id = id_
        return self.row


# ── Fixtures ────────────────────────────────────────────────────────────


def _ts(day: int = 1) -> _dt.datetime:
    return _dt.datetime(2026, 1, day, tzinfo=_dt.UTC)


def _episode_row(entry: str) -> Episode:
    return Episode(
        id=f"u1_{entry}",
        entry_id=entry,
        owner_id="u1",
        owner_type="user",
        session_id="sess_a",
        timestamp=_ts(),
        parent_type="memcell",
        parent_id="mc_1",
        sender_ids=["u1", "assistant"],
        subject=f"subj {entry}",
        summary=f"summary {entry}",
        episode=f"body of {entry}",
        episode_tokens=f"body of {entry}",
        md_path=f"users/u1/episodes/{entry}.md",
        content_sha256="abc",
        vector=[0.0] * 1024,
    )


def _agent_case_row(entry: str) -> AgentCase:
    return AgentCase(
        id=f"a1_{entry}",
        entry_id=entry,
        owner_id="a1",
        owner_type="agent",
        session_id="sess_x",
        timestamp=_ts(),
        parent_type="memcell",
        parent_id="mc_99",
        quality_score=0.8,
        task_intent=f"intent {entry}",
        task_intent_tokens=f"intent {entry}",
        approach=f"approach {entry}",
        approach_tokens=f"approach {entry}",
        key_insight=None,
        md_path=f"agents/a1/cases/{entry}.md",
        content_sha256="abc",
        vector=[0.0] * 1024,
    )


def _agent_skill_row(name: str) -> AgentSkill:
    return AgentSkill(
        id=f"a1_{name}",
        owner_id="a1",
        owner_type="agent",
        name=name,
        description=f"desc {name}",
        description_tokens=f"desc {name}",
        content=f"content {name}",
        content_tokens=f"content {name}",
        confidence=0.9,
        maturity_score=0.7,
        source_case_ids=["a1_ac_1"],
        md_path=f"agents/a1/skills/{name}/SKILL.md",
        content_sha256="abc",
        vector=[0.0] * 1024,
    )


def _user_profile_row(owner: str = "u1") -> UserProfile:
    return UserProfile(
        id=owner,
        owner_id=owner,
        owner_type="user",
        app_id="default",
        project_id="default",
        summary=f"{owner} loves climbing in Yosemite",
        explicit_info_json='[{"category": "Hobby", "description": "climbing"}]',
        implicit_traits_json='[{"trait": "Outdoorsy"}]',
        profile_timestamp_ms=1780304400000,
        md_path=f"users/{owner}/user.md",
        content_sha256="abc",
    )


@pytest.fixture
def profile_repo() -> _ProfileStubRepo:
    return _ProfileStubRepo()


@pytest.fixture
def manager(
    profile_repo: _ProfileStubRepo,
) -> tuple[GetManager, _StubRepo, _StubRepo, _StubRepo]:
    ep = _StubRepo()
    ac = _StubRepo()
    sk = _StubRepo()
    mgr = GetManager(
        episode_repo=ep,  # type: ignore[arg-type]
        agent_case_repo=ac,  # type: ignore[arg-type]
        agent_skill_repo=sk,  # type: ignore[arg-type]
        user_profile_repo=profile_repo,  # type: ignore[arg-type]
    )
    return mgr, ep, ac, sk


# ── Episode dispatch ────────────────────────────────────────────────────


async def test_episodic_memory_populates_episodes_and_counts(
    manager: tuple[GetManager, _StubRepo, _StubRepo, _StubRepo],
) -> None:
    mgr, ep, _, _ = manager
    ep.rows = [_episode_row("ep_1"), _episode_row("ep_2")]
    ep.total = 17  # filtered total may exceed the page
    req = GetRequest(
        user_id="u1",
        memory_type=GetMemoryType.EPISODE,
    )
    resp = await mgr.get(req)

    assert len(resp.request_id) == 32 and all(
        c in "0123456789abcdef" for c in resp.request_id
    )
    assert resp.data.total_count == 17
    assert resp.data.count == 2
    assert [item.id for item in resp.data.episodes] == ["u1_ep_1", "u1_ep_2"]
    assert resp.data.profiles == []
    assert resp.data.agent_cases == []
    assert resp.data.agent_skills == []
    # The shaper maps the lance row's owner_id onto the item's user_id field.
    assert all(item.user_id == "u1" for item in resp.data.episodes)


async def test_episodic_memory_passes_where_and_sort_to_repo(
    manager: tuple[GetManager, _StubRepo, _StubRepo, _StubRepo],
) -> None:
    """The compiled ``where`` must include owner_id + filter clauses."""
    mgr, ep, _, _ = manager
    req = GetRequest(
        user_id="u1",
        memory_type=GetMemoryType.EPISODE,
        sort_by="timestamp",
        sort_order="asc",
        page=2,
        page_size=10,
        filters=FilterNode.model_validate({"session_id": "sess_a"}),
    )
    await mgr.get(req)
    assert "owner_id = 'u1'" in ep.last.where
    assert "owner_type = 'user'" in ep.last.where
    assert "session_id = 'sess_a'" in ep.last.where
    assert ep.last.sort_by == "timestamp"
    assert ep.last.descending is False  # asc
    assert ep.last.page == 2
    assert ep.last.page_size == 10


# ── Profile dispatch ────────────────────────────────────────────────────


async def test_profile_miss_returns_empty(
    manager: tuple[GetManager, _StubRepo, _StubRepo, _StubRepo],
) -> None:
    """Cold start (no profile row yet) → empty list + total_count=0."""
    mgr, ep, ac, sk = manager  # profile_repo.row defaults to None
    req = GetRequest(
        user_id="u1",
        memory_type=GetMemoryType.PROFILE,
    )
    resp = await mgr.get(req)
    assert resp.data.profiles == []
    assert resp.data.total_count == 0
    assert resp.data.count == 0
    # The profile path never touches the paginated (episode/case/skill) repos.
    assert ep.last.where == ""
    assert ac.last.where == ""
    assert sk.last.where == ""


async def test_profile_hit_shapes_row_into_item(
    manager: tuple[GetManager, _StubRepo, _StubRepo, _StubRepo],
    profile_repo: _ProfileStubRepo,
) -> None:
    """A present profile row is fetched by owner and shaped + json-decoded."""
    mgr, *_ = manager
    profile_repo.row = _user_profile_row("u1")
    req = GetRequest(user_id="u1", memory_type=GetMemoryType.PROFILE)
    resp = await mgr.get(req)

    assert resp.data.total_count == 1
    assert resp.data.count == 1
    assert len(resp.data.profiles) == 1
    item = resp.data.profiles[0]
    assert item.id == "u1"
    assert item.user_id == "u1"
    # KV fetch keys on owner_id.
    assert profile_repo.last_id == "u1"
    # json buckets are decoded back into structured profile_data.
    assert item.profile_data["summary"] == "u1 loves climbing in Yosemite"
    assert item.profile_data["explicit_info"] == [
        {"category": "Hobby", "description": "climbing"}
    ]
    assert item.profile_data["implicit_traits"] == [{"trait": "Outdoorsy"}]
    assert item.profile_data["profile_timestamp_ms"] == 1780304400000


# ── Agent case dispatch ─────────────────────────────────────────────────


async def test_agent_case_populates_agent_cases(
    manager: tuple[GetManager, _StubRepo, _StubRepo, _StubRepo],
) -> None:
    mgr, _, ac, _ = manager
    ac.rows = [_agent_case_row("ac_1"), _agent_case_row("ac_2")]
    ac.total = 2
    req = GetRequest(
        agent_id="a1",
        memory_type=GetMemoryType.AGENT_CASE,
    )
    resp = await mgr.get(req)
    assert resp.data.total_count == 2
    assert resp.data.count == 2
    assert [item.id for item in resp.data.agent_cases] == ["a1_ac_1", "a1_ac_2"]
    assert resp.data.episodes == []
    assert resp.data.agent_skills == []


# ── Agent skill dispatch — sort_by silent override ──────────────────────


async def test_agent_skill_sort_by_silently_overridden_to_updated_at(
    manager: tuple[GetManager, _StubRepo, _StubRepo, _StubRepo],
) -> None:
    """``agent_skill`` always sorts by ``updated_at`` (no ``timestamp`` column)."""
    mgr, _, _, sk = manager
    sk.rows = [_agent_skill_row("planner")]
    sk.total = 1
    req = GetRequest(
        agent_id="a1",
        memory_type=GetMemoryType.AGENT_SKILL,
        # User passes the default — should be silently downgraded.
        sort_by="timestamp",
    )
    resp = await mgr.get(req)
    assert sk.last.sort_by == "updated_at"
    assert resp.data.total_count == 1
    assert resp.data.agent_skills[0].name == "planner"


async def test_agent_skill_explicit_updated_at_is_respected(
    manager: tuple[GetManager, _StubRepo, _StubRepo, _StubRepo],
) -> None:
    """``updated_at`` passes through unchanged (no double-override surprise)."""
    mgr, _, _, sk = manager
    req = GetRequest(
        agent_id="a1",
        memory_type=GetMemoryType.AGENT_SKILL,
        sort_by="updated_at",
    )
    await mgr.get(req)
    assert sk.last.sort_by == "updated_at"
