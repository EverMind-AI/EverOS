"""Unit tests for :class:`GetManager` with in-memory stub repos.

These tests exercise the dispatch / shape / sort-override logic without
LanceDB. Each repo is replaced by a minimal stub that records the call
and returns canned rows; the manager's job is to:

* dispatch on ``memory_type`` to the matching repo,
* compile filters once and pass the same ``where`` to the repo,
* shape rows into the correct ``GetItem`` (lossless except score),
* fetch the owner's single profile row (KV-by-owner) and shape it into
  ``GetProfileItem``, or return ``[]`` on a cold-start miss.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import pytest

from everos.infra.persistence.index.adapters import (
    EpisodeIndexAdapter,
    ProfileIndexAdapter,
)
from everos.infra.persistence.lancedb import (
    Episode,
    UserProfile,
)
from everos.memory.access import AccessPolicy
from everos.memory.get import (
    GetManager,
    GetMemoryType,
    GetRequest,
)
from everos.memory.search import FilterNode
from tests.fakes import InMemoryGroupStore

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
    """Stub ``storage.repos.user_profile`` — returns its configured row by id."""

    row: Any = None
    last_key: tuple[str, str, str] | None = None

    async def find_one_where(self, where: Any) -> Any:
        wanted = {c.field: c.value for c in where.children}
        self.last_key = (wanted["app_id"], wanted["project_id"], wanted["id"])
        if self.row is None or (self.row.app_id, self.row.project_id) != (
            wanted["app_id"],
            wanted["project_id"],
        ):
            return None
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


def _user_profile_row(owner: str = "u1") -> UserProfile:
    return UserProfile(
        id=owner,
        owner_id=owner,
        owner_type="user",
        app_id="default",
        project_id="default",
        summary=f"{owner} loves climbing in Yosemite",
        explicit_info_json='[{"category": "Hobby", "description": "climbing"}]',
        implicit_traits_json='[{"trait": "Outdoorsy", "description": "outdoorsy"}]',
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
) -> tuple[GetManager, _StubRepo]:
    ep = _StubRepo()
    mgr = GetManager(
        access=AccessPolicy(InMemoryGroupStore()),
        episode_index=EpisodeIndexAdapter(ep),  # type: ignore[arg-type]
        profile_index=ProfileIndexAdapter(profile_repo),  # type: ignore[arg-type]
    )
    return mgr, ep


# ── Episode dispatch ────────────────────────────────────────────────────


async def test_episodic_memory_populates_episodes_and_counts(
    manager: tuple[GetManager, _StubRepo],
) -> None:
    mgr, ep = manager
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
    # 2.0 has no agent memory: the response carries no agent arrays at all.
    assert "agent_cases" not in resp.data.model_dump()
    assert "agent_skills" not in resp.data.model_dump()
    # The shaper maps the lance row's owner_id onto the item's user_id field.
    assert all(item.user_id == "u1" for item in resp.data.episodes)


async def test_get_uses_propagated_request_id_when_bound(
    manager: tuple[GetManager, _StubRepo],
) -> None:
    """When a request id is bound upstream (middleware), ``get`` reuses it
    instead of minting a fresh one, so the response id matches the trace."""
    from everos.core.context import reset_request_id, set_request_id

    mgr, ep = manager
    ep.rows = [_episode_row("ep_1")]
    token = set_request_id("deadbeef" * 4)
    try:
        resp = await mgr.get(
            GetRequest(user_id="u1", memory_type=GetMemoryType.EPISODE)
        )
        assert resp.request_id == "deadbeef" * 4
    finally:
        reset_request_id(token)


async def test_episodic_memory_passes_where_and_sort_to_repo(
    manager: tuple[GetManager, _StubRepo],
) -> None:
    """The compiled ``where`` must include owner_id + filter clauses."""
    mgr, ep = manager
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
    from everos.infra.persistence.lancedb import render_predicate

    rendered = render_predicate(ep.last.where)
    assert "owner_id = 'u1'" in rendered
    assert "owner_type = 'user'" in rendered
    assert "session_id = 'sess_a'" in rendered
    assert ep.last.sort_by == "timestamp"
    assert ep.last.descending is False  # asc
    assert ep.last.page == 2
    assert ep.last.page_size == 10


# ── Profile dispatch ────────────────────────────────────────────────────


async def test_profile_miss_returns_empty(
    manager: tuple[GetManager, _StubRepo],
) -> None:
    """Cold start (no profile row yet) → empty list + total_count=0."""
    mgr, ep = manager  # profile_repo.row defaults to None
    req = GetRequest(
        user_id="u1",
        memory_type=GetMemoryType.PROFILE,
    )
    resp = await mgr.get(req)
    assert resp.data.profiles == []
    assert resp.data.total_count == 0
    assert resp.data.count == 0
    # The profile path never touches the paginated episode repo.
    assert ep.last.where == ""


async def test_profile_hit_shapes_row_into_item(
    manager: tuple[GetManager, _StubRepo],
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
    assert profile_repo.last_key == ("default", "default", "u1")
    # json buckets are decoded back into structured profile_data.
    assert item.profile_data["summary"] == "u1 loves climbing in Yosemite"
    assert item.profile_data["explicit_info"] == [
        {
            "category": "Hobby",
            "description": "climbing",
            "evidence": None,
            "observed_at": None,
        }
    ]
    assert item.profile_data["implicit_traits"] == [
        {
            "trait": "Outdoorsy",
            "description": "outdoorsy",
            "basis": None,
            "observed_at": None,
        }
    ]
    assert item.profile_data["profile_timestamp_ms"] == 1780304400000


async def test_profile_item_without_description_is_dropped_not_fatal(
    manager: tuple[GetManager, _StubRepo],
    profile_repo: _ProfileStubRepo,
) -> None:
    mgr, *_ = manager
    row = _user_profile_row("u1")
    row.implicit_traits_json = '[{"trait": "Outdoorsy"}, "patient"]'
    profile_repo.row = row

    resp = await mgr.get(GetRequest(user_id="u1", memory_type=GetMemoryType.PROFILE))

    (item,) = resp.data.profiles
    assert [i["description"] for i in item.profile_data["implicit_traits"]] == [
        "patient"
    ]


async def test_profile_missing_in_scope_is_empty(manager, profile_repo) -> None:  # type: ignore[no-untyped-def]
    """A profile row in another project does not answer this project's get."""
    mgr, *_ = manager
    profile_repo.row = _user_profile_row("u1")  # lives in default/default
    req = GetRequest(user_id="u1", memory_type=GetMemoryType.PROFILE, project_id="beta")
    resp = await mgr.get(req)
    assert resp.data.profiles == []
    assert profile_repo.last_key == ("default", "beta", "u1")


async def test_merged_episode_serialises_null_session_id(
    manager: tuple[GetManager, _StubRepo],
) -> None:
    """B10: an episode an earlier 2.0 build's Reflection merged (``parent_type``
    ``cluster``) has no session; /get lists it with ``"session_id": null``
    instead of failing the whole page."""
    mgr, ep = manager
    merged = _episode_row("ep_m").model_copy(
        update={"session_id": None, "parent_type": "cluster", "parent_id": "cl_1"}
    )
    ep.rows, ep.total = [merged, _episode_row("ep_1")], 2

    resp = await mgr.get(GetRequest(user_id="u1", memory_type=GetMemoryType.EPISODE))

    assert [item.session_id for item in resp.data.episodes] == [None, "sess_a"]
    assert '"session_id":null' in resp.model_dump_json()
