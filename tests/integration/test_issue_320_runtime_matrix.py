"""Real-LanceDB regression matrix for owner-scoped storage identities.

The six owner-scoped business schemas must use the complete logical scope as
their physical primary key.  These tests intentionally exercise the concrete
repositories against a temporary on-disk LanceDB rather than mocking the
storage boundary.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from everalgo.types import Candidate

from everos.core.persistence.lancedb.repository import LanceRepoBase
from everos.core.persistence.lancedb.row_id import (
    agent_skill_storage_id,
    agent_skill_wire_id,
    daily_log_storage_id,
    daily_log_wire_id,
    user_profile_storage_id,
    user_profile_wire_id,
)
from everos.infra.persistence.lancedb import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    UserProfile,
    agent_case_repo,
    agent_skill_repo,
    atomic_fact_repo,
    dispose_connection,
    episode_repo,
    foresight_repo,
    user_profile_repo,
)
from everos.memory.search.recall.base import row_to_candidate
from everos.memory.search.recall.profile import ProfileRecaller
from everos.memory.search.shaper import (
    shape_agent_case_from_candidate,
    shape_agent_skill_from_candidate,
    shape_atomic_fact_from_candidate,
    shape_episode_from_candidate,
)

_TS = dt.datetime(2026, 8, 12, 12, 0, tzinfo=dt.UTC)
_VECTOR = [0.0] * 1024


@pytest.fixture(autouse=True)
async def _fresh_lancedb() -> Any:
    """Keep singleton tables and event-loop-bound write locks test-local."""
    LanceRepoBase._reset_locks_for_tests()
    await dispose_connection()
    yield
    await dispose_connection()


def _episode(
    app_id: str, project_id: str, owner_id: str, logical_id: str, payload: str
) -> Episode:
    return Episode(
        id=daily_log_storage_id(
            app_id=app_id,
            project_id=project_id,
            owner_id=owner_id,
            entry_id=logical_id,
        ),
        entry_id=logical_id,
        owner_id=owner_id,
        owner_type="user",
        app_id=app_id,
        project_id=project_id,
        session_id="session",
        timestamp=_TS,
        parent_id="mc_1",
        sender_ids=[owner_id],
        subject="subject",
        summary="summary",
        episode=payload,
        episode_tokens=payload,
        md_path=f"{app_id}/{project_id}/users/{owner_id}/episodes/day.md",
        content_sha256=payload,
        vector=_VECTOR,
    )


def _atomic_fact(
    app_id: str, project_id: str, owner_id: str, logical_id: str, payload: str
) -> AtomicFact:
    return AtomicFact(
        id=daily_log_storage_id(
            app_id=app_id,
            project_id=project_id,
            owner_id=owner_id,
            entry_id=logical_id,
        ),
        entry_id=logical_id,
        owner_id=owner_id,
        owner_type="user",
        app_id=app_id,
        project_id=project_id,
        session_id="session",
        timestamp=_TS,
        parent_id="mc_1",
        sender_ids=[owner_id],
        fact=payload,
        fact_tokens=payload,
        md_path=f"{app_id}/{project_id}/users/{owner_id}/facts/day.md",
        content_sha256=payload,
        vector=_VECTOR,
    )


def _foresight(
    app_id: str, project_id: str, owner_id: str, logical_id: str, payload: str
) -> Foresight:
    return Foresight(
        id=daily_log_storage_id(
            app_id=app_id,
            project_id=project_id,
            owner_id=owner_id,
            entry_id=logical_id,
        ),
        entry_id=logical_id,
        owner_id=owner_id,
        owner_type="user",
        app_id=app_id,
        project_id=project_id,
        session_id="session",
        timestamp=_TS,
        parent_id="mc_1",
        sender_ids=[owner_id],
        foresight=payload,
        foresight_tokens=payload,
        md_path=f"{app_id}/{project_id}/users/{owner_id}/foresights/day.md",
        content_sha256=payload,
        vector=_VECTOR,
    )


def _agent_case(
    app_id: str, project_id: str, owner_id: str, logical_id: str, payload: str
) -> AgentCase:
    return AgentCase(
        id=daily_log_storage_id(
            app_id=app_id,
            project_id=project_id,
            owner_id=owner_id,
            entry_id=logical_id,
        ),
        entry_id=logical_id,
        owner_id=owner_id,
        owner_type="agent",
        app_id=app_id,
        project_id=project_id,
        session_id="session",
        timestamp=_TS,
        parent_id="mc_1",
        quality_score=0.9,
        task_intent=payload,
        task_intent_tokens=payload,
        approach="approach",
        approach_tokens="approach",
        md_path=f"{app_id}/{project_id}/agents/{owner_id}/cases/day.md",
        content_sha256=payload,
        vector=_VECTOR,
    )


def _agent_skill(
    app_id: str, project_id: str, owner_id: str, logical_id: str, payload: str
) -> AgentSkill:
    return AgentSkill(
        id=agent_skill_storage_id(
            app_id=app_id,
            project_id=project_id,
            owner_id=owner_id,
            name=logical_id,
        ),
        owner_id=owner_id,
        owner_type="agent",
        app_id=app_id,
        project_id=project_id,
        name=logical_id,
        description=payload,
        description_tokens=payload,
        content="content",
        content_tokens="content",
        confidence=0.9,
        maturity_score=0.8,
        source_case_ids=[],
        md_path=f"{app_id}/{project_id}/agents/{owner_id}/skills/{logical_id}.md",
        content_sha256=payload,
        vector=_VECTOR,
    )


def _user_profile(
    app_id: str, project_id: str, owner_id: str, logical_id: str, payload: str
) -> UserProfile:
    del logical_id
    return UserProfile(
        id=user_profile_storage_id(
            app_id=app_id,
            project_id=project_id,
            owner_id=owner_id,
        ),
        owner_id=owner_id,
        owner_type="user",
        app_id=app_id,
        project_id=project_id,
        summary=payload,
        explicit_info_json="[]",
        implicit_traits_json="[]",
        profile_timestamp_ms=1,
        md_path=f"{app_id}/{project_id}/users/{owner_id}/user.md",
        content_sha256=payload,
    )


def _daily_wire(owner_id: str, logical_id: str) -> str:
    return daily_log_wire_id(owner_id=owner_id, entry_id=logical_id)


def _skill_wire(owner_id: str, logical_id: str) -> str:
    return agent_skill_wire_id(owner_id=owner_id, name=logical_id)


def _profile_wire(owner_id: str, logical_id: str) -> str:
    del logical_id
    return user_profile_wire_id(owner_id=owner_id)


def _shape_episode(candidate: Candidate) -> Any:
    return shape_episode_from_candidate(candidate)


def _shape_fact(candidate: Candidate) -> Any:
    return shape_atomic_fact_from_candidate(candidate)


def _shape_case(candidate: Candidate) -> Any:
    return shape_agent_case_from_candidate(candidate)


def _shape_skill(candidate: Candidate) -> Any:
    return shape_agent_skill_from_candidate(candidate)


@dataclass(frozen=True)
class _SchemaCase:
    name: str
    repo: Any
    build: Callable[[str, str, str, str, str], Any]
    payload_field: str
    wire_id: Callable[[str, str], str]
    shaper: Callable[[Candidate], Any] | None
    profile_identity: bool = False

    async def fetch(
        self, *, app_id: str, project_id: str, owner_id: str, logical_id: str
    ) -> Any:
        if self.profile_identity:
            return await self.repo.get_by_id(
                user_profile_storage_id(
                    app_id=app_id,
                    project_id=project_id,
                    owner_id=owner_id,
                )
            )
        if self.name == "agent_skill":
            return await self.repo.get_by_id(
                agent_skill_storage_id(
                    app_id=app_id,
                    project_id=project_id,
                    owner_id=owner_id,
                    name=logical_id,
                )
            )
        return await self.repo.find_by_owner_entry(
            owner_id,
            logical_id,
            app_id=app_id,
            project_id=project_id,
        )


_CASES = [
    _SchemaCase(
        "episode", episode_repo, _episode, "episode", _daily_wire, _shape_episode
    ),
    _SchemaCase(
        "atomic_fact", atomic_fact_repo, _atomic_fact, "fact", _daily_wire, _shape_fact
    ),
    _SchemaCase(
        "foresight", foresight_repo, _foresight, "foresight", _daily_wire, None
    ),
    _SchemaCase(
        "agent_case",
        agent_case_repo,
        _agent_case,
        "task_intent",
        _daily_wire,
        _shape_case,
    ),
    _SchemaCase(
        "agent_skill",
        agent_skill_repo,
        _agent_skill,
        "description",
        _skill_wire,
        _shape_skill,
    ),
    _SchemaCase(
        "user_profile",
        user_profile_repo,
        _user_profile,
        "summary",
        _profile_wire,
        None,
        profile_identity=True,
    ),
]


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
async def test_owner_scoped_identity_runtime_matrix(case: _SchemaCase) -> None:
    """Prove coexistence, update isolation, injectivity, and wire containment."""
    app_id = "app-main"
    owner_id = "owner_same"
    logical_id = "entry_same"

    row_a = case.build(app_id, "project-a", owner_id, logical_id, "payload-a")
    row_b = case.build(app_id, "project-b", owner_id, logical_id, "payload-b")
    await case.repo.upsert([row_a])
    await case.repo.upsert([row_b])

    # Cross-project coexistence for an otherwise identical logical identity.
    assert await case.repo.count() == 2
    stored_a = await case.fetch(
        app_id=app_id,
        project_id="project-a",
        owner_id=owner_id,
        logical_id=logical_id,
    )
    stored_b = await case.fetch(
        app_id=app_id,
        project_id="project-b",
        owner_id=owner_id,
        logical_id=logical_id,
    )
    assert stored_a is not None and stored_b is not None
    assert stored_a.id != stored_b.id
    assert getattr(stored_a, case.payload_field) == "payload-a"
    assert getattr(stored_b, case.payload_field) == "payload-b"

    # Replaying an unchanged same-scope row is idempotent.
    await case.repo.upsert([row_a])
    assert await case.repo.count() == 2

    # Updating project A replaces only project A, never its project B sibling.
    updated_a = case.build(
        app_id, "project-a", owner_id, logical_id, "payload-a-updated"
    )
    await case.repo.upsert([updated_a])
    assert await case.repo.count() == 2
    stored_a = await case.fetch(
        app_id=app_id,
        project_id="project-a",
        owner_id=owner_id,
        logical_id=logical_id,
    )
    stored_b = await case.fetch(
        app_id=app_id,
        project_id="project-b",
        owner_id=owner_id,
        logical_id=logical_id,
    )
    assert getattr(stored_a, case.payload_field) == "payload-a-updated"
    assert getattr(stored_b, case.payload_field) == "payload-b"

    # These tuples collide under naive underscore joining. Length-prefixing
    # must remain injective for delimiter-heavy and Unicode scope segments.
    if case.profile_identity:
        injective_a = case.build("应用", "项目_甲", "乙_丙", "", "unicode-a")
        injective_b = case.build("应用", "项目", "甲_乙_丙", "", "unicode-b")
    else:
        injective_a = case.build("应用", "项目_甲", "乙", "丙_丁", "unicode-a")
        injective_b = case.build("应用", "项目", "甲_乙", "丙_丁", "unicode-b")
    assert injective_a.id != injective_b.id
    await case.repo.upsert([injective_a, injective_b])
    assert await case.repo.count() == 4
    assert await case.repo.get_by_id(injective_a.id) is not None
    assert await case.repo.get_by_id(injective_b.id) is not None

    # Every currently exposed DTO is shaped from logical identity fields, not
    # the physical storage key. Foresight has no public response shaper today.
    expected_wire_id = case.wire_id(owner_id, logical_id)
    assert stored_a.id != expected_wire_id
    if case.shaper is not None:
        candidate = row_to_candidate(
            stored_a.model_dump(mode="python"), source="vector", score=0.5
        )
        public_item = case.shaper(candidate)
        assert public_item is not None
        assert public_item.id == expected_wire_id
        assert stored_a.id not in public_item.model_dump_json()
    elif case.profile_identity:
        public_items = await ProfileRecaller().fetch(
            owner_id,
            app_id=app_id,
            project_id="project-a",
        )
        assert len(public_items) == 1
        assert public_items[0].id == expected_wire_id
        assert stored_a.id not in public_items[0].model_dump_json()
