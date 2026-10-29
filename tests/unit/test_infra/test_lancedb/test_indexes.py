"""LanceDB adapters of the memory index ports, against a real LanceDB.

The episode case goes all the way round: a domain episode is written by the
markdown store, projected by the cascade handler, and read back as a domain
episode through the index port.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.config import load_settings
from everos.infra.cascade.projections.episode import EpisodeHandler
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index.adapters import (
    AgentCaseIndexAdapter,
    AgentSkillIndexAdapter,
    AtomicFactIndexAdapter,
    EpisodeIndexAdapter,
    KnowledgeTopicIndexAdapter,
    ProfileIndexAdapter,
)
from everos.infra.persistence.lancedb import (
    AgentCase as AgentCaseRow,
)
from everos.infra.persistence.lancedb import (
    AgentSkill as AgentSkillRow,
)
from everos.infra.persistence.lancedb import (
    AtomicFact as AtomicFactRow,
)
from everos.infra.persistence.lancedb import (
    UserProfile as UserProfileRow,
)
from everos.infra.persistence.markdown import MarkdownEpisodeStore
from everos.memory.models import AgentCase, Episode, Profile
from everos.memory.ports import (
    AgentCaseIndex,
    AgentSkillIndex,
    AtomicFactIndex,
    EpisodeIndex,
    KnowledgeTopicIndex,
    Page,
    ProfileIndex,
    Scope,
    all_of,
    eq,
)
from tests.fakes import handler_deps
from tests.fakes.storage import Storage

_DIM = 1024
_TS = 1_747_216_800_000  # 2025-05-14T10:00:00Z
_USER = Scope(app_id="default", project_id="default", owner_id="u1", owner_type="user")
_AGENT = Scope(
    app_id="default", project_id="default", owner_id="a1", owner_type="agent"
)


class _WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def tokenize_batch(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]


class _ConstantEmbedder(EmbeddingProvider):
    dim = _DIM

    async def embed(self, text: str) -> list[float]:
        return [0.1] * _DIM

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture(autouse=True)
async def _real_lancedb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> AsyncIterator[None]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    yield
    await storage.lance.close()
    load_settings.cache_clear()


def test_adapters_satisfy_their_ports(storage: Storage) -> None:
    assert isinstance(EpisodeIndexAdapter(storage.repos.episode), EpisodeIndex)
    assert isinstance(
        AtomicFactIndexAdapter(storage.repos.atomic_fact), AtomicFactIndex
    )
    assert isinstance(AgentCaseIndexAdapter(storage.repos.agent_case), AgentCaseIndex)
    assert isinstance(
        AgentSkillIndexAdapter(storage.repos.agent_skill), AgentSkillIndex
    )
    assert isinstance(ProfileIndexAdapter(storage.repos.user_profile), ProfileIndex)
    assert isinstance(
        KnowledgeTopicIndexAdapter(storage.repos.knowledge_topic), KnowledgeTopicIndex
    )


async def test_episode_round_trips_through_markdown_and_the_cascade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    episode = Episode(
        owner_id="u1",
        episode="we planned a trip",
        timestamp=_TS,
        subject="Trip",
        summary="planning",
        session_id="s1",
        sender_ids=["u1", "u2"],
        parent_id="mc_1",
    )
    ref = await MarkdownEpisodeStore(root).append(_USER, episode)
    handler = EpisodeHandler(
        handler_deps(
            memory_root=root,
            tokenizer=_WhitespaceTokenizer(),
            index=storage.repos,
            embedding=_ConstantEmbedder(),
        )
    )
    await handler.handle_added_or_modified(ref.md_path)
    index = EpisodeIndexAdapter(storage.repos.episode)

    (stored,) = await index.find_by_entry_ids(_USER, [ref.entry_id])
    page = await index.paginate(_USER.to_filter(), Page(sort_by="timestamp"))
    after = await index.list_after(_USER, _TS - 1, parent_type="memcell")
    hits = await index.sparse_search(["trip"], _USER.to_filter(), limit=5)
    dense = await index.dense_search([0.1] * _DIM, _USER.to_filter(), limit=5)
    subject = await index.dense_search_subject([0.1] * _DIM, _USER.to_filter(), limit=5)
    every = await index.find(_USER.to_filter())

    assert stored.item == episode
    assert stored.ref == ref
    assert stored.scope == _USER
    assert page.total == 1 and page.items[0].item == episode
    assert [s.item for s in after] == [episode]
    assert await index.count(_USER, parent_type="memcell") == 1
    assert [h.id for h in hits] == [ref.id] and hits[0].score is not None
    assert [h.id for h in dense] == [ref.id] and "_distance" in dense[0].fields
    assert [h.id for h in subject] == [ref.id]
    assert [h.fields["entry_id"] for h in every] == [ref.entry_id]
    orphans = await index.orphans(
        all_of(_USER.to_filter(), eq("parent_type", "cluster")), limit=10
    )
    assert orphans == []


async def test_atomic_fact_find_with_and_without_limit(storage: Storage) -> None:
    await storage.repos.atomic_fact.upsert(
        [
            AtomicFactRow(
                id=f"u1_af_{i}",
                entry_id=f"af_{i}",
                owner_id="u1",
                owner_type="user",
                timestamp=dt.datetime(2025, 5, 14, tzinfo=dt.UTC),
                parent_id="ep_1",
                sender_ids=["u1"],
                fact=f"fact {i}",
                fact_tokens=f"fact {i}",
                md_path="users/u1/.atomic_facts/f.md",
                content_sha256="x" * 64,
            )
            for i in range(3)
        ]
    )
    index = AtomicFactIndexAdapter(storage.repos.atomic_fact)

    unbounded = await index.find(eq("parent_id", "ep_1"))
    bounded = await index.find(eq("parent_id", "ep_1"), limit=2)

    assert sorted(h.id for h in unbounded) == ["u1_af_0", "u1_af_1", "u1_af_2"]
    assert len(bounded) == 2
    assert all(h.fields["md_path"] for h in bounded)


async def test_agent_case_reads_return_domain_cases(storage: Storage) -> None:
    await storage.repos.agent_case.upsert(
        [
            AgentCaseRow(
                id="a1_ac_1",
                entry_id="ac_1",
                owner_id="a1",
                owner_type="agent",
                session_id="s1",
                timestamp=dt.datetime(2025, 5, 14, 10, tzinfo=dt.UTC),
                parent_id="mc_1",
                quality_score=0.8,
                task_intent="fix flaky test",
                task_intent_tokens="fix flaky test",
                approach="retry",
                approach_tokens="retry",
                md_path="agents/a1/.agent_cases/c.md",
                content_sha256="x" * 64,
            )
        ]
    )
    index = AgentCaseIndexAdapter(storage.repos.agent_case)
    expected = AgentCase(
        owner_id="a1",
        task_intent="fix flaky test",
        approach="retry",
        quality_score=0.8,
        timestamp=_TS,
        session_id="s1",
        parent_id="mc_1",
    )

    (stored,) = await index.find_by_entry_ids(_AGENT, ["ac_1"])
    page = await index.paginate(_AGENT.to_filter(), Page(sort_by="timestamp"))

    assert stored.item == expected
    assert stored.ref.id == "a1_ac_1"
    assert page.total == 1 and page.items[0].item == expected


async def test_agent_skill_reads_return_domain_skills(storage: Storage) -> None:
    await storage.repos.agent_skill.upsert(
        [
            AgentSkillRow(
                id=f"a1_{name}",
                owner_id="a1",
                owner_type="agent",
                name=name,
                description=f"desc {name}",
                description_tokens=f"desc {name}",
                content="body",
                content_tokens="body",
                confidence=0.5,
                maturity_score=0.6,
                source_case_ids=["ac_1"],
                cluster_id="cl_1",
                md_path=f"agents/a1/skills/skill_{name}/SKILL.md",
                content_sha256="x" * 64,
                vector=vector,
            )
            for name, vector in (
                ("near", [1.0] + [0.0] * (_DIM - 1)),
                ("far", [0.0] * (_DIM - 1) + [1.0]),
            )
        ]
    )
    index = AgentSkillIndexAdapter(storage.repos.agent_skill)

    top = await index.top_k_in_cluster(_AGENT, "cl_1", [1.0] + [0.0] * (_DIM - 1), k=1)
    page = await index.paginate(_AGENT.to_filter(), Page(sort_by="updated_at"))

    assert [s.item.name for s in top] == ["near"]
    assert top[0].ref.entry_id == "near"
    assert top[0].item.source_case_ids == ["ac_1"]
    assert await index.count() == 2
    assert await index.count_in_cluster(_AGENT, "cl_1") == 2
    assert page.total == 2


async def test_profile_get_decodes_the_json_buckets(storage: Storage) -> None:
    await storage.repos.user_profile.upsert(
        [
            UserProfileRow(
                id="u1",
                owner_id="u1",
                owner_type="user",
                summary="likes tea",
                explicit_info_json=json.dumps([{"description": "tea"}]),
                implicit_traits_json="{not json",
                profile_timestamp_ms=_TS,
                md_path="users/u1/user.md",
                content_sha256="x" * 64,
            )
        ]
    )

    stored = await ProfileIndexAdapter(storage.repos.user_profile).get(_USER)

    assert stored is not None
    assert stored.item == Profile(
        owner_id="u1",
        summary="likes tea",
        timestamp=_TS,
        explicit_info=[{"description": "tea"}],
        implicit_traits=[],
    )
    assert stored.ref.id == "u1"
    other = Scope(
        app_id="default", project_id="default", owner_id="u2", owner_type="user"
    )
    assert await ProfileIndexAdapter(storage.repos.user_profile).get(other) is None
