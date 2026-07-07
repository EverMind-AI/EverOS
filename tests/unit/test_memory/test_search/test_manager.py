"""Unit tests for ``SearchManager`` with in-memory stub recallers.

These tests exercise the orchestration without touching LanceDB. Every
recaller is replaced by a hand-rolled stub that returns a small
candidate list; the manager's job is to:

* honour the ``owner_type`` hard partition,
* run KEYWORD as sparse-only and leave ``atomic_facts`` empty,
* run VECTOR via MaxSim (ANN atomic_facts -> max-pool -> resolve episodes)
  and refuse when no embedding is wired,
* let HYBRID run without an LLM by default; require LLM only when the
  caller sets ``enable_llm_rerank=True``,
* refuse AGENTIC when reranker / LLM prerequisites are missing,
* delegate AGENTIC to ``search_episodes_agentic`` and return its result.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import pytest
from everalgo.types import Candidate, FactCandidate

from everos.memory.search.dto import SearchMethod, SearchRequest
from everos.memory.search.manager import SearchManager

# ── Stubs ───────────────────────────────────────────────────────────────


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _episode_row(
    eid: str, score: float = 0.8, memcell_id: str | None = None
) -> Candidate:
    return Candidate(
        id=eid,
        score=score,
        source="keyword",
        metadata={
            "owner_id": "alice",
            "owner_type": "user",
            "session_id": "sess_a",
            "timestamp": _ts(),
            "sender_ids": ["alice"],
            "subject": f"subj {eid}",
            "summary": f"summary {eid}",
            "episode": f"body {eid}",
            "entry_id": eid,
            "parent_id": memcell_id if memcell_id is not None else f"mc_{eid}",
        },
    )


def _case_row(cid: str) -> Candidate:
    return Candidate(
        id=cid,
        score=0.7,
        source="keyword",
        metadata={
            "owner_id": "agent_a",
            "owner_type": "agent",
            "session_id": "sess_b",
            "timestamp": _ts(),
            "task_intent": f"intent {cid}",
            "approach": f"approach {cid}",
            "quality_score": 0.8,
        },
    )


def _skill_row(sid: str) -> Candidate:
    return Candidate(
        id=sid,
        score=0.65,
        source="keyword",
        metadata={
            "owner_id": "agent_a",
            "owner_type": "agent",
            "name": f"skill_{sid}",
            "description": f"desc {sid}",
            "content": f"content {sid}",
            "confidence": 0.9,
            "maturity_score": 0.6,
            "source_case_ids": [],
        },
    )


class _StubEpisodeRecaller:
    kind: ClassVar[str] = "episode"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "episode"

    def __init__(self, sparse: list[Candidate], dense: list[Candidate]) -> None:
        self._sparse = sparse
        self._dense = dense
        self.last_where: str | None = None

    async def sparse_recall(
        self, query: str, where: str, *, limit: int
    ) -> list[Candidate]:
        self.last_where = where
        return list(self._sparse[:limit])

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
    ) -> list[Candidate]:
        self.last_where = where
        return list(self._dense[:limit])

    async def fetch_by_parent_ids(
        self, parent_ids: Sequence[str], where: str
    ) -> list[Candidate]:
        by_parent = {str(c.metadata.get("parent_id", "")): c for c in self._dense}
        return [by_parent[p] for p in parent_ids if p in by_parent]

    async def fetch_by_entry_ids(
        self, entry_ids: Sequence[str], where: str
    ) -> list[Candidate]:
        by_entry = {str(c.metadata.get("entry_id", "")): c for c in self._dense}
        return [by_entry[e] for e in entry_ids if e in by_entry]


class _StubAtomicFactRecaller:
    kind: ClassVar[str] = "atomic_fact"
    everalgo_memory_type: ClassVar[str] = "episodic"
    text_field: ClassVar[str] = "fact"

    def __init__(
        self,
        facts_map: dict[str, list[FactCandidate]] | None = None,
        dense: list[Candidate] | None = None,
    ) -> None:
        self._facts_map = facts_map or {}
        self._dense = dense or []

    async def sparse_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return []

    async def dense_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._dense)

    async def facts_for_episodes(
        self,
        ep_to_parents: Mapping[str, Sequence[str]],
        where: str,
        *,
        per_episode: int,
        query_vector: Any = None,
    ) -> dict[str, list[FactCandidate]]:
        # Accepted to match the real recaller signature; stub doesn't use it.
        return {
            eid: self._facts_map.get(eid, [])[:per_episode] for eid in ep_to_parents
        }


class _StubAgentCaseRecaller:
    kind: ClassVar[str] = "agent_case"
    everalgo_memory_type: ClassVar[str] = "case"
    text_field: ClassVar[str] = "task_intent"

    def __init__(self, sparse: list[Candidate], dense: list[Candidate]) -> None:
        self._sparse = sparse
        self._dense = dense

    async def sparse_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._sparse)

    async def dense_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._dense)


class _StubAgentSkillRecaller:
    kind: ClassVar[str] = "agent_skill"
    everalgo_memory_type: ClassVar[str] = "skill"
    text_field: ClassVar[str] = "description"

    def __init__(
        self,
        sparse: list[Candidate],
        dense: list[Candidate],
        by_case: list[Candidate] | None = None,
    ) -> None:
        self._sparse = sparse
        self._dense = dense
        # Bridge recall fixture: reverse-resolved skills (``fetch_by_case_ids``).
        # Default empty — only the bridge tests populate this.
        self._by_case = by_case or []

    async def sparse_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._sparse)

    async def dense_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._dense)

    async def fetch_by_case_ids(
        self, case_ids: Sequence[str], where: str, *, limit: int
    ) -> list[Candidate]:
        return list(self._by_case)


class _StubProfileRecaller:
    async def fetch(self, owner_id: str) -> list:
        return []


class _StubEmbedding:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


# ── Fixtures ────────────────────────────────────────────────────────────


def _build_manager(
    *,
    episode_sparse: list[Candidate] | None = None,
    episode_dense: list[Candidate] | None = None,
    case_sparse: list[Candidate] | None = None,
    case_dense: list[Candidate] | None = None,
    skill_sparse: list[Candidate] | None = None,
    skill_dense: list[Candidate] | None = None,
    skill_by_case: list[Candidate] | None = None,
    facts_map: dict[str, list[FactCandidate]] | None = None,
    atomic_fact_dense: list[Candidate] | None = None,
    embedding: _StubEmbedding | None = None,
    reranker: Any = None,
    llm_client: Any = None,
) -> SearchManager:
    ep_recaller = _StubEpisodeRecaller(episode_sparse or [], episode_dense or [])
    return SearchManager(
        episode_recaller=ep_recaller,
        atomic_fact_recaller=_StubAtomicFactRecaller(facts_map, atomic_fact_dense),
        agent_case_recaller=_StubAgentCaseRecaller(case_sparse or [], case_dense or []),
        agent_skill_recaller=_StubAgentSkillRecaller(
            skill_sparse or [], skill_dense or [], skill_by_case
        ),
        profile_recaller=_StubProfileRecaller(),
        embedding=embedding,
        reranker=reranker,
        llm_client=llm_client,
    )


def _user_req(
    method: SearchMethod = SearchMethod.KEYWORD, **kwargs: Any
) -> SearchRequest:
    return SearchRequest(user_id="alice", query="hi", method=method, **kwargs)


def _agent_req(
    method: SearchMethod = SearchMethod.KEYWORD, **kwargs: Any
) -> SearchRequest:
    return SearchRequest(agent_id="agent_a", query="hi", method=method, **kwargs)


# ── KEYWORD: user owner ────────────────────────────────────────────────


async def test_user_keyword_returns_episodes_only() -> None:
    mgr = _build_manager(episode_sparse=[_episode_row("ep_1")])
    resp = await mgr.search(_user_req())
    assert len(resp.request_id) == 32 and all(
        c in "0123456789abcdef" for c in resp.request_id
    )
    assert len(resp.data.episodes) == 1
    assert resp.data.episodes[0].id == "ep_1"
    assert resp.data.episodes[0].user_id == "alice"
    assert resp.data.episodes[0].type == "Conversation"
    # Agent paths stay empty.
    assert resp.data.agent_cases == []
    assert resp.data.agent_skills == []
    assert resp.data.profiles == []


async def test_user_keyword_leaves_atomic_facts_empty() -> None:
    """KEYWORD never back-fills facts — only HYBRID produces relevance-scored facts.

    Even if the facts repository would return rows for the matched
    episode, the keyword path must leave ``atomic_facts=[]``: there is
    no per-query score for those facts, so emitting them would muddy
    the contract (mirrors enterprise where event_log is a separate
    memory_type, not auto-attached to episodic results).
    """
    fact = FactCandidate(
        id="f1",
        parent_episode_id="ep_1",
        score=0.0,
        metadata={"fact": "Alice prefers oat milk"},
    )
    mgr = _build_manager(
        episode_sparse=[_episode_row("ep_1")],
        facts_map={"ep_1": [fact]},
    )
    resp = await mgr.search(_user_req())
    ep = resp.data.episodes[0]
    assert ep.atomic_facts == []


async def test_user_keyword_no_results() -> None:
    resp = await _build_manager().search(_user_req())
    assert resp.data.episodes == []


async def test_user_keyword_filters_compile_pinned_owner() -> None:
    """``compile_filters`` should pin owner_id / owner_type on the where."""
    recaller = _StubEpisodeRecaller([_episode_row("ep_1")], [])
    mgr = SearchManager(
        episode_recaller=recaller,
        atomic_fact_recaller=_StubAtomicFactRecaller(),
        agent_case_recaller=_StubAgentCaseRecaller([], []),
        agent_skill_recaller=_StubAgentSkillRecaller([], []),
        profile_recaller=_StubProfileRecaller(),
        embedding=None,
        reranker=None,
        llm_client=None,
    )
    await mgr.search(_user_req())
    assert recaller.last_where is not None
    assert "owner_id = 'alice'" in recaller.last_where
    assert "owner_type = 'user'" in recaller.last_where


def _atomic_fact_row(fid: str, *, parent_id: str, score: float) -> Candidate:
    """Atomic-fact candidate emitted by ``AtomicFactRecaller.dense_recall``."""
    return Candidate(
        id=fid,
        score=score,
        source="vector",
        metadata={
            "owner_id": "alice",
            "owner_type": "user",
            "session_id": "sess_a",
            "timestamp": _ts(),
            "sender_ids": ["alice"],
            "parent_id": parent_id,
            "fact": f"fact {fid}",
        },
    )


# ── VECTOR (MaxSim atomic) ────────────────────────────────────────────


async def test_vector_method_requires_embedding() -> None:
    mgr = _build_manager()  # embedding=None by default
    with pytest.raises(RuntimeError, match="embedding"):
        await mgr.search(_user_req(method=SearchMethod.VECTOR))


async def test_vector_method_returns_episodes_via_maxsim() -> None:
    mgr = _build_manager(
        episode_sparse=[_episode_row("should_not_appear")],
        episode_dense=[_episode_row("ep_dense")],
        atomic_fact_dense=[
            _atomic_fact_row("f1", parent_id="ep_dense", score=0.85),
        ],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.VECTOR))
    assert [e.id for e in resp.data.episodes] == ["ep_dense"]


async def test_vector_radius_filter_drops_below_threshold() -> None:
    mgr = _build_manager(
        episode_dense=[
            _episode_row("ep_low"),
            _episode_row("ep_high"),
        ],
        atomic_fact_dense=[
            _atomic_fact_row("f_low", parent_id="ep_low", score=0.3),
            _atomic_fact_row("f_high", parent_id="ep_high", score=0.9),
        ],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.VECTOR, radius=0.5))
    assert [e.id for e in resp.data.episodes] == ["ep_high"]


async def test_unlimited_mode_applies_default_radius_for_vector() -> None:
    """``top_k=-1`` without an explicit radius gets the project default 0.5.

    Mirrors enterprise's auto-floor behaviour — unlimited mode must not
    return arbitrarily low-similarity tail.
    """
    mgr = _build_manager(
        episode_dense=[
            _episode_row("ep_low"),
            _episode_row("ep_mid"),
            _episode_row("ep_high"),
        ],
        atomic_fact_dense=[
            _atomic_fact_row("f_low", parent_id="ep_low", score=0.3),  # below 0.5
            _atomic_fact_row("f_mid", parent_id="ep_mid", score=0.55),  # above 0.5
            _atomic_fact_row("f_high", parent_id="ep_high", score=0.9),
        ],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.VECTOR, top_k=-1))
    # Ordered by max-pooled fact score descending.
    assert [e.id for e in resp.data.episodes] == ["ep_high", "ep_mid"]


async def test_unlimited_mode_explicit_radius_overrides_default() -> None:
    """Caller-supplied radius (even ``0.0``) wins over the unlimited default."""
    mgr = _build_manager(
        episode_dense=[
            _episode_row("ep_low"),
            _episode_row("ep_high"),
        ],
        atomic_fact_dense=[
            _atomic_fact_row("f_low", parent_id="ep_low", score=0.2),
            _atomic_fact_row("f_high", parent_id="ep_high", score=0.9),
        ],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.VECTOR, top_k=-1, radius=0.1))
    # 0.1 threshold keeps both rows (the default 0.5 would have dropped ep_low).
    assert {e.id for e in resp.data.episodes} == {"ep_low", "ep_high"}


async def test_normal_mode_keeps_full_pool_when_no_radius() -> None:
    """``top_k > 0`` without a radius applies no threshold — truncation handles tail."""
    mgr = _build_manager(
        episode_dense=[
            _episode_row("ep_low"),
            _episode_row("ep_high"),
        ],
        atomic_fact_dense=[
            _atomic_fact_row("f_low", parent_id="ep_low", score=0.2),
            _atomic_fact_row("f_high", parent_id="ep_high", score=0.9),
        ],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.VECTOR, top_k=10))
    # No radius default in normal mode -> both kept.
    assert {e.id for e in resp.data.episodes} == {"ep_low", "ep_high"}


async def test_vector_maxsim_max_pools_facts_to_episodes() -> None:
    """ANN atomic_facts -> max-pool by episode entry_id -> resolve to
    episode, ordering episodes by the per-episode maximum fact score."""
    mgr = _build_manager(
        episode_dense=[
            _episode_row("ep_A", memcell_id="mc_A"),
            _episode_row("ep_B", memcell_id="mc_B"),
        ],
        atomic_fact_dense=[
            _atomic_fact_row("f_A1", parent_id="ep_A", score=0.95),
            _atomic_fact_row("f_A2", parent_id="ep_A", score=0.40),
            _atomic_fact_row("f_B1", parent_id="ep_B", score=0.75),
        ],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.VECTOR, top_k=5))
    eps = resp.data.episodes
    # Both episodes returned, ordered by max-pool score desc.
    assert [e.id for e in eps] == ["ep_A", "ep_B"]
    assert eps[0].score == pytest.approx(0.95)  # max(0.95, 0.40)
    assert eps[1].score == pytest.approx(0.75)


async def test_vector_returns_empty_when_no_facts() -> None:
    """No fact recall -> no episodes to score -> empty episode list."""
    mgr = _build_manager(
        episode_dense=[_episode_row("ep_A", memcell_id="mc_A")],
        atomic_fact_dense=[],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.VECTOR, top_k=5))
    assert resp.data.episodes == []


# ── HYBRID / AGENTIC: prerequisite errors ──────────────────────────────


async def test_hybrid_requires_embedding() -> None:
    mgr = _build_manager()
    with pytest.raises(RuntimeError, match="embedding"):
        await mgr.search(_user_req(method=SearchMethod.HYBRID))


async def test_hybrid_does_not_require_llm_by_default() -> None:
    """HYBRID no longer auto-pulls LLM. With enable_llm_rerank=False the
    fusion-only path (RRF / LR) should run without an LLM client."""
    mgr = _build_manager(embedding=_StubEmbedding())
    # Should not raise: no LLM needed when caller opts out of Phase-5 rerank.
    resp = await mgr.search(_user_req(method=SearchMethod.HYBRID))
    assert resp.data.episodes == []  # empty stub recallers → empty result


async def test_hybrid_requires_llm_when_enable_llm_rerank_true() -> None:
    """Setting ``enable_llm_rerank=True`` makes the LLM mandatory."""
    mgr = _build_manager(embedding=_StubEmbedding())
    with pytest.raises(RuntimeError, match="enable_llm_rerank"):
        await mgr.search(_user_req(method=SearchMethod.HYBRID, enable_llm_rerank=True))


async def test_user_hybrid_episode_fuses_and_evicts_facts() -> None:
    """HYBRID episode path: heap-expand pipeline (RRF -> LR -> expansion).

    ep_1 has a fact scoring higher than its LR score -> fact evicts episode.
    ep_2 has no facts -> episode emitted as-is.
    """
    ep1 = _episode_row("ep_1", score=0.8, memcell_id="mc_1")
    ep2 = _episode_row("ep_2", score=0.7, memcell_id="mc_2")
    fact1 = FactCandidate(
        id="f1",
        parent_episode_id="ep_1",
        score=0.95,
        metadata={"fact": "Alice prefers oat milk"},
    )
    mgr = _build_manager(
        episode_sparse=[ep1, ep2],
        episode_dense=[ep1, ep2],
        facts_map={"ep_1": [fact1]},
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.HYBRID, top_k=10))
    eps = resp.data.episodes
    assert len(eps) >= 1
    ep1_result = next((e for e in eps if e.id == "ep_1"), None)
    assert ep1_result is not None
    assert len(ep1_result.atomic_facts) == 1
    assert ep1_result.atomic_facts[0].id == "f1"


async def test_agentic_requires_reranker_and_llm() -> None:
    mgr = _build_manager(embedding=_StubEmbedding())
    with pytest.raises(RuntimeError, match="rerank provider"):
        await mgr.search(_user_req(method=SearchMethod.AGENTIC))


async def test_agent_hybrid_requires_reranker_without_llm_rerank() -> None:
    """``owner_type='agent'`` + HYBRID + ``enable_llm_rerank=False`` reaches
    the skill cross-encoder lane (``skill_hybrid``: rrf → cross-encoder),
    so a missing rerank provider must fail-fast with a config hint rather
    than crash deep inside the rerank callback.
    """
    mgr = _build_manager(embedding=_StubEmbedding())
    with pytest.raises(RuntimeError, match="rerank provider"):
        await mgr.search(_agent_req(method=SearchMethod.HYBRID))


async def test_agent_hybrid_with_llm_rerank_does_not_need_reranker() -> None:
    """The LLM-rerank lane skips the cross-encoder and dispatches through
    ``arank`` instead, so a missing reranker is fine as long as the LLM
    client is configured. Empty stub recallers → empty result; the call
    must not raise on the reranker-absence path.
    """
    mgr = _build_manager(embedding=_StubEmbedding(), llm_client=_StubLLM())
    resp = await mgr.search(
        _agent_req(method=SearchMethod.HYBRID, enable_llm_rerank=True)
    )
    assert resp.data.agent_skills == []
    assert resp.data.agent_cases == []


class _StubReranker:
    """Minimal reranker stub — returns trivial scores."""

    async def rerank(self, query: str, documents: Sequence[str]) -> list[Any]:
        from everos.component.rerank.protocol import RerankResult

        return [RerankResult(index=i, score=1.0) for i in range(len(documents))]


class _StubLLM:
    """Minimal LLM stub — satisfies protocol without making real calls."""

    async def chat(self, *args: Any, **kwargs: Any) -> Any:
        return ""


async def test_agentic_episode_delegates_to_search_episodes_agentic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENTIC method delegates to search_episodes_agentic and returns its result."""
    import datetime as _dt

    from everos.memory.search.dto import SearchEpisodeItem

    fake_result = [
        SearchEpisodeItem(
            id="ep_1",
            score=0.9,
            session_id="s",
            user_id="alice",
            timestamp=_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
            sender_ids=["alice"],
            subject="s",
            summary="s",
            episode="body",
            type="Conversation",
            atomic_facts=[],
        )
    ]

    async def _fake_agentic(*args: Any, **kwargs: Any) -> list[SearchEpisodeItem]:
        return fake_result

    monkeypatch.setattr(
        "everos.memory.search.manager.search_episodes_agentic", _fake_agentic
    )

    mgr = _build_manager(
        embedding=_StubEmbedding(),
        reranker=_StubReranker(),
        llm_client=_StubLLM(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.AGENTIC))
    assert resp.data.episodes == fake_result


# ── AGENT owner hard partition ─────────────────────────────────────────


async def test_agent_keyword_returns_cases_and_skills_only() -> None:
    mgr = _build_manager(
        case_sparse=[_case_row("c_1")],
        skill_sparse=[_skill_row("s_1")],
    )
    resp = await mgr.search(_agent_req())
    assert resp.data.episodes == []
    assert resp.data.profiles == []
    assert [c.id for c in resp.data.agent_cases] == ["c_1"]
    assert [s.id for s in resp.data.agent_skills] == ["s_1"]


async def test_agent_owner_ignores_include_profile() -> None:
    """Profile is user-only at this revision."""
    mgr = _build_manager()
    resp = await mgr.search(_agent_req(include_profile=True))
    assert resp.data.profiles == []


# ── Top-k behaviour ───────────────────────────────────────────────────


async def test_top_k_truncates_results() -> None:
    rows = [_episode_row(f"ep_{i}", score=1.0 - i * 0.01) for i in range(10)]
    mgr = _build_manager(episode_sparse=rows)
    resp = await mgr.search(_user_req(top_k=3))
    assert [e.id for e in resp.data.episodes] == ["ep_0", "ep_1", "ep_2"]


async def test_top_k_minus_one_caps_at_100() -> None:
    rows = [_episode_row(f"ep_{i}") for i in range(120)]
    mgr = _build_manager(episode_sparse=rows)
    resp = await mgr.search(_user_req(top_k=-1))
    assert len(resp.data.episodes) == 100


# ── AGENTIC agent_case / agent_skill delegation ───────────────────────────


async def test_agentic_agent_cases_delegates_to_search_agent_cases_agentic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENTIC method for agent owner delegates to search_agent_cases_agentic."""
    import datetime as _dt

    from everos.memory.search.dto import SearchAgentCaseItem

    fake_cases = [
        SearchAgentCaseItem(
            id="c_1",
            agent_id="agent_a",
            session_id="sess_b",
            timestamp=_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
            task_intent="handle login",
            approach="retry with backoff",
            quality_score=0.9,
            score=0.85,
        )
    ]

    async def _fake_cases_agentic(
        *args: Any, **kwargs: Any
    ) -> list[SearchAgentCaseItem]:
        return fake_cases

    monkeypatch.setattr(
        "everos.memory.search.manager.search_agent_cases_agentic",
        _fake_cases_agentic,
    )

    mgr = _build_manager(
        embedding=_StubEmbedding(),
        reranker=_StubReranker(),
        llm_client=_StubLLM(),
    )
    resp = await mgr.search(_agent_req(method=SearchMethod.AGENTIC))
    assert resp.data.agent_cases == fake_cases


async def test_agentic_agent_skills_delegates_to_search_agent_skills_agentic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AGENTIC method for agent owner delegates to search_agent_skills_agentic."""

    from everos.memory.search.dto import SearchAgentSkillItem

    fake_skills = [
        SearchAgentSkillItem(
            id="s_1",
            agent_id="agent_a",
            name="auth_refresh",
            description="Refreshes auth tokens",
            content="Retry with new token",
            confidence=0.9,
            maturity_score=0.7,
            source_case_ids=[],
            score=0.8,
        )
    ]

    async def _fake_skills_agentic(
        *args: Any, **kwargs: Any
    ) -> list[SearchAgentSkillItem]:
        return fake_skills

    monkeypatch.setattr(
        "everos.memory.search.manager.search_agent_skills_agentic",
        _fake_skills_agentic,
    )

    mgr = _build_manager(
        embedding=_StubEmbedding(),
        reranker=_StubReranker(),
        llm_client=_StubLLM(),
    )
    resp = await mgr.search(_agent_req(method=SearchMethod.AGENTIC))
    assert resp.data.agent_skills == fake_skills


# ── _merge_by_id_max / _case_bridged_skills helpers ──────────────────────


def test_merge_by_id_max_keeps_higher_score_on_collision() -> None:
    """Same-id collision → keep the higher score; non-colliding rows are
    unioned. Used to fold bridge candidates into the direct dense pool.
    """
    from everos.memory.search.manager import _merge_by_id_max

    primary = [
        Candidate(id="s1", score=0.5, source="vector", metadata={"src": "primary"}),
        Candidate(id="s2", score=0.7, source="vector", metadata={"src": "primary"}),
    ]
    extra = [
        Candidate(id="s1", score=0.9, source="vector", metadata={"src": "bridge"}),
        Candidate(id="s2", score=0.3, source="vector", metadata={"src": "bridge"}),
        Candidate(id="s3", score=0.6, source="vector", metadata={"src": "bridge"}),
    ]
    merged = {c.id: c for c in _merge_by_id_max(primary, extra)}
    # s1 collision → bridge wins (0.9 > 0.5); s2 collision → primary wins
    # (0.7 > 0.3); s3 fresh-from-bridge is added.
    assert merged["s1"].score == 0.9
    assert merged["s1"].metadata["src"] == "bridge"
    assert merged["s2"].score == 0.7
    assert merged["s2"].metadata["src"] == "primary"
    assert merged["s3"].score == 0.6


async def test_case_bridged_skills_max_pools_score_across_source_cases() -> None:
    """Each bridged skill inherits the highest score among its matched
    source cases (mirrors the ``maxsim_atomic`` fact→episode pooling).
    Source cases not present in the bridge pool are ignored.
    """
    skill_row = Candidate(
        id="agent_a_skill_x",
        score=0.0,  # bridge ignores the recaller-side score
        source="vector",
        metadata={"source_case_ids": ["c1", "c2", "c3"], "name": "x"},
    )
    mgr = _build_manager(skill_by_case=[skill_row])
    bridge_cases = [
        Candidate(id="c1", score=0.4, source="vector", metadata={}),
        Candidate(id="c2", score=0.9, source="vector", metadata={}),  # max wins
        Candidate(id="c_other", score=0.7, source="vector", metadata={}),
    ]
    bridged = await mgr._case_bridged_skills(bridge_cases, where="", top_k=5)
    assert len(bridged) == 1
    assert bridged[0].id == "agent_a_skill_x"
    # c1=0.4 and c2=0.9 are in the bridge pool; c3 is not → max-pool == 0.9.
    assert bridged[0].score == pytest.approx(0.9)
    # Metadata (incl. ``source_case_ids``) rides through so downstream
    # shaping doesn't need a second fetch.
    assert bridged[0].metadata["source_case_ids"] == ["c1", "c2", "c3"]


async def test_case_bridged_skills_returns_empty_for_none_or_empty_input() -> None:
    """No bridge cases ⇒ no bridge recall (skip the reverse fetch entirely).
    This is the cross-encoder lane / KEYWORD / VECTOR contract.
    """
    mgr = _build_manager(skill_by_case=[_skill_row("s1")])  # noise the stub
    assert await mgr._case_bridged_skills(None, where="", top_k=5) == []
    assert await mgr._case_bridged_skills([], where="", top_k=5) == []


# ── Agent HYBRID lane selection ──────────────────────────────────────────


async def test_agent_hybrid_no_llm_rerank_runs_cross_encoder_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enable_llm_rerank=False`` for agent HYBRID must dispatch through
    ``search_agent_skills_hybrid`` (rrf → cross-encoder lane) with the
    configured reranker, not through generic ``arank``.
    """
    captured: dict[str, Any] = {}

    async def _fake_hybrid(
        query: str,
        *,
        sparse: list[Candidate],
        dense: list[Candidate],
        reranker: Any,
        top_k: int,
    ) -> list:
        captured.update(
            query=query, sparse=sparse, dense=dense, reranker=reranker, top_k=top_k
        )
        return []

    monkeypatch.setattr(
        "everos.memory.search.manager.search_agent_skills_hybrid", _fake_hybrid
    )
    stub_reranker = _StubReranker()
    mgr = _build_manager(embedding=_StubEmbedding(), reranker=stub_reranker)
    await mgr.search(_agent_req(method=SearchMethod.HYBRID))

    assert captured["query"] == "hi"
    # Manager forwards its configured reranker to the cross-encoder lane.
    assert captured["reranker"] is stub_reranker
    # Agent kinds cap unlimited-mode top_k at _AGENT_TOP_K_CAP (10).
    assert captured["top_k"] == 10


async def test_agent_hybrid_llm_rerank_dispatches_arank_for_case_then_skill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM rerank lane: ``_search_cases_and_skills`` runs serially —
    ``arank`` is called once with ``memory_type="case"`` and once with
    ``memory_type="skill"``, both with ``enable_rerank=True`` + the LLM
    client. Order matters: the case call must precede the skill call so
    its results can feed the bridge.
    """
    from everalgo.types import RankOutput

    calls: list[tuple[str, dict[str, Any]]] = []

    async def _fake_arank(rank_input: Any, **kwargs: Any) -> RankOutput:
        calls.append((rank_input.memory_type, kwargs))
        return RankOutput(items=[], metadata={})

    monkeypatch.setattr("everos.memory.search.manager.arank", _fake_arank)
    mgr = _build_manager(embedding=_StubEmbedding(), llm_client=_StubLLM())
    await mgr.search(_agent_req(method=SearchMethod.HYBRID, enable_llm_rerank=True))

    # Two dispatches in the documented serial order.
    assert [c[0] for c in calls] == ["case", "skill"]
    # Both runs opt into rerank with the LLM client wired in.
    for _mt, kw in calls:
        assert kw["enable_rerank"] is True
        assert kw["llm"] is mgr._llm
        assert kw["rerank_top_k"] == 10  # _AGENT_TOP_K_CAP


async def test_agent_hybrid_llm_rerank_merges_bridged_skills_into_dense_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bridge must surface into the skill dispatch: skills resolved
    by ``fetch_by_case_ids`` are max-pooled into the dense candidates that
    ``arank`` sees on the second call, while the direct skill recall pool
    is preserved.
    """
    from everalgo.types import RankOutput, ScoredItem

    case_result = ScoredItem(
        id="agent_a_c1",
        score=0.85,
        item_type="case",
        # Shaper requires owner_type="agent" + timestamp + intent/approach;
        # otherwise the case is dropped and bridge_cases comes back empty.
        metadata={
            "owner_id": "agent_a",
            "owner_type": "agent",
            "session_id": "sess_b",
            "timestamp": _ts(),
            "task_intent": "intent c1",
            "approach": "approach c1",
            "quality_score": 0.8,
        },
    )
    skill_direct = _skill_row("s_direct")
    skill_bridged = Candidate(
        id="s_bridged",
        score=0.0,
        source="vector",
        metadata={"source_case_ids": ["agent_a_c1"], "name": "s_bridged"},
    )

    seen_skill_dense: dict[str, list[Candidate]] = {}

    async def _fake_arank(rank_input: Any, **_: Any) -> RankOutput:
        if rank_input.memory_type == "case":
            return RankOutput(items=[case_result], metadata={})
        # skill call — capture the merged dense pool the manager built.
        seen_skill_dense["dense"] = list(rank_input.dense_candidates)
        return RankOutput(items=[], metadata={})

    monkeypatch.setattr("everos.memory.search.manager.arank", _fake_arank)
    mgr = _build_manager(
        embedding=_StubEmbedding(),
        llm_client=_StubLLM(),
        skill_sparse=[],
        skill_dense=[skill_direct],
        skill_by_case=[skill_bridged],
    )
    await mgr.search(_agent_req(method=SearchMethod.HYBRID, enable_llm_rerank=True))

    dense_ids = {c.id for c in seen_skill_dense["dense"]}
    # Direct dense recall is preserved AND the case-bridged skill is unioned.
    assert dense_ids == {"s_direct", "s_bridged"}
    # The bridged skill inherits the matched case's score (0.85 from c1).
    by_id = {c.id: c for c in seen_skill_dense["dense"]}
    assert by_id["s_bridged"].score == pytest.approx(0.85)
