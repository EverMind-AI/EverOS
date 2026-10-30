"""Unit tests for ``SearchManager`` with in-memory stub recallers.

These tests exercise the orchestration without touching LanceDB. Every
recaller is replaced by a hand-rolled stub that returns a small
candidate list; the manager's job is to:

* honour the ``owner_type`` hard partition,
* run KEYWORD as sparse-only and leave ``atomic_facts`` empty,
* run VECTOR via MaxSim (ANN atomic_facts -> max-pool -> resolve episodes),
  fall back to the episode vector when no fact is in scope, and refuse
  when no embedding is wired,
* let HYBRID run without an LLM by default; require LLM only when the
  caller sets ``enable_llm_rerank=True``,
* refuse AGENTIC when reranker / LLM prerequisites are missing,
* delegate AGENTIC to ``search_episodes_agentic`` and return its result.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest

from everos.component.tokenizer import Tokenizer
from everos.config import load_settings
from everos.core.errors import ProviderNotConfiguredError
from everos.memory.models import CanonicalMessage, Cluster
from everos.memory.ports import Scope, SearchHit, SessionKey
from everos.memory.search.dto import SearchMethod, SearchRequest
from everos.memory.search.manager import SearchManager
from everos.memory.search.recall import (
    AgentCaseRecaller,
    AgentSkillRecaller,
    AtomicFactRecaller,
    EpisodeRecaller,
)
from everos.memory.search.recall.base import RecallerDeps
from tests.fakes import static_trace_path, static_tuning, unscripted_llm

from ._records import (
    ScriptedIndex,
    hit,
    stored_case,
    stored_episode,
    stored_fact,
    stored_skill,
)

# ── Stubs ───────────────────────────────────────────────────────────────


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _keyed(stored: Any, record_id: str) -> Any:
    """Give a stored record the global id ``record_id``."""
    return dataclasses.replace(
        stored, ref=dataclasses.replace(stored.ref, id=record_id)
    )


def _episode_row(
    eid: str, score: float = 0.8, memcell_id: str | None = None
) -> SearchHit[Any]:
    """An episode match whose global id and entry id are both ``eid``."""
    stored = stored_episode(
        eid, parent_id=memcell_id if memcell_id is not None else f"mc_{eid}"
    )
    return hit(_keyed(stored, eid), score)


def _case_row(cid: str) -> SearchHit[Any]:
    stored = stored_case(
        cid,
        owner_id="agent_a",
        task_intent=f"intent {cid}",
        approach=f"approach {cid}",
    )
    return hit(_keyed(stored, cid), 0.7)


def _skill_row(sid: str, source_case_ids: Sequence[str] = ()) -> SearchHit[Any]:
    stored = stored_skill(
        sid,
        owner_id="agent_a",
        name=f"skill_{sid}",
        description=f"desc {sid}",
        source_case_ids=source_case_ids,
    )
    return hit(stored, 0.65)


def _atomic_fact_row(fid: str, *, parent_id: str, score: float) -> SearchHit[Any]:
    """An atomic-fact match hanging off episode entry ``parent_id``."""
    return hit(_keyed(stored_fact(fid, parent_id=parent_id), fid), score)


def _tokenizer() -> Tokenizer:
    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = ["hi"]
    return tok


class _StubProfileRecaller:
    async def fetch(self, owner_id: str, **_: Any) -> list[Any]:
        return []


class _EmptySessionBuffer:
    """No session has buffered messages."""

    async def list(self, session: SessionKey) -> list[CanonicalMessage]:
        return []


class _NoClusters:
    """No owner has clusters."""

    async def list_for_owner(self, scope: Scope, kind: str) -> list[Cluster]:
        return []


_NO_SESSION_BUFFER: Any = _EmptySessionBuffer()
_NO_CLUSTERS: Any = _NoClusters()


class _StubEmbedding:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    async def embed(self, text: str) -> list[float]:
        return [0.0] * self.dim

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        return await self.embed(text)


# ── Fixtures ────────────────────────────────────────────────────────────


def _build_manager(
    *,
    episode_sparse: list[SearchHit[Any]] | None = None,
    episode_dense: list[SearchHit[Any]] | None = None,
    case_sparse: list[SearchHit[Any]] | None = None,
    case_dense: list[SearchHit[Any]] | None = None,
    skill_sparse: list[SearchHit[Any]] | None = None,
    skill_dense: list[SearchHit[Any]] | None = None,
    skill_by_case: list[SearchHit[Any]] | None = None,
    atomic_fact_dense: list[SearchHit[Any]] | None = None,
    embedding: _StubEmbedding | None = None,
    reranker: Any = None,
    llm_client: Any = None,
) -> SearchManager:
    """A manager over the real recallers, each reading a scripted index.

    ``llm_client`` (also the decider) defaults to one that fails the test if
    called.

    ``find`` on the episode index evaluates its filter (the MaxSim parent
    fetch narrows by entry id), and so does the skill index (the case→skill
    bridge narrows by ``source_case_ids`` membership); the fact index answers
    both the fact ANN and the HYBRID fact expansion from ``atomic_fact_dense``.
    """
    deps = RecallerDeps(tokenizer=_tokenizer())
    episodes = [*(episode_sparse or []), *(episode_dense or [])]
    return SearchManager(
        episode_recaller=EpisodeRecaller(
            deps,
            ScriptedIndex(  # type: ignore[arg-type]
                sparse=episode_sparse or [],
                dense=episode_dense or [],
                found=episodes,
                filter_found=True,
            ),
        ),
        atomic_fact_recaller=AtomicFactRecaller(
            deps,
            ScriptedIndex(  # type: ignore[arg-type]
                dense=atomic_fact_dense or [],
                found=atomic_fact_dense or [],
            ),
        ),
        agent_case_recaller=AgentCaseRecaller(
            deps,
            ScriptedIndex(sparse=case_sparse or [], dense=case_dense or []),  # type: ignore[arg-type]
        ),
        agent_skill_recaller=AgentSkillRecaller(
            deps,
            ScriptedIndex(  # type: ignore[arg-type]
                sparse=skill_sparse or [],
                dense=skill_dense or [],
                found=skill_by_case or [],
                filter_found=True,
            ),
        ),
        profile_recaller=_StubProfileRecaller(),  # type: ignore[arg-type]
        session_buffer=_NO_SESSION_BUFFER,
        cluster_store=_NO_CLUSTERS,
        embedding=embedding,
        reranker=reranker,
        llm_client=llm_client or unscripted_llm(),
        decider_client=llm_client or unscripted_llm(),
        tuning_source=static_tuning(),
        trace_path_source=static_trace_path(),
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


async def test_recall_hit_emitted_only_for_calibrated_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """recall_hit is meaningful only for calibrated-score methods (HYBRID LR
    / AGENTIC rerank). For KEYWORD (unbounded BM25) the code must emit
    top_score but NOT a hit verdict — a fixed 0.6 threshold against an
    unbounded score is an always-hit signal that inflates the dashboard."""
    import everos.memory.search.manager as mgr_mod

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(mgr_mod, "current_trace_ids", lambda: ("t" * 32, "s" * 16))
    monkeypatch.setattr(mgr_mod, "emit_recall_scores", lambda **kw: calls.append(kw))

    # KEYWORD: raw BM25 score well above the threshold, but uncalibrated.
    kw_mgr = _build_manager(episode_sparse=[_episode_row("ep_1", score=7.5)])
    await kw_mgr.search(_user_req(method=SearchMethod.KEYWORD))
    assert len(calls) == 1
    assert calls[0]["method"] == "keyword"
    assert calls[0]["top_score"] == 7.5
    assert calls[0]["hit"] is None  # no hit verdict for an uncalibrated method

    # HYBRID: calibrated LR score → a real boolean hit verdict is emitted.
    calls.clear()
    hy_mgr = _build_manager(embedding=_StubEmbedding())
    await hy_mgr.search(_user_req(method=SearchMethod.HYBRID))
    assert len(calls) == 1
    assert calls[0]["method"] == "hybrid"
    assert isinstance(calls[0]["hit"], bool)


async def test_search_uses_propagated_request_id_when_bound() -> None:
    """When a request id is bound upstream (middleware), ``search`` reuses it
    instead of minting a fresh one, so the response id matches the trace."""
    from everos.core.context import reset_request_id, set_request_id

    mgr = _build_manager(episode_sparse=[_episode_row("ep_1")])
    token = set_request_id("deadbeef" * 4)
    try:
        resp = await mgr.search(_user_req())
        assert resp.request_id == "deadbeef" * 4
    finally:
        reset_request_id(token)


async def test_user_keyword_leaves_atomic_facts_empty() -> None:
    """KEYWORD never back-fills facts — only HYBRID produces relevance-scored facts.

    Even if the facts repository would return rows for the matched
    episode, the keyword path must leave ``atomic_facts=[]``: there is
    no per-query score for those facts, so emitting them would muddy
    the contract (mirrors enterprise where event_log is a separate
    memory_type, not auto-attached to episodic results).
    """
    mgr = _build_manager(
        episode_sparse=[_episode_row("ep_1")],
        atomic_fact_dense=[_atomic_fact_row("f1", parent_id="ep_1", score=0.9)],
    )
    resp = await mgr.search(_user_req())
    ep = resp.data.episodes[0]
    assert ep.atomic_facts == []


async def test_user_keyword_no_results() -> None:
    resp = await _build_manager().search(_user_req())
    assert resp.data.episodes == []


async def test_user_keyword_filters_compile_pinned_owner() -> None:
    """``compile_filters`` should pin owner_id / owner_type on the where."""
    index = ScriptedIndex(sparse=[_episode_row("ep_1")])
    mgr = SearchManager(
        episode_recaller=EpisodeRecaller(RecallerDeps(tokenizer=_tokenizer()), index),  # type: ignore[arg-type]
        atomic_fact_recaller=MagicMock(),
        agent_case_recaller=MagicMock(),
        agent_skill_recaller=MagicMock(),
        profile_recaller=_StubProfileRecaller(),  # type: ignore[arg-type]
        session_buffer=_NO_SESSION_BUFFER,
        cluster_store=_NO_CLUSTERS,
        embedding=None,
        reranker=None,
        llm_client=unscripted_llm(),
        decider_client=unscripted_llm(),
        tuning_source=static_tuning(),
        trace_path_source=static_trace_path(),
    )
    await mgr.search(_user_req())
    from everos.infra.persistence.lancedb import render_predicate

    rendered = render_predicate(index.calls[-1][2])
    assert "owner_id = 'alice'" in rendered
    assert "owner_type = 'user'" in rendered


# ── VECTOR (MaxSim atomic) ────────────────────────────────────────────


async def test_embed_query_rejects_a_width_that_disagrees_with_dim() -> None:
    """A provider returning a width other than its declared ``dim`` must fail
    at the embed step, not inside LanceDB.

    Sending a mismatched query vector into the engine surfaces as an opaque
    ``ValueError: Invalid input, No vector column found to match…`` only after
    the query has been set up — 13-14s per request in a soak run, as an
    unhandled 500 with a ~6k-line traceback. Here it is microseconds and a
    named error. ``ConfigurationError`` (server-side) rather than
    ``InvalidInputError``: callers only ever send query *text*, so the width is
    entirely our provider's doing.
    """
    from everos.core.errors import ConfigurationError

    class _LyingEmbedding(_StubEmbedding):
        async def embed(self, text: str) -> list[float]:
            return [0.0] * (self.dim // 2)  # declares dim, returns half of it

    mgr = _build_manager(embedding=_LyingEmbedding(dim=8))

    with pytest.raises(ConfigurationError) as excinfo:
        await mgr._embed_query("anything")

    msg = str(excinfo.value)
    assert "4-dimension" in msg, msg
    assert "dim=8" in msg, msg


async def test_vector_method_requires_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _build_manager()  # embedding=None by default
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        await mgr.search(_user_req(method=SearchMethod.VECTOR))
    assert excinfo.value.provider == "embedding"
    assert excinfo.value.feature == "vector"


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


async def test_hybrid_radius_filters_the_dense_pool_before_fusion() -> None:
    """``radius`` is a cosine floor on dense recall for every vector route,
    HYBRID included (#394): a dense-only hit below it never reaches fusion."""
    mgr = _build_manager(
        episode_dense=[_episode_row("ep_low", score=0.3), _episode_row("ep_high", 0.9)],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.HYBRID, top_k=10, radius=0.5))
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


async def test_vector_falls_back_to_episode_vector_when_no_facts() -> None:
    """No fact in scope -> ANN over the episode's own vector, radius still applied.

    Used to return an empty list (issue #502): a store with no atomic facts
    in scope lost vector search silently while keyword kept working.
    """
    mgr = _build_manager(
        episode_dense=[
            _episode_row("ep_high", score=0.9),
            _episode_row("ep_low", score=0.3),
        ],
        atomic_fact_dense=[],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.VECTOR, top_k=5, radius=0.5))
    assert [e.id for e in resp.data.episodes] == ["ep_high"]


# ── HYBRID / AGENTIC: prerequisite errors ──────────────────────────────


async def test_hybrid_requires_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = _build_manager()
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        await mgr.search(_user_req(method=SearchMethod.HYBRID))
    assert excinfo.value.provider == "embedding"
    assert excinfo.value.feature == "user_hybrid"


async def test_hybrid_does_not_call_the_llm_by_default() -> None:
    """HYBRID no longer auto-pulls LLM. With enable_llm_rerank=False the
    fusion-only path (RRF / LR) runs without an LLM call."""
    mgr = _build_manager(embedding=_StubEmbedding())
    # The default LLM fails the test if called.
    resp = await mgr.search(_user_req(method=SearchMethod.HYBRID))
    assert resp.data.episodes == []  # empty stub recallers → empty result


async def test_user_hybrid_episode_fuses_and_evicts_facts() -> None:
    """HYBRID episode path: heap-expand pipeline (RRF -> LR -> expansion).

    ep_1 has a fact scoring higher than its LR score -> fact evicts episode.
    ep_2 has no facts -> episode emitted as-is.
    """
    ep1 = _episode_row("ep_1", score=0.8, memcell_id="mc_1")
    ep2 = _episode_row("ep_2", score=0.7, memcell_id="mc_2")
    mgr = _build_manager(
        episode_sparse=[ep1, ep2],
        episode_dense=[ep1, ep2],
        atomic_fact_dense=[_atomic_fact_row("f1", parent_id="ep_1", score=0.95)],
        embedding=_StubEmbedding(),
    )
    resp = await mgr.search(_user_req(method=SearchMethod.HYBRID, top_k=10))
    eps = resp.data.episodes
    assert len(eps) >= 1
    ep1_result = next((e for e in eps if e.id == "ep_1"), None)
    assert ep1_result is not None
    assert len(ep1_result.atomic_facts) == 1
    assert ep1_result.atomic_facts[0].id == "f1"


async def test_agentic_requires_reranker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mgr = _build_manager(embedding=_StubEmbedding())
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        await mgr.search(_user_req(method=SearchMethod.AGENTIC))
    assert excinfo.value.provider == "rerank"
    assert excinfo.value.feature == "agentic_search"


async def test_agent_hybrid_requires_reranker_without_llm_rerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``owner_type='agent'`` + HYBRID + ``enable_llm_rerank=False`` reaches
    the skill cross-encoder lane (``skill_hybrid``: rrf → cross-encoder),
    so a missing rerank provider must fail-fast with a config hint rather
    than crash deep inside the rerank callback.
    """
    mgr = _build_manager(embedding=_StubEmbedding())
    with pytest.raises(ProviderNotConfiguredError) as excinfo:
        await mgr.search(_agent_req(method=SearchMethod.HYBRID))
    assert excinfo.value.provider == "rerank"
    assert excinfo.value.feature == "agent_hybrid"
    assert excinfo.value.alternative_hint is not None
    assert "enable_llm_rerank" in excinfo.value.alternative_hint


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

    async def rerank(
        self, query: str, documents: Sequence[str], **kwargs: Any
    ) -> list[Any]:
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
            id="agent_a_c_1",
            entry_id="c_1",
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
        sparse: Any,
        dense: Any,
        reranker: Any,
        top_k: int,
    ) -> list[Any]:
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
    """The bridge must surface into the skill dispatch with the ids as they
    are stored: a case's index row id is ``<agent_id>_<entry_id>`` while a
    skill's ``source_case_ids`` holds the md-side entry ids, so the bridge
    has to be fed entry ids (issue #478). The bridged skill is max-pooled
    into the dense candidates that ``arank`` sees on the second call, while
    the direct skill recall pool is preserved.
    """
    from everalgo.types import RankOutput, ScoredItem

    case = stored_case("ac_20261009_0001", owner_id="agent_a")
    assert case.ref.id == "agent_a_ac_20261009_0001"
    case_result = ScoredItem(id=case.ref.id, score=0.85, item_type="case")
    skill_direct = _skill_row("s_direct")
    skill_bridged = _skill_row("s_bridged", source_case_ids=["ac_20261009_0001"])

    seen_skill_dense: dict[str, list[Any]] = {}

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
        case_dense=[hit(case, 0.7)],
        skill_sparse=[],
        skill_dense=[skill_direct],
        skill_by_case=[skill_bridged],
    )
    await mgr.search(_agent_req(method=SearchMethod.HYBRID, enable_llm_rerank=True))

    dense_ids = {c.id for c in seen_skill_dense["dense"]}
    # Direct dense recall is preserved AND the case-bridged skill is unioned.
    assert dense_ids == {"s_direct", "s_bridged"}
    # The bridged skill inherits the matched case's score (0.85 from the case).
    by_id = {c.id: c for c in seen_skill_dense["dense"]}
    assert by_id["s_bridged"].score == pytest.approx(0.85)


async def test_search_emits_memory_search_span() -> None:
    """search() opens an everos.memory.search retriever span carrying the
    langfuse.* attribute contract (observation type / user id / metadata)."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import (
        force_flush,
        init_tracing,
        shutdown_tracing,
    )

    exporter = InMemorySpanExporter()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    try:
        mgr = _build_manager(episode_sparse=[_episode_row("ep_1")])
        await mgr.search(_user_req())
        force_flush()
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert "everos.memory.search" in spans
        attrs = spans["everos.memory.search"].attributes
        assert attrs["langfuse.observation.type"] == "retriever"
        assert attrs["langfuse.user.id"] == "alice"
        assert attrs["langfuse.trace.metadata.owner_type"] == "user"
        assert list(attrs["langfuse.trace.tags"]) == ["everos", "memory"]
    finally:
        shutdown_tracing()


# ── Search sub-span decomposition (recall + rank phases) ────────────────


@pytest.fixture
def _search_spans():  # type: ignore[no-untyped-def]
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from everos.config.settings import ObservabilitySettings
    from everos.core.observability.tracing import init_tracing, shutdown_tracing

    exporter = InMemorySpanExporter()
    shutdown_tracing()
    init_tracing(
        ObservabilitySettings(enabled=True, endpoint="http://collector.invalid"),
        span_processor=SimpleSpanProcessor(exporter),
    )
    yield exporter
    shutdown_tracing()


def _span_index(exporter: Any) -> dict[str, Any]:
    from everos.core.observability.tracing import force_flush

    force_flush()
    return {s.name: s for s in exporter.get_finished_spans()}


async def test_keyword_user_emits_recall_no_rank(_search_spans: Any) -> None:
    mgr = _build_manager(episode_sparse=[_episode_row("ep_1")])
    await mgr.search(_user_req(method=SearchMethod.KEYWORD))
    spans = _span_index(_search_spans)
    assert "everos.search.recall" in spans
    assert "everos.search.rank" not in spans
    # recall nests under the search span (one trace).
    tid = spans["everos.memory.search"].context.trace_id
    assert spans["everos.search.recall"].context.trace_id == tid
    assert spans["everos.search.recall"].parent is not None


async def test_hybrid_user_emits_recall_and_rank(_search_spans: Any) -> None:
    mgr = _build_manager(
        episode_sparse=[_episode_row("ep_1")], embedding=_StubEmbedding()
    )
    await mgr.search(_user_req(method=SearchMethod.HYBRID))
    spans = _span_index(_search_spans)
    assert "everos.search.recall" in spans
    assert "everos.search.rank" in spans


async def test_keyword_agent_emits_recall_no_rank(_search_spans: Any) -> None:
    mgr = _build_manager(case_sparse=[_case_row("c1")], skill_sparse=[_skill_row("s1")])
    await mgr.search(_agent_req(method=SearchMethod.KEYWORD))
    spans = _span_index(_search_spans)
    assert "everos.search.recall" in spans
    assert "everos.search.rank" not in spans


async def test_hybrid_agent_emits_recall_and_rank(_search_spans: Any) -> None:
    mgr = _build_manager(
        case_sparse=[_case_row("c1")],
        skill_sparse=[_skill_row("s1")],
        embedding=_StubEmbedding(),
        reranker=_StubReranker(),
    )
    await mgr.search(_agent_req(method=SearchMethod.HYBRID))
    spans = _span_index(_search_spans)
    assert "everos.search.recall" in spans
    assert "everos.search.rank" in spans


async def test_search_emits_top_score_without_hit_for_keyword(
    _search_spans: Any,
) -> None:
    """KEYWORD sets everos.search.top_score (max item score) but NOT
    everos.search.hit — an unbounded BM25 score forced through a fixed
    threshold would be a misleading always-hit verdict."""
    mgr = _build_manager(episode_sparse=[_episode_row("ep_1", score=0.75)])
    await mgr.search(_user_req(method=SearchMethod.KEYWORD))
    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert attrs["everos.search.top_score"] == pytest.approx(0.75)
    assert "everos.search.hit" not in attrs


def _hit_threshold(monkeypatch: pytest.MonkeyPatch, threshold: float) -> None:
    monkeypatch.setenv("EVEROS_OBSERVABILITY__RECALL_HIT_THRESHOLD", str(threshold))
    load_settings.cache_clear()


async def test_search_emits_hit_when_calibrated_above_threshold(
    _search_spans: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HYBRID is calibrated ([0, 1]) so a top score >= recall_hit_threshold
    sets everos.search.hit = True on the retriever span."""
    _hit_threshold(monkeypatch, 0.0)
    mgr = _build_manager(embedding=_StubEmbedding())  # no candidates: top 0.0
    await mgr.search(_user_req(method=SearchMethod.HYBRID))
    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert attrs["everos.search.top_score"] == pytest.approx(0.0)
    assert attrs["everos.search.hit"] is True


async def test_search_hit_false_when_calibrated_below_threshold(
    _search_spans: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _hit_threshold(monkeypatch, 0.5)
    mgr = _build_manager(embedding=_StubEmbedding())  # no candidates: top 0.0
    await mgr.search(_user_req(method=SearchMethod.HYBRID))
    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert attrs["everos.search.top_score"] == pytest.approx(0.0)
    assert attrs["everos.search.hit"] is False


async def test_search_top_score_zero_without_hit_for_keyword(
    _search_spans: Any,
) -> None:
    mgr = _build_manager()  # no candidates
    await mgr.search(_user_req(method=SearchMethod.KEYWORD))
    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert attrs["everos.search.top_score"] == pytest.approx(0.0)
    assert "everos.search.hit" not in attrs


async def test_search_enqueues_recall_scores(
    _search_spans: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When tracing is active, search hands the top score to the score sink
    with the retriever span's trace_id (032x) + observation_id (016x).
    KEYWORD is uncalibrated so hit is None, which is what makes the sink
    report it as recall_top_score_raw and push no recall_hit."""
    import everos.memory.search.manager as mgr_mod

    captured: dict[str, Any] = {}

    def fake_emit(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(mgr_mod, "emit_recall_scores", fake_emit)
    mgr = _build_manager(episode_sparse=[_episode_row("ep_1", score=0.75)])
    await mgr.search(_user_req(method=SearchMethod.KEYWORD))

    assert captured["top_score"] == pytest.approx(0.75)
    assert captured["hit"] is None
    assert captured["method"] == "keyword"
    assert len(captured["trace_id"]) == 32
    assert len(captured["observation_id"]) == 16


async def test_search_captures_query_when_content_on(_search_spans: Any) -> None:
    from everos.core.observability.tracing import set_capture_content

    set_capture_content(True)
    try:
        mgr = _build_manager(episode_sparse=[_episode_row("ep_1")])
        await mgr.search(_user_req(method=SearchMethod.KEYWORD))  # query="hi"
    finally:
        set_capture_content(False)
    import json

    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert json.loads(attrs["langfuse.observation.input"])["query"] == "hi"


async def test_search_captures_returned_hits_when_content_on(
    _search_spans: Any,
) -> None:
    """capture_content on → the search span records the returned hit ids
    (episodes/agent_cases/agent_skills), not just the query."""
    from everos.core.observability.tracing import set_capture_content

    set_capture_content(True)
    try:
        mgr = _build_manager(episode_sparse=[_episode_row("ep_1")])
        await mgr.search(_user_req(method=SearchMethod.KEYWORD))
    finally:
        set_capture_content(False)
    import json

    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    out = json.loads(attrs["langfuse.observation.output"])
    assert out["episodes"] == ["ep_1"]
    assert out["agent_cases"] == [] and out["agent_skills"] == []


async def test_search_omits_query_when_content_off(_search_spans: Any) -> None:
    mgr = _build_manager(episode_sparse=[_episode_row("ep_1")])
    await mgr.search(_user_req(method=SearchMethod.KEYWORD))
    attrs = _span_index(_search_spans)["everos.memory.search"].attributes
    assert "langfuse.observation.input" not in attrs
