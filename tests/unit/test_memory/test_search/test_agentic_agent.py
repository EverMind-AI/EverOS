"""Unit tests for ``memory.search.agentic_agent``.

White-box: patches ``aagentic_retrieve`` to assert benchmark hyperparameters
are wired correctly, plus a shaping test to verify DTOs are built correctly.

The skill verify step has been removed from production code; this test
module covers the agentic retrieve flow only.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, ClassVar
from unittest.mock import patch

from everalgo.rank.protocols import AgenticDecision
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate

from everos.memory.search.agentic_agent import (
    search_agent_cases_agentic,
    search_agent_skills_agentic,
)
from everos.memory.search.dto import SearchAgentCaseItem, SearchAgentSkillItem

# ── Stubs ────────────────────────────────────────────────────────────────


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _case_candidate(cid: str, score: float = 0.8) -> Candidate:
    return Candidate(
        id=cid,
        score=score,
        source="vector",
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


def _skill_candidate(sid: str, score: float = 0.75) -> Candidate:
    return Candidate(
        id=sid,
        score=score,
        source="vector",
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


class _StubCaseRecaller:
    kind: ClassVar[str] = "agent_case"
    everalgo_memory_type: ClassVar[str] = "case"
    text_field: ClassVar[str] = "task_intent"

    def __init__(self, dense: list[Candidate]) -> None:
        self._dense = dense

    async def sparse_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._dense)

    async def dense_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._dense)


class _StubSkillRecaller:
    kind: ClassVar[str] = "agent_skill"
    everalgo_memory_type: ClassVar[str] = "skill"
    text_field: ClassVar[str] = "description"

    def __init__(self, dense: list[Candidate]) -> None:
        self._dense = dense

    async def sparse_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._dense)

    async def dense_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._dense)


class _StubReranker:
    async def rerank(self, query: str, passages: list[str]) -> list[Any]:
        class _R:
            def __init__(self, idx: int) -> None:
                self.index = idx
                self.score = 1.0 - idx * 0.1

        return [_R(i) for i in range(len(passages))]


async def _fake_embed(q: str) -> list[float]:
    return [0.1, 0.2, 0.3, 0.4]


# ── Tests ─────────────────────────────────────────────────────────────────


async def test_search_agent_cases_agentic_calls_aagentic_retrieve_with_benchmark_params() -> (  # noqa: E501
    None
):
    """Verify aagentic_retrieve called with benchmark hyperparams for agent_case."""
    captured: dict[str, Any] = {}

    async def fake_aagentic(
        query: str,
        *,
        base_retrieve: Any,
        llm: Any,
        rerank_fn: Any,
        round2_retrieve: Any,
        round2_cap: Any,
        top_n: int,
        round1_top_n: int,
        round1_rerank_top_n: int,
        refinement_strategy: str,
        multi_query_count: int,
        rrf_k: int,
    ) -> tuple[list[Candidate], AgenticDecision]:
        captured.update(
            top_n=top_n,
            round1_top_n=round1_top_n,
            round1_rerank_top_n=round1_rerank_top_n,
            round2_cap=round2_cap,
            round2_retrieve_is_none=round2_retrieve is None,
            multi_query_count=multi_query_count,
            rrf_k=rrf_k,
            refinement_strategy=refinement_strategy,
        )
        return [], AgenticDecision(is_multi_round=False)

    with patch("everos.memory.search.agentic_agent.aagentic_retrieve", fake_aagentic):
        await search_agent_cases_agentic(
            "How did agent handle login failure?",
            where="owner_id = 'agent_a' AND owner_type = 'agent'",
            case_recaller=_StubCaseRecaller([]),
            embed_query_fn=_fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    assert captured["top_n"] == 10
    assert captured["round1_top_n"] == 20
    assert captured["round1_rerank_top_n"] == 10
    assert captured["round2_cap"] == 40
    assert captured["round2_retrieve_is_none"] is True
    assert captured["multi_query_count"] == 3
    assert captured["rrf_k"] == 60
    assert captured["refinement_strategy"] == "multi_query"


async def test_search_agent_skills_agentic_calls_aagentic_retrieve_with_benchmark_params() -> (  # noqa: E501
    None
):
    """Verify aagentic_retrieve called with benchmark hyperparams for agent_skill."""
    captured: dict[str, Any] = {}

    async def fake_aagentic(
        query: str,
        *,
        base_retrieve: Any,
        llm: Any,
        rerank_fn: Any,
        round2_retrieve: Any,
        round2_cap: Any,
        top_n: int,
        round1_top_n: int,
        round1_rerank_top_n: int,
        refinement_strategy: str,
        multi_query_count: int,
        rrf_k: int,
    ) -> tuple[list[Candidate], AgenticDecision]:
        captured.update(
            top_n=top_n,
            round1_top_n=round1_top_n,
            round1_rerank_top_n=round1_rerank_top_n,
            round2_cap=round2_cap,
            round2_retrieve_is_none=round2_retrieve is None,
            multi_query_count=multi_query_count,
            rrf_k=rrf_k,
            refinement_strategy=refinement_strategy,
        )
        return [], AgenticDecision(is_multi_round=False)

    with patch("everos.memory.search.agentic_agent.aagentic_retrieve", fake_aagentic):
        await search_agent_skills_agentic(
            "What skill handles auth token refresh?",
            where="owner_id = 'agent_a' AND owner_type = 'agent'",
            skill_recaller=_StubSkillRecaller([]),
            embed_query_fn=_fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=5,
        )

    assert captured["top_n"] == 5
    assert captured["round1_top_n"] == 20
    assert captured["round1_rerank_top_n"] == 10
    assert captured["round2_cap"] == 40
    assert captured["round2_retrieve_is_none"] is True
    assert captured["multi_query_count"] == 3
    assert captured["rrf_k"] == 60
    assert captured["refinement_strategy"] == "multi_query"


async def test_search_agent_cases_agentic_shapes_result() -> None:
    """Output must be list[SearchAgentCaseItem] built from aagentic_retrieve results."""
    cand = _case_candidate("c_1")

    async def fake_aagentic(
        *_: Any, **__: Any
    ) -> tuple[list[Candidate], AgenticDecision]:
        return [cand], AgenticDecision(is_multi_round=False)

    with patch("everos.memory.search.agentic_agent.aagentic_retrieve", fake_aagentic):
        result = await search_agent_cases_agentic(
            "intent query",
            where="owner_id = 'agent_a' AND owner_type = 'agent'",
            case_recaller=_StubCaseRecaller([cand]),
            embed_query_fn=_fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    assert len(result) == 1
    assert isinstance(result[0], SearchAgentCaseItem)
    assert result[0].id == "c_1"
    assert result[0].task_intent == "intent c_1"


async def test_search_agent_skills_agentic_shapes_result() -> None:
    """Output must be list[SearchAgentSkillItem] from aagentic_retrieve results."""
    cand = _skill_candidate("s_1")

    async def fake_aagentic(
        *_: Any, **__: Any
    ) -> tuple[list[Candidate], AgenticDecision]:
        return [cand], AgenticDecision(is_multi_round=False)

    with patch("everos.memory.search.agentic_agent.aagentic_retrieve", fake_aagentic):
        result = await search_agent_skills_agentic(
            "skill query",
            where="owner_id = 'agent_a' AND owner_type = 'agent'",
            skill_recaller=_StubSkillRecaller([cand]),
            embed_query_fn=_fake_embed,
            reranker=_StubReranker(),
            llm=FakeLLMClient(responses=[]),
            top_k=10,
        )

    assert len(result) == 1
    assert isinstance(result[0], SearchAgentSkillItem)
    assert result[0].id == "s_1"
    assert result[0].name == "skill_s_1"
