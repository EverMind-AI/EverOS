"""Unit tests for ``memory.search.agentic_agent``.

Two groups of tests:

* White-box (patches ``aagentic_retrieve``): assert benchmark hyperparameters
  are wired correctly, plus a shaping test to verify DTOs are built
  correctly. These never execute the real ``everalgo._format_docs`` /
  rerank_fn wiring — they are dead coverage for the metadata bridge.
* Black-box (does NOT patch ``aagentic_retrieve``): exercises the real
  ``_format_docs`` prompt-rendering path and the real kind-shaped rerank_fn
  via ``everalgo.testing.fake_llm.FakeLLMClient``. These pin the regression
  fixed by the metadata-bridge refactor (empty-description skill -> 500)
  and the skill/case rerank-fn swap.

The skill verify step has been removed from production code; this test
module covers the agentic retrieve flow only.
"""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, ClassVar
from unittest.mock import patch

from everalgo.rank.protocols import AgenticDecision
from everalgo.testing.fake_llm import FakeLLMClient
from everalgo.types import Candidate

from everos.component.rerank import RerankResult
from everos.memory.search.agentic_agent import (
    search_agent_cases_agentic,
    search_agent_skills_agentic,
)
from everos.memory.search.callbacks import (
    _CASE_RERANK_INSTRUCTION,
    _SKILL_RERANK_INSTRUCTION,
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


# ── Black-box tests: real _format_docs + real kind-shaped rerank_fn ────────
#
# These deliberately do NOT patch ``aagentic_retrieve`` (unlike every test
# above), so the metadata bridge (``_to_everalgo_doc_metadata``) and the
# kind-shaped ``rerank_fn`` (``build_skill_rerank_fn`` / ``build_case_rerank_fn``)
# actually run.

# JSON body a real LLM would return for the sufficiency-check prompt; parsed
# by ``everalgo.rank.agentic._call_llm_for_sufficiency``. ``is_sufficient=True``
# short-circuits Round 2, so one LLM call is enough for every test below.
_SUFFICIENT_LLM_RESPONSE = json.dumps(
    {
        "is_sufficient": True,
        "reasoning": "single relevant candidate",
        "key_information_found": [],
        "missing_information": [],
    }
)


class _StubSkillRecallerAsym:
    """Like ``_StubSkillRecaller`` but with independently controllable
    dense/sparse routes, needed to pin an exact fused score."""

    kind: ClassVar[str] = "agent_skill"
    everalgo_memory_type: ClassVar[str] = "skill"
    text_field: ClassVar[str] = "description"

    def __init__(
        self, *, dense: list[Candidate], sparse: list[Candidate] | None = None
    ) -> None:
        self._dense = dense
        self._sparse = sparse if sparse is not None else list(dense)

    async def sparse_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._sparse)

    async def dense_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._dense)


class _StubCaseRecallerAsym:
    """Case-kind counterpart of :class:`_StubSkillRecallerAsym`."""

    kind: ClassVar[str] = "agent_case"
    everalgo_memory_type: ClassVar[str] = "case"
    text_field: ClassVar[str] = "task_intent"

    def __init__(
        self, *, dense: list[Candidate], sparse: list[Candidate] | None = None
    ) -> None:
        self._dense = dense
        self._sparse = sparse if sparse is not None else list(dense)

    async def sparse_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._sparse)

    async def dense_recall(self, *_: Any, **__: Any) -> list[Candidate]:
        return list(self._dense)


class _IdentityReranker:
    """Rerank stub that preserves input order; accepts ``instruction`` like
    a real :class:`RerankProvider` (unlike ``_StubReranker`` above, which
    only satisfies the white-box tests where rerank_fn is never called)."""

    async def rerank(
        self, query: str, passages: list[str], *, instruction: str | None = None
    ) -> list[RerankResult]:
        return [RerankResult(index=i, score=1.0) for i in range(len(passages))]


def _skill_metadata(*, name: str, description: str) -> dict[str, Any]:
    return {
        "owner_id": "agent_a",
        "owner_type": "agent",
        "name": name,
        "description": description,
        "content": "some remediation content",
        "confidence": 0.9,
        "maturity_score": 0.6,
        "source_case_ids": [],
    }


async def test_agentic_survives_name_only_skill() -> None:
    """A skill with an empty description is a valid everalgo output (see
    everalgo ``agent_memory/skill_ops.py`` — the guard is ``if not name and
    not description``). The metadata bridge must produce a non-empty
    passage for ``_format_docs``, otherwise everalgo raises ``ValueError``
    and the request 500s. This is the regression test for that defect."""
    skill = Candidate(
        id="s_name_only",
        score=0.9,
        source="vector",
        metadata=_skill_metadata(name="rotate_secrets", description=""),
    )
    recaller = _StubSkillRecallerAsym(dense=[skill])

    result = await search_agent_skills_agentic(
        "how to rotate credentials",
        where="owner_id = 'agent_a' AND owner_type = 'agent'",
        skill_recaller=recaller,
        embed_query_fn=_fake_embed,
        reranker=_IdentityReranker(),
        llm=FakeLLMClient(responses=[_SUFFICIENT_LLM_RESPONSE]),
        top_k=5,
    )

    assert len(result) == 1
    assert result[0].name == "rotate_secrets"


async def test_agentic_uses_skill_rerank_passage() -> None:
    """Rerank input passages must be the skill-shaped multi-field format
    (``build_skill_rerank_fn``'s ``"Agent Skill: {name} - {description}"``),
    not the raw single-field text the generic ``build_rerank_fn`` would use.
    Proves the rerank_fn swap in ``_run_agentic_retrieve`` actually happened."""
    captured_passages: list[str] = []
    captured_instruction: str | None = None

    class _SpyReranker:
        async def rerank(
            self,
            query: str,
            passages: list[str],
            *,
            instruction: str | None = None,
        ) -> list[RerankResult]:
            nonlocal captured_instruction
            captured_passages[:] = passages
            captured_instruction = instruction
            return [RerankResult(index=i, score=1.0) for i in range(len(passages))]

    skill = Candidate(
        id="s_revive",
        score=0.9,
        source="vector",
        metadata=_skill_metadata(name="revive_replica", description="restart node"),
    )
    recaller = _StubSkillRecallerAsym(dense=[skill])

    await search_agent_skills_agentic(
        "how to bring a replica back",
        where="owner_id = 'agent_a' AND owner_type = 'agent'",
        skill_recaller=recaller,
        embed_query_fn=_fake_embed,
        reranker=_SpyReranker(),
        llm=FakeLLMClient(responses=[_SUFFICIENT_LLM_RESPONSE]),
        top_k=5,
    )

    assert captured_passages == ["Agent Skill: revive_replica - restart node"]
    assert captured_instruction == _SKILL_RERANK_INSTRUCTION


async def test_agentic_uses_case_rerank_passage() -> None:
    """Symmetric to ``test_agentic_uses_skill_rerank_passage`` for the
    agent_case kind: passages must be ``"Agent Case: {task_intent} -
    {approach}"`` with ``_CASE_RERANK_INSTRUCTION``."""
    captured_passages: list[str] = []
    captured_instruction: str | None = None

    class _SpyReranker:
        async def rerank(
            self,
            query: str,
            passages: list[str],
            *,
            instruction: str | None = None,
        ) -> list[RerankResult]:
            nonlocal captured_instruction
            captured_passages[:] = passages
            captured_instruction = instruction
            return [RerankResult(index=i, score=1.0) for i in range(len(passages))]

    case = Candidate(
        id="c_restart",
        score=0.9,
        source="vector",
        metadata={
            "owner_id": "agent_a",
            "owner_type": "agent",
            "session_id": "sess_b",
            "timestamp": _ts(),
            "task_intent": "restart the pod",
            "approach": "kubectl rollout restart",
            "quality_score": 0.8,
        },
    )
    recaller = _StubCaseRecallerAsym(dense=[case])

    await search_agent_cases_agentic(
        "how to restart a stuck pod",
        where="owner_id = 'agent_a' AND owner_type = 'agent'",
        case_recaller=recaller,
        embed_query_fn=_fake_embed,
        reranker=_SpyReranker(),
        llm=FakeLLMClient(responses=[_SUFFICIENT_LLM_RESPONSE]),
        top_k=5,
    )

    assert captured_passages == [
        "Agent Case: restart the pod - kubectl rollout restart"
    ]
    assert captured_instruction == _CASE_RERANK_INSTRUCTION
