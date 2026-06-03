"""Unit tests for ``memory.search.skill_hybrid``.

skill_hybrid is the **cross-encoder lane** for skill HYBRID retrieval.
The LLM-rerank lane lives in ``SearchManager._search_agent_skills`` and
goes through ``everalgo.rank.skill.arank`` directly — covered by
``test_manager`` tests instead.

Covered surfaces:
    - ``search_agent_skills_hybrid`` (public function, MagicMock stubs)
    - ``_fuse``, ``_cross_encoder_rerank``, ``_shape_results``
      (via integration through the public function)

All I/O (reranker) is injected via MagicMock / stub objects. No LanceDB
or network calls are made.
"""

from __future__ import annotations

import datetime as _dt
from unittest.mock import AsyncMock, MagicMock

from everalgo.types import Candidate

from everos.memory.search.callbacks import _SKILL_RERANK_INSTRUCTION
from everos.memory.search.dto import SearchAgentSkillItem
from everos.memory.search.skill_hybrid import search_agent_skills_hybrid

# ── Helpers ───────────────────────────────────────────────────────────────


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _skill_candidate(
    sid: str,
    score: float = 0.8,
    name: str | None = None,
) -> Candidate:
    label = name or f"skill_{sid}"
    return Candidate(
        id=sid,
        score=score,
        source="vector",
        metadata={
            "owner_id": "agent_a",
            "owner_type": "agent",
            "name": label,
            "description": f"desc {sid}",
            "content": f"content {sid}",
            "confidence": 0.9,
            "maturity_score": 0.6,
            "source_case_ids": [],
        },
    )


def _make_reranker(candidates: list[Candidate]) -> MagicMock:
    """Stub reranker that returns identity-reranked results in the same order."""

    class _FakeResult:
        def __init__(self, index: int, score: float) -> None:
            self.index = index
            self.score = score

    reranker = MagicMock()
    # provider.rerank returns a list of result objects with index + score
    reranker.rerank = AsyncMock(
        return_value=[_FakeResult(i, c.score) for i, c in enumerate(candidates)]
    )
    return reranker


# ── Tests ─────────────────────────────────────────────────────────────────


class TestSearchAgentSkillsHybridRerank:
    """Cross-encoder rerank path."""

    async def test_returns_shaped_items_up_to_top_k(self) -> None:
        """rrf + rerank produces at most top_k SearchAgentSkillItem objects."""
        c1 = _skill_candidate("s1", score=0.9)
        c2 = _skill_candidate("s2", score=0.8)
        c3 = _skill_candidate("s3", score=0.7)

        reranker = _make_reranker([c1, c2, c3])

        result = await search_agent_skills_hybrid(
            "what skill handles auth?",
            sparse=[c1, c2, c3],
            dense=[c1, c2, c3],
            reranker=reranker,
            top_k=2,
        )

        assert len(result) == 2
        assert all(isinstance(item, SearchAgentSkillItem) for item in result)
        assert result[0].id == "s1"
        assert result[1].id == "s2"

    async def test_reranker_receives_skill_instruction_and_shaped_passages(
        self,
    ) -> None:
        """Reranker must see the skill-specific instruction and
        ``"Agent Skill: {name} - {description}"`` passage shape — matches
        the everosos-opensource contract for skill rerank.
        """
        c1 = _skill_candidate("s1", name="auth_middleware_refactor")
        c2 = _skill_candidate("s2", name="provider_lookup_split")

        reranker = _make_reranker([c1, c2])

        await search_agent_skills_hybrid(
            "how to split auth?",
            sparse=[c1],
            dense=[c1, c2],
            reranker=reranker,
            top_k=10,
        )

        reranker.rerank.assert_awaited_once()
        call = reranker.rerank.await_args
        assert call is not None
        positional = call.args
        kw = call.kwargs
        # Signature: rerank(query, passages, *, instruction=...)
        assert positional[0] == "how to split auth?"
        passages = positional[1]
        assert passages == [
            "Agent Skill: auth_middleware_refactor - desc s1",
            "Agent Skill: provider_lookup_split - desc s2",
        ]
        assert kw["instruction"] == _SKILL_RERANK_INSTRUCTION


class TestSearchAgentSkillsHybridEmpty:
    """Empty input / degenerate cases."""

    async def test_empty_sparse_and_dense_returns_empty_list(self) -> None:
        """No candidates → no items, no errors."""
        reranker = MagicMock()
        reranker.rerank = AsyncMock(return_value=[])

        result = await search_agent_skills_hybrid(
            "query",
            sparse=[],
            dense=[],
            reranker=reranker,
            top_k=10,
        )

        assert result == []
        # reranker.rerank must not be called when fused list is empty
        reranker.rerank.assert_not_called()
