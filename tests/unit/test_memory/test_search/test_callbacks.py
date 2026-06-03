"""Unit tests for ``memory.search.callbacks``."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
from everalgo.types import Candidate

from everos.memory.search.callbacks import (
    _SKILL_RERANK_INSTRUCTION,
    build_rerank_fn,
    build_skill_rerank_fn,
)


class _StubReranker:
    """Returns candidates in original order with scores 1.0, 0.9, 0.8, ...

    Records the ``instruction`` and ``passages`` from the most recent call so
    tests can assert that callback factories forward the right arguments.
    """

    def __init__(self) -> None:
        self.last_instruction: str | None = None
        self.last_passages: list[str] | None = None

    async def rerank(
        self, query: str, passages: list[str], *, instruction: str | None = None
    ) -> list[Any]:
        self.last_instruction = instruction
        self.last_passages = list(passages)

        class _R:
            def __init__(self, index: int, score: float) -> None:
                self.index = index
                self.score = score

        return [_R(i, 1.0 - i * 0.1) for i in range(len(passages))]


def _cand(cid: str, episode_text: str = "body") -> Candidate:
    return Candidate(
        id=cid,
        score=0.5,
        source="vector",
        metadata={"episode": episode_text},
    )


async def test_build_rerank_fn_returns_two_arg_callable() -> None:
    """build_rerank_fn must return a 2-arg async callable matching RerankFn."""
    rerank_fn = build_rerank_fn(_StubReranker(), text_field="episode")
    sig = inspect.signature(rerank_fn)
    params = list(sig.parameters)
    assert params == ["query", "candidates"], f"Expected 2-arg fn, got params: {params}"


async def test_build_rerank_fn_returns_all_candidates_without_truncation() -> None:
    """rerank_fn must return ALL reranked candidates; caller slices."""
    rerank_fn = build_rerank_fn(_StubReranker(), text_field="episode")
    cands = [_cand(f"c{i}") for i in range(5)]
    result = await rerank_fn("what did Alice eat?", cands)
    assert len(result) == 5


async def test_build_rerank_fn_attaches_scores_from_provider() -> None:
    """rerank_fn updates Candidate.score from RerankProvider results."""
    rerank_fn = build_rerank_fn(_StubReranker(), text_field="episode")
    cands = [_cand("a"), _cand("b")]
    result = await rerank_fn("q", cands)
    assert all(isinstance(c.score, float) for c in result)
    assert result[0].score == pytest.approx(1.0)
    assert result[1].score == pytest.approx(0.9)


async def test_build_rerank_fn_handles_empty_candidates() -> None:
    """Empty candidate list returns empty list without calling the provider."""
    rerank_fn = build_rerank_fn(_StubReranker(), text_field="episode")
    result = await rerank_fn("q", [])
    assert result == []


async def test_build_rerank_fn_forwards_instruction() -> None:
    """The task instruction is forwarded verbatim to the provider."""
    stub = _StubReranker()
    rerank_fn = build_rerank_fn(stub, text_field="episode", instruction="find facts")
    await rerank_fn("q", [_cand("a")])
    assert stub.last_instruction == "find facts"


# ── build_skill_rerank_fn ────────────────────────────────────────────────


def _skill_cand(cid: str, *, name: str = "", description: str = "") -> Candidate:
    return Candidate(
        id=cid,
        score=0.5,
        source="vector",
        metadata={"name": name, "description": description},
    )


async def test_build_skill_rerank_fn_emits_shaped_passage() -> None:
    """Passage = ``"Agent Skill: {name} - {description}"`` when both present."""
    stub = _StubReranker()
    rerank_fn = build_skill_rerank_fn(stub)
    await rerank_fn(
        "q",
        [_skill_cand("s1", name="refactor_auth", description="split provider lookup")],
    )
    assert stub.last_passages == ["Agent Skill: refactor_auth - split provider lookup"]


async def test_build_skill_rerank_fn_omits_dash_when_description_missing() -> None:
    """When description is empty, drop ``" - {description}"`` suffix."""
    stub = _StubReranker()
    rerank_fn = build_skill_rerank_fn(stub)
    await rerank_fn("q", [_skill_cand("s1", name="refactor_auth", description="")])
    assert stub.last_passages == ["Agent Skill: refactor_auth"]


async def test_build_skill_rerank_fn_falls_back_when_name_missing() -> None:
    """When name is empty, passage degrades to bare description."""
    stub = _StubReranker()
    rerank_fn = build_skill_rerank_fn(stub)
    await rerank_fn("q", [_skill_cand("s1", name="", description="just text")])
    assert stub.last_passages == ["just text"]


async def test_build_skill_rerank_fn_forwards_skill_instruction() -> None:
    """The skill-specific instruction is hard-wired into the call."""
    stub = _StubReranker()
    rerank_fn = build_skill_rerank_fn(stub)
    await rerank_fn("q", [_skill_cand("s1", name="x", description="y")])
    assert stub.last_instruction == _SKILL_RERANK_INSTRUCTION


async def test_build_skill_rerank_fn_handles_empty_candidates() -> None:
    """Empty candidate list skips the provider call entirely."""
    stub = _StubReranker()
    rerank_fn = build_skill_rerank_fn(stub)
    result = await rerank_fn("q", [])
    assert result == []
    assert stub.last_passages is None  # provider never called


async def test_build_skill_rerank_fn_attaches_scores_and_preserves_metadata() -> None:
    """Reranked candidates carry the provider's score and original metadata."""
    stub = _StubReranker()
    rerank_fn = build_skill_rerank_fn(stub)
    cands = [
        _skill_cand("a", name="alpha", description="d-a"),
        _skill_cand("b", name="beta", description="d-b"),
    ]
    result = await rerank_fn("q", cands)
    assert [c.id for c in result] == ["a", "b"]
    assert result[0].score == pytest.approx(1.0)
    assert result[1].score == pytest.approx(0.9)
    # metadata round-trips intact — the shape function only reads it, never mutates.
    assert result[0].metadata["name"] == "alpha"
    assert result[1].metadata["description"] == "d-b"
