"""Unit tests for ``memory.search.shaper``.

Tests are pure: no LanceDB, no everalgo, just dataclass-in / DTO-out.
"""

from __future__ import annotations

import datetime as _dt

from everalgo.types import Candidate, ScoredItem

from everos.memory.search.shaper import (
    reshape_hybrid_output,
    shape_agent_case_from_candidate,
    shape_agent_skill_from_candidate,
    shape_atomic_fact_from_candidate,
    shape_episode_from_candidate,
)

# ── Fixtures ────────────────────────────────────────────────────────────


def _ts(year: int = 2026) -> _dt.datetime:
    return _dt.datetime(year, 1, 1, tzinfo=_dt.UTC)


def _episode_candidate(*, id: str = "alice_ep_1", score: float = 0.9) -> Candidate:
    return Candidate(
        id=id,
        score=score,
        source="vector",
        metadata={
            "owner_id": "alice",
            "owner_type": "user",
            "session_id": "sess_a",
            "timestamp": _ts(),
            "sender_ids": ["alice", "assistant_1"],
            "subject": "Coffee chat",
            "summary": "Discussed coffee preferences.",
            "episode": "Alice said she prefers oat milk.",
        },
    )


def _agent_case_candidate() -> Candidate:
    return Candidate(
        id="agent_a_case_1",
        score=0.8,
        source="keyword",
        metadata={
            "owner_id": "agent_a",
            "owner_type": "agent",
            "session_id": "sess_a",
            "timestamp": _ts(),
            "task_intent": "Draft a follow-up email",
            "approach": "1. summarise...",
            "quality_score": 0.92,
            "key_insight": "User prefers brief tone",
        },
    )


def _agent_skill_candidate() -> Candidate:
    return Candidate(
        id="agent_a_skill_1",
        score=0.7,
        source="keyword",
        metadata={
            "owner_id": "agent_a",
            "owner_type": "agent",
            "name": "contract_redline",
            "description": "Spot risky clauses",
            "content": "Step 1: ...",
            "confidence": 0.9,
            "maturity_score": 0.5,
            "source_case_ids": ["agent_a_case_1"],
        },
    )


# ── Episode shaping ─────────────────────────────────────────────────────


def test_shape_episode_basic() -> None:
    item = shape_episode_from_candidate(_episode_candidate())
    assert item is not None
    assert item.id == "alice_ep_1"
    assert item.user_id == "alice"
    assert item.type == "Conversation"
    assert item.score == 0.9
    assert item.atomic_facts == []
    assert item.sender_ids == ["alice", "assistant_1"]


def test_shape_episode_drops_when_owner_type_wrong() -> None:
    cand = _episode_candidate()
    cand.metadata["owner_type"] = "agent"
    assert shape_episode_from_candidate(cand) is None


def test_shape_episode_drops_when_timestamp_missing() -> None:
    cand = _episode_candidate()
    del cand.metadata["timestamp"]
    assert shape_episode_from_candidate(cand) is None


def test_shape_episode_attaches_facts() -> None:
    facts = [
        shape_atomic_fact_from_candidate(
            Candidate(
                id="f1",
                score=0.5,
                source="other",
                metadata={"fact": "Alice prefers oat milk"},
            )
        )
    ]
    item = shape_episode_from_candidate(_episode_candidate(), atomic_facts=facts)
    assert item is not None
    assert len(item.atomic_facts) == 1
    assert item.atomic_facts[0].content == "Alice prefers oat milk"


# ── Agent case / skill shaping ──────────────────────────────────────────


def test_shape_agent_case_basic() -> None:
    item = shape_agent_case_from_candidate(_agent_case_candidate())
    assert item is not None
    assert item.agent_id == "agent_a"
    assert item.task_intent == "Draft a follow-up email"
    assert item.quality_score == 0.92
    assert item.key_insight == "User prefers brief tone"


def test_shape_agent_case_drops_when_owner_type_wrong() -> None:
    cand = _agent_case_candidate()
    cand.metadata["owner_type"] = "user"
    assert shape_agent_case_from_candidate(cand) is None


def test_shape_agent_skill_basic() -> None:
    item = shape_agent_skill_from_candidate(_agent_skill_candidate())
    assert item is not None
    assert item.name == "contract_redline"
    assert item.maturity_score == 0.5
    assert item.source_case_ids == ["agent_a_case_1"]


# ── Hybrid reshape ──────────────────────────────────────────────────────


def _scored_episode(eid: str, score: float) -> ScoredItem:
    return ScoredItem(
        id=eid,
        score=score,
        item_type="episode",
        metadata={
            "owner_id": "alice",
            "owner_type": "user",
            "session_id": "s1",
            "timestamp": _ts(),
            "sender_ids": ["alice"],
            "subject": "subj",
            "summary": "summ",
            "episode": "body",
        },
    )


def _scored_fact(fid: str, parent: str, score: float) -> ScoredItem:
    return ScoredItem(
        id=fid,
        score=score,
        item_type="atomic_fact",
        parent_episode_id=parent,
        metadata={"fact": f"fact text {fid}"},
    )


def test_reshape_hybrid_nests_facts_under_kept_episode() -> None:
    scored = [
        _scored_episode("ep_1", 0.9),
        _scored_fact("f_1", "ep_1", 0.95),
        _scored_fact("f_2", "ep_1", 0.85),
    ]
    out = reshape_hybrid_output(scored, episode_pool={})
    assert len(out) == 1
    assert out[0].id == "ep_1"
    # Facts sorted descending by score.
    assert [f.id for f in out[0].atomic_facts] == ["f_1", "f_2"]


def test_reshape_hybrid_backfills_evicted_episode_from_pool() -> None:
    # Episode ep_2 was evicted (only facts present),
    # but it is in episode_pool — should be restored as a result.
    scored = [
        _scored_episode("ep_1", 0.7),
        _scored_fact("f_a", "ep_2", 0.95),
    ]
    pool_episode = _episode_candidate(id="ep_2", score=0.0)
    out = reshape_hybrid_output(scored, episode_pool={"ep_2": pool_episode})
    assert len(out) == 2
    # Output sorted by score descending — ep_2 takes fact's max score (0.95).
    assert out[0].id == "ep_2"
    assert out[0].score == 0.95
    assert len(out[0].atomic_facts) == 1
    assert out[1].id == "ep_1"


def test_reshape_hybrid_drops_orphan_facts_with_no_pool_parent() -> None:
    scored = [_scored_fact("f_x", "ep_missing", 0.5)]
    out = reshape_hybrid_output(scored, episode_pool={})
    assert out == []
