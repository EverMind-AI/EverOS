"""Unit tests for ``memory.search.hierarchy``: lane fusion and episode scoring."""

from __future__ import annotations

import datetime as _dt

import pytest
from everalgo.rank.fusion import cosine_to_lr_score
from everalgo.types import Candidate

from everos.memory.search._candidates import FactCandidate
from everos.memory.search.hierarchy import (
    build_ep_to_fact_parents,
    fuse_lanes,
    score_episodes,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _ep(
    *,
    ep_id: str = "ep-1",
    score: float = 0.7,
    memcell_id: str = "mc-1",
    entry_id: str | None = None,
    source: str = "vector",
) -> Candidate:
    metadata: dict = {
        "parent_id": memcell_id,
        "owner_id": "u1",
        "owner_type": "user",
        "session_id": "sess-1",
        "timestamp": _ts(),
        "episode": "Some episode text.",
        "sender_ids": ["u1"],
        "subject": "Test subject",
        "summary": "Test summary",
    }
    if entry_id is not None:
        metadata["entry_id"] = entry_id
    return Candidate(id=ep_id, score=score, source=source, metadata=metadata)


def _fact(
    *,
    fact_id: str = "fact-1",
    parent_episode_id: str = "ep-1",
    score: float = 0.9,
) -> FactCandidate:
    return FactCandidate(
        id=fact_id,
        parent_episode_id=parent_episode_id,
        score=score,
        metadata={"fact": "Some fact text."},
    )


# ── build_ep_to_fact_parents ─────────────────────────────────────────────


class TestBuildEpToFactParents:
    def test_entry_id_only(self) -> None:
        """entry_id equal to parent_id → deduped to a single parent."""
        ep = _ep(ep_id="ep-1", memcell_id="ent-1", entry_id="ent-1")
        result = build_ep_to_fact_parents([ep])
        assert result == {"ep-1": ["ent-1"]}

    def test_parent_id_only(self) -> None:
        ep = _ep(ep_id="ep-1", memcell_id="mc-1")
        result = build_ep_to_fact_parents([ep])
        assert result == {"ep-1": ["mc-1"]}

    def test_both_entry_and_parent(self) -> None:
        ep = _ep(ep_id="ep-1", memcell_id="mc-1", entry_id="ent-1")
        result = build_ep_to_fact_parents([ep])
        assert result == {"ep-1": ["ent-1", "mc-1"]}

    def test_empty_list(self) -> None:
        assert build_ep_to_fact_parents([]) == {}

    def test_no_ids_in_metadata(self) -> None:
        ep = Candidate(id="ep-1", score=0.5, metadata={})
        assert build_ep_to_fact_parents([ep]) == {}


# ── fuse_lanes ───────────────────────────────────────────────────────────


class TestFuseLanes:
    def test_empty_lanes(self) -> None:
        assert fuse_lanes([], [], [], top_k=10) == []

    def test_keeps_top_k_episodes(self) -> None:
        lane = [_ep(ep_id=f"ep-{i}", score=1.0 - i / 10) for i in range(5)]
        assert [c.id for c in fuse_lanes(lane, [], [], top_k=3)] == [
            "ep-0",
            "ep-1",
            "ep-2",
        ]

    def test_fact_lane_alone_can_surface_an_episode(self) -> None:
        """An episode only the fact lane recalled still makes the cut."""
        kept = fuse_lanes(
            [_ep(ep_id="ep-a")], [_ep(ep_id="ep-a")], [_ep(ep_id="ep-b")], top_k=2
        )
        assert [c.id for c in kept] == ["ep-a", "ep-b"]

    def test_agreement_across_lanes_ranks_first(self) -> None:
        sparse = [_ep(ep_id="ep-a"), _ep(ep_id="ep-b")]
        dense = [_ep(ep_id="ep-b"), _ep(ep_id="ep-a")]
        fact_lane = [_ep(ep_id="ep-b")]
        assert fuse_lanes(sparse, dense, fact_lane, top_k=1)[0].id == "ep-b"


# ── score_episodes ───────────────────────────────────────────────────────


class TestScoreEpisodes:
    def test_own_vector_wins_attaches_no_fact(self) -> None:
        scored = score_episodes(
            [_ep(ep_id="ep-a")],
            {"ep-a": [_fact(fact_id="f-a", parent_episode_id="ep-a", score=0.6)]},
            sparse=[_ep(ep_id="ep-a", score=5.0)],
            dense=[_ep(ep_id="ep-a", score=0.8)],
            fact_lane=[],
        )
        assert [(s.item_type, s.id) for s in scored] == [("episode", "ep-a")]
        assert scored[0].score == pytest.approx(cosine_to_lr_score(0.8, 5.0))

    def test_better_fact_is_attached_with_the_episode_score(self) -> None:
        """The fact lane's 0.7 loses to the episode's own best fact, 0.75."""
        scored = score_episodes(
            [_ep(ep_id="ep-b")],
            {"ep-b": [_fact(fact_id="f-b", parent_episode_id="ep-b", score=0.75)]},
            sparse=[],
            dense=[_ep(ep_id="ep-b", score=0.5)],
            fact_lane=[_ep(ep_id="ep-b", score=0.7)],
        )
        episode, fact = scored
        assert episode.score == pytest.approx(cosine_to_lr_score(0.75, 0.0))
        assert (fact.item_type, fact.id, fact.parent_episode_id) == (
            "atomic_fact",
            "f-b",
            "ep-b",
        )
        assert fact.score == episode.score

    def test_fact_lane_score_counts_without_fetched_facts(self) -> None:
        """A fact-lane hit lifts the score even when no fact was fetched for it."""
        scored = score_episodes(
            [_ep(ep_id="ep-c")],
            {},
            sparse=[],
            dense=[_ep(ep_id="ep-c", score=0.4)],
            fact_lane=[_ep(ep_id="ep-c", score=0.9)],
        )
        assert [s.item_type for s in scored] == ["episode"]
        assert scored[0].score == pytest.approx(cosine_to_lr_score(0.9, 0.0))

    def test_keeps_fused_order(self) -> None:
        kept = [_ep(ep_id="ep-x"), _ep(ep_id="ep-y")]
        scored = score_episodes(kept, {}, sparse=[], dense=[], fact_lane=[])
        assert [s.id for s in scored] == ["ep-x", "ep-y"]
