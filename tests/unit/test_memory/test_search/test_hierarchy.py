"""Unit tests for ``memory.search.hierarchy``."""

from __future__ import annotations

import datetime as _dt

import pytest
from everalgo.rank.fusion import cosine_to_lr_score
from everalgo.types import Candidate, FactCandidate

from everos.memory.search.hierarchy import (
    build_ep_to_fact_parents,
    heap_expand,
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


# ── heap_expand ──────────────────────────────────────────────────────────


class TestHeapExpand:
    def test_empty_inputs_returns_empty(self) -> None:
        assert heap_expand(sparse=[], dense=[], episode_to_facts={}) == []

    def test_episodes_only_no_facts_sorted_by_lr(self) -> None:
        """No facts → all episodes survive, sorted by LR score descending."""
        sparse = [_ep(ep_id="ep-a", score=5.0), _ep(ep_id="ep-b", score=3.0)]
        dense = [_ep(ep_id="ep-a", score=0.8), _ep(ep_id="ep-b", score=0.6)]

        result = heap_expand(sparse=sparse, dense=dense, episode_to_facts={}, top_k=2)

        assert len(result) == 2
        assert result[0].id == "ep-a"
        assert result[1].id == "ep-b"
        assert all(r.item_type == "episode" for r in result)
        assert result[0].score == pytest.approx(cosine_to_lr_score(0.8, 5.0))
        assert result[1].score == pytest.approx(cosine_to_lr_score(0.6, 3.0))

    def test_fact_evicts_parent_episode(self) -> None:
        """A high-scoring fact enters top-N and evicts its parent episode."""
        sparse = [_ep(ep_id="ep-1", score=2.0)]
        dense = [_ep(ep_id="ep-1", score=0.5)]
        facts = {"ep-1": [_fact(fact_id="f1", parent_episode_id="ep-1", score=0.95)]}

        result = heap_expand(
            sparse=sparse, dense=dense, episode_to_facts=facts, top_k=2
        )

        fact_items = [r for r in result if r.item_type == "atomic_fact"]
        ep_items = [r for r in result if r.item_type == "episode"]
        assert len(fact_items) == 1
        assert fact_items[0].id == "f1"
        assert fact_items[0].parent_episode_id == "ep-1"
        assert len(ep_items) == 0

    def test_global_competition_fact_evicts_weaker_episode(self) -> None:
        """Fact from ep-a can push ep-b out of top-N if ep-b scores lower."""
        sparse = [
            _ep(ep_id="ep-a", score=4.0),
            _ep(ep_id="ep-b", score=1.0),
        ]
        dense = [
            _ep(ep_id="ep-a", score=0.7),
            _ep(ep_id="ep-b", score=0.3),
        ]
        facts = {
            "ep-a": [
                _fact(fact_id="f1", parent_episode_id="ep-a", score=0.95),
                _fact(fact_id="f2", parent_episode_id="ep-a", score=0.90),
            ],
        }

        result = heap_expand(
            sparse=sparse, dense=dense, episode_to_facts=facts, top_k=1
        )

        assert len(result) == 1
        assert result[0].item_type == "atomic_fact"
        assert result[0].id == "f1"

    def test_top_k_caps_output(self) -> None:
        sparse = [_ep(ep_id=f"ep-{i}", score=float(5 - i)) for i in range(5)]
        dense = [_ep(ep_id=f"ep-{i}", score=0.9 - i * 0.1) for i in range(5)]

        result = heap_expand(sparse=sparse, dense=dense, episode_to_facts={}, top_k=3)

        assert len(result) == 3

    def test_convergence_stops_loop(self) -> None:
        """With max_convergence_rounds=1, loop stops after 1 round of no change."""
        sparse = [_ep(ep_id="ep-1", score=2.0), _ep(ep_id="ep-2", score=1.0)]
        dense = [_ep(ep_id="ep-1", score=0.5), _ep(ep_id="ep-2", score=0.4)]

        result = heap_expand(
            sparse=sparse,
            dense=dense,
            episode_to_facts={},
            top_k=2,
            max_convergence_rounds=1,
        )

        assert len(result) == 2

    def test_alpha_blending(self) -> None:
        """alpha=0.5 blends child and parent LR scores equally."""
        sparse = [_ep(ep_id="ep-1", score=2.0)]
        dense = [_ep(ep_id="ep-1", score=0.5)]
        facts = {"ep-1": [_fact(fact_id="f1", parent_episode_id="ep-1", score=0.95)]}

        result = heap_expand(
            sparse=sparse,
            dense=dense,
            episode_to_facts=facts,
            top_k=2,
            alpha=0.5,
        )

        child_lr = cosine_to_lr_score(0.95, 2.0)
        parent_lr = cosine_to_lr_score(0.5, 2.0)
        expected = 0.5 * child_lr + 0.5 * parent_lr
        fact_items = [r for r in result if r.item_type == "atomic_fact"]
        assert fact_items
        assert fact_items[0].score == pytest.approx(expected)
