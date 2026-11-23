"""Unit tests for the search-stage trace collector (``memory.search.trace``).

Pins the collection protocol the MemTrace instrumentation relies on:

* ``record_stage`` is a no-op outside a ``collect_search_trace`` scope and
  never raises on malformed candidates;
* candidates are reduced to their ``(id, score, source)`` triple and each
  stage is capped at ``STAGE_CANDIDATE_CAP``; the pre-truncation pool size
  rides on ``total_count`` (``-1`` when the iterable has no length);
* concurrent search requests collect independently (``ContextVar``
  isolation), while the ``asyncio.gather`` fan-out inside one request
  shares the request's collector.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from everos.memory.search.trace import (
    STAGE_CANDIDATE_CAP,
    SearchStageRecord,
    collect_search_trace,
    record_stage,
)


@dataclass(frozen=True)
class _Candidate:
    """Minimal stand-in for the everalgo ``Candidate`` triple."""

    id: str
    score: float
    source: str


def _candidates(n: int, *, prefix: str = "c") -> list[_Candidate]:
    return [
        _Candidate(id=f"{prefix}_{i}", score=0.5 + i * 0.01, source="keyword")
        for i in range(n)
    ]


def test_record_stage_is_noop_without_collector() -> None:
    # No collect_search_trace scope installed — must silently do nothing.
    record_stage("sparse", "episode", _candidates(3))


def test_record_stage_never_raises_on_malformed_candidates() -> None:
    with collect_search_trace() as stages:
        record_stage("sparse", "episode", [object(), None])
    assert len(stages) == 1
    assert [(c.id, c.score, c.source) for c in stages[0].candidates] == [
        ("", 0.0, ""),
        ("", 0.0, ""),
    ]


def test_record_stage_swallows_candidate_attribute_errors() -> None:
    """A candidate whose attribute access raises must not break search —
    the whole stage record is dropped, the exception never propagates."""

    class _Exploding:
        @property
        def id(self) -> str:
            raise RuntimeError("boom")

    with collect_search_trace() as stages:
        record_stage("sparse", "episode", [_Exploding()])
    assert stages == []


def test_collect_search_trace_records_stages_in_order() -> None:
    with collect_search_trace() as stages:
        record_stage("sparse", "episode", _candidates(2))
        record_stage(
            "fusion",
            "episode",
            _candidates(1, prefix="f"),
            metadata={"fusion_mode": "rrf", "enable_rerank": False},
        )
    assert [s.phase for s in stages] == ["sparse", "fusion"]
    assert all(s.kind == "episode" for s in stages)
    assert [(c.id, c.score, c.source) for c in stages[0].candidates] == [
        ("c_0", 0.5, "keyword"),
        ("c_1", 0.51, "keyword"),
    ]
    assert stages[1].metadata == {"fusion_mode": "rrf", "enable_rerank": False}


def test_record_stage_defaults_metadata_to_empty() -> None:
    with collect_search_trace() as stages:
        record_stage("sparse", "episode", _candidates(1))
    assert stages[0].metadata == {}


def test_candidates_capped_at_stage_cap() -> None:
    with collect_search_trace() as stages:
        record_stage("dense", "episode", _candidates(STAGE_CANDIDATE_CAP + 20))
    assert len(stages[0].candidates) == STAGE_CANDIDATE_CAP
    # The first STAGE_CANDIDATE_CAP candidates (pool order) are kept.
    assert stages[0].candidates[-1].id == f"c_{STAGE_CANDIDATE_CAP - 1}"


def test_total_count_reflects_the_pre_truncation_pool_size() -> None:
    with collect_search_trace() as stages:
        record_stage("dense", "episode", _candidates(STAGE_CANDIDATE_CAP + 20))
        record_stage("sparse", "episode", _candidates(3))
    assert stages[0].total_count == STAGE_CANDIDATE_CAP + 20
    assert stages[0].total_count > len(stages[0].candidates)
    assert stages[1].total_count == 3
    assert stages[1].total_count == len(stages[1].candidates)


def test_total_count_is_unknown_when_candidates_expose_no_length() -> None:
    class _SliceableOnly:
        """Supports slicing but no ``len`` — the length stays unknown."""

        def __init__(self, items: list[_Candidate]) -> None:
            self._items = items

        def __getitem__(self, key: object) -> object:
            return self._items[key]

    with collect_search_trace() as stages:
        record_stage("sparse", "episode", _SliceableOnly(_candidates(2)))
    assert stages[0].total_count == -1
    assert len(stages[0].candidates) == 2


def test_stage_record_total_count_defaults_to_zero() -> None:
    record = SearchStageRecord(phase="sparse", kind="episode", candidates=())
    assert record.total_count == 0


def test_collector_uninstalled_on_scope_exit() -> None:
    with collect_search_trace():
        record_stage("sparse", "episode", _candidates(1))
    record_stage("dense", "episode", _candidates(1))  # no-op again
    with collect_search_trace() as stages:
        pass
    assert stages == []


def test_nested_scopes_collect_independently() -> None:
    with collect_search_trace() as outer:
        record_stage("sparse", "episode", _candidates(1, prefix="o"))
        with collect_search_trace() as inner:
            record_stage("dense", "episode", _candidates(1, prefix="i"))
        record_stage("fusion", "episode", _candidates(1, prefix="o"))
    assert [c.id for s in outer for c in s.candidates] == ["o_0", "o_0"]
    assert [s.phase for s in outer] == ["sparse", "fusion"]
    assert [s.phase for s in inner] == ["dense"]


async def test_collectors_isolated_across_concurrent_tasks() -> None:
    async def run(tag: str) -> list[SearchStageRecord]:
        with collect_search_trace() as stages:
            record_stage("sparse", "episode", _candidates(1, prefix=tag))
            await asyncio.sleep(0)  # force interleaving
            record_stage("fusion", "episode", _candidates(1, prefix=tag))
            return stages

    first, second = await asyncio.gather(run("a"), run("b"))
    assert [s.phase for s in first] == ["sparse", "fusion"]
    assert [s.phase for s in second] == ["sparse", "fusion"]
    assert [c.id for c in first[0].candidates] == ["a_0"]
    assert [c.id for c in second[0].candidates] == ["b_0"]


async def test_gather_children_share_the_request_collector() -> None:
    """One request's ``asyncio.gather`` fan-out lands in one collector."""

    async def child(kind: str) -> None:
        record_stage("sparse", kind, _candidates(1))

    with collect_search_trace() as stages:
        await asyncio.gather(child("agent_case"), child("agent_skill"))
    assert {s.kind for s in stages} == {"agent_case", "agent_skill"}
