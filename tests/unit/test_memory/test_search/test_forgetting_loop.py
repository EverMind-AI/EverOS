"""The multi-round loop with the forgetting curve: reweighs each block, drops none.

A :class:`~everos.memory.search.forgetting.CurveRanking` handed to the loop
multiplies each fused score by the candidate's factor and sorts the block
again *before* it is cut, so an old memory slips below a fresher one of
about the same fused score -- and can fall out of the block -- while a
clearly better one keeps its place. Without a ranking the loop is exactly the
plain one (every other test of the loop runs without one). The result says
which core the decider chose itself: a round that fell back to the ranking's
own top candidates is not the decider's choice, and is not reinforced.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.config.settings import DeciderSettings
from everos.memory.models import FORGOTTEN_MARK
from everos.memory.ports import SearchHit
from everos.memory.search.forgetting import DAY_MS, Curve, CurveRanking
from everos.memory.search.llm_multiround import run_llm_multiround
from everos.memory.search.recall import RecallResult

from ._records import TS_MS, hit, recalled, stored_episode

NOW = TS_MS + 720 * DAY_MS


def _ep(entry_id: str, *, age_days: float, mark: str | None = None) -> SearchHit[Any]:
    stored = stored_episode(
        entry_id,
        parent_id=f"mc_{entry_id}",
        session_id=f"s_{entry_id}",
        timestamp=NOW - int(age_days * DAY_MS),
    )
    return hit(dataclasses.replace(stored, mark=mark), 1.0)


class _Recaller:
    """Sparse recall only, in the order given, so ranks set the fused scores."""

    def __init__(self, per_query: dict[str, list[SearchHit[Any]]]) -> None:
        self._per_query = per_query

    async def sparse_recall(self, query: str, where: Any, *, limit: int) -> Any:
        return recalled(*self._per_query.get(query, []))

    async def dense_recall(
        self, vector: Sequence[float], where: Any, *, limit: int
    ) -> RecallResult[Any]:
        return recalled()


async def _no_vector(_q: str) -> list[float]:
    return []


def _decider(*replies: str) -> FakeLLMClient:
    script = list(replies)
    state = {"i": 0}

    def answer(_messages: Any, **_kwargs: Any) -> str:
        reply = script[min(state["i"], len(script) - 1)]
        state["i"] += 1
        return reply

    return FakeLLMClient(handler=answer)


def _keep(core: list[int], next_queries: list[str] | None = None) -> str:
    return json.dumps({"core": core, "next_queries": next_queries or []})


async def _run(
    per_query: dict[str, list[SearchHit[Any]]],
    *,
    ranking: CurveRanking | None,
    llm: FakeLLMClient | None = None,
    trace: Path | None = None,
    top_k: int = 10,
    **tuning: Any,
) -> Any:
    settings = DeciderSettings(**tuning)
    return await run_llm_multiround(
        "q",
        owner_id="alice",
        where="owner_id = 'alice'",  # type: ignore[arg-type]
        episode_recaller=_Recaller(per_query),  # type: ignore[arg-type]
        embed_query_fn=_no_vector,
        llm=llm or _decider(_keep([0])),
        top_k=top_k,
        tuning_source=lambda: settings,
        trace_path_source=lambda: str(trace) if trace else None,
        ranking=ranking,
    )


def _curve(*, retain: bool = True) -> CurveRanking:
    return CurveRanking(Curve(), now=NOW, retain=retain, store=None)


def _ids(result: Any) -> list[str]:
    return [e.id.removeprefix("alice_") for e in result.episodes]


async def test_an_old_memory_slips_below_a_fresh_one_of_about_the_same_score() -> None:
    seed = {"q": [_ep("old", age_days=700), _ep("new", age_days=1)]}
    plain = await _run(seed, ranking=None)
    curved = await _run(seed, ranking=_curve())
    assert _ids(plain) == ["old", "new"]  # the decider kept index 0
    assert _ids(curved) == ["new", "old"]


async def test_a_clearly_better_old_memory_keeps_its_place() -> None:
    """Retention takes at most 30%: rank 0 against rank 20 is not close."""
    filler = [_ep(f"f{i}", age_days=1) for i in range(19)]
    seed = {"q": [_ep("old", age_days=700), *filler, _ep("new", age_days=0)]}
    curved = await _run(seed, ranking=_curve(), llm=_decider(_keep([])), top_k=30)
    ids = _ids(curved)
    assert ids.index("old") < ids.index("new")


async def test_the_block_is_cut_after_the_reweighing() -> None:
    """With room for one candidate, the fresher one gets it."""
    seed = {"q": [_ep("old", age_days=700), _ep("new", age_days=1)]}
    prompts: list[str] = []

    def answer(messages: Any, **_kwargs: Any) -> str:
        prompts.append(str(messages[-1].content))
        return _keep([0])

    await _run(seed, ranking=_curve(), llm=FakeLLMClient(handler=answer), seed_topk=1)
    assert "subject new" in prompts[0]
    assert "subject old" not in prompts[0]


async def test_scores_are_the_weighted_ones_and_nothing_is_dropped() -> None:
    seed = {"q": [_ep("a", age_days=700), _ep("b", age_days=30), _ep("c", age_days=0)]}
    plain = await _run(seed, ranking=None, llm=_decider(_keep([])))
    curved = await _run(seed, ranking=_curve(), llm=_decider(_keep([])))
    assert sorted(_ids(plain)) == sorted(_ids(curved))
    fused = {e.id: e.score for e in plain.episodes}
    weighted = {e.id: e.score for e in curved.episodes}
    assert weighted["alice_c"] == fused["alice_c"]
    assert weighted["alice_b"] < fused["alice_b"]


async def test_a_past_version_is_scaled_down_without_retention() -> None:
    seed = {
        "q": [
            _ep("old_version", age_days=0, mark="ep_20260101_0002"),
            _ep("forgotten", age_days=0, mark=FORGOTTEN_MARK),
            _ep("current", age_days=0),
        ]
    }
    curved = await _run(seed, ranking=_curve(retain=False), llm=_decider(_keep([])))
    assert _ids(curved)[0] == "current"
    by_id = {e.id: e.score for e in curved.episodes}
    assert by_id["alice_old_version"] == pytest.approx(0.3 / 61)
    assert by_id["alice_forgotten"] == pytest.approx(0.3 / 62)
    assert by_id["alice_current"] == pytest.approx(1 / 63)


async def test_the_trace_records_what_the_curve_did(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    seed = {"q": [_ep("old", age_days=700), _ep("new", age_days=1)]}
    await _run(seed, ranking=_curve(), trace=trace)
    first = json.loads(trace.read_text().splitlines()[0])
    ranked = first["recall"]["blocks"][0]["rrf_ranked"]
    assert [r["id"] for r in ranked] == ["alice_new", "alice_old"]
    assert ranked[0]["curve"]["state"] == "current"
    assert 0.7 < ranked[1]["curve"]["factor"] < 0.8
    assert "retention" in ranked[1]["curve"]


async def test_without_a_ranking_the_trace_is_as_before(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    await _run({"q": [_ep("a", age_days=1)]}, ranking=None, trace=trace)
    first = json.loads(trace.read_text().splitlines()[0])
    assert "curve" not in first["recall"]["blocks"][0]["rrf_ranked"][0]


async def test_the_decider_chose_only_what_a_decided_round_kept() -> None:
    seed = {
        "q": [_ep("a", age_days=1), _ep("b", age_days=1)],
        "gap": [_ep("z", age_days=1)],
    }
    decided = await _run(
        seed, ranking=None, llm=_decider(_keep([1], ["gap"]), _keep([0]))
    )
    assert decided.chosen == ("alice_b", "alice_z")
    failing = _decider(_keep([1], ["gap"]), "not json at all")
    fell_back = await _run(seed, ranking=None, llm=failing, retries=0)
    assert fell_back.failures and fell_back.failures[-1].outcome == "fallback"
    assert "alice_b" in fell_back.core
    assert fell_back.chosen == ("alice_b",)


class _CountingStore:
    """A strength store that records each read and holds nothing."""

    def __init__(self) -> None:
        self.reads: list[int] = []

    async def read(self, keys: Any) -> dict:
        self.reads.append(len(keys))
        return {}

    async def write(self, rows: Any) -> None:
        raise AssertionError("a search never writes strengths")


async def test_strengths_are_read_once_per_round_never_per_block() -> None:
    seed = {
        "q": [_ep("a", age_days=1), _ep("b", age_days=2)],
        "g1": [_ep("c", age_days=3), _ep("a", age_days=1)],
        "g2": [_ep("d", age_days=4)],
    }
    store = _CountingStore()
    ranking = CurveRanking(Curve(), now=NOW, retain=True, store=store)
    llm = _decider(_keep([0], ["g1", "g2"]), _keep([0]))
    await _run(seed, ranking=ranking, llm=llm)
    # Round 0 reads a and b; round 1 (two sub-query blocks) reads c and d once.
    assert store.reads == [2, 2]
