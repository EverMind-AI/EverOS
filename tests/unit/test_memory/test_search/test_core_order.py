"""The order of the decider's core at the front of a multi-round result.

``[decider].core_order`` picks it. ``score`` (the default, and the only order
before 2026-11-13) sorts the core by each item's best RRF score over the
sub-queries that showed it; ``round`` sorts it by the round in which the
decider first kept the item, earlier rounds first, then by that score. Ties
keep the order the decider kept the items in, under both.

The order is not cosmetic. It is what the reader sees first, and when the core
outgrows ``top_k`` the cap keeps the head of it, so the order also decides
WHICH core items are returned. ``round`` is the offline replay's rule iii
(scratch/ops/decider_fixes_1111/tools/order_replay.py): the property test below
compares this module's ordering with that rule's expression and with the
``score`` expression the loop used before the switch existed, so the default
cannot drift from what every recorded run was measured with.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Sequence
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from everos.config.search_runtime import SearchConfigSource
from everos.config.settings import DeciderSettings, load_settings
from everos.memory.ports import SearchHit
from everos.memory.search.llm_multiround import (
    _ordered_core,
    search_episodes_llm_multiround,
)
from everos.memory.search.recall import RecallResult

from ._records import hit, recalled, stored_episode

_WHERE = "owner_id = 'alice' AND owner_type = 'user'"


def _ep(entry_id: str) -> SearchHit[Any]:
    stored = stored_episode(entry_id, session_id=f"s_{entry_id}")
    keyed = dataclasses.replace(
        stored, ref=dataclasses.replace(stored.ref, id=f"alice__{entry_id}")
    )
    return hit(keyed, 1.0)


class _SparseOnly:
    """Recall by query text alone, so an item's RRF score is set by its rank.

    The dense leg is skipped (the query embeds to an empty vector), so the RRF
    score of the item at rank ``r`` of a block is ``1 / (rrf_k + r + 1)``: the
    tests place items by rank to give them the scores they need.
    """

    def __init__(self, per_query: dict[str, list[str]]) -> None:
        self._per_query = per_query

    async def sparse_recall(
        self, query: str, where: str, *, limit: int
    ) -> RecallResult[Any]:
        return recalled(*[_ep(e) for e in self._per_query.get(query, [])])

    async def dense_recall(
        self, vector: Sequence[float], where: str, *, limit: int
    ) -> RecallResult[Any]:
        return recalled()


async def _no_vector(_q: str) -> list[float]:
    return []


def _llm(*replies: tuple[list[int], list[str]]) -> FakeLLMClient:
    """The decider's replies, one per round; the last repeats."""
    script = [json.dumps({"core": c, "next_queries": n}) for c, n in replies]
    state = {"i": 0}

    def answer(_messages: Any, **_kwargs: Any) -> str:
        text = script[min(state["i"], len(script) - 1)]
        state["i"] += 1
        return text

    return FakeLLMClient(handler=answer)


async def _search(
    per_query: dict[str, list[str]],
    llm: FakeLLMClient,
    *,
    top_k: int = 10,
    **tuning: Any,
) -> list[str]:
    settings = DeciderSettings(**tuning)
    found = await search_episodes_llm_multiround(
        "q",
        owner_id="alice",
        where=_WHERE,
        episode_recaller=_SparseOnly(per_query),
        embed_query_fn=_no_vector,
        llm=llm,
        top_k=top_k,
        tuning_source=lambda: settings,
        trace_path_source=lambda: None,
    )
    return [e.id.removeprefix("alice__") for e in found]


# Round 0 keeps the seed's rank-2 item (RRF 1/63); round 1 keeps the rank-0 item
# of the sub-query "gap" (RRF 1/61), which therefore outscores it.
_TWO_ROUNDS = {"q": ["hi", "mid", "lo"], "gap": ["z"]}


def _two_rounds() -> FakeLLMClient:
    return _llm(([2], ["gap"]), ([0], []))


async def test_score_order_is_the_default_and_puts_the_higher_rrf_first() -> None:
    assert DeciderSettings().core_order == "score"
    got = await _search(_TWO_ROUNDS, _two_rounds())
    # core by score, then the seed's guaranteed top-1, then the max-score fill
    assert got == ["z", "lo", "hi", "mid"]


async def test_round_order_puts_the_earlier_round_first() -> None:
    got = await _search(_TWO_ROUNDS, _two_rounds(), core_order="round")
    assert got == ["lo", "z", "hi", "mid"]


async def test_round_order_falls_back_to_score_inside_a_round() -> None:
    # Round 1 keeps two items of one sub-query, the lower-ranked one listed first.
    per_query = {"q": ["a", "b", "c"], "gap": ["x", "y"]}
    llm = _llm(([2], ["gap"]), ([1, 0], []))
    got = await _search(per_query, llm, core_order="round", top_k=3)
    assert got == ["c", "x", "y"]  # round 0's c; then x (1/61) before y (1/62)


async def test_ties_keep_the_order_the_decider_kept_the_items_in() -> None:
    # Two sub-queries' rank-0 items score 1/61 each, as does the seed's rank 0.
    # The decider lists y before x; neither order may resort them by id.
    per_query = {"q": ["s"], "g1": ["x"], "g2": ["y"]}
    for core_order in ("score", "round"):
        llm = _llm(([0], ["g1", "g2"]), ([1, 0], []))
        got = await _search(per_query, llm, core_order=core_order)
        assert got == ["s", "y", "x"], core_order


# ── the cap: with more core than top_k, the order decides what is kept ─────────

# Round 0 keeps the seed's ranks 2-3 (1/63, 1/64); round 1 keeps the sub-query's
# ranks 0-1 (1/61, 1/62). Four core items, top_k = 2.
_CAPPED = {"q": ["s0", "s1", "s2", "s3"], "gap": ["z0", "z1"]}


def _capped() -> FakeLLMClient:
    return _llm(([2, 3], ["gap"]), ([0, 1], []))


async def test_the_cap_keeps_the_best_scored_core_under_score_order() -> None:
    got = await _search(_CAPPED, _capped(), top_k=2)
    assert got == ["z0", "z1"]


async def test_the_cap_keeps_the_earliest_rounds_core_under_round_order() -> None:
    got = await _search(_CAPPED, _capped(), top_k=2, core_order="round")
    assert got == ["s2", "s3"]


async def test_the_cap_inside_one_round_keeps_its_best_scored_items() -> None:
    # Round 0 alone keeps three items for a top_k of 2: the round has no order of
    # its own, so the score decides, as it does under score order.
    per_query = {"q": ["a", "b", "c"], "gap": ["z"]}
    llm = _llm(([2, 0, 1], ["gap"]), ([0], []))
    got = await _search(per_query, llm, top_k=2, core_order="round")
    assert got == ["a", "b"]


@pytest.mark.parametrize(
    ("core_order", "want"),
    [("score", ["z0", "z1", "s2", "s3"]), ("round", ["s2", "s3", "z0", "z1"])],
)
async def test_core_overflow_returns_every_core_item_in_the_same_order(
    core_order: str, want: list[str]
) -> None:
    got = await _search(
        _CAPPED, _capped(), top_k=2, core_order=core_order, core_overflow=True
    )
    assert got == want


# ── the setting ───────────────────────────────────────────────────────────────


def test_the_setting_reads_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_DECIDER__CORE_ORDER", "round")
    assert load_settings().decider.core_order == "round"
    assert SearchConfigSource().tuning().core_order == "round"


def test_an_unknown_order_is_refused() -> None:
    with pytest.raises(ValidationError):
        DeciderSettings(core_order="decider")  # type: ignore[arg-type]


# ── the ordering itself, against the expressions it must reproduce ────────────


def _reference_score(core: list[str], score: dict[str, float]) -> list[str]:
    """The loop's order before the switch: llm_multiround at origin/feat/v2_chat
    abd4599, ``sorted(core_order, key=max_score, reverse=True)``."""
    return sorted(core, key=lambda c: score[c], reverse=True)


def _reference_round(
    core: list[str], rounds: dict[str, int], score: dict[str, float]
) -> list[str]:
    """Rule iii of the offline replay (order_replay.py, ``core_orders``)."""
    return sorted(core, key=lambda c: (-rounds[c], score[c]), reverse=True)


_SCORES = (0.0, 1 / 61, 1 / 62, 1 / 63, 0.05)
"""Few distinct values, so ties in score, in round and in both are common."""


@st.composite
def _cores(draw: st.DrawFn) -> tuple[list[str], dict[str, int], dict[str, float]]:
    n = draw(st.integers(min_value=0, max_value=30))
    picked = [f"c{i}" for i in draw(st.permutations(range(n)))]
    rounds = {c: draw(st.integers(min_value=0, max_value=2)) for c in picked}
    score = {c: draw(st.sampled_from(_SCORES)) for c in picked}
    # The loop appends each round's picks after the earlier rounds' picks.
    core = sorted(picked, key=rounds.__getitem__)
    return core, rounds, score


@given(_cores())
def test_score_order_is_the_order_before_the_switch(
    case: tuple[list[str], dict[str, int], dict[str, float]],
) -> None:
    core, rounds, score = case
    got = _ordered_core(core, rounds, score.__getitem__, "score")
    assert got == _reference_score(core, score)


@given(_cores())
def test_round_order_is_the_replays_rule_iii(
    case: tuple[list[str], dict[str, int], dict[str, float]],
) -> None:
    core, rounds, score = case
    got = _ordered_core(core, rounds, score.__getitem__, "round")
    assert got == _reference_round(core, rounds, score)
