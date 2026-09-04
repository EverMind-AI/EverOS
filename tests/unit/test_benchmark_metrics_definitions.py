"""Small hand-computable cases that pin what the benchmark metrics mean.

Both modules here are post-processors, so a wrong definition does not crash and does
not show up in a run: it produces a number of the right shape that answers a different
question. The two pinned here were each wrong in a way that flattered the system.

* ``ir.score`` built nDCG's ideal ranking out of the relevance values it had
  *retrieved*, so gold the search missed was missing from the denominator too. Half
  the gold recalled scored ``ndcg@2 = 1.0`` next to ``recall@2 = 0.5``.
* ``core.core_sessions_from_trace`` keyed by owner, so on a dataset that asks many
  questions per owner -- LoCoMo asks ~200 per conversation -- every question but the
  last was silently dropped, and which one survived depended on trace order.

Every expected value below is written as the arithmetic that produces it, so the test
states the definition rather than echoing whatever the code returns.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from metrics import core as core_metrics  # noqa: E402
from metrics import ir as ir_metrics  # noqa: E402

# ---------------------------------------------------------------------------
# nDCG
# ---------------------------------------------------------------------------


def test_missing_gold_is_in_the_ideal_ranking() -> None:
    """The reported case: one of two gold sessions found must not score a perfect 1.

    ``ranked=[a]`` against ``gold={a, b}`` earns DCG ``1/log2(2) = 1``. The ideal
    ranking for k=2 holds both gold items, so IDCG is ``1 + 1/log2(3)``.
    """
    out = ir_metrics.score([["a"]], [{"a", "b"}], ks=(2,))
    expected = 1.0 / (1.0 + 1.0 / math.log2(3))
    assert out["recall@2"] == 0.5
    assert out["ndcg@2"] == pytest.approx(expected)
    assert out["ndcg@2"] < 1.0


def test_a_repeated_session_is_credited_once() -> None:
    """Otherwise the IDCG fix hands out more gain than the ideal ranking can hold.

    ``ranked=[a, a]`` against ``gold={a}`` would earn ``1 + 1/log2(3) = 1.63``
    against a one-item ideal of ``1`` -- an nDCG above 1, which is not a ratio.
    """
    out = ir_metrics.score([["a", "a"]], [{"a"}], ks=(2,))
    assert out["ndcg@2"] == 1.0


def test_a_perfect_ranking_is_one_and_a_miss_is_zero() -> None:
    assert ir_metrics.score([["a", "b"]], [{"a", "b"}], ks=(2,))["ndcg@2"] == 1.0
    assert ir_metrics.score([["x", "y"]], [{"a"}], ks=(2,))["ndcg@2"] == 0.0


def test_a_gold_item_below_k_still_discounts() -> None:
    """Rank matters: the same hit is worth less further down the list."""
    out = ir_metrics.score([["x", "a"]], [{"a"}], ks=(5,))
    assert out["ndcg@5"] == pytest.approx(1.0 / math.log2(3))


def test_questions_with_no_gold_are_not_scored() -> None:
    """A question with no resolvable evidence says nothing about retrieval."""
    assert ir_metrics.score([["a"]], [set()], ks=(2,)) == {"n": 0}


@pytest.mark.parametrize(
    ("ranked", "gold", "k"),
    [
        (["a"], {"a", "b"}, 2),
        (["a", "a"], {"a"}, 2),
        (["a", "b", "a"], {"a", "b"}, 2),
        (["a"], {"a"}, 20),
        ([], {"a"}, 3),
        (["x", "y", "z"], {"a", "b", "c"}, 1),
    ],
)
def test_ndcg_stays_a_ratio(ranked: list[str], gold: set[str], k: int) -> None:
    """Across every shape the reviewer named, including k past the end of the list."""
    out = ir_metrics.score([ranked], [gold], ks=(k,))
    assert 0.0 <= out[f"ndcg@{k}"] <= 1.0


# ---------------------------------------------------------------------------
# core selection
# ---------------------------------------------------------------------------


def _trace(tmp_path: Path, *records: dict) -> str:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return str(path)


def _injected(owner: str, question: str, session: str) -> dict:
    return {
        "owner_id": owner,
        "question": question,
        "injected": [{"session_id": session, "is_core": True}],
    }


def test_two_questions_under_one_owner_both_survive(tmp_path: Path) -> None:
    """The reported case, which used to collapse to ``{"same": {"s2"}}``."""
    path = _trace(
        tmp_path, _injected("same", "q1", "s1"), _injected("same", "q2", "s2")
    )
    assert core_metrics.core_sessions_from_trace(path) == {
        ("same", "q1"): {"s1"},
        ("same", "q2"): {"s2"},
    }


def test_the_aggregate_does_not_depend_on_trace_order(tmp_path: Path) -> None:
    """Order decided the answer before, which is the part that made it untrustworthy."""
    forward = _trace(
        tmp_path / "a", _injected("o", "q1", "s1"), _injected("o", "q2", "s2")
    )
    reverse = _trace(
        tmp_path / "b", _injected("o", "q2", "s2"), _injected("o", "q1", "s1")
    )
    assert core_metrics.core_sessions_from_trace(
        forward
    ) == core_metrics.core_sessions_from_trace(reverse)


def test_a_later_round_for_one_question_still_replaces_it(tmp_path: Path) -> None:
    """The final injection is the state that was scored -- that part was right."""
    path = _trace(tmp_path, _injected("o", "q", "early"), _injected("o", "q", "final"))
    assert core_metrics.core_sessions_from_trace(path) == {("o", "q"): {"final"}}


def test_scoring_keeps_every_question(tmp_path: Path) -> None:
    """Two questions, one core right and one wrong, average to 0.5 -- not to 0 or 1."""
    core = {("o", "q1"): {"s1"}, ("o", "q2"): {"wrong"}}
    gold = {("o", "q1"): {"s1"}, ("o", "q2"): {"s2"}}
    out = core_metrics.score(core, gold)
    assert out["n"] == 2
    assert out["recall"] == 0.5
    assert out["precision"] == 0.5


def test_gold_keyed_the_old_way_reports_nothing_rather_than_something(
    tmp_path: Path,
) -> None:
    """The migration hazard, made loud.

    A caller still keying gold by owner gets ``n == 0`` and a skip count, not a
    plausible score computed over one arbitrary question.
    """
    core = {("o", "q1"): {"s1"}, ("o", "q2"): {"s2"}}
    out = core_metrics.score(core, {"o": {"s1"}})
    assert out == {"n": 0, "skipped_no_gold": 2}


def test_a_record_without_an_owner_is_skipped(tmp_path: Path) -> None:
    path = _trace(tmp_path, {"question": "q", "injected": [], "owner_id": ""})
    assert core_metrics.core_sessions_from_trace(path) == {}


def test_a_question_id_is_preferred_when_the_trace_has_one(tmp_path: Path) -> None:
    """A labelling step can fill ``question_id``; it is the better key when present."""
    rec = _injected("o", "the question text", "s1") | {"question_id": "qid-7"}
    assert core_metrics.core_sessions_from_trace(_trace(tmp_path, rec)) == {
        ("o", "qid-7"): {"s1"}
    }
