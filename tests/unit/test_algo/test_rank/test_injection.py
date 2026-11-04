"""Evidence selection respects budgets, facet coverage, and state ownership."""

from __future__ import annotations

from copy import deepcopy

import pytest

from everos.algo.rank import select_injection


def test_core_then_guarantee_then_max_score_preserves_facet_coverage() -> None:
    selection = select_injection(
        top_k=3,
        core_ids={"core"},
        core_order=["core"],
        candidate_ids=["core", "facet", "consensus", "specialist"],
        subq_hits={
            "core": {"original": 0.01},
            "facet": {"followup": 0.02},
            "consensus": {"original": 0.04, "followup": 0.04},
            "specialist": {"followup": 0.06},
        },
        guarantee=[("original", "core"), ("followup", "facet")],
    )

    assert selection.chosen == ("core", "facet", "specialist")
    assert selection.slot_source == {
        "core": "core",
        "facet": "guarantee-top1",
        "specialist": "maxscore-fill",
    }
    assert (selection.n_core, selection.n_guaranteed, selection.n_filled) == (1, 1, 1)


@pytest.mark.parametrize(
    ("core_overflow", "expected"), [(False, ("high",)), (True, ("high", "low"))]
)
def test_core_budget_is_an_explicit_per_call_option(
    core_overflow: bool, expected: tuple[str, ...]
) -> None:
    selection = select_injection(
        top_k=1,
        core_ids={"low", "high"},
        core_order=["low", "high"],
        candidate_ids=["low", "high", "extra"],
        subq_hits={"low": {"q": 0.01}, "high": {"q": 0.03}},
        guarantee=[("q", "extra")],
        core_overflow=core_overflow,
    )

    assert selection.chosen == expected
    assert selection.n_core_selected == 2
    assert selection.n_core == len(expected)
    assert selection.n_guaranteed == selection.n_filled == 0


def test_inputs_are_preserved_and_results_do_not_leak_between_calls() -> None:
    state = {
        "core_ids": {"a"},
        "core_order": ["a"],
        "candidate_ids": ["a", "b"],
        "subq_hits": {"a": {"q": 0.1}, "b": {"q": 0.2}},
        "guarantee": [("q", "b")],
    }
    before = deepcopy(state)
    first = select_injection(top_k=2, **state)
    empty = select_injection(
        top_k=2,
        core_ids=set(),
        core_order=[],
        candidate_ids=[],
        subq_hits={},
        guarantee=[],
    )
    again = select_injection(top_k=2, **state)

    assert state == before
    assert empty.chosen == ()
    assert empty.slot_source == {}
    assert again == first
    assert again.slot_source is not first.slot_source


def test_caller_owned_policy_is_read_at_each_core_decision() -> None:
    policy = iter([False, True])
    selection = select_injection(
        top_k=1,
        core_ids={"a", "b"},
        core_order=["a", "b"],
        candidate_ids=["a", "b"],
        subq_hits={"a": {"q": 0.2}, "b": {"q": 0.1}},
        guarantee=[],
        core_overflow=lambda: next(policy),
    )

    assert selection.chosen == ("a", "b")
    with pytest.raises(StopIteration):
        next(policy)


@pytest.mark.parametrize("top_k", [0, 1, 2])
def test_budget_and_discovery_order_break_score_ties(top_k: int) -> None:
    selection = select_injection(
        top_k=top_k,
        core_ids=set(),
        core_order=[],
        candidate_ids=["b", "a"],
        subq_hits={"a": {"q": 0.1}, "b": {"q": 0.1}},
        guarantee=[],
    )

    assert selection.chosen == ("b", "a")[:top_k]
