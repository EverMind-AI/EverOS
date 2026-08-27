"""Asker detection and per-subject profile injection for EverMemBench.

EverMemBench's owner is a whole project group, so the store holds one profile per
participant. A search that does not say which participant gets every one of them
-- 38 on topic 01 -- which is the composite-profile failure again, only longer.
These tests pin the two halves of avoiding that: the adapter reading the asker out
of the question, and the harness turning that into a ``profile_subject`` on the
search payload.

Measured against the real dataset in the same session: 434 of the 2400 questions
name an asker, and they are almost entirely the P_* persona families. The rest
name nobody and must therefore get NO profile block rather than all of them.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

emb = importlib.import_module("adapters.evermembench")


# ── asker detection ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # The three opening forms the dataset actually uses, verbatim heads of
        # P_Skill_Top01_001, P_Style_Top01_001 and a P_Title variant.
        ("I'm Lan Ye from the operations team. I heard the company ...", "Lan Ye"),
        ("I (Xinhao Yao) have completed the design of the caching ...", "Xinhao Yao"),
        ("As Jing Lv, I am preparing a technical presentation ...", "Jing Lv"),
        # Three-part names stay whole.
        ("I'm Mary Jane Watson from ops.", "Mary Jane Watson"),
    ],
)
def test_asker_is_read_from_the_opening_clause(question: str, expected: str) -> None:
    assert emb.profile_subject_of({"question": question}) == expected


@pytest.mark.parametrize(
    "question",
    [
        # F_SH: a fact question, no asker.
        "After SQL optimization of the data submission API, what was the peak CPU?",
        # MA_U: third-person scenario, names somebody who is NOT the asker.
        "Li, a new backend engineer, is developing the report generation module ...",
        # Lowercase / mid-sentence mentions must not be mistaken for the asker.
        "The team discussed whether I'm right about the cache strategy.",
        "",
    ],
)
def test_questions_that_name_no_asker_get_no_subject(question: str) -> None:
    """No subject means no profile block -- 38 profiles is worse than none."""
    assert emb.profile_subject_of({"question": question}) is None


def test_missing_question_key_is_not_an_error() -> None:
    assert emb.profile_subject_of({}) is None


# ── the payload the harness sends ────────────────────────────────────────


def _payload(**kwargs: object) -> dict:
    """Build the search payload the way ``_search_one`` does, without the HTTP call."""
    run = importlib.import_module("run")
    captured: dict = {}

    class _Client:
        def post(self, _path: str, payload: dict) -> tuple[int, dict]:
            captured.update(payload)
            return 200, {"data": {"episodes": [], "profiles": []}}

    run._search_one(
        0,
        {"question": "q", "answer": "a"},
        client=_Client(),
        method="hybrid",
        top_k=20,
        owner_id="01",
        app_id="default",
        project_id="default",
        **kwargs,  # type: ignore[arg-type]
    )
    return captured


def test_subject_rides_the_payload_only_with_include_profile() -> None:
    """``profile_subject`` is meaningless without asking for profiles at all."""
    assert "profile_subject" not in _payload(
        include_profile=False, profile_subject="Lan Ye"
    )
    assert (
        _payload(include_profile=True, profile_subject="Lan Ye")["profile_subject"]
        == "Lan Ye"
    )


def test_absent_subject_is_omitted_not_sent_empty() -> None:
    """An empty string is a filter that matches nothing, not "no filter"."""
    sent = _payload(include_profile=True, profile_subject=None)
    assert sent["include_profile"] is True
    assert "profile_subject" not in sent


def test_locomo_declares_neither_hook() -> None:
    """A single-person owner has one profile; naming a subject would be wrong."""
    locomo = importlib.import_module("adapters.locomo")
    assert getattr(locomo, "profile_subject_of", None) is None
    assert getattr(locomo, "INCLUDE_PROFILE", False) is False


def test_only_the_three_documented_opening_forms_are_matched() -> None:
    """The patterns themselves, not just their output.

    Measured on the full set: 434 of 2400 questions carry an opening that names their
    asker, and 431 of those are P_Skill / P_Style / P_Title -- the persona families.
    Widening these patterns would start attaching a profile to questions that grade
    something else, which is the composite-profile failure by a slower route.
    """
    assert len(emb._ASKER_PATTERNS) == 3
    # Every pattern is anchored: an asker named mid-sentence is not the asker.
    assert all(p.pattern.startswith("^") for p in emb._ASKER_PATTERNS)
    assert (
        emb.profile_subject_of({"question": "The report says I'm Lan Ye was late"})
        is None
    )
