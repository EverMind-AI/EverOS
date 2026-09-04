"""The headline accuracy is a micro accuracy, and the categories are not averaged.

`benchmarks/README.md` described EverMemBench's headline as "the mean of its nine
category columns" -- a macro average -- while `_collect_method_summary` computes
`sum(correct) / sum(total)`, a micro accuracy. The two differ whenever the categories
have unequal sizes and unequal accuracy, which is every real run, so the published
number could not be reproduced from the documented definition.

The measured numbers stand and the guide was corrected to match them. That makes this
file the place the definition lives: the counts below are chosen so macro and micro
give visibly different answers, and both are written out as arithmetic, so a change of
definition fails here with the two values side by side instead of silently restating
one benchmark's results in another's terms.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import run as run_mod  # noqa: E402
from config import BenchmarkConfig  # noqa: E402

# Deliberately lopsided: a large category the system is bad at, and a small one it is
# perfect on. Micro = 24/104 = 0.2308; macro = (0.20 + 1.00) / 2 = 0.60. Any
# implementation reporting ~0.60 here is averaging the columns.
_CATEGORIES: tuple[tuple[str, int, int], ...] = (
    ("F_SH", 100, 20),
    ("F_MH", 4, 4),
)
_MICRO = 24 / 104
_MACRO = (20 / 100 + 4 / 4) / 2


def _judge_rows(spec: tuple[tuple[str, int, int], ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0
    for category, total, correct in spec:
        for i in range(total):
            rows.append(
                {
                    "index": idx,
                    "question": f"q{idx}",
                    "golden_answer": "a",
                    "generated_answer": "a",
                    "category": category,
                    "is_correct": i < correct,
                    "judgments": [i < correct],
                }
            )
            idx += 1
    return rows


def _summary(
    tmp_path: Path, spec: tuple[tuple[str, int, int], ...] = _CATEGORIES
) -> dict[str, Any]:
    conv = tmp_path / "conv0"
    conv.mkdir(parents=True, exist_ok=True)
    rows = _judge_rows(spec)
    (conv / "judge_hybrid.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    out = run_mod._collect_method_summary(
        "hybrid", tmp_path, [0], BenchmarkConfig(adapter="evermembench")
    )
    assert out is not None
    return out


def test_the_headline_is_the_micro_accuracy(tmp_path: Path) -> None:
    """`sum(correct) / sum(total)`, so a category's weight is its question count."""
    out = _summary(tmp_path)
    assert (out["correct"], out["total"]) == (24, 104)
    assert out["accuracy"] == pytest.approx(round(_MICRO, 4))


def test_the_headline_is_not_the_mean_of_the_category_columns(tmp_path: Path) -> None:
    """The definition the guide used to state, kept here as the thing it is not.

    Named separately from the assertion above so a refactor to macro reports "this
    became the category mean" rather than only "this number moved".
    """
    out = _summary(tmp_path)
    assert out["accuracy"] != pytest.approx(round(_MACRO, 4))
    assert abs(_MACRO - _MICRO) > 0.35, "the fixture stopped telling the two apart"


def test_every_category_is_reported_even_though_none_is_averaged(
    tmp_path: Path,
) -> None:
    """Reported and not averaged is the whole arrangement; both halves are asserted."""
    stats = _summary(tmp_path)["category_stats"]
    assert stats == {
        "F_SH": {"correct": 20, "total": 100},
        "F_MH": {"correct": 4, "total": 4},
    }


def test_a_category_with_no_questions_neither_divides_nor_shifts(
    tmp_path: Path,
) -> None:
    """An empty category contributes no rows, so it cannot reach either ratio.

    Under a macro average it would be a division by zero, or an invented 0.0 dragging
    the headline down; under this one it simply is not there.
    """
    out = _summary(tmp_path, (*_CATEGORIES, ("F_TP", 0, 0)))
    assert "F_TP" not in out["category_stats"]
    assert out["accuracy"] == pytest.approx(round(_MICRO, 4))


def test_a_method_with_no_graded_rows_reports_nothing(tmp_path: Path) -> None:
    """No rows means no summary, so the ratio is never taken at all.

    The other way to survive an empty denominator -- report 0.0 -- would put a method
    that never ran into the table at the bottom of it.
    """
    conv = tmp_path / "conv0"
    conv.mkdir(parents=True, exist_ok=True)
    (conv / "judge_hybrid.jsonl").write_text("", encoding="utf-8")
    assert (
        run_mod._collect_method_summary(
            "hybrid", tmp_path, [0], BenchmarkConfig(adapter="evermembench")
        )
        is None
    )


def test_the_guide_states_the_definition_the_code_implements() -> None:
    """The mismatch was in the documentation, so the documentation is under test.

    Cheap to keep true, and it is the artifact a reader compares numbers against; the
    wording it replaces described a statistic this code has never computed.
    """
    guide = (_BENCH / "README.md").read_text(encoding="utf-8")
    assert "micro accuracy" in guide
    assert "sum(correct) / sum(total)" in guide
    assert "mean of its nine category columns, which is" not in guide
