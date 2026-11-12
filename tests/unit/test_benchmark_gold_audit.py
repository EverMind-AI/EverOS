"""JUDGE re-judges what a gold audit disputes; the report shows both scores.

LoCoMo's gold answers are wrong for about 6% of its questions: a third-party audit
(dial481/locomo-audit) disputes 99 of 1,540 and gives a corrected answer for each. A
single headline cannot tell those apart from system errors, so JUDGE grades the same
answers a second time against the corrected answers -- same judge, same runs, same
majority vote -- and the report gives `Re-judged` beside `Accuracy`. Only the gold
changes: nothing is searched or answered again, and the original score is untouched.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

import run as run_mod  # noqa: E402
from adapters import locomo  # noqa: E402
from adapters.base import question_key  # noqa: E402
from config import AnswerResult, BenchmarkConfig  # noqa: E402

_AUDIT = [
    {
        "question_id": "locomo_0_qa3",
        "question": "Who is  Tim's coach?",
        "error_type": "ATTRIBUTION_ERROR",
        "correct_answer": "Coach Ann",
    },
    {
        "question_id": "locomo_1_qa0",
        "question": "When did Jon visit Paris?",
        "error_type": "TEMPORAL_ERROR",
        "correct_answer": "May 2023",
    },
    {
        "question_id": "locomo_0_qa5",
        "question": "What does Tim collect?",
        "error_type": "WRONG_CITATION",
        "correct_answer": "evidence D2:3",
    },
]

# conversation -> (question, gold, generated answer, category). Each disputed question
# was answered with the audit's corrected answer, so it is wrong under the dataset's
# gold and right under the audit's; the WRONG_CITATION one keeps its gold.
_QA = {
    0: [
        ("who is tim's coach?", "Coach Bob", "Coach Ann trains Tim", 4),
        ("What does Tim collect?", "stamps", "Tim collects stamps", 4),
        ("Where does Tim live?", "Boston", "Tim lives in Denver", 2),
    ],
    1: [
        ("When did Jon visit Paris?", "June 2023", "In May 2023", 2),
        ("What does Jon dance?", "salsa", "Jon dances salsa", 4),
    ],
}


class _GoldEchoJudge:
    """A judge that says CORRECT exactly when the generated answer contains the gold.

    Counts its calls, so a test can tell a re-judged row from a reused one.
    """

    def __init__(self) -> None:
        self.chat = self
        self.completions = self
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        self.calls += 1
        prompt = kwargs["messages"][-1]["content"]
        # The last ones: the prompt's own worked example carries a gold answer too.
        gold = re.findall(r"Gold answer: (.*)\n", prompt)[-1]
        generated = re.findall(r"Generated answer: (.*)\n", prompt)[-1]
        label = "CORRECT" if gold.lower() in generated.lower() else "WRONG"

        class _Message:
            content = json.dumps({"label": label})

        class _Choice:
            message = _Message()

        class _Reply:
            choices = [_Choice()]
            usage = None

        return _Reply()


def _write_audit(tmp_path: Path) -> Path:
    path = tmp_path / "errors.json"
    path.write_text(json.dumps(_AUDIT), encoding="utf-8")
    return path


def _config(audit_path: str, **over: Any) -> BenchmarkConfig:
    return BenchmarkConfig(
        adapter="locomo",
        gold_audit_path=audit_path,
        judge_runs=3,
        eval_concurrency=1,
        **over,
    )


def _write_answers(out: Path, *, search_error_conv1: bool = False) -> None:
    for conv, rows in _QA.items():
        d = out / f"conv{conv}"
        d.mkdir(parents=True)
        lines = []
        for idx, (question, gold, generated, category) in enumerate(rows):
            row = AnswerResult(
                index=idx,
                question=question,
                golden_answer=gold,
                category=category,
                generated_answer=generated,
                answer_time_s=0.0,
                answer_attempts=1,
                search_error="boom" if search_error_conv1 and conv == 1 else "",
            )
            lines.append(row.model_dump_json() + "\n")
        (d / "answer_agentic.jsonl").write_text("".join(lines), encoding="utf-8")


def _judge(
    out: Path, config: BenchmarkConfig, judge: _GoldEchoJudge, *, rejudge: bool = True
) -> None:
    """JUDGE as run_conversation drives it: judge, then re-judge the disputed."""
    corrections, _note = run_mod._gold_audit(config)
    for conv in _QA:
        conv_dir = out / f"conv{conv}"
        answers = conv_dir / "answer_agentic.jsonl"
        run_mod.run_evaluate_phase(
            answers, judge, config, config.judge_runs, conv_dir, method_label="agentic"
        )
        if rejudge and corrections:
            run_mod.run_rejudge_phase(
                answers,
                corrections,
                judge,
                config,
                judge_runs=config.judge_runs,
                conv_dir=conv_dir,
                conv_index=conv,
                method_label="agentic",
            )


def _run(tmp_path: Path, audit_path: str, **kw: Any) -> tuple[Path, dict[str, Any]]:
    out = tmp_path / "run"
    _write_answers(out, search_error_conv1=kw.pop("search_error_conv1", False))
    config = _config(audit_path)
    _judge(out, config, _GoldEchoJudge(), **kw)
    summary = run_mod._collect_method_summary("agentic", out, [0, 1], config)
    assert summary is not None
    return out, summary


# ---------------------------------------------------------------------------
# reading the audit
# ---------------------------------------------------------------------------


def test_every_entry_but_a_wrong_citation_carries_a_corrected_gold(
    tmp_path: Path,
) -> None:
    """A WRONG_CITATION only moves the evidence pointer; its gold answer stands."""
    assert locomo.gold_corrections(str(_write_audit(tmp_path))) == {
        question_key(0, "Who is Tim's coach?"): "Coach Ann",
        question_key(1, "When did Jon visit Paris?"): "May 2023",
    }


def test_an_unknown_question_id_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "errors.json"
    path.write_text(
        json.dumps(
            [
                {
                    "question_id": "q7",
                    "question": "x",
                    "error_type": "AMBIGUOUS",
                    "correct_answer": "y",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="locomo_<conv>_qa<n>"):
        locomo.gold_corrections(str(path))


# ---------------------------------------------------------------------------
# the re-judge stage
# ---------------------------------------------------------------------------


def test_only_the_disputed_questions_are_judged_again_with_the_corrected_gold(
    tmp_path: Path,
) -> None:
    out, _summary = _run(tmp_path, str(_write_audit(tmp_path)))
    conv0 = [
        json.loads(line)
        for line in (out / "conv0" / "rejudge_agentic.jsonl").read_text().splitlines()
    ]
    conv1 = [
        json.loads(line)
        for line in (out / "conv1" / "rejudge_agentic.jsonl").read_text().splitlines()
    ]
    assert [(r["index"], r["golden_answer"]) for r in conv0] == [(0, "Coach Ann")]
    assert [(r["index"], r["golden_answer"]) for r in conv1] == [(0, "May 2023")]
    # Same judge runs as JUDGE, and the answer itself is the one JUDGE graded.
    assert conv0[0]["judgments"] == [True, True, True]
    assert conv0[0]["generated_answer"] == "Coach Ann trains Tim"


def test_a_resumed_judge_stage_does_not_judge_twice(tmp_path: Path) -> None:
    out = tmp_path / "run"
    _write_answers(out)
    config = _config(str(_write_audit(tmp_path)))
    _judge(out, config, _GoldEchoJudge())
    again = _GoldEchoJudge()
    _judge(out, config, again)
    assert again.calls == 0
    rows = (out / "conv0" / "rejudge_agentic.jsonl").read_text().splitlines()
    assert len(rows) == 1


def test_a_failed_search_stays_wrong_without_a_judge_call(tmp_path: Path) -> None:
    """The re-judge goes through JUDGE's own rule, so it cannot rescue such a row."""
    out, summary = _run(tmp_path, str(_write_audit(tmp_path)), search_error_conv1=True)
    [row] = [
        json.loads(line)
        for line in (out / "conv1" / "rejudge_agentic.jsonl").read_text().splitlines()
    ]
    assert (row["is_correct"], row["judgments"]) == (False, [])
    # conv0's two right answers; both of conv1's rows are graded WRONG either way.
    assert summary["rejudged"]["correct"] == 2


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------


def test_the_rejudged_score_replaces_only_the_disputed_verdicts(
    tmp_path: Path,
) -> None:
    _out, s = _run(tmp_path, str(_write_audit(tmp_path)))
    assert (s["correct"], s["total"]) == (2, 5)
    rv = s["rejudged"]
    assert (rv["correct"], rv["total"], rv["rejudged"]) == (4, 5, 2)
    assert rv["category_stats"] == {
        "2": {"correct": 1, "total": 2},
        "4": {"correct": 3, "total": 3},
    }
    assert rv["per_run_accuracies"] == [0.8, 0.8, 0.8]


def test_a_missing_audit_is_reported_not_raised(tmp_path: Path) -> None:
    _out, s = _run(tmp_path, str(tmp_path / "absent.json"))
    assert s["rejudged"] is None
    assert s["rejudged_note"].startswith("not re-judged: gold audit not found")
    assert "download_data.sh locomo" in s["rejudged_note"]


def test_a_benchmark_without_an_audit_says_so(tmp_path: Path) -> None:
    _out, s = _run(tmp_path, "")
    assert s["rejudged"] is None
    assert s["rejudged_note"] == "no gold audit exists for this benchmark"


def test_a_run_judged_before_the_audit_gives_no_partial_number(
    tmp_path: Path,
) -> None:
    """Substituting some disputed verdicts and not others would read as finished."""
    _out, s = _run(tmp_path, str(_write_audit(tmp_path)), rejudge=False)
    assert s["rejudged"] is None
    assert "2 of 2 disputed question(s) not re-judged yet" in s["rejudged_note"]


def _report(tmp_path: Path, audit_path: str) -> str:
    _out, s = _run(tmp_path, audit_path)
    txt = tmp_path / "report.txt"
    run_mod._write_report_txt(txt, {"agentic": s}, [0, 1], _config(audit_path), {}, "")
    return txt.read_text(encoding="utf-8")


def test_the_report_shows_both_scores_side_by_side(tmp_path: Path) -> None:
    audit = _write_audit(tmp_path)
    text = _report(tmp_path, str(audit))
    assert "Accuracy:         40.0% (2/5, majority of 3)" in text
    assert (
        "Re-judged:        80.0% (4/5, majority of 3; mean 80.0%) -- 2 disputed "
        f"question(s) re-judged against the corrected answers in {audit}"
    ) in text
    assert "    2. temporal         0.0% (0/2)       re-judged  50.0% (1/2)" in text


def test_the_report_names_why_the_rejudged_score_is_missing(tmp_path: Path) -> None:
    text = _report(tmp_path, "")
    assert "Re-judged:        unavailable -- no gold audit exists" in text
    assert "    2. temporal         0.0% (0/2)\n" in text


def test_the_terminal_summary_shows_the_rejudged_score(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    audit = str(_write_audit(tmp_path))
    _out, s = _run(tmp_path, audit)
    run_mod._print_terminal_summary({"agentic": s}, tmp_path, "", _config(audit))
    out = capsys.readouterr().out
    assert "Re-judged:  80.0% (4/5)" in out
    assert "    2. temporal         0.0% (0/2)  re-judged  50.0%" in out


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------


def test_judge_re_judges_right_after_judging() -> None:
    """In the JUDGE stage, on the answers JUDGE just graded, and only with an audit."""
    src = inspect.getsource(run_mod.run_conversation)
    judge_at = src.index("run_evaluate_phase(")
    rejudge_at = src.index("run_rejudge_phase(")
    assert judge_at < rejudge_at
    assert "if gold_corrections:" in src[judge_at:rejudge_at]


def test_the_run_reads_the_audit_once_and_hands_it_to_each_conversation() -> None:
    src = inspect.getsource(run_mod._main_inner)
    assert "_gold_audit(config)" in src
    assert "gold_corrections=_corrections" in src


def test_locomo_reads_the_audit_download_data_fetches() -> None:
    path = BenchmarkConfig.from_toml("locomo").gold_audit_path
    assert path.endswith("benchmarks/data/locomo-audit-errors.json")
    script = (_BENCH / "download_data.sh").read_text(encoding="utf-8")
    assert '"$DATA/locomo-audit-errors.json"' in script


@pytest.mark.parametrize("name", ["longmemeval", "evermembench", "subtlememory"])
def test_the_other_benchmarks_have_no_audit(name: str) -> None:
    assert BenchmarkConfig.from_toml(name).gold_audit_path == ""
