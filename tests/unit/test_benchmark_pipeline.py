"""End-to-end contract of the four benchmark stages, driven against stubs.

No network, no server, no model: the EverOS client and both LLM clients are stubs,
so this runs in milliseconds and pins the parts that broke in practice rather than
the parts that are slow.

What it pins:

* every stage writes exactly one row per question -- the count is the denominator, and a
  question lost to a crash or a resume silently RAISES the reported accuracy
* resume is idempotent: interrupting a stage and rerunning it yields the same rows, once
  each, in the same order. An earlier version keyed resume on question text, so a
  conversation with two identical questions lost rows and reported "0 to go"
* a failed retrieval is carried through as [SEARCH_FAILED] and graded WRONG rather than
  raised, keeping the question in the denominator
* the aggregate reports majority accuracy over the full question count
"""

from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import run as run_mod  # noqa: E402
from config import BenchmarkConfig  # noqa: E402


class StubEveros:
    """Returns two episodes for every search, or a 500 for questions in ``fail``."""

    def __init__(self, fail: set[int] | None = None) -> None:
        self.fail = fail or set()
        self.calls = 0

    def post(self, path: str, payload: dict, **_kw: Any) -> tuple[int, dict]:
        self.calls += 1
        if int(payload.get("_q", -1)) in self.fail:
            return 500, {"detail": "stub failure"}
        return 200, {
            "data": {
                "episodes": [
                    {
                        "subject": "S1",
                        "episode": "Episode one text.",
                        "session_id": "s1",
                    },
                    {
                        "subject": "S2",
                        "episode": "Episode two text.",
                        "session_id": "s2",
                    },
                ],
                "profiles": [],
            }
        }


class StubLLM:
    """Answers every prompt, and judges by whether the gold string is echoed back."""

    def __init__(self, answer: str = "FINAL ANSWER: yes") -> None:
        self.answer = answer
        self.chat = self
        self.completions = self
        self.n = 0

    def create(self, **kwargs: Any) -> Any:
        self.n += 1
        prompt = kwargs["messages"][-1]["content"]
        reply = (
            '{"label": "CORRECT"}'
            if "label" in prompt or "CORRECT" in prompt
            else self.answer
        )

        class _M:
            content = reply

        class _C:
            message = _M()

        class _R:
            choices = [_C()]
            usage = type(
                "U",
                (),
                {"total_tokens": 30, "prompt_tokens": 25, "completion_tokens": 5},
            )()

        return _R()


QA = [
    {"question": "Same question?", "answer": "yes", "category": 1, "question_id": "q0"},
    # Deliberately identical text: resume must key on the index, not the question.
    {"question": "Same question?", "answer": "yes", "category": 1, "question_id": "q1"},
    {"question": "Third?", "answer": "yes", "category": 4, "question_id": "q2"},
]


@pytest.fixture
def cfg() -> BenchmarkConfig:
    return BenchmarkConfig.from_toml("locomo").model_copy(
        update={"search_concurrency": 2, "eval_concurrency": 2, "judge_runs": 1}
    )


def _search(conv: Path, cfg: BenchmarkConfig, client: Any) -> list:
    return run_mod.run_search_phase(
        client,
        QA,
        "owner_0",
        "hybrid",
        cfg.top_k,
        "default",
        "default",
        conv,
        cfg,
        method_label="hybrid",
    )


def _answer(conv: Path, cfg: BenchmarkConfig, _results: list, llm: Any) -> list:
    # The stage reads its input from the previous stage's file, not from a list.
    return run_mod.run_answer_phase(
        conv / "search_hybrid.jsonl", "A", "B", llm, cfg, conv, method_label="hybrid"
    )


def _judge(conv: Path, cfg: BenchmarkConfig, _answers: list, llm: Any) -> list:
    return run_mod.run_evaluate_phase(
        conv / "answer_hybrid.jsonl",
        llm,
        cfg,
        cfg.judge_runs,
        conv,
        method_label="hybrid",
    )


def test_every_stage_writes_one_row_per_question(
    tmp_path: Path, cfg: BenchmarkConfig
) -> None:
    conv = tmp_path / "conv0"
    conv.mkdir()
    llm = StubLLM()

    searches = _search(conv, cfg, StubEveros())
    answers = _answer(conv, cfg, searches, llm)
    judged = _judge(conv, cfg, answers, llm)

    assert len(searches) == len(QA)
    assert len(answers) == len(QA)
    assert len(judged) == len(QA)
    for name in ("search_hybrid", "answer_hybrid", "judge_hybrid"):
        rows = (conv / f"{name}.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(rows) == len(QA), (
            f"{name} wrote {len(rows)} rows for {len(QA)} questions"
        )
        assert [json.loads(r)["index"] for r in rows] == list(range(len(QA)))


def test_resume_is_idempotent(tmp_path: Path, cfg: BenchmarkConfig) -> None:
    """Two questions share their text; resume must still count three rows, not two."""
    conv = tmp_path / "conv0"
    conv.mkdir()

    # First pass: only the first question, as if the stage had been interrupted.
    partial = run_mod.run_search_phase(
        StubEveros(),
        QA[:1],
        "owner_0",
        "hybrid",
        cfg.top_k,
        "default",
        "default",
        conv,
        cfg,
        method_label="hybrid",
    )
    assert len(partial) == 1

    full = _search(conv, cfg, StubEveros())
    assert len(full) == len(QA)
    indices = [r.index for r in full]
    assert indices == sorted(set(indices)), "resume duplicated or dropped a row"

    llm = StubLLM()
    answers = _answer(conv, cfg, full, llm)
    judged = _judge(conv, cfg, answers, llm)
    assert len(judged) == len(QA)


def test_failed_search_is_graded_not_dropped(
    tmp_path: Path, cfg: BenchmarkConfig
) -> None:
    """A question whose retrieval fails stays in the denominator, scored WRONG."""
    conv = tmp_path / "conv0"
    conv.mkdir()

    # The stub keys its failure off a marker the payload carries for question 1 only.
    class FailingSecond(StubEveros):
        def post(self, path: str, payload: dict, **_kw: Any) -> tuple[int, dict]:
            if payload.get("query") == "Third?":
                return 500, {"detail": "stub failure"}
            return super().post(path, payload, **_kw)

    searches = _search(conv, cfg, FailingSecond())
    assert len(searches) == len(QA)
    failed = [s for s in searches if s.search_error]
    assert len(failed) == 1 and failed[0].question == "Third?"

    llm = StubLLM()
    answers = _answer(conv, cfg, searches, llm)
    assert any(a.generated_answer == "[SEARCH_FAILED]" for a in answers)

    judged = _judge(conv, cfg, answers, llm)
    assert len(judged) == len(QA)
    bad = [j for j in judged if j.search_error]
    assert len(bad) == 1 and bad[0].is_correct is False
    assert bad[0].judgments == [], "a failed search must not consume a judge call"


def test_summary_denominator_is_the_question_count(
    tmp_path: Path, cfg: BenchmarkConfig
) -> None:
    conv = tmp_path / "conv0"
    conv.mkdir()
    llm = StubLLM()
    judged = _judge(
        conv, cfg, _answer(conv, cfg, _search(conv, cfg, StubEveros()), llm), llm
    )

    summary = run_mod._collect_method_summary("hybrid", conv.parent, [0], cfg)
    assert summary["total"] == len(QA)
    assert summary["correct"] == len(judged)
    assert summary["answer"]["avg_prompt_tokens"] > 0, "Average Tokens must be reported"


def test_transport_error_is_retried_then_recorded(
    tmp_path: Path, cfg: BenchmarkConfig
) -> None:
    """A connection blip must be retried, not end the stage.

    The client reports a transport failure as a negative status so the retry loop
    treats it like a 5xx. Letting the exception escape would abandon every
    remaining question.
    """
    conv = tmp_path / "conv0"
    conv.mkdir()

    class Flaky(StubEveros):
        def __init__(self) -> None:
            super().__init__()
            self.seen = 0

        def post(self, path: str, payload: dict, **kw: Any) -> tuple[int, dict]:
            self.seen += 1
            if self.seen == 1:  # first call looks like a dropped connection
                return -1, {"error": "Connection aborted"}
            return super().post(path, payload, **kw)

    client = Flaky()
    searches = _search(conv, cfg, client)
    assert len(searches) == len(QA)
    assert not any(s.search_error for s in searches), "a retried blip is not an error"
    assert client.seen == len(QA) + 1, "the failed call should have been retried once"


def test_persistent_transport_error_is_recorded(
    tmp_path: Path, cfg: BenchmarkConfig
) -> None:
    """When it never recovers, the question is kept and marked, not dropped."""
    conv = tmp_path / "conv0"
    conv.mkdir()

    class Dead(StubEveros):
        def post(self, path: str, payload: dict, **_kw: Any) -> tuple[int, dict]:
            return -1, {"error": "Connection refused"}

    searches = _search(conv, cfg, Dead())
    assert len(searches) == len(QA)
    assert all(s.search_error for s in searches)


def test_report_writes_and_prints(
    tmp_path: pathlib.Path, cfg: BenchmarkConfig, capsys: Any
) -> None:
    """Both report paths must run to completion for a finished conversation.

    They execute only after every stage has succeeded, so a defect here wastes the
    whole run and leaves a non-zero exit on data that is actually complete. Three
    NameErrors lived here -- the category label, the terminal summary and the search
    worker -- because nothing called it.
    """
    conv = tmp_path / "conv0"
    conv.mkdir()
    llm = StubLLM()
    _judge(conv, cfg, _answer(conv, cfg, _search(conv, cfg, StubEveros()), llm), llm)

    summary = run_mod._collect_method_summary("hybrid", tmp_path, [0], cfg)
    assert summary is not None
    all_summaries = {"hybrid": summary}

    txt = tmp_path / "report.txt"
    run_mod._write_report_txt(
        txt, all_summaries, [0], cfg, {"git_hash": "test"}, "0h 0m 1s"
    )
    body = txt.read_text(encoding="utf-8")
    assert "Accuracy:" in body
    assert "Average tokens:" in body, "the context-efficiency figure must be reported"
    # The label comes from the adapter; LoCoMo's category 1 is multi-hop, which is
    # counter-intuitive and was printed wrongly by a second map inside run.py.
    assert "multi-hop" in body

    run_mod._print_terminal_summary(all_summaries, tmp_path, "0h 0m 1s", cfg)
    out = capsys.readouterr().out
    assert "Accuracy:" in out
    assert "Avg tokens:" in out


def test_report_labels_categories_from_the_adapter(cfg: BenchmarkConfig) -> None:
    """One category map, in the adapter. A second copy in run.py contradicted it."""
    from adapters import locomo

    for key, want in locomo.categories().items():
        assert run_mod._category_label(cfg, key) == want
