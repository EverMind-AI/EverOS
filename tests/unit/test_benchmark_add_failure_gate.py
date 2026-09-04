"""A conversation whose ingest failed is not searched, answered or judged.

`_run_pass` already said this was the rule -- "A conversation that failed its ingest
must not be read: its store is short whatever the failure dropped, and searching it
would score the gap as a retrieval miss" -- and then recorded the outcome without
acting on it. The read pass ran over every conversation in the group, so a failed ADD
produced a full set of rows against a partial store, and those rows landed in the
report's denominator. The run does exit non-zero, but the report was written first and
outlives the exit code: nothing in the file said which conversations it could not
cover.

Structural rather than end-to-end, because the subject is one sequencing decision
inside `main()`, whose surroundings are a server fleet, a thread pool and four LLM
clients. `_run_server_group` is reproduced here from `run.py`'s own source, so the copy
cannot silently drift from the original.
"""

from __future__ import annotations

import ast
import sys
import threading
from pathlib import Path
from typing import Any

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

SRC = (_BENCH / "run.py").read_text(encoding="utf-8")


def _source_of(name: str) -> str:
    """The named nested function, exactly as ``run.py`` defines it."""
    for node in ast.walk(ast.parse(SRC)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(SRC, node) or ""
    raise AssertionError(f"{name} is gone from run.py; this test's subject moved")


def _harness(add_fails: set[int]) -> dict[str, Any]:
    """Run the real ``_run_server_group`` body against recording stand-ins."""
    calls: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
    results: dict[int, bool] = {}
    written: list[str] = []

    def _run_pass(convs: list[int], stages: list[str]) -> None:
        calls.append((tuple(convs), tuple(stages)))
        if stages == ["add"]:
            for ci in convs:
                results[ci] = ci not in add_fails

    ns: dict[str, Any] = {
        "_run_pass": _run_pass,
        "_does_add": True,
        "_read_stages": ["search", "answer", "judge"],
        "budget_stopped": False,
        "_quiesce_servers": lambda _urls: None,
        "_urls": ["http://127.0.0.1:9000"],
        "_res_lock": threading.Lock(),
        "results": results,
        "_tqdm": type("_T", (), {"write": staticmethod(written.append)}),
    }
    exec(_source_of("_run_server_group"), ns)
    ns["_run_server_group"](0, [0, 1, 2])
    return {"calls": calls, "written": written}


def test_a_failed_ingest_skips_its_read_stages() -> None:
    """The one that fabricates numbers: no search, no answer, no judgment for it."""
    out = _harness(add_fails={1})
    read = [c for c in out["calls"] if c[1] != ("add",)]
    assert read == [((0, 2), ("search", "answer", "judge"))]


def test_a_clean_ingest_still_reads_every_conversation() -> None:
    """The guard must not cost the normal path anything."""
    out = _harness(add_fails=set())
    read = [c for c in out["calls"] if c[1] != ("add",)]
    assert read == [((0, 1, 2), ("search", "answer", "judge"))]


def test_no_read_pass_at_all_when_every_ingest_failed() -> None:
    """An empty pass would submit nothing and report a clean group."""
    out = _harness(add_fails={0, 1, 2})
    assert [c for c in out["calls"] if c[1] != ("add",)] == []


def test_the_skip_is_announced() -> None:
    """Silently scoring fewer conversations is the failure being fixed."""
    out = _harness(add_fails={1})
    assert any("SKIPPING" in line and "[1]" in line for line in out["written"])


def test_the_report_records_what_it_could_not_cover() -> None:
    """The exit code does not survive into the file, so the file has to say it."""
    assert 'summary["unscorable_conversations"] = sorted(unscorable)' in SRC
    assert "aggregate_report(output_dir, args.conv, config, unscorable=failed)" in SRC
