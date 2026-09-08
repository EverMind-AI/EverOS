"""A benchmark report is valid only when every multi-round decider round ran."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

run = importlib.import_module("run")


def _write_trace(root: Path, *records: dict[str, object]) -> None:
    trace_dir = root / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "trace_port9400.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_other_methods_need_no_decider_trace(tmp_path: Path) -> None:
    run._assert_clean_decider_trace(tmp_path, ["hybrid"])


def test_a_clean_round_trace_passes(tmp_path: Path) -> None:
    _write_trace(
        tmp_path,
        {"round_idx": 0, "decider_failed": False},
        {"round_idx": None, "injected": []},
    )
    run._assert_clean_decider_trace(tmp_path, ["llm_multiround"])


def test_a_fallback_round_blocks_the_report(tmp_path: Path) -> None:
    _write_trace(tmp_path, {"round_idx": 0, "decider_failed": True})
    with pytest.raises(run.DeciderTraceValidationError, match="1 decider fallback"):
        run._assert_clean_decider_trace(tmp_path, ["llm_multiround"])


def test_missing_trace_blocks_the_report(tmp_path: Path) -> None:
    with pytest.raises(run.DeciderTraceValidationError, match="no retrieval trace"):
        run._assert_clean_decider_trace(tmp_path, ["llm_multiround"])


def test_malformed_trace_blocks_the_report(tmp_path: Path) -> None:
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    (trace_dir / "trace.jsonl").write_text("not json\n", encoding="utf-8")
    with pytest.raises(run.DeciderTraceValidationError, match="malformed"):
        run._assert_clean_decider_trace(tmp_path, ["llm_multiround"])
