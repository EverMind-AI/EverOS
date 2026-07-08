"""Unit tests for BenchmarkConfig loading."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_defaults_construct_without_error() -> None:
    """BenchmarkConfig() with defaults must construct and have all fields."""
    from benchmarks.config import BenchmarkConfig

    cfg = BenchmarkConfig()
    assert isinstance(cfg.methods, str)
    assert cfg.top_k > 0
    assert cfg.cascade_timeout > 0
    assert cfg.batch_size > 0
    assert isinstance(cfg.answer_model, str)
    assert isinstance(cfg.judge_model, str)
    assert cfg.judge_runs >= 1
    assert cfg.conversations_concurrency >= 1
    assert cfg.eval_concurrency >= 1
    assert cfg.search_concurrency >= 1


def test_from_toml_loads_default_config() -> None:
    """from_toml('config') loads benchmarks/config.toml successfully."""
    from benchmarks.config import BenchmarkConfig

    cfg = BenchmarkConfig.from_toml("config")
    assert isinstance(cfg.methods, str)
    assert isinstance(cfg.judge_model, str)


def test_from_toml_override(tmp_path: Path) -> None:
    """A custom TOML overrides specific fields, rest stay default."""
    from benchmarks.config import BenchmarkConfig

    (tmp_path / "custom.toml").write_text(
        'methods = "hybrid"\ntop_k = 20\n', encoding="utf-8"
    )
    cfg = BenchmarkConfig.from_toml("custom", config_dir=tmp_path)
    assert cfg.methods == "hybrid"
    assert cfg.top_k == 20
    assert cfg.judge_model == "gpt-4o-mini"  # untouched


def test_from_toml_missing_file_raises() -> None:
    """from_toml with nonexistent name raises FileNotFoundError."""
    from benchmarks.config import BenchmarkConfig

    with pytest.raises(FileNotFoundError):
        BenchmarkConfig.from_toml("nonexistent")


def test_frozen_model() -> None:
    """BenchmarkConfig instances are immutable."""
    from benchmarks.config import BenchmarkConfig

    cfg = BenchmarkConfig()
    with pytest.raises(Exception):  # noqa: B017 — ValidationError
        cfg.top_k = 999


def test_concurrency_fields_exist() -> None:
    """Concurrency fields exist and are positive; no legacy 'concurrency' attr."""
    from benchmarks.config import BenchmarkConfig

    cfg = BenchmarkConfig()
    assert cfg.conversations_concurrency >= 1
    assert cfg.eval_concurrency >= 1
    assert cfg.search_concurrency >= 1
    assert not hasattr(cfg, "concurrency")


def test_search_result_frozen() -> None:
    from benchmarks.config import SearchResult

    sr = SearchResult(
        index=0,
        question="q",
        golden_answer="a",
        category=1,
        evidence=[],
        episodes=[],
        profiles=[],
        search_time_s=0.5,
        method="agentic",
    )
    assert sr.index == 0
    with pytest.raises(Exception):  # noqa: B017
        sr.index = 99


def test_search_result_jsonl_roundtrip() -> None:
    from benchmarks.config import SearchResult

    sr = SearchResult(
        index=0,
        question="q",
        golden_answer="a",
        category=1,
        evidence=["e1"],
        episodes=[{"id": "ep1"}],
        profiles=[],
        search_time_s=0.5,
        method="agentic",
    )
    line = sr.model_dump_json()
    restored = SearchResult.model_validate_json(line)
    assert restored == sr


def test_answer_result_tracks_tokens() -> None:
    from benchmarks.config import AnswerResult

    ar = AnswerResult(
        index=0,
        question="q",
        golden_answer="a",
        category=1,
        generated_answer="ans",
        answer_time_s=1.0,
        answer_attempts=1,
        answer_tokens=500,
    )
    assert ar.answer_tokens == 500


def test_judge_result_jsonl_roundtrip() -> None:
    from benchmarks.config import JudgeResult

    jr = JudgeResult(
        index=0,
        question="q",
        golden_answer="a",
        generated_answer="ans",
        category=1,
        is_correct=True,
        judgments=[True, True, False],
        judge_tokens=200,
    )
    line = jr.model_dump_json()
    restored = JudgeResult.model_validate_json(line)
    assert restored == jr
    assert restored.is_correct is True


def test_run_spec_includes_git_hash() -> None:
    from benchmarks.config import RunSpec

    spec = RunSpec(
        run_name="v1",
        config={},
        conversations=[0, 1],
        stages=["add", "search"],
        git_hash="abc1234",
        python_version="3.12.11",
        everos_version="1.1.0",
        started_at="2026-06-28T22:30:00Z",
    )
    assert spec.git_hash == "abc1234"
