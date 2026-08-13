"""Benchmark configuration.

Frozen Pydantic model providing all tunable parameters for the LoCoMo
benchmark pipeline.  Defaults are aligned with the upstream evaluation
reference so that numbers are directly comparable.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class BenchmarkConfig(BaseModel):
    """Immutable benchmark configuration.

    Args:
        cascade_timeout: Max seconds to wait for cascade queue to drain after flush.
        batch_size: Messages per /add request.
        methods: Comma-separated search methods.
        top_k: Number of episodes to retrieve per question.
        eval_owner: Which speaker's memory partition to query.
        answer_model: LLM model for the Answer phase.
        answer_temperature: Sampling temperature for answers.
        answer_max_tokens: Max output tokens per answer call.
        answer_timeout: Per-request timeout (seconds) for the answer LLM.
        answer_max_retries: Retry budget for the answer phase.
        judge_model: LLM model for the Judge phase.
        judge_temperature: Sampling temperature for judging.
        judge_timeout: Per-request timeout (seconds) for the judge LLM.
        judge_max_retries: Retry budget for the judge phase.
        judge_runs: Independent judge evaluations per question (majority vote).
        conversations_concurrency: How many conversations run at the same time.
        eval_concurrency: How many questions are processed in parallel within each conversation.
    """

    model_config = ConfigDict(frozen=True)

    # --- EverOS server ---
    cascade_timeout: int = 7200
    batch_size: int = 25

    # --- Search ---
    methods: str = "agentic"
    top_k: int = 10
    eval_owner: Literal["speaker_a", "speaker_b"] = "speaker_a"

    # --- Answer LLM ---
    answer_model: str = "gpt-4.1-mini"
    answer_temperature: float = 0.0
    answer_max_tokens: int = 32768
    answer_timeout: float = 300.0
    answer_max_retries: int = 5

    # --- Judge LLM ---
    judge_model: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    judge_timeout: float = 300.0
    judge_max_retries: int = 5
    judge_runs: int = 3

    # --- Concurrency ---
    conversations_concurrency: int = 10
    eval_concurrency: int = 20
    search_concurrency: int = 5

    @property
    def parsed_methods(self) -> list[str]:
        """Split comma-separated methods string into a list."""
        return [m.strip() for m in self.methods.split(",") if m.strip()]

    @classmethod
    def from_toml(
        cls, name: str = "config", *, config_dir: Path | None = None
    ) -> BenchmarkConfig:
        """Load config from a TOML file.

        Args:
            name: Config name without .toml extension.
            config_dir: Directory containing config files.
                Falls back to ``benchmarks/`` relative to the repo root.

        Raises:
            FileNotFoundError: When the TOML file does not exist.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        path = config_dir / f"{name}.toml"
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "rb") as f:
            overrides = tomllib.load(f)
        return cls(**overrides)


class SearchResult(BaseModel):
    """One QA pair's search stage output."""

    model_config = ConfigDict(frozen=True)

    index: int
    question: str
    golden_answer: str
    category: int | None
    evidence: list[str]
    episodes: list[dict]
    profiles: list[dict]
    search_time_s: float
    method: str


class AnswerResult(BaseModel):
    """One QA pair's answer stage output."""

    model_config = ConfigDict(frozen=True)

    index: int
    question: str
    golden_answer: str
    category: int | None
    generated_answer: str
    answer_time_s: float
    answer_attempts: int
    answer_tokens: int = 0


class JudgeResult(BaseModel):
    """One QA pair's judge stage output."""

    model_config = ConfigDict(frozen=True)

    index: int
    question: str
    golden_answer: str
    generated_answer: str
    category: int | None
    is_correct: bool
    judgments: list[bool]
    judge_tokens: int = 0


class RunSpec(BaseModel):
    """Reproducibility snapshot serialized at run start."""

    model_config = ConfigDict(frozen=True)

    run_name: str
    config: dict
    conversations: list[int]
    stages: list[str]
    git_hash: str
    python_version: str
    everos_version: str
    started_at: str
