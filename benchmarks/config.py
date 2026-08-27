"""Benchmark configuration.

Frozen Pydantic model providing all tunable parameters for the LoCoMo
benchmark pipeline.  Defaults are aligned with the upstream evaluation
reference so that numbers are directly comparable.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def unresolved(value: str) -> bool:
    """Whether a string still contains an unexpanded ``${VAR}`` reference.

    Needed because leaving the literal in place is the right call for a required field
    -- an endpoint that silently became "" is a connection error thirty minutes in,
    while ``${BENCH_DECIDER_BASE_URL}`` names what is missing -- and the wrong call for
    an optional one, where the literal becomes a directory named ``${BENCH_EVAL_ROOT}``
    or a model no provider serves. Callers decide which of the two a field is.
    """
    return "${" in (value or "")


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """Substitute ``${VAR}`` / ``${VAR:-default}`` in every string in a config tree.

    Exists so a shipped config can name what a run needs without containing it. The
    files used to carry a private gateway URL and an absolute path into one workspace,
    which made them unusable to anyone else and leaked the deployment either way.

    An unset variable with no default is left as the literal ``${VAR}``, not replaced by
    the empty string: an endpoint silently becoming "" is a connection error thirty
    minutes into a run, while the unexpanded text names the variable that is missing.
    """
    if isinstance(value, str):
        return _ENV_REF.sub(
            lambda m: (
                os.environ.get(m.group(1))
                or (m.group(2) if m.group(2) is not None else m.group(0))
            ),
            value,
        )
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


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
        eval_concurrency: How many questions are processed in parallel within each
        conversation.
    """

    model_config = ConfigDict(frozen=True)

    # --- EverOS server ---
    cascade_timeout: int = 7200
    batch_size: int = 25

    # --- Search ---
    methods: str = "agentic"
    top_k: int = 10
    eval_owner: Literal["speaker_a", "speaker_b"] = "speaker_a"
    adapter: str = "locomo"
    """Which dataset adapter drives the run: locomo | longmemeval | evermembench |
    subtlememory. The adapter owns everything that differs between benchmarks -- how
    questions load, which owner they live under, how gold maps to store session ids, and
    which judge grades them."""
    data_path: str = ""
    """Dataset location. Empty keeps the CLI default; datasets are NOT vendored into
    this
    repository (LongMemEval alone is 278 MB)."""
    conversations: int = 10
    """How many conversation units the benchmark has."""
    servers: int = 1
    """How many EverOS servers `--servers` starts when the flag is not given explicitly.

    ADD is serial within a conversation -- one flush per session, one extraction per
    flush -- so the only parallelism is across conversations, and run.py clamps this to
    the number of conversations being run. The server is not the constraint: it waits on
    remote extraction
    calls, so the useful setting follows the dataset's shape, not the machine's."""
    app_id: str = "default"
    project_id: str = "default"
    """Store partition keys -- the values written INTO the index, not the directory
    names:
    the tree is <root>/default_app/default_project/... while these fields read
    `default`.
    Passing a directory name matches no rows and returns zero episodes without raising.
    """

    # --- EverOS-side backbone ---
    backbone_model: str = ""
    """The model EverOS itself runs, applied when this harness starts the servers.

    EverOS has a single `[llm]` setting, so this ONE model does both jobs: it extracts
    memories during ADD and it is the decider inside the multi-round retrieval loop
    (`extract_agent_case`, `extract_user_profile`, `extract_atomic_facts`,
    `reflect_episodes` and the search loop all resolve the same `get_llm_client()`).
    There is no separate decider knob to set.

    Empty means "leave EVEROS_LLM__MODEL as the environment already has it" -- which is
    the only option when attaching to servers via --base-url, since their backbone was
    fixed when they were launched. It matters which one it was: a store extracted by one
    backbone is not comparable to a store extracted by another, and that fact is not
    recoverable
    from the store afterwards."""

    providers: dict[str, str] = Field(default_factory=dict)
    """OpenRouter provider allow-list per model vendor: ``{"openai": "openai"}``.

    The key is the vendor prefix of a model id (``openai`` in ``openai/gpt-4.1-mini``),
    the value a comma-separated list of OpenRouter provider slugs. A vendor listed here
    is pinned with ``provider={"only": [...], "allow_fallbacks": false}``; a vendor NOT
    listed is free-routed by OpenRouter across every provider serving that model, which
    can mean a different quantisation per request -- the slug carries it
    (``gmicloud/fp8`` vs ``baseten/fp4``). ``OPENROUTER_PROVIDER_ONLY`` overrides this
    for every vendor.

    Applies to the answer and judge clients this harness owns. The extraction backbone
    runs inside the EverOS servers, whose LLM client sends no provider block at all, so
    pinning it would need a change in everos.component.llm."""
    results_root: str = ""
    """Where this benchmark's runs are written.

    Per benchmark rather than global: a run belongs with the dataset it scored, next to
    the store it read, not in a directory keyed by whichever harness happened to produce
    it. Empty falls back to the CLI default, which keeps a bare checkout runnable.
    """
    include_profile: bool | None = None
    """Send ``include_profile`` on every search, so the answer prompt gets the owner's
    profile block alongside the retrieved episodes.

    ``None`` (the default) defers to the adapter's ``INCLUDE_PROFILE`` attribute, which
    is where the decision belongs when it follows from the benchmark's own task set --
    EverMemBench grades 541 persona questions and declares ``True``; a benchmark whose
    reference never sent the flag declares nothing and gets ``False``. Setting it here
    overrides that, which is what a profile ablation needs. The server defaults the
    field to ``False``, so a run that wants profiles has to ask."""
    trace: bool = True
    """Export the per-round retrieval trace and the profile-extraction trace.

    ``EVEROS_LLMMR_TRACE_DUMP`` and ``EVEROS_PROFILE_TRACE_DUMP`` are derived per
    server under ``results/<run>/traces/``. On by default: a run without them cannot be
    attributed afterwards, and the two files are the only record of what the decider saw
    and which memcells each profile was built from. Turn it off only for a throughput
    measurement where the writes themselves are being measured."""

    retrieval_env: dict[str, str] = Field(default_factory=dict)
    """Retrieval knobs pushed into the servers this harness starts, as EVEROS_*
    variables.

    The multi-round route reads twelve `EVEROS_LLMMR_*` settings straight from the
    process environment (round cap, seed/sub-query top-k, RRF k, patience, decider
    retries, whether the decider sees full text, ...). None of them is a model -- the
    decider runs the same `[llm].model` as extraction -- but every one of them changes
    the result, so they belong in the benchmark's config where they are part of the
    recorded run rather than in a launch script where they leave no trace.

    Ignored when attaching with --base-url: those servers read their environment at
    startup, so their values are already fixed."""

    backbone_base_url: str = ""
    """Endpoint for the backbone. Empty keeps whatever the environment already has.

    Set it to a local server (`http://127.0.0.1:8000/v1` for vLLM / SGLang) to run the
    backbone on-box instead of through a hosted API -- the store is then extracted, and
    every retrieval decision made, by that model. Which one it was is not recoverable
    from
    the store afterwards, so it belongs in the config next to the model name."""
    backbone_api_key: str = ""
    decider_model: str = ""
    """Model that makes the retrieval decisions, if different from the backbone.

    Every published run used two different models here -- the directory names record
    them apart, e.g. `backbone-GPT-4.1-mini_retrieval-Qwen3.6-27B`. Collapsing them into
    one field was wrong: the model that extracted a store and the model that decides
    which episodes are core are separate choices with separate costs. Empty falls back
    to the
    backbone, which is what EverOS did before it grew a `[decider]` section."""
    decider_api_key: str = ""
    """Key for `decider_base_url`. The self-hosted gateway accepts any non-empty value.
    """
    decider_base_url: str = ""
    """Endpoint for `decider_model`. The 27B decider is served locally, so this is
    normally
    a local address; empty inherits the backbone's endpoint."""
    """Key for `backbone_base_url`. Local servers usually accept any non-empty string;
    leave empty to inherit `EVEROS_LLM__API_KEY` from the environment."""

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
            config_dir = Path(__file__).parent / "configs"
        # Accept a bare benchmark name ("locomo") as well as a legacy "config.locomo"
        # form, so the CLI can take one identifier instead of a name plus a file path.
        stem = name[len("config.") :] if name.startswith("config.") else name
        # "config" was the file's name before the per-benchmark layout; keep it
        # resolving to the shared defaults rather than breaking callers that still ask
        # for it.
        stem = "default" if stem in ("", "config") else stem
        path = config_dir / f"{stem}.toml"
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "rb") as f:
            overrides = _expand_env(tomllib.load(f))
        # `[answer]` / `[judge]` group the settings by pipeline stage in the file; the
        # model keeps them flat so a field is named the same everywhere it is read.
        for section, prefix in (("answer", "answer_"), ("judge", "judge_")):
            for key, value in (overrides.pop(section, None) or {}).items():
                overrides[f"{prefix}{key}"] = value
        # `concurrency` under [answer] covers both answering and judging.
        if "answer_concurrency" in overrides:
            overrides["eval_concurrency"] = overrides.pop("answer_concurrency")
        return cls(**overrides)


class SearchResult(BaseModel):
    """One QA pair's search stage output."""

    model_config = ConfigDict(frozen=True)

    index: int
    question: str
    golden_answer: str
    question_id: str = ""
    """Carried through so JUDGE can ask the adapter for this question's grading
    rule. LongMemEval marks unanswerable questions with an `_abs` suffix here, and that
    is not recoverable from the question text."""
    qa_meta: dict[str, Any] = Field(default_factory=dict)
    search_error: str = ""
    """Set when retrieval failed for this question after retries.

    The record is kept rather than dropped: the reference answers it `[SEARCH_FAILED]`
    and grades it WRONG, so the question stays in the denominator. Dropping it instead
    would
    silently raise the reported accuracy."""
    """Benchmark-specific fields its own judge reads, passed through untouched.

    SubtleMemory grades against relation type/subtype, the extracted facts, and both
    the accepted and known-incorrect answer lists, so question text and a single gold
    string are not enough. Carried search -> answer -> judge because the judge stage
    reads its input from the previous stage's file, not from the dataset."""
    question_date: str = ""
    """Anchors temporal questions to when they were asked; empty for benchmarks that
    have no such field."""
    category: int | str | None
    evidence: list[str]
    """Raw citation as the dataset writes it (LoCoMo's ``D<session>:<turn>``). Only
    LoCoMo
    populates this; it is kept for traceability, not for scoring."""
    gold_sessions: list[str] = Field(default_factory=list)
    """Store session ids the adapter resolved this question's gold to. run.py never
    computes metrics itself -- metrics/ir.py and metrics/core.py are post-processors
    that read these files -- so gold that is not written here cannot be scored at all.
    `evidence`
    alone is not enough: it is empty for every benchmark except LoCoMo."""
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
    question_id: str = ""
    """Carried through so JUDGE can ask the adapter for this question's grading
    rule. LongMemEval marks unanswerable questions with an `_abs` suffix here, and that
    is not recoverable from the question text."""
    qa_meta: dict[str, Any] = Field(default_factory=dict)
    search_error: str = ""
    """Set when retrieval failed for this question after retries.

    The record is kept rather than dropped: the reference answers it `[SEARCH_FAILED]`
    and grades it WRONG, so the question stays in the denominator. Dropping it instead
    would
    silently raise the reported accuracy."""
    """Benchmark-specific fields its own judge reads, passed through untouched.

    SubtleMemory grades against relation type/subtype, the extracted facts, and both
    the accepted and known-incorrect answer lists, so question text and a single gold
    string are not enough. Carried search -> answer -> judge because the judge stage
    reads its input from the previous stage's file, not from the dataset."""
    category: int | str | None
    generated_answer: str
    answer_time_s: float
    answer_attempts: int
    answer_tokens: int = 0
    """Prompt + completion tokens across every attempt, i.e. what the run cost."""
    answer_prompt_tokens: int = 0
    """Context tokens on the attempt that produced the answer.

    This is the paper's Average Tokens metric. It is deliberately NOT the retry-
    inclusive total: the point of the number is how much context the retrieval put in
    front of the
    answer model, which a retry does not change."""
    answer_completion_tokens: int = 0
    """Completion tokens on the attempt that produced the answer."""
    answer_prompt_tokens_total: int = 0
    """Context tokens across every attempt, retries included."""
    answer_completion_tokens_total: int = 0
    """Completion tokens across every attempt, retries included."""


class JudgeResult(BaseModel):
    """One QA pair's judge stage output."""

    model_config = ConfigDict(frozen=True)

    index: int
    question: str
    golden_answer: str
    generated_answer: str
    qa_meta: dict[str, Any] = Field(default_factory=dict)
    """Carried through so the graded rows can be split the way the benchmark splits
    them. Without it the judge file cannot be broken down by SubtleMemory's relation
    subtype or EverMemBench's multiple-choice/open-ended halves after the fact."""
    question_id: str = ""
    """Carried through so JUDGE can ask the adapter for this question's grading
    rule. LongMemEval marks unanswerable questions with an `_abs` suffix here, and that
    is not recoverable from the question text."""
    category: int | str | None
    search_error: str = ""
    """Non-empty when retrieval failed. Such a row is graded WRONG without a judge call
    and
    stays in the denominator, so a run degraded by retrieval failures cannot read as
    clean.
    """
    judge_failed: bool = False
    """Set when the judge could not be read after every retry AND this benchmark's
    protocol excludes such a row from the denominator rather than grading it wrong.
    LongMemEval's harness does that (rejudge_hybrid.py:79 keeps only rows whose verdict
    is not None); LoCoMo's does not. Excluding rows can only ever raise the reported
    number, so the count is printed with the result."""
    is_correct: bool
    judgments: list[bool]
    judge_tokens: int = 0


class ServingSpec(BaseModel):
    """What was actually serving one role during a run.

    This exists because a whole batch of decider experiments had to be thrown away: the
    launch parameters lived only in hand-written shell scripts, two of them (
    ``reasoning_parser`` and prefix caching) were omitted on some runs, and afterwards
    there was no way to tell which results came from which stack. Recording the model
    name is not enough -- record the endpoint, the engine version and the full argv.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    """``decider`` | ``answer`` | ``judge`` | ``embedding`` | ``reranker`` |
    ``extraction``.
    """
    model: str
    endpoint: str = ""
    local: bool = False
    engine_version: str = ""
    """vLLM / SGLang version when locally served; empty for a remote endpoint."""
    launch_argv: list[str] = Field(default_factory=list)
    """Complete command, unsummarised. Empty for a remote endpoint."""
    extra: dict = Field(default_factory=dict)
    """Anything role-specific: tensor_parallel, max_model_len, reasoning_parser,
    prefix_caching, chat_template_kwargs."""


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
    benchmark: str = "locomo"
    """Which adapter drove the run."""
    store_root: str = ""
    """The memory root queried. A run that skips ADD is scored against a store somebody
    else built, so the root is part of the result's identity."""
    packages: dict[str, str] = Field(default_factory=dict)
    """Installed versions of the packages that can change the numbers -- the everalgo
    family above all. `everos_version` alone is not enough: EverOS 1.2.3 on
    everalgo-user-memory 0.3.1 and on 0.4.0 are two different experiments, because that
    package IS the extraction algorithm. The shipped stores were extracted with 0.3.1
    while this repository pins 0.4.0, so a from-scratch end-to-end run cannot reproduce
    published numbers by construction."""
    serving: list[ServingSpec] = Field(default_factory=list)
