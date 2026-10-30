"""OMEConfig (engine-level) + TomlRoot (per-strategy override schema).

All models forbid extra keys so configuration typos surface at startup
as StartupValidationError instead of being silently ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Self

from apscheduler.triggers.cron import CronTrigger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CounterOverride(BaseModel):
    """TOML override for a strategy's Counter gate (per-key None means keep)."""

    model_config = ConfigDict(extra="forbid")

    threshold: Annotated[int, Field(gt=0)] | None = None
    cooldown_seconds: Annotated[int, Field(ge=0)] | None = None
    event_field: Annotated[str, Field(min_length=1)] | None = None


class StrategyOverride(BaseModel):
    """TOML override for one strategy's decorator parameters."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    max_retries: Annotated[int, Field(ge=0)] | None = None
    gate: CounterOverride | None = None
    cron: str | None = None
    idle_seconds: Annotated[int, Field(gt=0)] | None = None
    scan_interval_seconds: Annotated[int, Field(gt=0)] | None = None

    @field_validator("cron")
    @classmethod
    def _validate_crontab(cls, v: str | None) -> str | None:
        if v is not None:
            CronTrigger.from_crontab(v)
        return v

    @model_validator(mode="after")
    def _check_idle_pair_consistency(self) -> Self:
        # One-sided overrides are merged with existing meta downstream,
        # so cross-check only when both fields are in this payload.
        if (
            self.idle_seconds is not None
            and self.scan_interval_seconds is not None
            and self.scan_interval_seconds > self.idle_seconds // 2
        ):
            raise ValueError(
                "StrategyOverride: scan_interval_seconds "
                f"({self.scan_interval_seconds}) must be <= idle_seconds // 2 "
                f"({self.idle_seconds // 2})"
            )
        return self


class TomlRoot(BaseModel):
    """Top-level TOML schema for ome.toml."""

    model_config = ConfigDict(extra="forbid")

    strategies: dict[str, StrategyOverride] = Field(default_factory=dict)


class OMEConfig(BaseModel):
    """Engine-level configuration consumed by OfflineEngine."""

    model_config = ConfigDict(extra="forbid")

    jobstore_path: Path = Field(
        description="SQLite DB path holding OME's own state (run records, "
        "counter store, idle store). The application passes the memory "
        "root's ``ome_db`` (``<memory-root>/.index/sqlite/ome.db``).",
    )
    aps_jobstore_path: Path | None = Field(
        default=None,
        description="SQLite DB path holding the APScheduler jobstore. Kept "
        "in a separate file from ``jobstore_path`` so APS's sync SQLAlchemy "
        "writer never contends with OME's async aiosqlite writer for the "
        "same SQLite file lock. When unset, defaults to a sibling "
        "``<stem>.aps.db`` next to ``jobstore_path``.",
    )
    max_concurrent_runs: Annotated[
        int,
        Field(
            gt=0,
            description="Engine-wide cap on concurrent strategy invocations "
            "(asyncio.Semaphore in Runner). The application passes "
            "``[ome] max_concurrent_runs``. The default suits an interactive "
            "install; a bulk ingest of many conversations is throttled by it long "
            "before the machine is, since each slot spends almost all its time "
            "waiting on a remote extraction call rather than using CPU.",
        ),
    ] = 20
    run_timeout_seconds: Annotated[
        float | None,
        Field(
            description="Wall-clock ceiling on one strategy attempt; None "
            "disables it. The application passes ``[ome] run_timeout_seconds``. "
            "Without a ceiling, a coroutine parked on an "
            "await that carries no deadline of its own -- an asyncio.Lock held "
            "by another stuck coroutine, a connection-pool wait -- keeps its "
            "max_concurrent_runs slot forever: it never raises, so it never "
            "retries, and its record stays RUNNING. Enough of them and the "
            "engine runs nothing at all, observed with 60 of 64 slots parked on "
            "one lock, starving every other strategy for 6.7 hours until it was "
            "killed by hand. Crash recovery does not cover this -- it reclaims "
            "orphans from a PREVIOUS process at startup, not live coroutines in "
            "this one. The 30-minute default is a deadlock backstop, not a "
            "latency target; killing slow-but-healthy work would trade a stall "
            "for lost extractions.",
        ),
    ] = 1800.0
    max_retries: Annotated[
        int,
        Field(
            ge=0,
            description="Default retry budget per run, overridable via "
            "StrategySpec(max_retries=...) or StrategyOverride.max_retries. "
            "0 disables retries.",
        ),
    ] = 1
    retry_backoff_base_seconds: Annotated[
        float,
        Field(
            ge=0.0,
            description=(
                "Base seconds for exponential retry backoff (sleep between "
                "attempts). attempt N waits base * 2**(N-1), capped at "
                "retry_backoff_cap_seconds, plus up to retry_jitter_seconds "
                "of random jitter. 0.0 disables backoff."
            ),
        ),
    ] = 1.0
    retry_backoff_cap_seconds: Annotated[
        float,
        Field(
            ge=0.0,
            description="Upper bound on the exponential backoff sleep before jitter.",
        ),
    ] = 10.0
    retry_jitter_seconds: Annotated[
        float,
        Field(
            ge=0.0,
            description=(
                "Uniform [0, retry_jitter_seconds] noise added to each "
                "backoff sleep to spread retry storms."
            ),
        ),
    ] = 0.5
    max_records_per_strategy: Annotated[
        int,
        Field(
            gt=0,
            description="Per-strategy RunRecord ring-buffer size; oldest "
            "entries are pruned on insert.",
        ),
    ] = 1000
    crash_recovery_enabled: bool = Field(
        default=True,
        description=(
            "Run ``OfflineEngine._run_crash_recovery`` on ``engine.start()``. "
            "Default ``True`` — the primary server engine "
            "(built by ``service.bootstrap``) needs to resume its own "
            "prior sessions' RUNNING work after a crash / restart. "
            "Set to ``False`` for one-shot engines that share the jobstore "
            "path (e.g. backfill's replay engines, built by "
            "``everos.service.bootstrap.build_replay_engine``). Those "
            "engines register only the "
            "Phase 2/3 subset of strategies; if they ran crash recovery on "
            "startup they would re-enqueue the server's stale RUNNING rows "
            "into their own APS scheduler, whose registry doesn't know "
            "those strategy names — causing the re-enqueued event to be "
            "permanently lost via ``KeyError`` at dispatch time. The "
            "server's stale rows stay in ``run_record`` untouched; the "
            "next server restart resumes them normally."
        ),
    )
    config_path: Path | None = Field(
        default=None,
        description="Path to ome.toml for per-strategy overrides. None "
        "disables TOML-driven hot reload.",
    )
    config_watch: bool = Field(
        default=True,
        description="When true and config_path is set, watch the file for "
        "edits and apply overrides at runtime.",
    )
    config_watch_debounce_ms: Annotated[
        int,
        Field(
            gt=0,
            description="Debounce window collapsing bursts of filesystem "
            "events (e.g. editor saves) into one reload.",
        ),
    ] = 1600

    @model_validator(mode="after")
    def _derive_aps_jobstore_path(self) -> Self:
        # When unset, materialize as a sibling of jobstore_path so callers
        # that pass only jobstore_path (e.g. tests using tmp_path) still get
        # an isolated APS db rather than the global default root.
        if self.aps_jobstore_path is None:
            self.aps_jobstore_path = self.jobstore_path.with_name(
                self.jobstore_path.stem + ".aps.db"
            )
        return self
