"""What the operator use cases return: plain data, no behaviour."""

from __future__ import annotations

__all__ = [
    "Capabilities",
    "CascadeHealthReport",
    "CascadeStatus",
    "HealthReport",
    "OmeRun",
    "OmeTriggerResult",
    "ProviderStatus",
    "QuiesceResult",
    "RebuildResult",
]

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everos.infra.persistence.sqlite import QueueSummary
    from everos.service.memory import QueueHealth


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """Which optional providers the application was assembled with.

    A snapshot taken by :func:`everos.service.build_application`; a provider
    configured later needs a restart, as before.
    """

    llm: bool
    decider: bool
    embedding: bool
    rerank: bool
    multimodal_llm: bool
    parser: bool
    index_backend: str


@dataclasses.dataclass(frozen=True)
class ProviderStatus:
    """The five provider flags ``GET /health`` reports.

    ``embed`` is the startup dimension probe's verdict, not whether a
    provider was built; ``llm`` is always true, because the server refuses
    to start without an extraction LLM.
    """

    llm: bool
    embed: bool
    rerank: bool
    multimodal_llm: bool
    parser: bool


@dataclasses.dataclass(frozen=True)
class CascadeHealthReport:
    """Readiness of the md -> index projection, as ``GET /health`` shows it."""

    healthy: bool
    reasons: list[str]
    pending: int
    failed_permanent: int
    failed_retryable: int
    drain_consecutive_failures: int
    unrecoverable_total: int
    optimize_failure_streak: int
    prune_stale_seconds: float
    index_rows: dict[str, int] = dataclasses.field(default_factory=dict)
    index_emptied: bool = False


@dataclasses.dataclass(frozen=True)
class HealthReport:
    """Everything ``GET /health`` reports besides the version.

    ``cascade`` / ``memorize_queue`` are ``None`` while the subsystem is not
    running; a probe that failed reads as an unhealthy block rather than an
    error, so the liveness endpoint never turns into a 500.
    """

    providers: ProviderStatus
    disabled_features: list[str]
    cascade: CascadeHealthReport | None
    memorize_queue: QueueHealth | None


@dataclasses.dataclass(frozen=True)
class CascadeStatus:
    """``everos cascade status``: the ledger, the index, and the live sources."""

    summary: QueueSummary
    index_rows: dict[str, int]
    live_sources: int


@dataclasses.dataclass(frozen=True)
class QuiesceResult:
    """What the final drain before stopping the cascade accomplished."""

    drained: int
    pending_before: int
    pending_after: int
    failed_permanent: int


@dataclasses.dataclass(frozen=True)
class RebuildResult:
    """What ``everos cascade rebuild`` reset, dropped and re-indexed."""

    ledger_rows_reset: int
    dropped_tables: list[str]
    processed: int


@dataclasses.dataclass(frozen=True)
class OmeRun:
    """One strategy run attempt within a manual trigger."""

    run_id: str
    status: str
    error: str | None


@dataclasses.dataclass(frozen=True)
class OmeTriggerResult:
    """Outcome of a manual OME trigger: ``ok`` / ``timeout`` / ``not_dispatched``."""

    status: str
    dispatched: int
    runs: list[OmeRun]
