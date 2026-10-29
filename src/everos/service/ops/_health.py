"""Health signals: which features a tier disables, and the probe blocks.

Two feature-name vocabularies exist and deliberately do not share a source
of truth:

- ``ProviderNotConfiguredError.feature`` (raised per request by search and
  the knowledge endpoints): a granular tag for the one search mode or
  endpoint that failed -- ``"vector"`` | ``"user_hybrid"`` |
  ``"agent_hybrid"`` | ``"agentic_search"`` | ``"knowledge"`` |
  ``"skill_extraction_backfill"``. It appears in the HTTP 422 ``message``.
- :func:`compute_disabled_features`: a capability-level tag for a whole
  feature category the current tier disables -- ``"vector_search"`` |
  ``"hybrid_search"`` | ``"agentic_search"`` | ``"reflection"`` |
  ``"skill_extraction"`` | ``"knowledge"`` | ``"multimodal_upload"``. It
  appears in ``GET /health``'s ``disabled_features``.

Both are stable client contracts; unifying them needs a coordinated client
migration.
"""

from __future__ import annotations

__all__ = [
    "QueueHealthProbe",
    "cascade_probe_failed",
    "cascade_report",
    "compute_disabled_features",
    "health_report",
    "no_queue",
    "queue_probe_failed",
]

import dataclasses
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger
from everos.service.memory import QueueHealth
from everos.service.ops._models import (
    Capabilities,
    CascadeHealthReport,
    HealthReport,
    ProviderStatus,
)

if TYPE_CHECKING:
    from everos.infra.cascade import CascadeAdmin, CascadeHealth, CascadeOrchestrator

logger = get_logger("everos.service.ops")

QueueHealthProbe = Callable[[], Awaitable[QueueHealth | None]]


async def no_queue() -> QueueHealth | None:
    """The queue probe while no memorize queue is wired in."""
    return None


async def health_report(
    *,
    admin: CascadeAdmin,
    capabilities: Capabilities,
    cascade: CascadeOrchestrator | None,
    queue_health: QueueHealthProbe,
) -> HealthReport:
    """Provider flags, disabled features, and the cascade / queue blocks.

    ``cascade`` is ``None`` while the server is not running it. A probe that
    raises (a locked or full SQLite, say) is reported as an unhealthy block
    with the error as its reason instead of propagating: the health endpoint
    is a liveness signal, and turning it into a 500 would get the process
    restarted, which fixes neither.
    """
    providers = ProviderStatus(
        llm=True,
        embed=admin.embedding_probed(),
        rerank=capabilities.rerank,
        multimodal_llm=capabilities.multimodal_llm,
        parser=capabilities.parser,
    )
    return HealthReport(
        providers=providers,
        # ``embed`` reports the startup probe, but ``disabled_features`` is
        # a contract about what the server will refuse, and every gate
        # (search, the embedding strategies, backfill) asks whether a
        # provider was built.
        disabled_features=compute_disabled_features(
            {**dataclasses.asdict(providers), "embed": capabilities.embedding}
        ),
        cascade=await _probe_cascade(cascade),
        memorize_queue=await _probe_queue(queue_health),
    )


async def _probe_cascade(
    cascade: CascadeOrchestrator | None,
) -> CascadeHealthReport | None:
    if cascade is None:
        return None
    try:
        return cascade_report(await cascade.health())
    except Exception as exc:
        logger.warning("cascade_health_probe_failed", error=repr(exc))
        return cascade_probe_failed(exc)


async def _probe_queue(queue_health: QueueHealthProbe) -> QueueHealth | None:
    try:
        return await queue_health()
    except Exception as exc:
        logger.warning("memorize_queue_health_probe_failed", error=repr(exc))
        return queue_probe_failed(exc)


def compute_disabled_features(caps: Mapping[str, bool]) -> list[str]:
    """Derive the features the configured providers leave disabled.

    Args:
        caps: Availability of ``"llm"``, ``"embed"``, ``"rerank"``,
            ``"multimodal_llm"`` and ``"parser"``. ``caps["llm"]`` is
            accepted for shape symmetry but not read: the server refuses to
            start without an extraction LLM, so any process asking has one.

    Returns:
        A subset of ``"vector_search"``, ``"hybrid_search"``,
        ``"agentic_search"``, ``"reflection"``, ``"skill_extraction"``,
        ``"knowledge"`` and ``"multimodal_upload"``.
    """
    disabled: list[str] = []
    if not caps["embed"]:
        disabled.extend(
            ["vector_search", "hybrid_search", "reflection", "skill_extraction"]
        )
    if not caps["rerank"]:
        disabled.append("agentic_search")
    if not (caps["embed"] and caps["rerank"]):
        disabled.append("knowledge")
    if not (caps["multimodal_llm"] and caps["parser"]):
        disabled.append("multimodal_upload")
    return disabled


def cascade_report(health: CascadeHealth) -> CascadeHealthReport:
    """The cascade verdict as ``GET /health`` reports it."""
    return CascadeHealthReport(
        healthy=health.healthy,
        reasons=health.reasons,
        pending=health.pending,
        failed_permanent=health.failed_permanent,
        failed_retryable=health.failed_retryable,
        drain_consecutive_failures=health.drain_consecutive_failures,
        unrecoverable_total=health.unrecoverable_total,
        optimize_failure_streak=health.optimize_failure_streak,
        prune_stale_seconds=round(health.prune_stale_seconds, 1),
        index_rows=health.index_rows,
        index_emptied=health.index_emptied,
    )


def cascade_probe_failed(exc: Exception) -> CascadeHealthReport:
    """An unhealthy cascade block naming the probe's failure."""
    return CascadeHealthReport(
        healthy=False,
        reasons=[f"cascade health probe failed: {exc!r}"],
        pending=0,
        failed_permanent=0,
        failed_retryable=0,
        drain_consecutive_failures=0,
        unrecoverable_total=0,
        optimize_failure_streak=0,
        prune_stale_seconds=0.0,
    )


def queue_probe_failed(exc: Exception) -> QueueHealth:
    """An unhealthy memorize-queue block naming the probe's failure."""
    return QueueHealth(
        healthy=False,
        reasons=[f"memorize queue health probe failed: {exc!r}"],
        loop_alive=False,
        accepting=False,
        pending=0,
        oldest_pending_age_s=None,
        failed=0,
        loop_restarts=0,
        last_loop_error_at=None,
    )
