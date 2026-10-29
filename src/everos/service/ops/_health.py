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
    "cascade_probe_failed",
    "cascade_report",
    "compute_disabled_features",
    "queue_probe_failed",
]

from collections.abc import Mapping
from typing import TYPE_CHECKING

from everos.service.memorize_queue import QueueHealth
from everos.service.ops._models import CascadeHealthReport

if TYPE_CHECKING:
    from everos.infra.cascade import CascadeHealth


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
