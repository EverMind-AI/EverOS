"""The worker's in-memory health snapshot and the verdict it implies.

Pure value object: no IO, no clock. :class:`CascadeWorker` fills it in;
:meth:`CascadeWorkerHealth.reasons` turns it into operator-facing reasons
against the fixed alert thresholds in :mod:`._tuning`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._tuning import (
    DRAIN_FAILURE_ALERT_THRESHOLD,
    OPTIMIZE_FAILURE_ALERT_THRESHOLD,
    PRUNE_STALE_SECONDS_ALERT,
)


@dataclass(frozen=True)
class CascadeWorkerHealth:
    """Snapshot of the worker's in-memory health signals.

    Cheap to produce (no IO — pure in-process counters). The orchestrator
    combines it with the SQLite queue summary for the full verdict.
    """

    drain_consecutive_failures: int
    """Consecutive :meth:`CascadeWorker.drain_once` exceptions; 0 when the last drain
    completed cleanly."""

    unrecoverable_total: int
    """Cumulative unrecoverable handler failures since worker start —
    each is a md file whose projection to LanceDB permanently failed
    and now needs a user edit (surfaced by ``cascade fix``)."""

    optimize_failure_streak: int
    """Max consecutive non-benign optimize/prune failure count across
    kinds; benign light-beat commit conflicts do not count."""

    prune_stale_seconds: float
    """Staleness of the **worst** kind — seconds since that kind's last
    successful prune (version cleanup), measured from worker start if it
    has never pruned. ``0`` when no kind has an optimizer state yet (no
    write activity to prune).

    Deliberately the worst kind, not the newest prune across kinds: a
    per-kind cleanup that dies (hung lance cleanup, lost commit races)
    grows *that table's* index dir unbounded, and every other kind
    pruning normally must not mask it."""

    prune_stale_kind: str | None = None
    """The kind :attr:`prune_stale_seconds` belongs to; ``None`` when no
    kind has state yet. Named in :meth:`reasons` so an operator knows
    which table to look at."""

    def reasons(self) -> list[str]:
        """Operational degradation reasons; empty when healthy.

        Combined with the SQLite ``failed_permanent`` count in
        :meth:`CascadeOrchestrator.health` to produce the ``/health``
        verdict. These in-memory signals do not include the permanent-
        failure backlog (that needs a query)."""
        out: list[str] = []
        if self.drain_consecutive_failures >= DRAIN_FAILURE_ALERT_THRESHOLD:
            out.append(
                f"drain loop failing ({self.drain_consecutive_failures} in a row)"
            )
        if self.optimize_failure_streak >= OPTIMIZE_FAILURE_ALERT_THRESHOLD:
            out.append(f"optimize stuck ({self.optimize_failure_streak} in a row)")
        if self.prune_stale_seconds >= PRUNE_STALE_SECONDS_ALERT:
            kind = self.prune_stale_kind or "unknown"
            out.append(
                f"version cleanup stalled for kind '{kind}' "
                f"({int(self.prune_stale_seconds)}s since its last prune "
                "— that table's index dir may grow)"
            )
        return out
