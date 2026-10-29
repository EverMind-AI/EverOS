"""The backfill run: the selected phases, in order, until one stops it."""

from __future__ import annotations

__all__ = ["run_backfill"]

from collections.abc import Awaitable, Callable

from everos.core.observability.logging import get_logger

from ._backfill import (
    BACKFILL_LOGGER,
    BACKFILL_PHASES,
    BackfillDeps,
    BackfillOutcome,
    BackfillPresenter,
    BackfillSummary,
    PhaseResult,
)
from ._backfill_clusters import run_clusters_phase
from ._backfill_skills import run_skills_phase
from ._backfill_vectors import run_vectors_phase

backfill_logger = get_logger(BACKFILL_LOGGER)

_PHASE_RUNNERS: dict[str, Callable[..., Awaitable[PhaseResult]]] = {
    "vectors": run_vectors_phase,
    "clusters": run_clusters_phase,
    "skills": run_skills_phase,
}


async def run_backfill(
    deps: BackfillDeps,
    phase: str,
    *,
    auto_yes: bool,
    presenter: BackfillPresenter,
    summary: BackfillSummary,
) -> BackfillOutcome:
    """Run ``phase`` (or ``"all"``: every phase, in order) and say how it ended.

    A phase blocked by a missing provider or a running server, or one
    the user declines, stops the run: declining the first phase's
    estimate is a "no" to the phases whose estimates were never shown.
    Each phase's result lands in ``summary`` as soon as it finishes, so a
    caller interrupted mid-run still has every phase that completed.

    Raises:
        ValueError: ``phase`` is neither ``"all"`` nor a phase name.
    """
    if phase != "all" and phase not in BACKFILL_PHASES:
        raise ValueError(f"unknown backfill phase: {phase!r}")
    selected = BACKFILL_PHASES if phase == "all" else (phase,)
    for name in selected:
        presenter.phase_started(name)
        result = await _PHASE_RUNNERS[name](
            deps, auto_yes=auto_yes, presenter=presenter
        )
        summary.record(name, result)
        stop = _stopping_outcome(result)
        if stop is not None:
            _log_stop(name, stop, result)
            return stop
    if summary.rows_failed:
        backfill_logger.warning(
            "cascade_backfill_completed_with_failures",
            total_failed=summary.rows_failed,
        )
        return BackfillOutcome.COMPLETED_WITH_FAILURES
    return BackfillOutcome.COMPLETED


def _stopping_outcome(result: PhaseResult) -> BackfillOutcome | None:
    """How a phase result ends the run, or ``None`` when the next phase runs.

    A missing provider wins over a running server: the first is a permanent
    configuration gap, the second goes away when the server stops.
    """
    if result.blocked_by_capability:
        return BackfillOutcome.BLOCKED_BY_CAPABILITY
    if result.blocked_by_server:
        return BackfillOutcome.BLOCKED_BY_SERVER
    if result.aborted:
        return BackfillOutcome.ABORTED
    return None


def _log_stop(phase: str, outcome: BackfillOutcome, result: PhaseResult) -> None:
    if outcome is BackfillOutcome.BLOCKED_BY_CAPABILITY:
        backfill_logger.warning(
            "cascade_backfill_blocked_by_capability",
            phase=phase,
            provider=result.blocked_by_capability,
        )
    elif outcome is BackfillOutcome.BLOCKED_BY_SERVER:
        backfill_logger.warning("cascade_backfill_blocked_by_server", phase=phase)
    else:
        backfill_logger.info("cascade_backfill_aborted", phase=phase)
