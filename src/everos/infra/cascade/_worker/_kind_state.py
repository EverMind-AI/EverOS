"""Per-kind bookkeeping that the index maintainer keeps for one LanceDB table.

One :class:`KindOptimizerState` per kind: the throttle clock, the two prune
clocks, the failure counters behind the health verdict, the rebuild retry
deadline and the two in-flight task slots that keep optimize and rebuild from
committing on the same manifest at once.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class KindOptimizerState:
    """Per-kind throttle state for LanceDB ``optimize()``.

    ``dirty`` is the trailing-edge signal: every write sets it, the
    runner consumes it before each ``optimize()`` call. If a write
    arrives mid-optimize, ``dirty`` is re-raised and the runner loops
    once more after honouring the throttle interval.

    ``task`` holds the in-flight runner; at most one runner exists
    per kind so concurrent LanceDB writes never collide on the same
    table's manifest.

    Two prune clocks, deliberately split:

    - ``last_prune_attempt_at`` gates **scheduling** — the monotonic time
      of the last heavy (prune) beat *attempt*, advanced whether it
      succeeds or times out. So a prune that hangs and is killed by the
      write-lock timeout backs off a full cadence before the next attempt
      instead of retrying every ~10s and holding the lock ~97% of the
      time (review N1).
    - ``last_prune_at`` records the last *successful* prune, advanced only
      after the call returns. It drives the prune-staleness **health**
      signal (:meth:`IndexMaintainer.prune_staleness`), so a
      persistently failing/hanging prune still surfaces as degraded
      instead of being masked by advancing the schedule clock.

    Both default ``0`` ("never") — the first run after worker startup
    always prunes, catching up from a prior session.
    """

    last_run_at: float = 0.0
    last_prune_attempt_at: float = 0.0
    last_prune_at: float = 0.0
    dirty: bool = False
    optimize_failures: int = 0
    """Consecutive ``optimize()`` failures **since the last success**.

    Drives the health verdict and escalation to ``error`` at
    :data:`OPTIMIZE_FAILURE_ALERT_THRESHOLD` so a stuck optimize (which stalls
    version cleanup and grows the index dir) is not swallowed as a silent
    warning stream.

    Reset **only** by a successful optimize — never by the fallback rebuild it
    triggers. That distinction is the whole point of splitting this from
    ``failures_since_fallback``: zeroing the alert counter inside the branch
    that fires at the threshold made ``optimize_failure_streak >= threshold``
    effectively unobservable. A table failing 100% of the time cycled
    1..threshold -> 0 -> 1.., and the only window where a poller could read the
    threshold value was the sub-second fallback rebuild itself. Same shape as
    the cross-kind ``max()`` masking bug from run7: a remediation path
    refreshing the very signal that is supposed to report it.
    """
    failures_since_fallback: int = 0
    """Failures since the last fallback rebuild — the rate limiter.

    Reset by the fallback rebuild so it fires at most once per
    :data:`OPTIMIZE_FAILURE_ALERT_THRESHOLD` failures instead of on every
    failure. This is the job the reset was originally there for; it just used
    to share a field with the alert signal.
    """
    rebuild_retry_at: float = 0.0
    """Monotonic deadline for re-attempting a rebuild that lost a commit race.
    ``0`` means nothing pending. Set instead of sleeping so the rebuild loop
    stays free to serve the other kinds — see
    :data:`REBUILD_CONFLICT_BACKOFFS_SECONDS`."""
    rebuild_attempt: int = 0
    """Consecutive lost commit races for this kind; indexes into the backoff
    schedule and resets on any completed rebuild."""
    task: asyncio.Task[None] | None = None
    rebuild_task: asyncio.Task[None] | None = None
    """In-flight rebuild task slot, separate from ``task`` so ordinary
    ``_schedule_optimize`` calls during a rebuild can still register
    ``dirty`` + spawn an optimize runner. The runner itself waits for
    ``rebuild_task`` before touching the LanceDB manifest, so the two
    operations never race on commits — only the dispatch slot is split.
    """
