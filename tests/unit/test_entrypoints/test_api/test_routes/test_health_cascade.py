"""Pins the cascade section of ``GET /health`` (#364).

Contract under test: an HTTP host must be able to detect that memory
writes are landing in markdown but no longer reaching the index. The
write path is deliberately decoupled from indexing, so ``/memory/flush``
keeps answering ``extracted`` while the cascade worker is stuck — the
counters here are the only signal an API-only host gets.

Two facts are pinned:

* terminal failures surface as ``failed_permanent`` and the LSN gap
  surfaces as ``lag``, so a host can alert on either;
* an unreadable queue yields ``cascade: null`` rather than a 500 or
  all-zero counters — ``/health`` stays a liveness probe, and "unknown"
  must not be mistaken for "clean".
"""

from __future__ import annotations

import pytest

from everos.entrypoints.api.routes import health as health_route
from everos.infra.persistence.sqlite.repos.md_change_state import QueueSummary


async def test_cascade_counters_surface_terminal_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stuck queue is visible: failed_permanent and lag are reported."""

    async def fake_summary() -> QueueSummary:
        return QueueSummary(
            pending=3,
            done=41,
            failed_retryable=2,
            failed_permanent=7,
            max_lsn=50,
            last_processed_lsn=44,
        )

    monkeypatch.setattr(
        health_route.md_change_state_repo, "queue_summary", fake_summary
    )

    cascade = await health_route._cascade_health()

    assert cascade is not None
    assert cascade.failed_permanent == 7
    assert cascade.failed_retryable == 2
    assert cascade.pending == 3
    assert cascade.done == 41
    assert cascade.lag == 6


async def test_lag_never_goes_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    """A processed LSN ahead of max reports zero lag, not a negative number."""

    async def fake_summary() -> QueueSummary:
        return QueueSummary(
            pending=0,
            done=10,
            failed_retryable=0,
            failed_permanent=0,
            max_lsn=5,
            last_processed_lsn=9,
        )

    monkeypatch.setattr(
        health_route.md_change_state_repo, "queue_summary", fake_summary
    )

    cascade = await health_route._cascade_health()

    assert cascade is not None
    assert cascade.lag == 0


async def test_unreadable_queue_degrades_to_null(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When queue state cannot be read, the probe degrades instead of failing.

    ``None`` rather than zeros: a host must not read "unknown" as "clean",
    which is the exact confusion that let the original outage hide.
    """

    async def boom() -> QueueSummary:
        raise RuntimeError("lance error: LanceError(IO)")

    monkeypatch.setattr(health_route.md_change_state_repo, "queue_summary", boom)

    assert await health_route._cascade_health() is None
