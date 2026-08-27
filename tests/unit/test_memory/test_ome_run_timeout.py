"""A strategy that hangs must lose its slot, not the whole engine.

Regression cover for a full-process stall. ``OMEConfig`` bounded retries and
recovered orphans left by a *previous* process, but nothing bounded a live
attempt. A coroutine parked on an await with no deadline of its own -- an
``asyncio.Lock`` held by another stuck coroutine, a connection-pool wait -- kept
its ``max_concurrent_runs`` slot indefinitely: it never raised, so it never
retried, and its record stayed RUNNING forever.

Observed in a benchmark run: 60 of 64 slots parked on one lock, which starved
every other strategy in the process. Extraction stopped for 6.7 hours while the
server still answered HTTP and every liveness signal read healthy -- the queue
depth was the only thing that moved, and it moved the wrong way.
"""

from __future__ import annotations

import asyncio

import pytest

from everos.infra.ome.config import OMEConfig, _env_float


def _cfg(**kw: object) -> OMEConfig:
    return OMEConfig(jobstore_path="/tmp/x.db", max_concurrent_runs=4, **kw)  # type: ignore[arg-type]


def test_a_ceiling_is_on_by_default() -> None:
    """Opt-out, not opt-in: the failure it prevents is silent and total."""
    assert _cfg().run_timeout_seconds == 1800.0


def test_the_ceiling_leaves_room_for_slow_but_healthy_work() -> None:
    """Measured worst case is a ~7-minute 38-subject profile pass.

    Pinned as an inequality rather than a constant: the point is the margin, so
    tuning the default stays free while shrinking it below the known-good
    workload does not.
    """
    assert (_cfg().run_timeout_seconds or 0) >= 4 * 7 * 60


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("300", 300.0),
        ("0", None),
        ("off", None),
        ("none", None),
        ("false", None),
        ("", 1800.0),
        ("garbage", 1800.0),
        ("-5", None),
    ],
)
def test_env_parsing(
    raw: str, expected: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operators need an off switch, and a typo must not take the process down."""
    monkeypatch.setenv("EVEROS_OME_RUN_TIMEOUT_SECONDS", raw)
    assert _env_float("EVEROS_OME_RUN_TIMEOUT_SECONDS", 1800.0) == expected


def test_disabled_ceiling_is_representable() -> None:
    assert _cfg(run_timeout_seconds=None).run_timeout_seconds is None


async def test_a_hung_attempt_raises_timeout_error_and_frees_its_slot() -> None:
    """The mechanism, end to end: hang -> cancel -> Exception -> slot released.

    ``asyncio.timeout`` cancels the body, and it is ``TimeoutError`` (an
    ``Exception``, unlike ``CancelledError``) that escapes -- which is what lets
    the runner's existing ``except Exception`` path retry or dead-letter the
    attempt instead of leaking the slot. Written against the primitives so it
    pins the property the runner depends on.
    """
    sem = asyncio.Semaphore(1)
    never = asyncio.Event()  # stands in for a lock nobody will release

    async def hangs() -> None:
        await never.wait()

    with pytest.raises(TimeoutError):
        async with sem:
            async with asyncio.timeout(0.05):
                await hangs()

    # The slot is back: without the ceiling this acquire would block forever.
    async with asyncio.timeout(1):
        async with sem:
            pass


async def test_cancelled_error_alone_would_not_have_been_caught() -> None:
    """Why the fix is a timeout and not a bare cancel.

    ``CancelledError`` derives from ``BaseException``, so cancelling the attempt
    directly would slip past ``except Exception`` and abandon the record in
    RUNNING -- the same leak, with extra steps.
    """
    assert not issubclass(asyncio.CancelledError, Exception)
    assert issubclass(TimeoutError, Exception)
