"""Unit tests for memory_root_lock async context manager."""

from __future__ import annotations

import multiprocessing
import time
from pathlib import Path

import anyio
import pytest

from everos.core.persistence import LockError, MemoryRoot, memory_root_lock


async def test_lock_creates_anchor_file(tmp_path: Path) -> None:
    mr = MemoryRoot(tmp_path)
    async with memory_root_lock(mr):
        assert mr.lock_file.exists()


async def test_lock_acquire_release_acquire(tmp_path: Path) -> None:
    """Same process can re-acquire after release (no leftover state)."""
    mr = MemoryRoot(tmp_path)
    async with memory_root_lock(mr):
        pass
    async with memory_root_lock(mr):
        pass


def _hold_lock(memory_root_path: str, ready: object, release: object) -> None:
    """Subprocess helper: acquire blocking lock, signal, wait, release.

    The subprocess runs its own event loop via :func:`anyio.run` since
    :func:`memory_root_lock` is now async.
    """

    async def _run() -> None:
        mr = MemoryRoot(memory_root_path)
        async with memory_root_lock(mr, blocking=True):
            ready.set()
            # Use a thread-offloaded wait so we don't block the event loop.
            await anyio.to_thread.run_sync(release.wait, 5)

    anyio.run(_run)


async def test_nonblocking_raises_when_held_by_other_process(tmp_path: Path) -> None:
    """Different process holding the lock → blocking=False raises LockError."""
    mr = MemoryRoot(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(mr.root), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=5), "subprocess failed to acquire lock"
        with pytest.raises(LockError):
            async with memory_root_lock(mr, blocking=False):
                pass
    finally:
        release.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()


async def test_blocking_waits_for_release(tmp_path: Path) -> None:
    """Different process holding lock + main process blocking=True waits."""
    mr = MemoryRoot(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(mr.root), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=5)
        # Schedule the subprocess to release shortly; main process should
        # acquire the lock after that.
        release_started = time.monotonic()

        def release_after_short_delay() -> None:
            time.sleep(0.2)
            release.set()

        import threading

        threading.Thread(target=release_after_short_delay, daemon=True).start()
        async with memory_root_lock(mr, blocking=True):
            elapsed = time.monotonic() - release_started
            # Should have waited at least roughly the delay.
            assert elapsed >= 0.1
    finally:
        release.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()


async def test_blocking_wait_is_bounded_and_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A held lock must produce a bounded, *visible* wait — not a silent hang.

    The wait itself is correct by design (the second process is supposed to
    wait, then find the migration already done). What is not correct is doing it
    with no upper bound and no log line: an ``everos server start`` that lands
    on a held lock looks like a hang whose last message is
    ``lifespan_provider_startup name=lancedb``.
    """
    from everos.core.persistence import locking

    events: list[str] = []

    class _SpyLogger:
        def __getattr__(self, _level: str):  # type: ignore[no-untyped-def]
            def rec(event: str, **_kw) -> None:  # type: ignore[no-untyped-def]
                events.append(event)

            return rec

    monkeypatch.setattr(locking, "logger", _SpyLogger())
    monkeypatch.setattr(locking, "_LOCK_POLL_INTERVAL_SECONDS", 0.01)

    mr = MemoryRoot(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(mr.root), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=5)
        started = time.monotonic()
        with pytest.raises(LockError, match="timed out"):
            async with memory_root_lock(mr, timeout_seconds=0.2):
                pass
        elapsed = time.monotonic() - started
        assert 0.2 <= elapsed < 5.0, f"must give up near the budget, took {elapsed}s"
        assert "memory_root_lock_waiting" in events, (
            "waiting on another process must be announced, or the startup "
            "stall has no explanation in the log"
        )
    finally:
        release.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()

    # The timed-out attempt must not have left a lock behind. Acquisition polls
    # with LOCK_NB precisely so a give-up cannot leave a worker thread blocked
    # in ``flock`` that later acquires the lock with nobody left to release it.
    with anyio.fail_after(2):
        async with memory_root_lock(mr, timeout_seconds=1.0):
            pass


async def test_successful_wait_logs_how_long_it_waited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the wait does succeed, say so — silence is what made it opaque."""
    import threading

    from everos.core.persistence import locking

    events: list[str] = []

    class _SpyLogger:
        def __getattr__(self, _level: str):  # type: ignore[no-untyped-def]
            def rec(event: str, **_kw) -> None:  # type: ignore[no-untyped-def]
                events.append(event)

            return rec

    monkeypatch.setattr(locking, "logger", _SpyLogger())
    monkeypatch.setattr(locking, "_LOCK_POLL_INTERVAL_SECONDS", 0.01)

    mr = MemoryRoot(tmp_path)
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_lock, args=(str(mr.root), ready, release))
    proc.start()
    try:
        assert ready.wait(timeout=5)
        threading.Timer(0.2, release.set).start()
        async with memory_root_lock(mr, timeout_seconds=5.0):
            pass
        assert events == [
            "memory_root_lock_waiting",
            "memory_root_lock_acquired_after_wait",
        ]
    finally:
        release.set()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()


async def test_uncontended_acquisition_stays_silent(tmp_path: Path) -> None:
    """No contention → no log noise. The wait lines must be signal, not chatter
    on the hot startup path."""
    from everos.core.persistence import locking

    events: list[str] = []

    class _SpyLogger:
        def __getattr__(self, _level: str):  # type: ignore[no-untyped-def]
            def rec(event: str, **_kw) -> None:  # type: ignore[no-untyped-def]
                events.append(event)

            return rec

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(locking, "logger", _SpyLogger())
    try:
        async with memory_root_lock(MemoryRoot(tmp_path)):
            pass
    finally:
        monkeypatched.undo()

    assert events == []
