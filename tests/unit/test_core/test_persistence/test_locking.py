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
