"""Unit tests for memory_root_lock async context manager."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import anyio
import pytest

from everos.core.persistence import LockError, MemoryRoot, memory_root_lock

_LOCK_HOLDER_SCRIPT = """
import fcntl
import os
import sys
import time

from pathlib import Path

lock_path, ready_path, release_path = sys.argv[1:]
Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
try:
    fcntl.flock(fd, fcntl.LOCK_EX)
    Path(ready_path).write_text("ready")
    while not Path(release_path).exists():
        time.sleep(0.05)
finally:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
"""
LOCK_HOLDER_READY_TIMEOUT = 5.0


async def _assert_subprocess_ready(
    ready_path: Path,
    proc: subprocess.Popen[str],
) -> None:
    deadline = time.monotonic() + LOCK_HOLDER_READY_TIMEOUT
    while time.monotonic() < deadline:
        if await anyio.to_thread.run_sync(ready_path.exists):
            return
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise AssertionError(
                "subprocess exited before acquiring lock "
                f"(exitcode={proc.returncode}, stdout={stdout!r}, stderr={stderr!r})"
            )
        await anyio.sleep(0.05)

    proc.terminate()
    stdout, stderr = proc.communicate(timeout=1)
    raise AssertionError(
        "subprocess failed to acquire lock "
        f"(exitcode={proc.returncode}, stdout={stdout!r}, stderr={stderr!r})"
    )


def _spawn_lock_holder(mr: MemoryRoot) -> tuple[subprocess.Popen[str], Path, Path]:
    ready_path = mr.root / ".test-lock-ready"
    release_path = mr.root / ".test-lock-release"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _LOCK_HOLDER_SCRIPT,
            str(mr.lock_file),
            str(ready_path),
            str(release_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc, ready_path, release_path


async def _start_lock_holder(mr: MemoryRoot) -> tuple[subprocess.Popen[str], Path]:
    proc, ready_path, release_path = _spawn_lock_holder(mr)
    await _assert_subprocess_ready(ready_path, proc)
    return proc, release_path


def _stop_lock_holder(proc: subprocess.Popen[str], release_path: Path) -> None:
    release_path.write_text("release")
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        proc.wait(timeout=5)


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


async def test_nonblocking_raises_when_held_by_other_process(tmp_path: Path) -> None:
    """Different process holding the lock → blocking=False raises LockError."""
    mr = MemoryRoot(tmp_path)
    proc, release_path = await _start_lock_holder(mr)
    try:
        with pytest.raises(LockError):
            async with memory_root_lock(mr, blocking=False):
                pass
    finally:
        _stop_lock_holder(proc, release_path)


async def test_blocking_waits_for_release(tmp_path: Path) -> None:
    """Different process holding lock + main process blocking=True waits."""
    mr = MemoryRoot(tmp_path)
    proc, release_path = await _start_lock_holder(mr)
    try:
        # Schedule the subprocess to release shortly; main process should
        # acquire the lock after that.
        release_started = time.monotonic()

        def release_after_short_delay() -> None:
            time.sleep(0.2)
            release_path.write_text("release")

        import threading

        threading.Thread(target=release_after_short_delay, daemon=True).start()
        async with memory_root_lock(mr, blocking=True):
            elapsed = time.monotonic() - release_started
            # Should have waited at least roughly the delay.
            assert elapsed >= 0.1
    finally:
        _stop_lock_holder(proc, release_path)
