"""Process-wide exclusive lock on a memory-root.

Uses ``fcntl.flock`` (POSIX advisory locking, available on Linux + macOS;
Windows is not supported — see project README on platform scope). The
public surface is an :func:`contextlib.asynccontextmanager` so callers
use ``async with memory_root_lock(mr):``; the underlying syscalls have
no async equivalent so they run in a worker thread via
:func:`anyio.to_thread.run_sync`.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio

from .memory_root import MemoryRoot


class LockError(RuntimeError):
    """Raised when the memory-root lock cannot be acquired in non-blocking mode."""


@asynccontextmanager
async def memory_root_lock(
    memory_root: MemoryRoot,
    *,
    blocking: bool = True,
) -> AsyncIterator[None]:
    """Acquire an exclusive process lock on the memory-root.

    Args:
        memory_root: The memory-root to lock. The lock anchor file
            (``<root>/.lock``) is created on first use.
        blocking: If ``True`` (default), wait until the lock is free. If
            ``False``, raise :class:`LockError` immediately when another
            process holds it.

    Raises:
        LockError: When ``blocking=False`` and the lock is already held.
    """
    await anyio.Path(memory_root.root).mkdir(parents=True, exist_ok=True)
    lock_path = memory_root.lock_file

    # Open the anchor file (create on first use). The fd, not the path, is
    # what fcntl operates on. ``os.open`` is microsecond-fast but offloaded
    # for consistency with the rest of the lock acquisition flow.
    fd = await anyio.to_thread.run_sync(
        lambda: os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    )

    flags = fcntl.LOCK_EX
    if not blocking:
        flags |= fcntl.LOCK_NB

    try:
        await anyio.to_thread.run_sync(fcntl.flock, fd, flags)
    except BlockingIOError as exc:
        await anyio.to_thread.run_sync(os.close, fd)
        raise LockError(
            f"another process already holds the memory-root lock at {lock_path}"
        ) from exc

    # Lock acquired — release + close strictly on exit. The BlockingIOError
    # path above already cleaned up its fd, so it must NOT enter this
    # finally block (otherwise we'd double-close).
    try:
        yield
    finally:
        try:
            await anyio.to_thread.run_sync(fcntl.flock, fd, fcntl.LOCK_UN)
        finally:
            await anyio.to_thread.run_sync(os.close, fd)
