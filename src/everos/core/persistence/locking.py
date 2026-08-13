"""Process-wide exclusive lock on a memory-root.

Uses ``fcntl.flock`` (POSIX advisory locking, available on Linux + macOS;
Windows is not supported — see project README on platform scope). The
public surface is an :func:`contextlib.asynccontextmanager` so callers
use ``async with memory_root_lock(mr):``; the underlying syscalls have
no async equivalent so they run in a worker thread via
:func:`anyio.to_thread.run_sync`.

**Acquisition polls with ``LOCK_NB`` instead of blocking in the thread.** A
blocking ``flock`` cannot be bounded or cancelled: the syscall runs in a worker
thread, and cancelling the awaiting coroutine leaves that thread to acquire the
lock later with nobody left to release it — strictly worse than waiting. Short
non-blocking attempts on a poll interval give the same semantics while making
the wait bounded, cancellable, and visible in the log. Visibility is the point:
the wait itself is by design (see :func:`ensure_business_indexes` — the second
process is *supposed* to wait, then find the work already done), but without a
log line a server startup that waits on it looks like a hang with no last
message beyond ``lifespan_provider_startup``.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio

from everos.core.observability.logging import get_logger

from .memory_root import MemoryRoot

logger = get_logger(__name__)

DEFAULT_LOCK_TIMEOUT_SECONDS = 1800.0
"""Default upper bound on waiting for the memory-root lock.

Deliberately an order of magnitude above the legitimate hold, not near it:
the legitimate holder is a one-shot FTS/schema migration whose runtime is
O(rows), so a large memory-root on a slow disk can hold it for minutes — a
bound *near* that (an earlier draft used 300s) turns the worst legitimate
migration into a startup crash for every process waiting on it.

Generosity costs almost nothing here, because this timeout's job is
diagnosis, not recovery. The wait is already visible from the first poll
(``memory_root_lock_waiting``), and when the holder is genuinely stuck —
alive but wedged inside its critical section, the only case this bounds —
giving up sooner does not un-stick it: the error and the operator's next
move (inspect the holding process) are the same at 5 minutes or 30.
``flock`` is released by the kernel on process exit, so a *dead* holder
never needs this.
"""

_LOCK_POLL_INTERVAL_SECONDS = 0.5
"""Gap between non-blocking acquisition attempts. Startup-path latency, so
sub-second is imperceptible; keeping it off zero avoids a spin."""


class LockError(RuntimeError):
    """Raised when the memory-root lock cannot be acquired."""


@asynccontextmanager
async def memory_root_lock(
    memory_root: MemoryRoot,
    *,
    blocking: bool = True,
    timeout_seconds: float | None = DEFAULT_LOCK_TIMEOUT_SECONDS,
) -> AsyncIterator[None]:
    """Acquire an exclusive process lock on the memory-root.

    Args:
        memory_root: The memory-root to lock. The lock anchor file
            (``<root>/.lock``) is created on first use.
        blocking: If ``True`` (default), wait until the lock is free or
            ``timeout_seconds`` elapses. If ``False``, raise
            :class:`LockError` immediately when another process holds it.
        timeout_seconds: Upper bound on the wait when ``blocking=True``;
            ``None`` waits indefinitely. Ignored when ``blocking=False``.

    Raises:
        LockError: When the lock is held and either ``blocking=False`` or the
            timeout elapsed.
    """
    await anyio.Path(memory_root.root).mkdir(parents=True, exist_ok=True)
    lock_path = memory_root.lock_file

    # Open the anchor file (create on first use). The fd, not the path, is
    # what fcntl operates on. ``os.open`` is microsecond-fast but offloaded
    # for consistency with the rest of the lock acquisition flow.
    fd = await anyio.to_thread.run_sync(
        lambda: os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    )

    started = time.monotonic()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    announced = False
    try:
        while True:
            try:
                await anyio.to_thread.run_sync(
                    fcntl.flock, fd, fcntl.LOCK_EX | fcntl.LOCK_NB
                )
                break
            except BlockingIOError as exc:
                if not blocking:
                    raise LockError(
                        "another process already holds the memory-root lock "
                        f"at {lock_path}"
                    ) from exc
                if not announced:
                    logger.info(
                        "memory_root_lock_waiting",
                        path=str(lock_path),
                        timeout_seconds=timeout_seconds,
                    )
                    announced = True
                if deadline is not None and time.monotonic() >= deadline:
                    raise LockError(
                        "timed out after "
                        f"{time.monotonic() - started:.1f}s waiting for the "
                        f"memory-root lock at {lock_path}. The holder is "
                        "still alive (the kernel releases a dead process's "
                        "flock automatically) — inspect the process holding "
                        f"{lock_path} rather than retrying this one"
                    ) from exc
                await anyio.sleep(_LOCK_POLL_INTERVAL_SECONDS)
    except BaseException:
        await anyio.to_thread.run_sync(os.close, fd)
        raise

    if announced:
        logger.info(
            "memory_root_lock_acquired_after_wait",
            path=str(lock_path),
            waited_seconds=round(time.monotonic() - started, 1),
        )

    # Lock acquired — release + close strictly on exit. The failure paths above
    # already closed their fd, so they must NOT enter this finally block
    # (otherwise we'd double-close).
    try:
        yield
    finally:
        try:
            await anyio.to_thread.run_sync(fcntl.flock, fd, fcntl.LOCK_UN)
        finally:
            await anyio.to_thread.run_sync(os.close, fd)
