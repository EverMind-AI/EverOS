"""Cross-process lifecycle lock for the rebuildable LanceDB projection.

Normal servers hold a shared lock for the complete LanceDB lifespan. An
offline rebuild requires the exclusive side of the same lock, so it cannot
invalidate or replace tables while a current server has open table handles.

Rebuild also holds the OfflineEngine's existing portalocker anchor. That
second lock detects an already-running older server which predates the
projection lock but already owns the OME lock. It cannot coordinate the
startup window of an old binary that does not implement this protocol, so
operators must stop old servers before rebuilding during an upgrade.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TextIO

import anyio
import portalocker

from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot

logger = get_logger(__name__)

_PROJECTION_LOCK_NAME = ".projection.lock"
_POLL_INTERVAL_SECONDS = 0.25
_SERVER_LOCK_TIMEOUT_SECONDS = 1800.0


class ProjectionLockUnavailableError(RuntimeError):
    """Raised when a projection lifecycle lock cannot be acquired."""


def projection_lock_path(memory_root: MemoryRoot) -> Path:
    """Return the stable shared/exclusive projection-lock anchor."""
    return memory_root.root / _PROJECTION_LOCK_NAME


def ome_lock_path(memory_root: MemoryRoot) -> Path:
    """Return the exact lock anchor used by :class:`OfflineEngine`."""
    return Path(str(memory_root.ome_db) + ".lock")


@asynccontextmanager
async def projection_server_lock(
    memory_root: MemoryRoot,
    *,
    timeout_seconds: float | None = _SERVER_LOCK_TIMEOUT_SECONDS,
) -> AsyncIterator[None]:
    """Hold a shared projection lock for one server's LanceDB lifespan.

    Shared holders coexist, so deployments that intentionally run more than
    one API process are not serialized. An active rebuild holds the exclusive
    side; startup waits for it to finish and verifies the generation marker
    only after the rebuild has released the lock.
    """
    async with _portalocker_lock(
        projection_lock_path(memory_root),
        flags=portalocker.LOCK_SH | portalocker.LOCK_NB,
        blocking=True,
        timeout_seconds=timeout_seconds,
        label="projection shared",
    ):
        yield


@asynccontextmanager
async def projection_rebuild_lock(memory_root: MemoryRoot) -> AsyncIterator[None]:
    """Hold all cross-version exclusions required by an offline rebuild.

    Acquisition is non-blocking and completes before the caller may publish a
    ``REBUILDING`` marker or mutate either store. The projection lock excludes
    current servers and concurrent rebuilds. The OME lock excludes an older
    server once that process has acquired its legacy OME guard; it does not
    make a concurrently starting old binary participate in this protocol.
    """
    async with (
        _portalocker_lock(
            projection_lock_path(memory_root),
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
            blocking=False,
            timeout_seconds=None,
            label="projection exclusive",
        ),
        _portalocker_lock(
            ome_lock_path(memory_root),
            flags=portalocker.LOCK_EX | portalocker.LOCK_NB,
            blocking=False,
            timeout_seconds=None,
            label="OME exclusive",
        ),
    ):
        yield


@asynccontextmanager
async def _portalocker_lock(
    path: Path,
    *,
    flags: int,
    blocking: bool,
    timeout_seconds: float | None,
    label: str,
) -> AsyncIterator[None]:
    """Acquire one retained portalocker handle with bounded async polling."""
    handle = await anyio.to_thread.run_sync(_open_lock_handle, path)
    started = time.monotonic()
    deadline = None if timeout_seconds is None else started + timeout_seconds
    announced = False
    try:
        while True:
            try:
                await anyio.to_thread.run_sync(portalocker.lock, handle, flags)
                break
            except portalocker.LockException as exc:
                if not blocking:
                    raise ProjectionLockUnavailableError(
                        f"{label} lock is held at {path}"
                    ) from exc
                if not announced:
                    logger.info(
                        "projection_lock_waiting",
                        label=label,
                        path=str(path),
                        timeout_seconds=timeout_seconds,
                    )
                    announced = True
                if deadline is not None and time.monotonic() >= deadline:
                    raise ProjectionLockUnavailableError(
                        f"timed out waiting for {label} lock at {path}"
                    ) from exc
                await anyio.sleep(_POLL_INTERVAL_SECONDS)
    except BaseException:
        await anyio.to_thread.run_sync(handle.close)
        raise

    if announced:
        logger.info(
            "projection_lock_acquired_after_wait",
            label=label,
            path=str(path),
            waited_seconds=round(time.monotonic() - started, 2),
        )

    try:
        yield
    finally:
        with anyio.CancelScope(shield=True):
            try:
                await anyio.to_thread.run_sync(portalocker.unlock, handle)
            finally:
                await anyio.to_thread.run_sync(handle.close)


def _open_lock_handle(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a+", encoding="utf-8")


__all__ = [
    "ProjectionLockUnavailableError",
    "ome_lock_path",
    "projection_lock_path",
    "projection_rebuild_lock",
    "projection_server_lock",
]
