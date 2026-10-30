"""The memory root's exclusive lock, taken from outside a running engine.

:class:`OfflineEngine` holds ``<jobstore>.lock`` for as long as it runs, so a
second engine (a second server) on the same root fails at startup. Commands
that rewrite shared storage without starting an engine -- the CLI's index
writers, backfill -- take the same file with the same flags through
:class:`OmeLock`, so they and a server exclude each other.
"""

from __future__ import annotations

__all__ = ["OmeLock"]

import contextlib
from collections.abc import Iterator
from pathlib import Path
from typing import IO

import portalocker

from everos.infra.ome.exceptions import EngineLockHeldError


class OmeLock:
    """The ``<jobstore>.lock`` file an :class:`OfflineEngine` holds while running.

    Args:
        jobstore_path: The engine's jobstore (``ome.db``); the lock file sits
            next to it.
    """

    def __init__(self, jobstore_path: Path) -> None:
        self._path = Path(str(jobstore_path) + ".lock")
        self._handle: IO[str] | None = None

    @property
    def held(self) -> bool:
        """Whether this instance currently holds the lock."""
        return self._handle is not None

    def is_free(self) -> bool:
        """Whether no other process holds the lock right now.

        Takes it with ``LOCK_EX | LOCK_NB`` and releases it at once. A probe,
        not a reservation: the lock can be taken between this call and
        whatever the caller does next, which then surfaces as
        :class:`EngineLockHeldError` from :meth:`hold` or the engine's start.
        """
        with self._open() as handle:
            try:
                portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
            except portalocker.LockException:
                return False
            portalocker.unlock(handle)
            return True

    def acquire(self) -> None:
        """Take the lock and keep it until :meth:`release`; a no-op when held.

        The engine holds it from ``start`` to ``stop``; :meth:`hold` wraps the
        pair for callers whose critical section is one block.

        Raises:
            EngineLockHeldError: Another process already holds it.
        """
        if self._handle is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self._path, "a+")  # noqa: SIM115  # closed by release()
        try:
            portalocker.lock(handle, portalocker.LOCK_EX | portalocker.LOCK_NB)
        except portalocker.LockException as exc:
            handle.close()
            raise EngineLockHeldError(f"another process holds {self._path}") from exc
        self._handle = handle

    def release(self) -> None:
        """Give the lock back; a no-op when not held."""
        if self._handle is None:
            return
        try:
            portalocker.unlock(self._handle)
        finally:
            self._handle.close()
            self._handle = None

    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        """Hold the lock for the duration of the block.

        Raises:
            EngineLockHeldError: Another process already holds it.
        """
        self.acquire()
        try:
            yield
        finally:
            self.release()

    @contextlib.contextmanager
    def _open(self) -> Iterator[IO[str]]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a+") as handle:
            yield handle
