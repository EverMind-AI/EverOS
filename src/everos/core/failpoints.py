"""Test-only failure points: crash, stall, fail or skip at a named step.

Some windows of a deletion or a purge are two adjacent system calls apart,
and a random kill or a timed request never lands in them. A real-server test
starts the process with ``EVEROS_TEST_FAILPOINTS`` set to a comma-separated
list of items, each ``name`` or ``name=<mode>``:

- ``name`` -- exit the process at once with status 137 (no cleanup runs,
  like a hard kill);
- ``name=pause`` -- wait until ``$EVEROS_TEST_FAILPOINT_DIR/<name>.release``
  exists, then go on;
- ``name=skip`` -- tell the caller to skip the step that follows;
- ``name=raise:<alias>:<n>`` -- raise the aliased exception the first ``n``
  times the point is reached (see ``_ALIASES``).

The code calls :func:`failpoint` (a synchronous step: only the crash and
raise modes apply) or awaits :func:`afailpoint` (all four). The variable is
read once, at import. Unset -- always, outside such a test -- nothing
happens.
"""

from __future__ import annotations

__all__ = ["afailpoint", "failpoint"]

import asyncio
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path

import sqlalchemy.exc

from everos.core.errors import (
    DomainError,
    InfrastructureError,
    VectorStoreBusyError,
)

_POLL_S = 0.05


def _sqlite_locked(message: str) -> Exception:
    return sqlalchemy.exc.OperationalError(
        message, None, sqlite3.OperationalError("database is locked")
    )


_ALIASES: dict[str, Callable[[str], Exception]] = {
    "infra": InfrastructureError,
    "domain": DomainError,
    "busy": VectorStoreBusyError,
    "sqlite_locked": _sqlite_locked,
    "runtime": RuntimeError,
}


def _parse(raw: str) -> dict[str, str]:
    armed: dict[str, str] = {}
    for item in raw.split(","):
        name, _, mode = item.strip().partition("=")
        if name:
            armed[name.strip()] = mode.strip() or "kill"
    return armed


_ARMED = _parse(os.environ.get("EVEROS_TEST_FAILPOINTS", ""))
_HITS: dict[str, int] = {}


def _maybe_raise(name: str, mode: str) -> None:
    alias, _, times = mode.removeprefix("raise:").partition(":")
    _HITS[name] = _HITS.get(name, 0) + 1
    if _HITS[name] <= int(times or "1"):
        raise _ALIASES[alias](f"failpoint {name}")


def failpoint(name: str) -> None:
    """Crash (status 137) or raise if ``name`` is armed for it."""
    mode = _ARMED.get(name)
    if mode == "kill":
        os._exit(137)
    if mode is not None and mode.startswith("raise:"):
        _maybe_raise(name, mode)


async def afailpoint(name: str) -> bool:
    """:func:`failpoint`, plus stalling until released and skipping.

    Returns:
        ``True`` when the caller should skip the step that follows.
    """
    failpoint(name)
    mode = _ARMED.get(name)
    if mode == "pause":
        release = Path(os.environ.get("EVEROS_TEST_FAILPOINT_DIR", ".")) / (
            f"{name}.release"
        )
        while True:  # the signal is a file another process writes
            if release.exists():
                break
            await asyncio.sleep(_POLL_S)
    return mode == "skip"
