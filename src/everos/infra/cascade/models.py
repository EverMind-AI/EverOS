"""Cascade value types — small dataclasses shared by registry / reconciler / handler.

All types here are :class:`dataclasses.dataclass(frozen=True)` so the
pure-function modules (``reconciler``) stay deterministic and easy to
unit-test without an event loop or IO mocks.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Literal

ChangeType = Literal["added", "modified", "deleted"]

RETIRED_KINDS: Final[Mapping[str, str]] = MappingProxyType({"foresight": ".foresights"})
"""Kinds a later release removed, each with its daily-log folder name.

Startup (``CascadeAdmin.retire_removed_kinds``) drops their index table and
their folders. Until then, and for a row whose file is gone, the worker settles
such a ``md_change_state`` row as a done deletion (which reconcile skips)
instead of failing it on every scan."""


@dataclasses.dataclass(frozen=True)
class ScanInput:
    """One scanner observation.

    ``md_path`` is **relative to the memory root** (e.g.
    ``users/u1/episodes/episode-2026-05-14.md``). ``mtime`` is the
    POSIX timestamp captured at scan time; ``kind`` is the
    registry-matched name (``"episode"`` / ``"agent_skill"`` / …).
    """

    md_path: str
    mtime: float
    kind: str


@dataclasses.dataclass(frozen=True)
class ReconcileDecision:
    """One md_path's reconcile outcome.

    Drives the watcher / scanner / sync entry into
    :meth:`MdChangeStateRepo.upsert`. ``change_type`` is a hint —
    handlers re-derive truth from the actual md file state at run
    time (DD-3 in 12 doc).
    """

    md_path: str
    kind: str
    change_type: ChangeType
    mtime: float


@dataclasses.dataclass(frozen=True)
class HandlerOutcome:
    """Per-handler-run summary, returned to the worker for telemetry.

    The worker uses ``upserted`` / ``deleted`` to log how much work
    each md change translated into downstream. ``skipped`` counts
    entries whose ``content_sha256`` matched the existing row — the
    no-op case the diff is meant to optimise.
    """

    md_path: str
    kind: str
    upserted: int
    deleted: int
    skipped: int
