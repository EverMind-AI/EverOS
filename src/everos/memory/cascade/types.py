"""Cascade value types — small dataclasses shared by registry / reconciler / handler.

All types here are :class:`dataclasses.dataclass(frozen=True)` so the
pure-function modules (``reconciler``) stay deterministic and easy to
unit-test without an event loop or IO mocks.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

ChangeType = Literal["added", "modified", "deleted"]


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
