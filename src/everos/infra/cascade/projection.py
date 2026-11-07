"""Settle the index projection of markdown a memorize call just wrote.

``extracted`` means the markdown is on disk. The index is a later step,
and a caller that only sees ``extracted`` cannot tell that the cascade
worker failed those files (#364). This waits, bounded, for the ledger
rows of the paths the call wrote and reports how they finished.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Collection
from pathlib import Path

from everos.infra.persistence.sqlite import MTIME_TOLERANCE_SECONDS, MdChangeStateRepo

from .registry import match_kind

PROJECTION_SETTLE_TIMEOUT_SECONDS = 15.0
"""How long a write call waits to learn whether its files indexed.

Sized to cover one pass of the worker's inline retry ladder (default
``max_retry=3`` and a 2s backoff is 12s of sleep) plus a little room for
the handler itself. An unrecoverable failure is marked on the first
attempt, so it shows up well inside this. The wait ends as soon as every
path has left ``pending`` / ``processing``; the budget is only the cap
when a row never settles. Past it, a row still in flight is lag, not a
failure, and the call keeps ``extracted``.
"""

_POLL_SECONDS = 0.05


@dataclasses.dataclass(frozen=True)
class ProjectionVerdict:
    """How the files a call wrote finished in ``md_change_state``.

    ``observed`` is false when there was nothing to wait for (cascade not
    running, or no path belongs to a kind). ``pending`` counts paths that
    were still queued when the budget ran out.
    """

    observed: bool
    failed_permanent: int = 0
    failed_retryable: int = 0
    pending: int = 0

    @staticmethod
    def unobserved() -> ProjectionVerdict:
        """No projection ran; the caller should keep its own status."""
        return ProjectionVerdict(observed=False)


async def settle_written(
    ledger: MdChangeStateRepo,
    root: Path,
    paths: Collection[str],
    *,
    started: bool,
    budget_seconds: float = PROJECTION_SETTLE_TIMEOUT_SECONDS,
    poll: float = _POLL_SECONDS,
) -> ProjectionVerdict:
    """Wait until ``paths`` are indexed or failed, then count the failures.

    A path that is not a registered kind is ignored. When the cascade is
    not running, the verdict is unobserved. A missing ledger row is
    enqueued from disk first, so the answer does not depend on the
    watcher having already fired.
    """
    relevant = [path for path in dict.fromkeys(paths) if match_kind(path) is not None]
    if not started or not relevant:
        return ProjectionVerdict.unobserved()
    for path in relevant:
        await _ensure_enqueued(ledger, root, path)
    deadline = time.monotonic() + budget_seconds
    while True:
        permanent, retryable, pending = await _classify(ledger, relevant)
        if pending == 0 or time.monotonic() >= deadline:
            return ProjectionVerdict(
                observed=True,
                failed_permanent=permanent,
                failed_retryable=retryable,
                pending=pending,
            )
        await asyncio.sleep(poll)


async def _classify(
    ledger: MdChangeStateRepo, paths: list[str]
) -> tuple[int, int, int]:
    permanent = retryable = pending = 0
    for path in paths:
        row = await ledger.read_one(path)
        if row is None or row.status in ("pending", "processing"):
            pending += 1
        elif row.status == "failed":
            if row.retryable:
                retryable += 1
            else:
                permanent += 1
    return permanent, retryable, pending


async def _ensure_enqueued(ledger: MdChangeStateRepo, root: Path, path: str) -> None:
    """Queue ``path`` when the ledger does not already describe this mtime."""
    spec = match_kind(path)
    if spec is None:
        return
    full = root / path
    try:
        exists = full.is_file()
        mtime = full.stat().st_mtime if exists else 0.0
    except OSError:
        exists = False
        mtime = 0.0
    row = await ledger.read_one(path)
    change = "deleted" if not exists else "modified"
    if row is None:
        await ledger.upsert(
            path,
            kind=spec.name,
            change_type="added" if exists else "deleted",
            mtime=mtime,
        )
        return
    stale = abs(row.mtime - mtime) >= MTIME_TOLERANCE_SECONDS
    vanished = not exists and row.change_type != "deleted"
    if stale or vanished:
        await ledger.upsert(path, kind=spec.name, change_type=change, mtime=mtime)
