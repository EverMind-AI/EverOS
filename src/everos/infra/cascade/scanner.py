"""Periodic md tree scanner.

The watcher catches realtime events but misses:

- files created while the daemon was down,
- ``cp`` / external editors that move-replace and confuse inotify,
- WSL2 / network mounts where fsevents don't propagate.

The scanner closes those gaps by walking the memory root every
``scan_interval`` seconds (default 30s, ``[cascade] scan_interval_seconds``), matching
paths against the kind registry, reading prior state, and running the
pure :func:`reconcile` function to emit the upsert plan.

Walking happens off the event loop via ``asyncio.to_thread`` since
``pathlib.Path.rglob`` is sync; the prior-state fetch + upsert calls
stay async on the loop.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from pathlib import Path

from sqlmodel import select

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    MdChangeState,
    MdChangeStateRepo,
    session_scope,
)

from .models import ReconcileDecision, ScanInput
from .reconciler import PriorState, reconcile
from .registry import KIND_REGISTRY

logger = get_logger(__name__)


class CascadeScanner:
    """Periodic walker — owns its asyncio task."""

    def __init__(
        self,
        memory_root: MemoryRoot,
        ledger: MdChangeStateRepo,
        *,
        scan_interval_seconds: float = 30.0,
    ) -> None:
        self._memory_root = memory_root
        self._ledger = ledger
        self._interval = scan_interval_seconds
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run_loop(), name="cascade-scanner")
        logger.info("cascade_scanner_started", interval=self._interval)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._task
        self._task = None
        logger.info("cascade_scanner_stopped")

    async def scan_once(self) -> list[ReconcileDecision]:
        """One scan + reconcile pass over every kind in :data:`KIND_REGISTRY`;
        returns the decisions that were upserted into :class:`MdChangeState`.

        Exposed so the CLI ``cascade sync`` command can trigger a sweep
        without owning a long-lived scanner task.

        Only the ledger rows of registered kinds are reconciled. A row of a
        kind this build no longer has (an older store's agent, knowledge,
        fact or foresight files) names a file the walk never visits, so
        reconcile would read it as deleted, the re-stat would find the file
        and queue it again, and the worker would fail it -- on every pass.
        Such a row is left as it is.
        """
        scan_inputs = await asyncio.to_thread(
            _collect_scan_inputs, self._memory_root.root
        )
        known = {spec.name for spec in KIND_REGISTRY}
        state = {
            path: row
            for path, row in (await _load_state_snapshot(self._ledger)).items()
            if row.kind in known
        }
        decisions = reconcile(scan_inputs, state, now=get_utc_now())
        decisions = await asyncio.to_thread(
            _recheck_deletions, self._memory_root.root, decisions
        )
        for decision in decisions:
            await self._ledger.upsert(
                decision.md_path,
                kind=decision.kind,
                change_type=decision.change_type,
                mtime=decision.mtime,
            )
        if decisions:
            logger.info(
                "cascade_scanner_decisions",
                count=len(decisions),
                added=sum(1 for d in decisions if d.change_type == "added"),
                modified=sum(1 for d in decisions if d.change_type == "modified"),
                deleted=sum(1 for d in decisions if d.change_type == "deleted"),
            )
        return decisions

    async def _run_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.scan_once()
            except Exception as exc:
                logger.exception("cascade_scanner_failed", error=str(exc))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self._interval)
            except TimeoutError:
                continue


def _collect_scan_inputs(root: Path) -> list[ScanInput]:
    """Walk ``root`` once per registered kind, returning every match.

    ``stat()`` failure mode discrimination is **load-bearing**: the
    reconciler treats "in state but not in scan" as a deletion signal,
    so if we silently drop a path here under a *transient* OS error,
    the next reconcile sweep will emit ``change_type='deleted'`` for
    that healthy md and the handler will wipe its LanceDB rows.

    Two errno classes:

    - :class:`FileNotFoundError` (ENOENT) — the file was unlinked
      between ``glob`` and ``stat``. This is a genuine deletion; drop
      from inputs so the reconciler emits ``deleted`` (correct).
    - any other :class:`OSError` (EMFILE / ENFILE — FD exhaustion,
      EACCES — perms, EIO — disk error, etc.) — we don't know whether
      the file is gone. **Raise** to abort the whole sweep; the
      ``_run_loop`` outer ``try / except Exception`` catches and logs
      it, and we retry on the next interval. A partial scan is worse
      than no scan, because the reconciler can't tell the difference.

    Symptom this guards against (observed 2026-05-28 on LoCoMo
    benchmark conv_2): a search-time FD exhaustion bled into the
    concurrent scanner sweep, and 8 healthy md files got marked
    ``change_type=deleted, status=done`` with their LanceDB rows
    cleared — single-direction data loss until external intervention.
    """
    inputs: list[ScanInput] = []
    for spec in KIND_REGISTRY:
        for absolute in root.glob(spec.path_glob()):
            try:
                mtime = absolute.stat().st_mtime
            except FileNotFoundError:
                # Race between glob and stat; treat as a genuine deletion
                # by leaving this path out of inputs.
                continue
            # Any other OSError (EMFILE / EACCES / EIO ...) — propagate.
            # Partial inputs would trigger spurious deletes in reconcile().
            try:
                rel = absolute.relative_to(root).as_posix()
            except ValueError:
                continue
            inputs.append(ScanInput(md_path=rel, mtime=mtime, kind=spec.name))
    return inputs


def _recheck_deletions(
    root: Path, decisions: list[ReconcileDecision]
) -> list[ReconcileDecision]:
    """Downgrade a tombstone when the file is present at decision time.

    The walk runs in a thread while the watcher and other writers keep the
    root changing. An editor that replaces a file by rename can hide it for
    the duration of the walk; the watcher's upsert for the landed file then
    reaches the state snapshot before reconcile runs, so a path that is on
    disk right now looks "in state, not on disk". Upserting that as
    ``deleted`` hands the worker the file's real mtime, and the reconciler
    trusts the resulting ``done/deleted`` row forever (#544).

    Re-stat only the would-be deletions. A path that exists is emitted as
    ``modified`` with its current mtime instead; a path that is still gone
    keeps the deletion. A non-ENOENT stat error aborts the sweep, matching
    :func:`_collect_scan_inputs`' all-or-nothing rule.
    """
    checked: list[ReconcileDecision] = []
    for decision in decisions:
        if decision.change_type != "deleted":
            checked.append(decision)
            continue
        path = root / decision.md_path
        try:
            if not path.is_file():
                checked.append(decision)
                continue
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            checked.append(decision)
            continue
        except OSError:
            raise
        checked.append(
            ReconcileDecision(
                md_path=decision.md_path,
                kind=decision.kind,
                change_type="modified",
                mtime=mtime,
            )
        )
    return checked


async def _load_state_snapshot(ledger: MdChangeStateRepo) -> dict[str, PriorState]:
    """Project every row in ``md_change_state`` into :class:`PriorState`."""
    async with session_scope(ledger.session_factory) as s:
        rows: Iterable[MdChangeState] = (
            (await s.execute(select(MdChangeState))).scalars().all()
        )
        return {
            row.md_path: PriorState(
                md_path=row.md_path,
                kind=row.kind,
                mtime=row.mtime,
                status=row.status,
                change_type=row.change_type,
                retryable=row.retryable,
                retry_count=row.retry_count,
                retry_at=row.retry_at,
            )
            for row in rows
        }
