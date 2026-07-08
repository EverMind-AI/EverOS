"""Periodic md tree scanner.

The watcher catches realtime events but misses:

- files created while the daemon was down,
- ``cp`` / external editors that move-replace and confuse inotify,
- WSL2 / network mounts where fsevents don't propagate.

The scanner closes those gaps by walking the memory root every
``scan_interval`` seconds (default 30s, configurable later), matching
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
from everos.core.persistence import MemoryRoot
from everos.core.persistence.sqlite import session_scope
from everos.infra.persistence.sqlite import (
    MdChangeState,
    get_session_factory,
    md_change_state_repo,
)

from .reconciler import PriorState, reconcile
from .registry import KIND_REGISTRY
from .types import ReconcileDecision, ScanInput

logger = get_logger(__name__)


class CascadeScanner:
    """Periodic walker — owns its asyncio task."""

    def __init__(
        self,
        memory_root: MemoryRoot,
        *,
        scan_interval_seconds: float = 30.0,
    ) -> None:
        self._memory_root = memory_root
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
        """One scan + reconcile pass; returns the decisions that were
        upserted into :class:`MdChangeState`.

        Exposed so the CLI ``cascade sync`` command can trigger a sweep
        without owning a long-lived scanner task.
        """
        scan_inputs = await asyncio.to_thread(
            _collect_scan_inputs, self._memory_root.root
        )
        state = await _load_state_snapshot()
        decisions = reconcile(scan_inputs, state)
        for decision in decisions:
            await md_change_state_repo.upsert(
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


async def _load_state_snapshot() -> dict[str, PriorState]:
    """Project every row in ``md_change_state`` into :class:`PriorState`."""
    factory = get_session_factory()
    async with session_scope(factory) as s:
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
            )
            for row in rows
        }
