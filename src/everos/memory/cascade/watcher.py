"""Filesystem watcher — emits cascade enqueue events on md changes.

watchdog 6's :class:`Observer` runs in its own native thread; the
event handler callback fires there too. We bridge those events back
onto the orchestrator's asyncio loop via
:func:`asyncio.run_coroutine_threadsafe` so every state-table write
goes through the same async repo as the scanner / CLI sync paths.

The handler is intentionally cheap: pattern-match the path against
the kind registry, then enqueue. The watcher does **not** read the
file content — that's the worker's job after :meth:`claim_one`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from watchdog.events import FileMovedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from everos.core.observability.logging import get_logger
from everos.core.persistence import MemoryRoot
from everos.infra.persistence.sqlite import md_change_state_repo

from .registry import KindSpec, match_kind

logger = get_logger(__name__)


class CascadeWatcher:
    """Bridge watchdog → md_change_state for the configured memory root.

    The watchdog observer is started on :meth:`start` and stopped on
    :meth:`stop`. Events outside the registered kind paths are silently
    ignored — DD-7 (single whitelist layer) keeps the watcher free of
    bespoke exclusion rules.
    """

    def __init__(
        self,
        memory_root: MemoryRoot,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._memory_root = memory_root
        self._loop = loop
        self._observer = Observer()
        self._handler = _Handler(memory_root, loop)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        # The memory root is created lazily by other layers; watchdog
        # rejects non-existent paths so we ensure it exists here.
        self._memory_root.ensure()
        self._observer.schedule(
            self._handler, str(self._memory_root.root), recursive=True
        )
        self._observer.start()
        self._started = True
        logger.info("cascade_watcher_started", root=str(self._memory_root.root))

    def stop(self) -> None:
        if not self._started:
            return
        self._observer.stop()
        self._observer.join(timeout=5)
        self._started = False
        logger.info("cascade_watcher_stopped")


class _Handler(FileSystemEventHandler):
    """Watchdog callback — fires in the watchdog thread."""

    def __init__(
        self, memory_root: MemoryRoot, loop: asyncio.AbstractEventLoop
    ) -> None:
        self._memory_root = memory_root
        self._loop = loop

    def on_created(self, event: FileSystemEvent) -> None:
        self._enqueue(event.src_path, "added")

    def on_modified(self, event: FileSystemEvent) -> None:
        self._enqueue(event.src_path, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        # macOS FSEvents fires a synthetic deletion for the OLD inode
        # whenever ``os.replace`` overwrites an existing file — the path
        # itself is still present, now pointing at the new inode, and the
        # paired ``on_moved`` has already enqueued the dest as 'added'.
        # Propagating this false-positive 'deleted' drives the worker to
        # call ``delete_by_md_path`` and wipe LanceDB while md is fine.
        # The stat is on the watcher thread but cheap on APFS (~µs);
        # real unlinks still surface because the path is truly gone.
        if Path(event.src_path).exists():
            return
        self._enqueue(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        # A rename emits both a `moved` for the src and effectively a
        # `created` for the dest. We materialise both sides so the
        # state table tracks the source as deleted and the destination
        # as added.
        #
        # Symmetric to ``on_deleted``: stat src first. If the path still
        # exists (e.g. macOS reports a synthetic move for the old inode
        # of an atomic-replace pair, or a hardlink survives the rename
        # so the named path is still bound), the 'deleted' enqueue
        # would wipe LanceDB while the file is intact. Real renames
        # (src genuinely gone, dest the new home) keep both legs.
        if not Path(event.src_path).exists():
            self._enqueue(event.src_path, "deleted")
        if isinstance(event, FileMovedEvent):
            self._enqueue(event.dest_path, "added")

    def _enqueue(self, raw_path: str, change_type: str) -> None:
        rel = _relative_to_root(self._memory_root.root, raw_path)
        if rel is None:
            return
        spec = match_kind(rel)
        if spec is None:
            return
        mtime = _safe_mtime(raw_path)
        asyncio.run_coroutine_threadsafe(
            _enqueue_async(spec, rel, change_type, mtime),
            self._loop,
        )


async def _enqueue_async(
    spec: KindSpec, rel: str, change_type: str, mtime: float
) -> None:
    """Coroutine variant — runs on the orchestrator's event loop."""
    try:
        await md_change_state_repo.upsert(
            rel,
            kind=spec.name,
            change_type=change_type,
            mtime=mtime,
        )
    except Exception as exc:
        logger.warning(
            "cascade_watcher_upsert_failed",
            md_path=rel,
            kind=spec.name,
            error=str(exc),
        )


def _relative_to_root(root: Path, raw: str) -> str | None:
    """Return ``raw`` relative to ``root`` using POSIX separators.

    ``None`` when the path is outside the memory root (defensive — the
    watcher only watches inside ``root``, but external symlinks could
    surface).
    """
    try:
        rel = Path(raw).resolve().relative_to(root)
    except ValueError:
        return None
    return rel.as_posix()


def _safe_mtime(raw: str) -> float:
    """Return mtime in seconds, falling back to 0.0 on stat failure."""
    try:
        return Path(raw).stat().st_mtime
    except OSError:
        return 0.0
