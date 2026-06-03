"""Markdown file writer with atomic write semantics.

Atomicity is provided by writing to a same-directory temp file
(``.<name>.tmp.<uuid>``) and using :func:`os.replace` to rename it onto
the target. Keeping the temp file in the same directory guarantees the
rename is on the same filesystem (POSIX rename is atomic only within a
single fs).

All public methods are async. File I/O (``read_text`` / ``write_text``
/ ``mkdir``) goes through :class:`anyio.Path`; the few syscalls without
a native async equivalent (``os.fsync`` / ``os.replace`` / ``unlink``
in the cleanup path) are offloaded via :func:`anyio.to_thread.run_sync`.

In-process per-path locking
---------------------------
:meth:`append_entry` / :meth:`append_entries` are read-modify-write of
the whole file (load frontmatter+body, merge an entry block, atomic
write the result). The atomic write itself is safe, but the read→write
window crosses ``await`` points. Concurrent asyncio tasks targeting the
same path would otherwise lose-update each other (both read N entries,
both produce N+1, second write overwrites the first → 1 entry lost).

To prevent this, an in-process per-path :class:`asyncio.Lock` is held
across the entire read-modify-write sequence. Lock objects live on the
writer instance (not class-level) so they bind to the event loop active
when the writer was constructed — this avoids the
"Lock bound to different loop" failure mode that surfaces when
pytest-asyncio rebuilds the loop between tests but module-level writer
singletons leak Lock objects across boundaries.

Process-level coordination (multi-process writers against the same
memory-root) remains the job of
:func:`everos.core.persistence.locking.memory_root_lock`, which uses
``fcntl.flock``. The two locks compose: per-path async lock serialises
tasks within one process, ``memory_root_lock`` serialises processes
against each other.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio

from ..memory_root import MemoryRoot
from .entries import EntryId
from .frontmatter import dump_frontmatter
from .reader import MarkdownReader


class MarkdownWriter:
    """Atomic writer for markdown files inside a memory-root.

    The ``memory_root`` reference is held to enable future enforcement that
    targets stay within the configured root; current writes do not depend on
    it for the rename itself (same-dir temp file).
    """

    def __init__(self, memory_root: MemoryRoot) -> None:
        self._memory_root = memory_root
        # Per-path async lock registry. ``setdefault`` is GIL-atomic, so
        # concurrent callers race only on the dict insert (resolved by
        # ``setdefault`` returning the existing value), not on the Lock.
        # Plain dict (not WeakValueDictionary): a Lock with pending waiters
        # must outlive any task awaiting it; ref-counted GC would race with
        # those waiters. See Python bpo-28427 for the WeakValueDictionary
        # multithreading hazard that bites the weak-ref approach.
        self._path_locks: dict[Path, asyncio.Lock] = {}

    @property
    def memory_root(self) -> MemoryRoot:
        return self._memory_root

    def lock_for(self, path: Path) -> asyncio.Lock:
        """Return the per-path lock; create on first use.

        Public so that higher-level writers (e.g. :class:`BaseDailyWriter`)
        can serialise their own multi-step ``read → compute → write``
        sequences against this writer's single-step ``append`` paths.
        Pair with :meth:`_append_entries_unlocked` to avoid reentrant
        re-acquisition of the same lock from within an already-locked
        critical section (``asyncio.Lock`` is *not* reentrant).
        """
        # Resolve to an absolute canonical path so aliases (relative vs.
        # absolute, symlinks) share the same lock object.
        key = Path(path).resolve()
        lock = self._path_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._path_locks[key] = lock
        return lock

    async def write(self, path: Path, content: str) -> Path:
        """Atomically write ``content`` to ``path``.

        Steps:
            1. ``mkdir -p`` the parent directory.
            2. Write to ``<parent>/.<name>.tmp.<uuid>``.
            3. ``flush`` + ``fsync`` the temp file.
            4. ``os.replace`` the temp file onto ``path`` (atomic on POSIX).

        Returns:
            ``path`` (resolved as written).
        """
        target = Path(path)
        await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)
        tmp = target.parent / f".{target.name}.tmp.{uuid.uuid4().hex}"
        try:
            await anyio.to_thread.run_sync(_write_and_fsync, tmp, content)
            await anyio.to_thread.run_sync(os.replace, tmp, target)
        except Exception:
            # Best-effort cleanup of the staging file on failure.
            await _unlink_quiet(tmp)
            raise
        return target

    async def write_markdown(
        self,
        path: Path,
        *,
        frontmatter: Mapping[str, Any] | None = None,
        body: str = "",
    ) -> Path:
        """Assemble ``frontmatter`` + ``body`` then atomic-write to ``path``."""
        head = dump_frontmatter(frontmatter or {})
        return await self.write(path, head + body)

    async def append_entry(
        self,
        path: Path,
        *,
        entry_body: str,
        entry_id: EntryId,
        frontmatter_updates: Mapping[str, Any] | None = None,
    ) -> Path:
        """Append a single entry block to a markdown file, merging frontmatter.

        Convenience wrapper around :meth:`append_entries` for single-entry
        callers. See that method for full semantics.

        Args:
            path: Target markdown file. Created if missing.
            entry_body: Content between the open and close markers.
                One leading and trailing newline are added automatically.
            entry_id: The id to stamp on this entry. The caller normally
                builds it with :meth:`EntryId.next_for`.
            frontmatter_updates: Mapping shallow-merged into existing
                frontmatter (later wins). ``None`` skips the merge.

        Returns:
            ``path`` (resolved as written).
        """
        return await self.append_entries(
            path,
            [(entry_body, entry_id)],
            frontmatter_updates=frontmatter_updates,
        )

    async def append_entries(
        self,
        path: Path,
        entries: Sequence[tuple[str, EntryId]],
        *,
        frontmatter_updates: Mapping[str, Any] | None = None,
    ) -> Path:
        """Append ``N`` entry blocks in a single locked read-modify-write cycle.

        Compared with calling :meth:`append_entry` ``N`` times, this:

        * Performs one file read + one file write instead of ``N`` of each
          (IO complexity drops from ``O(N²)`` to ``O(N)`` when the file
          already holds many entries).
        * Holds the per-path lock for one short critical section instead of
          ``N`` separate acquisitions.
        * Updates ``frontmatter`` once at the end (no intermediate
          ``entry_count`` flapping).

        The caller assigns and supplies all :class:`EntryId` values — see
        :meth:`append_entry` for the rationale. The order in ``entries`` is
        the order the blocks land in the file.

        Args:
            path: Target markdown file. Created if missing.
            entries: ``(entry_body, entry_id)`` pairs to append, in order.
                Empty sequence is allowed; the file is still touched for
                frontmatter updates if any are supplied.
            frontmatter_updates: Mapping shallow-merged into existing
                frontmatter once after all entries are appended.

        Returns:
            ``path`` (resolved as written).
        """
        target = Path(path)
        async with self.lock_for(target):
            return await self._append_entries_unlocked(
                target,
                entries,
                frontmatter_updates=frontmatter_updates,
            )

    async def _append_entries_unlocked(
        self,
        path: Path,
        entries: Sequence[tuple[str, EntryId]],
        *,
        frontmatter_updates: Mapping[str, Any] | None = None,
    ) -> Path:
        """Same as :meth:`append_entries` but assumes the caller already
        holds :meth:`lock_for` ``(path)``.

        For use by higher-level writers that perform a multi-step
        ``read → compute eid → write`` sequence and need to keep the lock
        held across the read and the write. Public ``append_entries`` /
        ``append_entry`` always wrap this with the lock.

        Reentrant re-acquisition is unsafe — ``asyncio.Lock`` is not
        reentrant, so calling this without holding the lock yourself
        breaks the safety contract.
        """
        target = Path(path)

        # 1. Load existing markdown (or initialise empty).
        if await anyio.Path(target).is_file():
            parsed = await MarkdownReader.read(target)
            meta: dict[str, Any] = dict(parsed.frontmatter)
            body = parsed.body
        else:
            meta = {}
            body = ""

        # 2. Shallow-merge frontmatter updates.
        if frontmatter_updates:
            meta.update(frontmatter_updates)

        # 3. Append all entry blocks in order.
        if entries:
            if body and not body.endswith("\n"):
                body += "\n"
            appended_blocks: list[str] = []
            for entry_body, entry_id in entries:
                eid_str = entry_id.format()
                appended_blocks.append(
                    f"<!-- entry:{eid_str} -->\n{entry_body}\n"
                    f"<!-- /entry:{eid_str} -->\n"
                )
            body = body + "".join(appended_blocks)

        # 4. Atomic write.
        return await self.write_markdown(target, frontmatter=meta, body=body)


def _write_and_fsync(tmp: Path, content: str) -> None:
    """Sync helper: write + fsync the staging file. Offloaded to a thread."""
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(content)
        fh.flush()
        os.fsync(fh.fileno())


async def _unlink_quiet(tmp: Path) -> None:
    """Best-effort unlink — swallow OSError so the original exception wins."""
    with contextlib.suppress(OSError):
        await anyio.Path(tmp).unlink(missing_ok=True)
