"""Owner directories: who holds a conversation, and an old deletion's trash.

EverOS 2.0 deletes no owner directory (a delete is a forget). Two things stay:

* :meth:`MarkdownOwnerDirStore.owners_with_session` -- the users whose memory
  holds a conversation, which a delete by ``session_id`` alone forgets in;
* :meth:`MarkdownOwnerDirStore.sweep_trash` (run at startup, before anything
  writes) finishes a whole-owner deletion an earlier build left half done: it
  had moved the directory into ``<root>/.tmp/delete-<random>/`` next to a
  ``scope.json`` naming the owner. Today's daily-log files go back empty,
  carrying their id counters (the next entry must not reuse a deleted id),
  and the trash folder is removed.
"""

from __future__ import annotations

__all__ = ["MarkdownOwnerDirStore"]

import json
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

import anyio

from everos.core.errors import PathTraversalError
from everos.core.locks import LOCKS
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import today_with_timezone
from everos.infra.persistence.memory_root import MemoryRoot
from everos.memory.ports import Scope

from ._base import Entry, EntryId, MarkdownReader, MarkdownWriter
from ._writers import (
    BaseDailyWriter,
    EpisodeWriter,
    ImageMemoryWriter,
    SceneWriter,
)
from .stores import owner_lock

logger = get_logger(__name__)

_TRASH_PREFIX = "delete-"
_MANIFEST = "scope.json"
_MOVED = "owner"


class MarkdownOwnerDirStore:
    """:class:`~everos.memory.ports.OwnerDirStore` over the memory root.

    Args:
        root: Memory root the owner directories live under.
    """

    def __init__(self, root: MemoryRoot) -> None:
        self._root = root
        self._md = MarkdownWriter(root)
        self._logs: dict[str, dict[str, BaseDailyWriter]] = {
            "user": {
                "episode": EpisodeWriter(root),
                "image_memory": ImageMemoryWriter(root),
                "scene": SceneWriter(root),
            },
        }

    def _trash_folders(self) -> list[tuple[Path, Scope | None]]:
        """Every trash folder with the owner it holds (``None``: nothing moved)."""
        tmp = self._root.tmp_dir
        if not tmp.is_dir():
            return []
        found: list[tuple[Path, Scope | None]] = []
        for trash in sorted(tmp.glob(f"{_TRASH_PREFIX}*")):
            try:
                fields = json.loads((trash / _MANIFEST).read_text(encoding="utf-8"))
                scope: Scope | None = Scope(**fields)
            except (OSError, ValueError, TypeError):
                scope = None  # died before the manifest was written: nothing moved
            found.append((trash, scope if (trash / _MOVED).is_dir() else None))
        return found

    async def sweep_trash(self) -> int:
        finished = 0
        for trash, scope in self._trash_folders():
            if scope is not None:
                async with LOCKS.hold("owner", owner_lock(scope)):
                    await self._keep_today_counters(scope, trash / _MOVED)
            try:
                await anyio.to_thread.run_sync(shutil.rmtree, trash)
            except OSError as exc:
                # The counters are back; what is left is only bytes. Starting
                # matters more: the next sweep (or a resend) removes it.
                logger.warning(
                    "memory_delete_trash_sweep_failed",
                    trash=trash.name,
                    error=repr(exc)[:200],
                )
                continue
            logger.info("memory_delete_trash_swept", trash=trash.name)
            finished += 1
        return finished

    async def owners_with_session(
        self, app_id: str, project_id: str, session_id: str
    ) -> list[Scope]:
        needle = f"**session_id**: {session_id}"
        owners: list[Scope] = []
        for owner_type, logs in self._logs.items():
            base = self._base(owner_type, app_id, project_id)
            if not base.is_dir():
                continue
            for owner_dir in sorted(p for p in base.iterdir() if p.is_dir()):
                scope = Scope(
                    app_id=app_id,
                    project_id=project_id,
                    owner_id=owner_dir.name,
                    owner_type=owner_type,  # type: ignore[arg-type]
                )
                if await self._mentions(scope, logs.values(), needle):
                    owners.append(scope)
        return owners

    # ── internals ─────────────────────────────────────────────────────────

    def _base(self, owner_type: str, app_id: str, project_id: str) -> Path:
        if owner_type == "user":
            return self._root.users_dir(app_id, project_id)
        raise ValueError(f"no owner directory for {owner_type!r} owners")

    def _owner_dir(self, scope: Scope) -> Path:
        """The owner's directory; refuses one that resolves anywhere else.

        The routes already reject ids with a path separator or ``..``; this
        is the second line: the directory must be exactly the owner's entry
        under ``users/`` of the request's app and project.
        """
        base = self._base(scope.owner_type, scope.app_id, scope.project_id)
        target = base / scope.owner_id
        # ``base / "."`` (or ``""``) is ``base`` itself: the name check refuses
        # it before anything could move the whole users/ folder.
        if (
            target.name != scope.owner_id
            or target.resolve() != base.resolve() / scope.owner_id
        ):
            raise PathTraversalError(
                f"owner directory escapes {base}: {scope.owner_id!r}"
            )
        return target

    async def _mentions(
        self, scope: Scope, logs: Iterable[BaseDailyWriter], needle: str
    ) -> bool:
        for writer in logs:
            folder = anyio.Path(
                writer.path_for(
                    scope.owner_id, app_id=scope.app_id, project_id=scope.project_id
                ).parent
            )
            if not await folder.is_dir():
                continue
            async for path in folder.glob("*.md"):
                try:
                    if needle in await MarkdownReader.read_text(path):
                        return True
                except FileNotFoundError:
                    continue
        return False

    async def _keep_today_counters(self, scope: Scope, moved: Path) -> None:
        """Put today's daily-log files back, empty, at their old id counters."""
        owner_dir = self._owner_dir(scope)
        today = today_with_timezone()
        for writer in self._logs[scope.owner_type].values():
            live = writer.path_for(
                scope.owner_id, today, app_id=scope.app_id, project_id=scope.project_id
            )
            old = moved / live.relative_to(owner_dir)
            if not old.is_file():
                continue
            parsed = await MarkdownReader.read(old)
            count = _counter(parsed.frontmatter.get("entry_count"), parsed.entries)
            if live.is_file():  # written to since (only after a crash)
                current = await MarkdownReader.read(live)
                if (
                    _counter(current.frontmatter.get("entry_count"), current.entries)
                    < count
                ):
                    await self._md.patch_frontmatter(live, {"entry_count": count})
                continue
            frontmatter = {
                k: v for k, v in parsed.frontmatter.items() if k != "deprecated_entries"
            }
            frontmatter["entry_count"] = count
            await self._md.write_markdown(live, frontmatter=frontmatter, body="")


def _counter(raw: object, entries: Sequence[Entry]) -> int:
    """The id counter a daily-log file stands at: never below its ids."""
    count = raw if isinstance(raw, int) and not isinstance(raw, bool) else 0
    seqs = []
    for entry in entries:
        try:
            seqs.append(EntryId.parse(entry.id).seq)
        except ValueError:
            continue
    return max([count, *seqs])
