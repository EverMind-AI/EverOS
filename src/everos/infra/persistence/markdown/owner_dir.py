"""Deleting one memory owner's whole directory.

The directory is moved, in one rename under the owner's lock, into a
per-deletion trash folder ``<root>/.tmp/delete-<random>/`` next to a
``scope.json`` naming the owner. Still under the lock, today's daily-log
files are put back empty, carrying their id counters: entries are only ever
appended to today's file, and the next one must not reuse a deleted id.
Then the lock is released, the trash is read (what was removed) and removed.

A crash can stop this between any two steps. :meth:`sweep_trash` (run at
startup, before anything writes) finishes every trash folder it finds: puts
today's counters back where they are missing and removes the folder.
"""

from __future__ import annotations

__all__ = ["MarkdownOwnerDirStore"]

import json
import shutil
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

import anyio

from everos.core.errors import PathTraversalError
from everos.core.failpoints import failpoint
from everos.core.locks import LOCKS
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import today_with_timezone
from everos.infra.persistence.memory_root import MemoryRoot
from everos.memory.ports import MemoryRef, OwnerDirDeletion, Scope

from ._base import Entry, EntryId, MarkdownReader, MarkdownWriter
from ._mds import AgentSkillFrontmatter, UserProfileFrontmatter
from ._writers import (
    AgentCaseWriter,
    AtomicFactWriter,
    BaseDailyWriter,
    EpisodeWriter,
    ForesightWriter,
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
                "atomic_fact": AtomicFactWriter(root),
                "foresight": ForesightWriter(root),
            },
            "agent": {"agent_case": AgentCaseWriter(root)},
        }

    async def delete(self, scope: Scope) -> OwnerDirDeletion:
        owner_dir = self._owner_dir(scope)
        key = owner_lock(scope)
        trash = self._root.tmp_dir / f"{_TRASH_PREFIX}{uuid.uuid4().hex}"
        async with LOCKS.hold("owner", key):
            if not await anyio.Path(owner_dir).is_dir():
                return OwnerDirDeletion(refs={}, count=0)
            await anyio.Path(trash).mkdir(parents=True)
            await anyio.Path(trash / _MANIFEST).write_text(
                json.dumps(
                    {
                        "app_id": scope.app_id,
                        "project_id": scope.project_id,
                        "owner_type": scope.owner_type,
                        "owner_id": scope.owner_id,
                    }
                ),
                encoding="utf-8",
            )
            await anyio.Path(owner_dir).rename(trash / _MOVED)
            failpoint("delete_owner.after_rename")
            await self._keep_today_counters(scope, trash / _MOVED)
            failpoint("delete_owner.after_restore")
            LOCKS.bump_delete_count(key)
        deletion = await self._inventory(scope, trash / _MOVED)
        await anyio.to_thread.run_sync(shutil.rmtree, trash)
        return deletion

    async def sweep_trash(self) -> int:
        tmp = anyio.Path(self._root.tmp_dir)
        if not await tmp.is_dir():
            return 0
        finished = 0
        async for folder in tmp.glob(f"{_TRASH_PREFIX}*"):
            trash = Path(str(folder))
            moved = trash / _MOVED
            try:
                fields = json.loads((trash / _MANIFEST).read_text(encoding="utf-8"))
                scope = Scope(**fields)
            except (OSError, ValueError, TypeError):
                scope = None  # died before the manifest was written: nothing moved
            if scope is not None and moved.is_dir():
                async with LOCKS.hold("owner", owner_lock(scope)):
                    await self._keep_today_counters(scope, moved)
            await anyio.to_thread.run_sync(shutil.rmtree, trash)
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
        return self._root.agents_dir(app_id, project_id)

    def _owner_dir(self, scope: Scope) -> Path:
        """The owner's directory; refuses one that resolves anywhere else.

        The routes already reject ids with a path separator or ``..``; this
        is the second line: the directory must be exactly the owner's entry
        under ``users/`` or ``agents/`` of the request's app and project.
        """
        base = self._base(scope.owner_type, scope.app_id, scope.project_id)
        target = base / scope.owner_id
        # ``base / "."`` (or ``""``) is ``base`` itself: the name check refuses
        # it before anything could move the whole users/ or agents/ folder.
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
                    if needle in await path.read_text(encoding="utf-8"):
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

    async def _inventory(self, scope: Scope, moved: Path) -> OwnerDirDeletion:
        """What the moved directory held, with refs at their old paths."""
        owner_dir = self._owner_dir(scope)
        refs: dict[str, list[MemoryRef]] = {}
        count = 0
        for table, writer in self._logs[scope.owner_type].items():
            folder = writer.path_for(
                scope.owner_id, app_id=scope.app_id, project_id=scope.project_id
            ).parent.relative_to(owner_dir)
            found: list[MemoryRef] = []
            for path in sorted((moved / folder).glob("*.md")):
                md_path = (owner_dir / path.relative_to(moved)).relative_to(
                    self._root.root
                )
                for entry in (await MarkdownReader.read(path)).entries:
                    found.append(
                        MemoryRef(
                            id=f"{scope.owner_id}_{entry.id}",
                            entry_id=entry.id,
                            md_path=md_path.as_posix(),
                        )
                    )
            refs[table] = found
            count += len(found)
        if (moved / UserProfileFrontmatter.PROFILE_FILENAME).is_file():
            count += 1
        skills = moved / AgentSkillFrontmatter.SKILLS_CONTAINER_NAME
        if skills.is_dir():
            count += sum(
                1
                for d in skills.iterdir()
                if d.is_dir()
                and d.name.startswith(AgentSkillFrontmatter.SKILL_DIR_PREFIX)
            )
        return OwnerDirDeletion(refs=refs, count=count)


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
