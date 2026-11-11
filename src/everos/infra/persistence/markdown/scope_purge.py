"""Clearing a purge scope's markdown: move it out, keep today's counters, remove.

A purge moves each in-scope project's ``users/``, ``agents/`` and
``knowledge/`` (except ``.taxonomy.md``) in one rename each into
``<root>/.tmp/purge-<job>/<app_dir>/<project_dir>/``, puts today's daily-log
files back empty at their id counters (entries are only appended to today's
file; a later one must not reuse a purged id), then removes the folder. The
extra ``<app_dir>/<project_dir>`` level keeps the trash out of the cascade's
``<app>/<project>/...`` globs, which also match dot-named directories.

Every write here runs as the purge itself (the caller sets ``PURGING``).
"""

from __future__ import annotations

__all__ = ["MarkdownScopePurger"]

import shutil
from collections import Counter
from pathlib import Path

import anyio

from everos.core.failpoints import afailpoint
from everos.core.observability.logging import get_logger
from everos.core.purge_marks import PurgeScope
from everos.infra.persistence.memory_root import (
    MemoryRoot,
    app_id_from_dir,
    project_id_from_dir,
)
from everos.memory.ports import Scope

from ._base import MarkdownReader
from ._mds import AgentSkillFrontmatter, UserProfileFrontmatter
from .owner_dir import MarkdownOwnerDirStore

logger = get_logger(__name__)

_SYSTEM_DIRS = frozenset({".index", ".tmp"})
_TRASH_PREFIX = "purge-"
_OWNER_TYPES = {"users": "user", "agents": "agent"}
_TAXONOMY = ".taxonomy.md"
_INDEX = "index.md"
_DAILY_KINDS = ("episode", "atomic_fact", "foresight", "agent_case")


def _children(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []


class MarkdownScopePurger:
    """Markdown side of a purge.

    Args:
        root: The memory root.
        owner_dirs: Owner-directory store (today's-counter restore, and the
            ``/delete`` trash folders it keeps).
    """

    def __init__(self, root: MemoryRoot, owner_dirs: MarkdownOwnerDirStore) -> None:
        self._root = root
        self._owner_dirs = owner_dirs

    # ── scope layout ─────────────────────────────────────────────────────

    def _project_dirs(self, scope: PurgeScope) -> list[Path]:
        """The scope's existing ``<app_dir>/<project_dir>`` directories.

        Root-level system directories are never an app here, even when an
        app id names one (``.tmp``): their contents are trash and the index.
        """
        root = self._root.root
        if scope.kind == "space":
            apps = [p for p in _children(root) if p.name not in _SYSTEM_DIRS]
        elif scope.app_dir in _SYSTEM_DIRS:
            return []
        else:
            apps = [root / str(scope.app_dir)]
        if scope.kind == "project":
            project = apps[0] / str(scope.project_dir)
            return [project] if project.is_dir() else []
        return [project for app in apps for project in _children(app)]

    def _trash(self, job_id: str) -> Path:
        return self._root.tmp_dir / f"{_TRASH_PREFIX}{job_id}"

    def _delete_trash_of(self, scope: PurgeScope) -> list[tuple[Path, Scope]]:
        """``/delete`` trash folders holding an owner inside ``scope``."""
        return [
            (trash, owner)
            for trash, owner in self._owner_dirs.trash_folders()
            if owner is not None and scope.covers(owner.app_id, owner.project_id)
        ]

    # ── counting ─────────────────────────────────────────────────────────

    async def count(self, scope: PurgeScope) -> dict[str, int]:
        """Memory entries per kind (not today's empty counter files, not
        ``.taxonomy.md``); a profile, a skill and a document count one each."""
        counts: Counter[str] = Counter(
            {k: 0 for k in (*_DAILY_KINDS, "user_profile", "agent_skill")}
        )
        counts["knowledge_document"] = 0
        for project in self._project_dirs(scope):
            for path in sorted(project.rglob("*.md")):
                rel = path.relative_to(project)
                if rel.parts[0] == "knowledge":
                    counts["knowledge_document"] += path.name == _INDEX
                elif rel.parts[0] not in _OWNER_TYPES:
                    continue
                elif path.name == UserProfileFrontmatter.PROFILE_FILENAME:
                    counts["user_profile"] += 1
                elif path.name == AgentSkillFrontmatter.SKILL_MAIN_FILENAME:
                    counts["agent_skill"] += 1
                elif (kind := path.stem.rsplit("-", 3)[0]) in _DAILY_KINDS:
                    counts[kind] += len((await MarkdownReader.read(path)).entries)
        return dict(counts)

    # ── step 2 ───────────────────────────────────────────────────────────

    async def move_out(self, scope: PurgeScope, job_id: str) -> int:
        """Move the scope's memory into this job's trash; entries moved."""
        moved = sum((await self.count(scope)).values())
        trash = self._trash(job_id)
        for project in self._project_dirs(scope):
            target = trash / project.relative_to(self._root.root)
            for name in (*_OWNER_TYPES, "knowledge"):
                source = project / name
                if not source.is_dir():
                    continue
                if name != "knowledge":
                    await self._rename(source, target / name)
                    continue
                for item in sorted(source.iterdir()):
                    if item.name != _TAXONOMY:
                        await self._rename(item, target / name / item.name)
        await afailpoint("purge.after_step2b")
        return moved

    async def restore_counters(self, scope: PurgeScope, job_id: str) -> int:
        """Put today's counters back from this job's trash and the scope's
        ``/delete`` trash folders; owners restored."""
        owners: list[tuple[Scope, Path]] = [
            (owner, trash / "owner") for trash, owner in self._delete_trash_of(scope)
        ]
        trash = self._trash(job_id)
        for app in _children(trash):
            for project in _children(app):
                for folder, owner_type in _OWNER_TYPES.items():
                    for moved in _children(project / folder):
                        owner = Scope(
                            app_id=app_id_from_dir(app.name),
                            project_id=project_id_from_dir(project.name),
                            owner_id=moved.name,
                            owner_type=owner_type,  # type: ignore[arg-type]
                        )
                        owners.append((owner, moved))
        for owner, moved in owners:
            await self._owner_dirs.restore_today_counters(owner, moved)
        return len(owners)

    async def empty(self, scope: PurgeScope, job_id: str) -> int:
        """Remove this job's trash, the scope's ``/delete`` trash and the
        directories the move left empty; trash folders removed."""
        folders = [self._trash(job_id), *(t for t, _ in self._delete_trash_of(scope))]
        removed = 0
        for folder in folders:
            if folder.exists():
                await anyio.to_thread.run_sync(shutil.rmtree, folder)
                removed += 1
        for project in self._project_dirs(scope):
            for name in (*_OWNER_TYPES, "knowledge"):
                _prune_empty(project / name)
            _prune_empty(project, recurse=False)
            _prune_empty(project.parent, recurse=False)
        return removed

    async def sweep_inactive(self, active_job_id: str | None) -> int:
        """Remove ``.tmp/purge-*`` folders of jobs that no longer run (startup)."""
        tmp = self._root.tmp_dir
        if not tmp.is_dir():
            return 0
        removed = 0
        for folder in sorted(tmp.glob(f"{_TRASH_PREFIX}*")):
            if folder.name == f"{_TRASH_PREFIX}{active_job_id}":
                continue
            await anyio.to_thread.run_sync(shutil.rmtree, folder)
            logger.info("purge_trash_swept", trash=folder.name)
            removed += 1
        return removed

    @staticmethod
    async def _rename(source: Path, target: Path) -> None:
        await anyio.Path(target.parent).mkdir(parents=True, exist_ok=True)
        await anyio.Path(source).rename(target)


def _prune_empty(path: Path, *, recurse: bool = True) -> None:
    """Remove ``path`` if it is (or, recursing, has become) an empty tree."""
    if not path.is_dir():
        return
    if recurse:
        for child in path.iterdir():
            if child.is_dir():
                _prune_empty(child)
    if not any(path.iterdir()):
        path.rmdir()
