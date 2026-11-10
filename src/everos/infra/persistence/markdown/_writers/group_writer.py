"""Group writers: ``group.md`` (meta + roster log) and the group daily logs.

``group.md`` is append-only below its frontmatter: a roster change is a new
entry, never an edit of an old one (docs/group-memory.md, section 4.3). The
frontmatter is the group's meta and is rewritten when the meta changes. Entry
ids are ``gr_<YYYYMMDD>_<seq>``: the date is the event's own UTC date and
``seq`` counts the file's entries of that date, so a backdated event minted
today still sorts under the day it happened.

The daily-log writers are :class:`BaseDailyWriter` over the group schemas; only
their frontmatter differs from the user kinds (``group_id`` / ``track:
group``).
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import anyio

from everos.core.utils.datetime import get_now_with_timezone, to_iso_format
from everos.infra.persistence.memory_root import MemoryRoot

from .._base import (
    Entry,
    EntryId,
    MarkdownReader,
    MarkdownWriter,
    render_structured_entry,
)
from .._mds import (
    GroupEpisodeDailyFrontmatter,
    GroupFrontmatter,
    GroupImageMemoryDailyFrontmatter,
    GroupSceneDailyFrontmatter,
)
from .base import BaseDailyWriter


class GroupWriter:
    """``groups/<group_id>/group.md``.

    Args:
        root: Memory root the groups live under.
        writer: Shared markdown writer (its per-path locks); one is built when
            omitted.
    """

    def __init__(
        self, root: MemoryRoot, *, writer: MarkdownWriter | None = None
    ) -> None:
        self._root = root
        self._writer = writer or MarkdownWriter(root)

    def path_for(
        self, group_id: str, *, app_id: str = "default", project_id: str = "default"
    ) -> Path:
        """Where the group's ``group.md`` lives."""
        return (
            self._root.groups_dir(app_id, project_id)
            / group_id
            / GroupFrontmatter.PROFILE_FILENAME
        )

    async def read(self, path: Path) -> tuple[dict[str, Any], list[Entry]] | None:
        """The frontmatter and the log entries, or ``None`` for no file."""
        if not await anyio.Path(path).is_file():
            return None
        parsed = await MarkdownReader.read(path)
        return dict(parsed.frontmatter), list(parsed.entries)

    async def append(
        self,
        path: Path,
        *,
        frontmatter: Mapping[str, Any],
        entries: Sequence[tuple[_dt.date, Mapping[str, object]]],
    ) -> list[EntryId]:
        """Merge ``frontmatter`` and append one entry per ``(date, inline)``.

        Ids are allocated under the file's lock, from the entries already in
        the file, so two writers can never mint the same id.

        Returns:
            The ids of the appended entries, in order.
        """
        async with self._writer.locked(path):
            existing = await self.read(path)
            seqs: dict[_dt.date, int] = {}
            for entry in existing[1] if existing else []:
                try:
                    parsed = EntryId.parse(entry.id)
                except ValueError:
                    continue
                seqs[parsed.date] = max(seqs.get(parsed.date, 0), parsed.seq)
            ids: list[EntryId] = []
            rendered: list[tuple[str, EntryId]] = []
            for day, inline in entries:
                seqs[day] = seqs.get(day, 0) + 1
                eid = EntryId(
                    prefix=GroupFrontmatter.ENTRY_ID_PREFIX, date=day, seq=seqs[day]
                )
                ids.append(eid)
                rendered.append((render_structured_entry(inline=inline), eid))
            count = (len(existing[1]) if existing else 0) + len(rendered)
            await self._writer._append_entries_unlocked(
                path,
                rendered,
                frontmatter_updates={**frontmatter, "entry_count": count},
            )
            return ids


class GroupEpisodeWriter(BaseDailyWriter):
    """Daily-log writer for group episodes (``gep_<YYYYMMDD>_<seq>``)."""

    schema = GroupEpisodeDailyFrontmatter

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"group_episode_log_{scope_id}_{date.isoformat()}",
            "type": "group_episode_daily",
            "file_type": "group_episode_daily",
            "schema_version": 1,
            "group_id": scope_id,
            "track": "group",
            "date": date.isoformat(),
            "entry_count": next_count,
            "last_appended_at": to_iso_format(get_now_with_timezone()),
        }

    async def _current_count(self, path: Path) -> int:
        return await self._count_from_frontmatter(path)


class GroupImageMemoryWriter(BaseDailyWriter):
    """Daily-log writer for group picture memories (``gim_<YYYYMMDD>_<seq>``)."""

    schema = GroupImageMemoryDailyFrontmatter

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"group_image_log_{scope_id}_{date.isoformat()}",
            "type": "group_image_memory_daily",
            "file_type": "group_image_memory_daily",
            "schema_version": 1,
            "group_id": scope_id,
            "track": "group",
            "date": date.isoformat(),
            "entry_count": next_count,
            "last_appended_at": to_iso_format(get_now_with_timezone()),
        }

    async def _current_count(self, path: Path) -> int:
        return await self._count_from_frontmatter(path)


class GroupSceneWriter(BaseDailyWriter):
    """Daily-log writer for group scene versions (``gsc_<YYYYMMDD>_<seq>``)."""

    schema = GroupSceneDailyFrontmatter

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        return {
            "id": f"group_scene_log_{scope_id}_{date.isoformat()}",
            "type": "group_scene_daily",
            "file_type": "group_scene_daily",
            "schema_version": 1,
            "group_id": scope_id,
            "track": "group",
            "date": date.isoformat(),
            "entry_count": next_count,
            "last_appended_at": to_iso_format(get_now_with_timezone()),
        }

    async def _current_count(self, path: Path) -> int:
        return await self._count_from_frontmatter(path)
