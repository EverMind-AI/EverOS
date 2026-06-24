"""Base business writer for daily-log markdown files.

Daily-log files (memcell / episode / case / atomic_fact / foresight)
share three things:

    * scope (user-track or agent-track, derived from the schema)
    * filename pattern: ``<FILE_PREFIX>-<YYYY-MM-DD>.md`` under
      ``<scope_root>/<scope_id>/<DIR_NAME>/``
    * entry id pattern: ``<ENTRY_ID_PREFIX>_<YYYYMMDD>_<NNN>``

:class:`BaseDailyWriter` factors out **path resolution + entry-id
construction + today's date default**, leaving frontmatter field
maintenance (e.g. ``entry_count`` / ``last_appended_at``) to concrete
business subclasses.

Subclass usage::

    class _MemcellWriter(BaseDailyWriter):
        schema = UserMemcellDailyFrontmatter

    writer = _MemcellWriter(layout)
    eid = writer.append("u_jason", body="...")
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar

import anyio

from everos.component.utils.datetime import today_with_timezone
from everos.core.persistence import (
    BaseFrontmatter,
    EntryId,
    MarkdownReader,
    MarkdownWriter,
    MemoryRoot,
    render_structured_entry,
)


class BaseDailyWriter:
    """Append a new entry to today's (or a given date's) daily-log file.

    Subclasses bind a single :class:`BaseFrontmatter` subclass via the
    ``schema`` ClassVar. The schema must declare ``SCOPE_DIR``,
    ``ENTRY_ID_PREFIX``, ``DIR_NAME``, and ``FILE_PREFIX`` —
    ``SCOPE_DIR`` is provided by inheriting :class:`UserScopedFrontmatter`
    or :class:`AgentScopedFrontmatter` (or by a custom scope mixin).

    Path resolution is driven entirely by the schema's ClassVars and
    the given :class:`MemoryRoot` — write, read, and addressing for a
    single record kind all live in this writer + its reader twin, no
    separate layout layer.
    """

    schema: ClassVar[type[BaseFrontmatter]]  # subclass must declare

    def __init__(
        self,
        root: MemoryRoot,
        *,
        writer: MarkdownWriter | None = None,
    ) -> None:
        schema = getattr(type(self), "schema", None)
        if schema is None:
            raise TypeError(
                f"{type(self).__name__} must declare a class-level ``schema`` attribute"
            )
        for attr in ("SCOPE_DIR", "ENTRY_ID_PREFIX", "DIR_NAME", "FILE_PREFIX"):
            if not getattr(schema, attr, None):
                raise TypeError(f"{schema.__name__} missing ClassVar {attr!r}")
        self._root = root
        self._writer = writer or MarkdownWriter(root)

    # ── Public API ────────────────────────────────────────────────────────

    async def append_entry(
        self,
        scope_id: str,
        *,
        inline: Mapping[str, object],
        sections: Mapping[str, str],
        date: _dt.date | None = None,
        app_id: str = "default",
        project_id: str = "default",
    ) -> EntryId:
        """Append a single rendered entry; return the freshly minted ``EntryId``.

        Unifies the per-schema ``append_entry`` previously duplicated across
        :class:`AtomicFactWriter` / :class:`ForesightWriter` /
        :class:`EpisodeWriter` / :class:`AgentCaseWriter`. The whole flow
        (read ``entry_count``, allocate ``EntryId``, render entry body,
        update frontmatter, atomic write) runs inside one per-path lock,
        eliminating the read-modify-write race that previously allowed
        concurrent callers to silently overwrite each other's appends.

        Args:
            scope_id: ``user_id`` or ``agent_id`` (matches the schema's
                scope flavour).
            inline: Inline metadata (``**key**: value`` lines under the
                H2 header).
            sections: ``{title: body}`` blocks rendered as ``### Title`` +
                body text.
            date: Date bucket — defaults to today in the configured TZ.

        Returns:
            The :class:`EntryId` assigned to the new entry. Caller can
            use it to write downstream state (sqlite row, lance index).
        """
        eids = await self.append_entries(
            scope_id,
            [(inline, sections)],
            date=date,
            app_id=app_id,
            project_id=project_id,
        )
        return eids[0]

    async def append_entries(
        self,
        scope_id: str,
        items: Sequence[tuple[Mapping[str, object], Mapping[str, str]]],
        *,
        date: _dt.date | None = None,
        app_id: str = "default",
        project_id: str = "default",
    ) -> list[EntryId]:
        """Append ``N`` rendered entries in one locked read-modify-write cycle.

        Compared with looping :meth:`append_entry` ``N`` times, this:

        * Performs one file read + one file write instead of ``N`` of each.
        * Holds the per-path lock for one short critical section.
        * Updates ``frontmatter`` (``entry_count`` / ``last_appended_at``)
          once at the end (no intermediate flapping).

        Order in ``items`` is the order entries land in the file (and the
        order ``EntryId``s are allocated). Empty ``items`` is a no-op
        that returns ``[]`` without touching the file.

        Args:
            scope_id: Subject scope (user / agent id).
            items: Sequence of ``(inline, sections)`` pairs.
            date: Date bucket — defaults to today in the configured TZ.

        Returns:
            ``N`` :class:`EntryId`s in the same order as ``items``.
        """
        bucket = date or today_with_timezone()
        path = self._resolve_path(scope_id, bucket, app_id, project_id)
        if not items:
            return []

        async with self._writer.lock_for(path):
            base_count = await self._current_count(path)
            eids = [
                EntryId.next_for(self.schema.ENTRY_ID_PREFIX, bucket, base_count + i)
                for i in range(len(items))
            ]
            rendered = [
                (
                    render_structured_entry(
                        header=eid.format(),
                        inline=inline,
                        sections=sections,
                    ),
                    eid,
                )
                for eid, (inline, sections) in zip(eids, items, strict=True)
            ]
            frontmatter_updates = self._frontmatter_updates(
                scope_id, bucket, next_count=base_count + len(items)
            )
            await self._writer._append_entries_unlocked(  # noqa: SLF001
                path,
                rendered,
                frontmatter_updates=frontmatter_updates,
            )
            return eids

    async def append(
        self,
        scope_id: str,
        entry_body: str,
        *,
        date: _dt.date | None = None,
        frontmatter_updates: Mapping[str, Any] | None = None,
        app_id: str = "default",
        project_id: str = "default",
    ) -> EntryId:
        """Append a pre-rendered ``entry_body`` to the daily-log file.

        Kept for back-compat with callers that hand in fully rendered
        bodies (rare — most callers should use :meth:`append_entry` and
        let this class do the rendering). The whole sequence (read
        ``entry_count``, allocate eid, write) runs inside the per-path
        lock.

        Args:
            scope_id: ``user_id`` or ``agent_id`` (matches the schema's
                scope flavour).
            entry_body: Content placed between the entry markers.
            date: Date bucket — defaults to today in the configured TZ.
            frontmatter_updates: Optional fields to merge into the file's
                frontmatter (e.g. ``entry_count`` / ``last_appended_at``).
                When ``None``, the subclass hook
                :meth:`_frontmatter_updates` is consulted to build
                default updates.

        Returns:
            The :class:`EntryId` assigned to the new entry.
        """
        bucket = date or today_with_timezone()
        path = self._resolve_path(scope_id, bucket, app_id, project_id)

        async with self._writer.lock_for(path):
            count = await self._current_count(path)
            eid = EntryId.next_for(self.schema.ENTRY_ID_PREFIX, bucket, count)

            # Subclass hook: derive defaults if caller passes nothing.
            if frontmatter_updates is None:
                frontmatter_updates = self._frontmatter_updates(
                    scope_id, bucket, next_count=count + 1
                )

            await self._writer._append_entries_unlocked(  # noqa: SLF001
                path,
                [(entry_body, eid)],
                frontmatter_updates=frontmatter_updates,
            )
            return eid

    async def patch_frontmatter(self, path: Path, updates: Mapping[str, Any]) -> None:
        """Merge ``updates`` into the frontmatter of an existing daily-log file.

        Delegates to the underlying :class:`MarkdownWriter` so that callers
        do not need to reach through the private ``_writer`` attribute.

        Args:
            path: Target markdown file (must exist).
            updates: Mapping of frontmatter keys to merge.
        """
        await self._writer.patch_frontmatter(path, updates)

    # ── Hooks (subclass override) ─────────────────────────────────────────

    async def _current_count(self, path: Path) -> int:
        """Return the current entry count for the file.

        Default: number of ``<!-- entry:... -->`` blocks already present.
        Subclasses may override to read a frontmatter field (e.g.
        ``entry_count``) when they trust that field over a marker scan.
        """
        if not await anyio.Path(path).is_file():
            return 0
        parsed = await MarkdownReader.read(path)
        return len(parsed.entries)

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        """Build the per-append frontmatter dict (subclass override).

        Called only when :meth:`append`'s ``frontmatter_updates`` is
        ``None``. Default returns ``None`` (no frontmatter mutation).
        Concrete business subclasses override to maintain fields like
        ``id`` / ``entry_count`` / ``last_appended_at`` automatically,
        so callers don't repeat themselves on every append.
        """
        return None

    # ── Path API ──────────────────────────────────────────────────────────

    def path_for(
        self,
        scope_id: str,
        date: _dt.date | None = None,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Return the daily-log path for ``scope_id`` on ``date`` (today default).

        Public counterpart of :meth:`_resolve_path` — callers (services,
        scripts) should use this rather than poking at private attrs.
        """
        return self._resolve_path(
            scope_id, date or today_with_timezone(), app_id, project_id
        )

    # ── Internals ─────────────────────────────────────────────────────────

    def _resolve_path(
        self, scope_id: str, date: _dt.date, app_id: str, project_id: str
    ) -> Path:
        """Build the daily-log path for ``scope_id`` on ``date``."""
        # SCOPE_DIR ("users" / "agents") names the matching MemoryRoot method,
        # which prepends the <app>/<project> business prefix.
        scope_dir = getattr(self._root, f"{self.schema.SCOPE_DIR}_dir")
        return (
            scope_dir(app_id, project_id)
            / scope_id
            / self.schema.DIR_NAME
            / f"{self.schema.FILE_PREFIX}-{date.isoformat()}.md"
        )
