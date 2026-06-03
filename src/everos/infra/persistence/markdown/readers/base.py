"""Base business reader for daily-log markdown files.

Symmetric to :class:`BaseDailyWriter`: reads the daily-log file for
a given ``(scope_id, date)``, locates entries by id within it, and
optionally upgrades them to :class:`StructuredEntry` so service-layer
callers don't have to re-do that plumbing each time.

Subclass usage::

    class _MemcellReader(BaseDailyReader):
        schema = UserMemcellDailyFrontmatter

    reader = _MemcellReader(root)
    parsed = reader.read_for("u_jason")               # today's file
    entry = reader.find_entry("u_jason", "umc_20260422_0001")
    structured = reader.find_structured("u_jason", entry.id)

The reader does **not** typed-parse the file's frontmatter dict — the
schema is used only for path resolution (matching what the appender
writes). Frontmatter validation belongs to higher-level callers that
know the business rules.

Path resolution is identical to :class:`BaseDailyWriter` (same
``SCOPE_DIR`` / ``DIR_NAME`` / ``FILE_PREFIX`` ClassVars), so a
reader and writer bound to the same schema agree on every path.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import ClassVar

import anyio

from everos.component.utils.datetime import today_with_timezone
from everos.core.persistence import (
    BaseFrontmatter,
    Entry,
    EntryId,
    MarkdownReader,
    MemoryRoot,
    ParsedMarkdown,
    StructuredEntry,
    find_entry,
)


class BaseDailyReader:
    """Single-record reader for daily-log markdown files.

    Subclasses bind a :class:`BaseFrontmatter` subclass via the
    ``schema`` ClassVar. The schema must declare ``SCOPE_DIR``,
    ``DIR_NAME``, and ``FILE_PREFIX`` (same set the appender uses); no
    ``ENTRY_ID_PREFIX`` requirement here because the reader takes the
    entry id from the caller, not the schema.
    """

    schema: ClassVar[type[BaseFrontmatter]]  # subclass must declare

    def __init__(self, root: MemoryRoot) -> None:
        schema = getattr(type(self), "schema", None)
        if schema is None:
            raise TypeError(
                f"{type(self).__name__} must declare a class-level ``schema`` attribute"
            )
        for attr in ("SCOPE_DIR", "DIR_NAME", "FILE_PREFIX"):
            if not getattr(schema, attr, None):
                raise TypeError(f"{schema.__name__} missing ClassVar {attr!r}")
        self._root = root

    # ── Public API ────────────────────────────────────────────────────────

    async def read_for(
        self,
        scope_id: str,
        date: _dt.date | None = None,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> ParsedMarkdown | None:
        """Read the daily-log file for ``(scope_id, date)``.

        Args:
            scope_id: ``user_id`` or ``agent_id``.
            date: Date bucket — defaults to today in the configured TZ.
            app_id: App scope segment (defaults to the ``"default"`` space).
            project_id: Project scope segment (defaults to ``"default"``).

        Returns:
            :class:`ParsedMarkdown` (frontmatter dict + body + entries),
            or ``None`` when the file does not exist on disk. ``None``
            avoids forcing every caller to wrap reads in try/except —
            "no file yet" is a normal early state.
        """
        path = self._resolve_path(
            scope_id, date or today_with_timezone(), app_id, project_id
        )
        if not await anyio.Path(path).is_file():
            return None
        return await MarkdownReader.read(path)

    async def find_entry(
        self,
        scope_id: str,
        entry_id: str | EntryId,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Entry | None:
        """Locate the entry with ``entry_id`` inside its daily-log file.

        The date bucket is taken from the entry id (an :class:`EntryId`
        encodes its own date), so the caller doesn't pass a date.
        Returns ``None`` if either the file or the entry is missing.
        """
        eid = entry_id if isinstance(entry_id, EntryId) else EntryId.parse(entry_id)
        eid_str = eid.format()
        parsed = await self.read_for(
            scope_id, eid.date, app_id=app_id, project_id=project_id
        )
        if parsed is None:
            return None
        return find_entry(parsed.body, eid_str)

    async def find_structured(
        self,
        scope_id: str,
        entry_id: str | EntryId,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> StructuredEntry | None:
        """Locate the entry and parse its body as audit-form data.

        Sugar over :meth:`find_entry` + :meth:`Entry.as_structured`.
        Returns ``None`` if the entry is missing.
        """
        entry = await self.find_entry(
            scope_id, entry_id, app_id=app_id, project_id=project_id
        )
        if entry is None:
            return None
        return entry.as_structured()

    def path_for(
        self,
        scope_id: str,
        date: _dt.date | None = None,
        *,
        app_id: str = "default",
        project_id: str = "default",
    ) -> Path:
        """Return the daily-log path for ``scope_id`` on ``date`` (today default).

        Public counterpart of :meth:`_resolve_path` — symmetric with
        :meth:`BaseDailyWriter.path_for`. Does not check existence.
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
