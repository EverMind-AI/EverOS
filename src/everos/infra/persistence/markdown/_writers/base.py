"""Base business writer for daily-log markdown files.

Daily-log files (memcell / episode / case / image memory) share three
things:

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
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, ClassVar, Literal

import anyio

from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import (
    get_now_with_timezone,
    to_iso_format,
    today_with_timezone,
)
from everos.infra.persistence.markdown._base.entries import (
    Entry,
    EntryId,
    render_structured_entry,
    split_entries,
)
from everos.infra.persistence.markdown._base.frontmatter import (
    BaseFrontmatter,
    parse_frontmatter,
)
from everos.infra.persistence.markdown._base.reader import MarkdownReader
from everos.infra.persistence.markdown._base.writer import MarkdownWriter
from everos.infra.persistence.memory_root import MemoryRoot

logger = get_logger(__name__)


class BaseDailyWriter:
    """Append a new entry to today's (or a given date's) daily-log file.

    Subclasses bind a single :class:`BaseFrontmatter` subclass via the
    ``schema`` attribute. The schema must declare ``SCOPE_DIR``,
    ``ENTRY_ID_PREFIX``, ``DIR_NAME``, and ``FILE_PREFIX`` —
    ``SCOPE_DIR`` is provided by inheriting :class:`UserScopedFrontmatter`
    or :class:`AgentScopedFrontmatter` (or by a custom scope mixin).

    Path resolution is driven entirely by the schema's ClassVars and
    the given :class:`MemoryRoot` — write, read, and addressing for a
    single record kind all live in this writer + its reader twin, no
    separate layout layer.
    """

    schema: type[BaseFrontmatter]
    """Bound by the subclass, or per instance before ``__init__`` runs
    (:class:`OwnedDailyWriter`)."""

    def __init__(
        self,
        root: MemoryRoot,
        *,
        writer: MarkdownWriter | None = None,
    ) -> None:
        schema = getattr(self, "schema", None)
        if schema is None:
            raise TypeError(f"{type(self).__name__} must declare ``schema``")
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
        the daily-log writers (:class:`EpisodeWriter`,
        :class:`AgentCaseWriter`, ...). The whole flow
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

        async with self._writer.locked(path):
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
            await self._writer._append_entries_unlocked(
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

        async with self._writer.locked(path):
            count = await self._current_count(path)
            eid = EntryId.next_for(self.schema.ENTRY_ID_PREFIX, bucket, count)

            # Subclass hook: derive defaults if caller passes nothing.
            if frontmatter_updates is None:
                frontmatter_updates = self._frontmatter_updates(
                    scope_id, bucket, next_count=count + 1
                )

            await self._writer._append_entries_unlocked(
                path,
                [(entry_body, eid)],
                frontmatter_updates=frontmatter_updates,
            )
            return eid

    async def remove_entries(
        self, path: Path, drop: Callable[[Entry], bool]
    ) -> list[str]:
        """Rewrite ``path`` without the entries ``drop`` selects; return their ids.

        Under the file's lock, atomically. ``entry_count`` is kept at least
        the highest sequence the file held before, and a file left with no
        entries stays (its frontmatter carries the count): the next append
        must not reuse a removed id. A missing file removes nothing; an
        entry ``drop`` raises on (an unparsable, hand-edited one) is kept.
        """
        async with self._writer.locked(path):
            try:
                raw = await MarkdownReader.read_text(path)
            except FileNotFoundError:
                return []
            meta, body = parse_frontmatter(raw)
            entries = split_entries(body)
            gone = [entry for entry in entries if _selected(drop, entry)]
            if not gone:
                return []
            highest = max((_seq(entry.id) for entry in entries), default=0)
            count = meta.get("entry_count", 0)
            if not isinstance(count, int) or isinstance(count, bool):
                count = 0
            meta["entry_count"] = max(count, highest)
            for entry in reversed(gone):  # offsets stay valid back to front
                end = entry.end + (body[entry.end : entry.end + 1] == "\n")
                body = body[: entry.start] + body[end:]
            await self._writer.write_markdown(path, frontmatter=meta, body=body)
            return [entry.id for entry in gone]

    async def take_entries(
        self, path: Path, entry_ids: Collection[str]
    ) -> list[tuple[str, str, str | None]]:
        """Remove ``entry_ids`` from ``path``; each as ``(id, body, mark)``.

        :meth:`remove_entries` that hands back what it removed, so the caller
        can keep it elsewhere and :meth:`put_back` it later: ``body`` is the
        text between the entry's markers exactly as the file held it, and
        ``mark`` the entry's own ``deprecated_entries`` value (dropped from the
        frontmatter with it). ``entry_count`` never falls, so no later append
        reuses a removed id. A missing file or id removes nothing.
        """
        wanted = set(entry_ids)
        async with self._writer.locked(path):
            try:
                raw = await MarkdownReader.read_text(path)
            except FileNotFoundError:
                return []
            meta, body = parse_frontmatter(raw)
            entries = split_entries(body)
            gone = [entry for entry in entries if entry.id in wanted]
            if not gone:
                return []
            highest = max((_seq(entry.id) for entry in entries), default=0)
            count = meta.get("entry_count", 0)
            if not isinstance(count, int) or isinstance(count, bool):
                count = 0
            meta["entry_count"] = max(count, highest)
            marks = meta.get("deprecated_entries")
            marks = dict(marks) if isinstance(marks, dict) else {}
            taken = [
                (entry.id, entry.body, _mark_of(marks.pop(entry.id, None)))
                for entry in gone
            ]
            if "deprecated_entries" in meta:
                meta["deprecated_entries"] = marks
            for entry in reversed(gone):  # offsets stay valid back to front
                end = entry.end + (body[entry.end : entry.end + 1] == "\n")
                body = body[: entry.start] + body[end:]
            await self._writer.write_markdown(path, frontmatter=meta, body=body)
            return taken

    async def put_back(
        self,
        scope_id: str,
        path: Path,
        entries: Sequence[tuple[str, str, str | None]],
    ) -> list[str]:
        """Insert entries :meth:`take_entries` removed, under their own ids.

        Each ``(id, body, mark)`` goes back where its sequence puts it among
        the file's entries, with its mark; an id the file already holds is
        left alone (a repeated undo restores nothing twice). A file that is
        gone is written again with this writer's frontmatter. Returns the ids
        put back.
        """
        if not entries:
            return []
        async with self._writer.locked(path):
            try:
                raw = await MarkdownReader.read_text(path)
            except FileNotFoundError:
                raw = None
            if raw is None:
                day = EntryId.parse(entries[0][0]).date
                meta = dict(
                    self._frontmatter_updates(scope_id, day, next_count=0) or {}
                )
                body = ""
            else:
                meta, body = parse_frontmatter(raw)
            present = {entry.id for entry in split_entries(body)}
            restored: list[str] = []
            marks = meta.get("deprecated_entries")
            marks = dict(marks) if isinstance(marks, dict) else {}
            for entry_id, entry_body, mark in sorted(entries, key=lambda e: _seq(e[0])):
                if entry_id in present:
                    continue
                body = _insert_block(body, entry_id, entry_body)
                present.add(entry_id)
                restored.append(entry_id)
                if mark:
                    marks[entry_id] = mark
            if not restored:
                return []
            if marks:
                meta["deprecated_entries"] = marks
            highest = max((_seq(entry_id) for entry_id in present), default=0)
            count = meta.get("entry_count", 0)
            if not isinstance(count, int) or isinstance(count, bool):
                count = 0
            meta["entry_count"] = max(count, highest)
            await self._writer.write_markdown(path, frontmatter=meta, body=body)
            return restored

    async def drop_deprecations(self, path: Path, keys: Collection[str]) -> bool:
        """Remove ``keys`` from the file's ``deprecated_entries``; whether any was."""
        async with self._writer.locked(path):
            try:
                raw = await MarkdownReader.read_text(path)
            except FileNotFoundError:
                return False
            meta, body = parse_frontmatter(raw)
            marks = meta.get("deprecated_entries")
            if not isinstance(marks, dict) or not marks.keys() & set(keys):
                return False
            meta["deprecated_entries"] = {
                k: v for k, v in marks.items() if k not in keys
            }
            await self._writer.write_markdown(path, frontmatter=meta, body=body)
            return True

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

    async def _count_from_frontmatter(self, path: Path) -> int:
        """``entry_count`` from the frontmatter, never behind the file's ids.

        The next entry id is allocated from this number, so it must not
        fall below the highest sequence already in the file: the file is
        the user's, and a hand-lowered ``entry_count`` would otherwise make
        the next append reuse an id (two entries with one id; the index
        keeps one). A value that is not an integer counts as absent. Both
        cases are logged, since they mean the frontmatter was edited by
        hand.
        """
        if not await anyio.Path(path).is_file():
            return 0
        parsed = await MarkdownReader.read(path)
        highest = 0
        for entry in parsed.entries:
            try:
                highest = max(highest, EntryId.parse(entry.id).seq)
            except ValueError:
                continue  # a foreign marker; not part of this writer's sequence
        raw = parsed.frontmatter.get("entry_count", 0)
        count = raw if isinstance(raw, int) and not isinstance(raw, bool) else 0
        if count != raw or count < highest:
            logger.warning(
                "daily_log_entry_count_behind_file",
                path=str(path),
                entry_count=raw,
                highest_seq=highest,
            )
        return max(count, highest)

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


OwnerKind = Literal["user", "group"]
"""The owners an :class:`OwnedDailyWriter` writes for."""


class OwnedDailyWriter(BaseDailyWriter):
    """A daily log a user or a group owns: one writer for both.

    The two owner types share the chassis and differ only in what the files
    say about their owner: the schema (directory, entry-id prefix, ``type``)
    and the frontmatter's owner key (``user_id`` / ``group_id``) with its
    ``track``. A subclass binds the kind's two schemas and its log name; an
    instance writes one owner type's files, so an index row id
    (``<owner_id>_<entry_id>``) never meets the other owner type's.

    A group's log ids and types are the user's with a ``group_`` prefix:
    ``episode_log_<id>_<date>`` / ``group_episode_log_<id>_<date>``.
    """

    schemas: ClassVar[Mapping[OwnerKind, type[BaseFrontmatter]]]
    log_name: ClassVar[str]

    def __init__(
        self,
        root: MemoryRoot,
        *,
        owner_type: OwnerKind = "user",
        writer: MarkdownWriter | None = None,
    ) -> None:
        self.schema = self.schemas[owner_type]
        self._owner_type: OwnerKind = owner_type
        super().__init__(root, writer=writer)

    def _frontmatter_updates(
        self,
        scope_id: str,
        date: _dt.date,
        *,
        next_count: int,
    ) -> Mapping[str, Any] | None:
        group = self._owner_type == "group"
        kind = self.schema.model_fields["type"].default
        return {
            "id": f"{'group_' if group else ''}{self.log_name}_{scope_id}_"
            f"{date.isoformat()}",
            "type": kind,
            "file_type": kind,
            "schema_version": 1,
            "group_id" if group else "user_id": scope_id,
            "track": self._owner_type,
            "date": date.isoformat(),
            "entry_count": next_count,
            "last_appended_at": to_iso_format(get_now_with_timezone()),
        }

    async def _current_count(self, path: Path) -> int:
        return await self._count_from_frontmatter(path)


def _mark_of(value: object) -> str | None:
    return str(value) if value else None


def _insert_block(body: str, entry_id: str, entry_body: str) -> str:
    """``body`` with one entry block inserted before the first later sequence."""
    block = f"<!-- entry:{entry_id} -->\n{entry_body}\n<!-- /entry:{entry_id} -->\n"
    seq = _seq(entry_id)
    for entry in split_entries(body):
        if _seq(entry.id) > seq:
            return body[: entry.start] + block + body[entry.start :]
    if body and not body.endswith("\n"):
        body += "\n"
    return body + block


def _selected(drop: Callable[[Entry], bool], entry: Entry) -> bool:
    try:
        return drop(entry)
    except Exception:  # a hand-edited entry that does not parse stays
        return False


def _seq(entry_id: str) -> int:
    try:
        return EntryId.parse(entry_id).seq
    except ValueError:
        return 0  # a foreign marker; not part of this writer's sequence
