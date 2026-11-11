"""Markdown adapters for the memory domain's truth ports.

Each ``Markdown*Store`` wraps one Reader / Writer and owns the domain <->
markdown-entry conversion, so the domain hands over domain objects and gets
back a :class:`~everos.memory.ports.MemoryRef`. Paths in refs are relative
to the memory root, the same convention as the index rows' ``md_path``.
"""

from __future__ import annotations

__all__ = [
    "MarkdownAgentCaseStore",
    "MarkdownAgentSkillStore",
    "MarkdownEpisodeStore",
    "MarkdownGroupProfileStore",
    "MarkdownImageMemoryStore",
    "MarkdownMemberProfileStore",
    "MarkdownProfileStore",
    "MarkdownSceneStore",
]

from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import anyio
from everalgo.user_memory import OutputLanguage

from everos.core.locks import LOCKS, owner_key
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import (
    from_iso_format,
    from_timestamp_ms,
    to_iso_format,
    to_timestamp_ms,
)
from everos.infra.persistence.memory_root import MemoryRoot
from everos.infra.persistence.records import member_profile_row_id
from everos.memory.models import (
    FORGOTTEN_MARK,
    AgentCase,
    AgentSkill,
    Episode,
    GroupProfileItem,
    ImageMemory,
    MemberProfile,
    Profile,
    Scene,
)
from everos.memory.ports import (
    MemoryRef,
    NoSourceCheck,
    PendingBatch,
    Scope,
    SourceCheck,
    Stored,
    Withdrawal,
    WithdrawnEntry,
)

from ._base import Entry, EntryId, MarkdownReader, MarkdownWriter
from ._mds import (
    AgentSkillFrontmatter,
    GroupMemberProfileFrontmatter,
    GroupProfileDailyFrontmatter,
    GroupProfilePendingFrontmatter,
    UserProfileFrontmatter,
    group_profile_item_from_entry,
    render_group_profile_entry,
)
from ._readers import AgentSkillReader, ProfileReader
from ._writers import (
    AgentCaseWriter,
    AgentSkillWriter,
    BaseDailyWriter,
    EpisodeWriter,
    GroupEpisodeWriter,
    GroupImageMemoryWriter,
    GroupProfileWriter,
    GroupSceneWriter,
    ImageMemoryWriter,
    ProfileWriter,
)

logger = get_logger(__name__)

type EntryBody = tuple[dict[str, object], dict[str, str]]
"""``(inline, sections)`` of one rendered daily-log entry."""


def owner_lock(scope: Scope) -> str:
    """The ``LOCKS`` key of the owner every markdown write of ``scope`` holds."""
    return owner_key(scope.app_id, scope.project_id, scope.owner_type, scope.owner_id)


class _DailyLogStore[T]:
    """Append / deprecate for one daily-log kind.

    Every write holds the owner's lock (``"owner"`` in ``LOCKS``) and runs its
    ``only_if`` first, inside the lock: a deletion removes the sources, then
    takes the same lock, so a write that passed its check has finished before
    the deletion reads the files, and one that runs later sees the sources
    gone.

    Args:
        root: Memory root the files live under.
        writer: The kind's daily-log writer.
        to_entry: Renders one domain object as ``(inline, sections)``.
    """

    _KIND: ClassVar[str]

    def __init__(
        self,
        root: MemoryRoot,
        writer: BaseDailyWriter,
        to_entry: Callable[[T], EntryBody],
        *,
        group_writer: BaseDailyWriter | None = None,
    ) -> None:
        self._root = root
        self._writer = writer
        self._group_writer = group_writer
        self._to_entry = to_entry

    def _writer_for(self, scope: Scope) -> BaseDailyWriter:
        """The writer of ``scope``'s owner type: its files and its id prefix.

        A group owns only the kinds that declare a group writer (episodes and
        picture memories); every other kind refuses a group scope, so nothing
        can write, say, an agent case under ``groups/``.
        """
        if scope.owner_type != "group":
            return self._writer
        if self._group_writer is None:
            raise ValueError(f"{self._KIND} has no group-owned records")
        return self._group_writer

    async def _append_many(
        self,
        scope: Scope,
        items: Sequence[T],
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        if not items:
            return []
        async with LOCKS.hold("owner", owner_lock(scope)):
            if not await self._sources_alive(scope, only_if, len(items)):
                return []
            eids = await self._writer_for(scope).append_entries(
                scope.owner_id,
                [self._to_entry(item) for item in items],
                app_id=scope.app_id,
                project_id=scope.project_id,
            )
        return [self._ref(scope, eid) for eid in eids]

    async def _deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        by_file: dict[str, dict[str, str]] = defaultdict(dict)
        for md_path, entry_id in entries:
            by_file[md_path][entry_id] = replaced_by
        async with LOCKS.hold("owner", owner_lock(scope)):
            if not await self._sources_alive(scope, only_if, len(entries)):
                return False
            for md_path, replaced in by_file.items():
                await self._writer_for(scope).patch_frontmatter(
                    self._root.root / md_path, {"deprecated_entries": replaced}
                )
        return True

    async def _sources_alive(
        self, scope: Scope, only_if: SourceCheck | NoSourceCheck, count: int
    ) -> bool:
        if isinstance(only_if, NoSourceCheck) or await only_if():
            return True
        logger.info(
            "memory_write_skipped_source_gone",
            kind=self._KIND,
            app_id=scope.app_id,
            project_id=scope.project_id,
            owner_id=scope.owner_id,
            count=count,
        )
        return False

    # ── Withdrawal (a group forget) ──────────────────────────────────────

    async def withdraw(self, scope: Scope, entry_ids: Collection[str]) -> Withdrawal:
        """Take entries out of the owner's files, keeping their exact text.

        Under the owner lock: each entry leaves its file with its own mark,
        and an entry whose mark named one of them -- the version it had
        replaced -- loses that mark and is current again (``released``). The
        returned :class:`Withdrawal` is what :meth:`reinstate` takes to put
        every byte back.
        """
        wanted = set(entry_ids)
        if not wanted:
            return Withdrawal()
        by_path: dict[Path, set[str]] = defaultdict(set)
        for entry_id in wanted:
            path = self._path_of(scope, entry_id)
            if path is not None:
                by_path[path].add(entry_id)
        writer = self._writer_for(scope)
        key = owner_lock(scope)
        taken: list[WithdrawnEntry] = []
        released: list[tuple[str, str, str]] = []
        async with LOCKS.hold("owner", key):
            for path, ids in sorted(by_path.items()):
                md_path = _relative(self._root, path)
                taken += [
                    WithdrawnEntry(entry_id=e, md_path=md_path, text=t, mark=m)
                    for e, t, m in await writer.take_entries(path, ids)
                ]
            gone = {entry.entry_id for entry in taken}
            for path in await self._daily_files(scope):
                marks = await _deprecations(path)
                freed = {
                    k: str(v) for k, v in marks.items() if v in gone and k not in gone
                }
                if freed and await writer.drop_deprecations(path, freed):
                    md_path = _relative(self._root, path)
                    released += [(md_path, k, v) for k, v in sorted(freed.items())]
            if taken:
                LOCKS.bump_delete_count(key)
        return Withdrawal(entries=tuple(taken), released=tuple(released))

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        """Put a :meth:`withdraw`'s entries back under their ids, marks included.

        An entry the file holds again already is left alone, so a repeated
        undo restores nothing twice; a released mark comes back only while
        its entry has none of its own.
        """
        by_path: dict[str, list[WithdrawnEntry]] = defaultdict(list)
        for entry in withdrawal.entries:
            by_path[entry.md_path].append(entry)
        writer = self._writer_for(scope)
        refs: list[MemoryRef] = []
        async with LOCKS.hold("owner", owner_lock(scope)):
            for md_path, entries in sorted(by_path.items()):
                put = await writer.put_back(
                    scope.owner_id,
                    self._root.root / md_path,
                    [(e.entry_id, e.text, e.mark) for e in entries],
                )
                refs += [
                    MemoryRef(
                        id=f"{scope.owner_id}_{entry_id}",
                        entry_id=entry_id,
                        md_path=md_path,
                    )
                    for entry_id in put
                ]
            restore: dict[str, dict[str, str]] = defaultdict(dict)
            for md_path, entry_id, mark in withdrawal.released:
                restore[md_path][entry_id] = mark
            for md_path, marks in restore.items():
                path = self._root.root / md_path
                current = await _deprecations(path)
                wanted = {k: v for k, v in marks.items() if not current.get(k)}
                if wanted and await anyio.Path(path).is_file():
                    await writer.patch_frontmatter(path, {"deprecated_entries": wanted})
        return refs

    async def _entries_where(
        self, scope: Scope, needles: Sequence[str], keep: Callable[[Entry], bool]
    ) -> list[tuple[str, Entry, str | None]]:
        """``(md_path, entry, mark)`` of the entries ``keep`` selects.

        Only the files whose text holds one of ``needles`` are parsed.
        """
        found: list[tuple[str, Entry, str | None]] = []
        for path in await self._files_mentioning(scope, *needles):
            try:
                parsed = await MarkdownReader.read(path)
            except FileNotFoundError:
                continue
            marks = _marks(parsed.frontmatter)
            md_path = _relative(self._root, path)
            for entry in parsed.entries:
                try:
                    selected = keep(entry)
                except Exception:  # a hand-edited entry that does not parse
                    continue
                if selected:
                    found.append((md_path, entry, marks.get(entry.id)))
        return found

    # ── Deletion ──────────────────────────────────────────────────────────

    async def remove_session(self, scope: Scope, session_id: str) -> list[MemoryRef]:
        """Remove the owner's entries of ``session_id`` from every file of this kind.

        The caller deleted the session's memcells first. Taking the owner lock
        once then waits out every write that passed its source check before
        that; any later one finds its memcell gone. So the files can be
        scanned without the lock, and each match is re-read and rewritten
        under it. An entry matches on its parsed ``session_id`` field, never
        on a text prefix (``s1`` does not match ``s10``).
        """
        async with LOCKS.hold("owner", owner_lock(scope)):
            pass
        needle = f"**session_id**: {session_id}"
        removed: list[MemoryRef] = []
        for path in await self._files_mentioning(scope, needle):
            removed += await self._rewrite(
                scope,
                path,
                lambda e: e.as_structured().inline.get("session_id") == session_id,
            )
        return removed

    async def _rewrite(
        self, scope: Scope, path: Path, drop: Callable[[Entry], bool]
    ) -> list[MemoryRef]:
        """Remove the selected entries of one file under the owner lock."""
        key = owner_lock(scope)
        async with LOCKS.hold("owner", key):
            removed = await self._writer_for(scope).remove_entries(path, drop)
            if removed:
                LOCKS.bump_delete_count(key)
        md_path = _relative(self._root, path)
        return [
            MemoryRef(id=f"{scope.owner_id}_{eid}", entry_id=eid, md_path=md_path)
            for eid in removed
        ]

    async def entries_for_session(
        self, scope: Scope, session_id: str
    ) -> list[tuple[str, str]]:
        """``(md_path, entry_id)`` of the owner's entries that came from ``session_id``.

        Read from markdown rather than the index: a forget is the truth-changing
        operation, and its target set must not depend on whether the cascade has
        caught up. Matches the parsed ``session_id`` inline field, never a text
        prefix (``s1`` must not match ``s10``).
        """
        found: list[tuple[str, str]] = []
        needle = f"**session_id**: {session_id}"
        for path in await self._files_mentioning(scope, needle):
            try:
                parsed = await MarkdownReader.read(path)
            except FileNotFoundError:
                continue
            md_path = _relative(self._root, path)
            for entry in parsed.entries:
                if entry.as_structured().inline.get("session_id") == session_id:
                    found.append((md_path, entry.id))
        return found

    def _folder(self, scope: Scope) -> Path:
        return (
            self._writer_for(scope)
            .path_for(scope.owner_id, app_id=scope.app_id, project_id=scope.project_id)
            .parent
        )

    async def _daily_files(self, scope: Scope) -> list[Path]:
        folder = anyio.Path(self._folder(scope))
        if not await folder.is_dir():
            return []
        return sorted([Path(str(f)) async for f in folder.glob("*.md")])

    async def _files_mentioning(self, scope: Scope, *needles: str) -> list[Path]:
        """The owner's files of this kind whose text contains any of ``needles``."""
        hits: list[Path] = []
        for path in await self._daily_files(scope):
            try:
                text = await MarkdownReader.read_text(path)
            except FileNotFoundError:
                continue
            if any(needle in text for needle in needles):
                hits.append(path)
        return hits

    def _path_of(self, scope: Scope, entry_id: str) -> Path | None:
        try:
            day = EntryId.parse(entry_id).date
        except ValueError:
            return None  # not an id this store ever minted
        return self._writer_for(scope).path_for(
            scope.owner_id, day, app_id=scope.app_id, project_id=scope.project_id
        )

    def _ref(self, scope: Scope, eid: EntryId) -> MemoryRef:
        path = self._writer_for(scope).path_for(
            scope.owner_id,
            eid.date,
            app_id=scope.app_id,
            project_id=scope.project_id,
        )
        entry_id = eid.format()
        return MemoryRef(
            id=f"{scope.owner_id}_{entry_id}",
            entry_id=entry_id,
            md_path=_relative(self._root, path),
        )


class MarkdownEpisodeStore(_DailyLogStore[Episode]):
    """Episode daily logs (``users/<id>/episodes/``, ``groups/<id>/episodes/``).

    Args:
        root: Memory root the files live under.
    """

    _KIND = "episode"

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(
            root,
            EpisodeWriter(root),
            _episode_entry,
            group_writer=GroupEpisodeWriter(root),
        )

    async def append(
        self,
        scope: Scope,
        episode: Episode,
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None:
        refs = await self._append_many(scope, [episode], only_if)
        return refs[0] if refs else None

    async def append_many(
        self,
        scope: Scope,
        episodes: Sequence[Episode],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        return await self._append_many(scope, episodes, only_if)

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        return await self._deprecate(
            scope, entries=entries, replaced_by=replaced_by, only_if=only_if
        )

    async def exists(self, scope: Scope, entry_ids: Collection[str]) -> set[str]:
        return {entry.id for entry in await self._entries(scope, entry_ids)}

    async def read_many(
        self, scope: Scope, entry_ids: Collection[str]
    ) -> list[Stored[Episode]]:
        found: list[Stored[Episode]] = []
        for path, entry in await self._entries_with_paths(scope, entry_ids):
            found.append(
                Stored(
                    ref=MemoryRef(
                        id=f"{scope.owner_id}_{entry.id}",
                        entry_id=entry.id,
                        md_path=_relative(self._root, path),
                    ),
                    scope=scope,
                    item=_episode_from_entry(entry, owner_id=scope.owner_id),
                )
            )
        return found

    async def read_for_segments(
        self, scope: Scope, memcell_ids: Collection[str]
    ) -> list[Stored[Episode]]:
        """The owner's current episodes written from one of ``memcell_ids``.

        Read from markdown (a forget changes the truth and must not wait for
        the index); an entry a supersession or forget mark retired is left out.
        """
        wanted = set(memcell_ids)
        if not wanted:
            return []
        found = await self._entries_where(
            scope,
            [f"**parent_id**: {memcell_id}" for memcell_id in sorted(wanted)],
            lambda e: (
                e.as_structured().inline.get("parent_id") in wanted
                and (e.as_structured().inline.get("parent_type") or "memcell")
                == "memcell"
            ),
        )
        return [
            self._stored(scope, md_path, entry)
            for md_path, entry, mark in found
            if not mark
        ]

    async def read_referring(
        self, scope: Scope, image_id: str
    ) -> list[Stored[Episode]]:
        """The owner's current episodes that refer to the picture ``image_id``."""
        found = await self._entries_where(
            scope,
            [image_id],
            lambda e: image_id in e.body,
        )
        return [
            self._stored(scope, md_path, entry)
            for md_path, entry, mark in found
            if not mark
        ]

    def _stored(self, scope: Scope, md_path: str, entry: Entry) -> Stored[Episode]:
        return Stored(
            ref=MemoryRef(
                id=f"{scope.owner_id}_{entry.id}",
                entry_id=entry.id,
                md_path=md_path,
            ),
            scope=scope,
            item=_episode_from_entry(entry, owner_id=scope.owner_id),
        )

    async def remove(self, scope: Scope, entry_ids: Collection[str]) -> list[MemoryRef]:
        """Remove these entries; each file is rewritten under the owner lock."""
        by_path: dict[Path, set[str]] = defaultdict(set)
        for entry_id in entry_ids:
            path = self._path_of(scope, entry_id)
            if path is not None:
                by_path[path].add(entry_id)
        removed: list[MemoryRef] = []
        for path, ids in sorted(by_path.items()):
            removed += await self._rewrite(scope, path, lambda e, ids=ids: e.id in ids)
        return removed

    async def merged_into(self, scope: Scope, targets: Collection[str]) -> set[str]:
        """The entries the owner's files mark as merged into any of ``targets``.

        Reads every episode file's ``deprecated_entries`` (no lock). One step:
        a caller that wants the closure repeats it with the result.
        """
        wanted = set(targets)
        found: set[str] = set()
        for path in await self._daily_files(scope):
            marks = await _deprecations(path)
            found |= {key for key, merged in marks.items() if merged in wanted}
        return found

    async def deprecations(self, scope: Scope, md_path: str) -> dict[str, str]:
        """One file's supersession marks, keyed by the entry they retire."""
        return await _deprecations(self._root.root / md_path)

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        """Drop ``keys`` from every episode file's ``deprecated_entries``."""
        wanted = set(keys)
        for path in await self._daily_files(scope):
            if not (await _deprecations(path)).keys() & wanted:
                continue
            async with LOCKS.hold("owner", owner_lock(scope)):
                await self._writer_for(scope).drop_deprecations(path, wanted)

    async def _entries(self, scope: Scope, entry_ids: Collection[str]) -> list[Entry]:
        return [entry for _, entry in await self._entries_with_paths(scope, entry_ids)]

    async def _entries_with_paths(
        self, scope: Scope, entry_ids: Collection[str]
    ) -> list[tuple[Path, Entry]]:
        wanted: dict[Path, set[str]] = defaultdict(set)
        for entry_id in entry_ids:
            path = self._path_of(scope, entry_id)
            if path is not None:
                wanted[path].add(entry_id)
        found: list[tuple[Path, Entry]] = []
        for path, ids in wanted.items():
            try:
                parsed = await MarkdownReader.read(path)
            except FileNotFoundError:
                continue
            found += [(path, entry) for entry in parsed.entries if entry.id in ids]
        return found


class MarkdownAgentCaseStore(_DailyLogStore[AgentCase]):
    """Agent-case daily logs (``agents/<id>/.agent_cases/``).

    Args:
        root: Memory root the files live under.
    """

    _KIND = "agent_case"

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root, AgentCaseWriter(root), _agent_case_entry)

    async def append(
        self,
        scope: Scope,
        case: AgentCase,
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None:
        refs = await self._append_many(scope, [case], only_if)
        return refs[0] if refs else None


_IMAGE_READ_CAP = 1_000
"""Entries one undo-of-a-forget reads for one picture. Same bound as a forget
marks, so the two can never disagree about how many occurrences there were."""


class MarkdownImageMemoryStore(_DailyLogStore[ImageMemory]):
    """Image-memory daily logs (``users/<id>/images/``, ``groups/<id>/images/``).

    A picture's memory is written once per occurrence and never edited in
    place: a superseded description is a new entry that points back at the old
    one through ``deprecated_entries``, which the cascade turns into the index
    row's ``deprecated_by`` column. That keeps the file append-only and keeps
    the old reading of the picture recoverable.

    Args:
        root: Memory root the files live under.
    """

    _KIND = "image_memory"

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(
            root,
            ImageMemoryWriter(root),
            _image_memory_entry,
            group_writer=GroupImageMemoryWriter(root),
        )

    async def append(
        self,
        scope: Scope,
        image: ImageMemory,
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None:
        refs = await self._append_many(scope, [image], only_if)
        return refs[0] if refs else None

    async def append_many(
        self,
        scope: Scope,
        images: Sequence[ImageMemory],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        return await self._append_many(scope, images, only_if)

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        """Mark ``(md_path, entry_id)`` pairs superseded by ``replaced_by``."""
        return await self._deprecate(
            scope, entries=entries, replaced_by=replaced_by, only_if=only_if
        )

    async def undeprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        """Drop the supersession marks of ``(md_path, entry_id)`` pairs."""
        by_file: dict[str, set[str]] = defaultdict(set)
        for md_path, entry_id in entries:
            by_file[md_path].add(entry_id)
        async with LOCKS.hold("owner", owner_lock(scope)):
            if not await self._sources_alive(scope, only_if, len(entries)):
                return False
            for md_path, keys in by_file.items():
                await self._writer_for(scope).drop_deprecations(
                    self._root.root / md_path, keys
                )
        return True

    async def entries_for_session(
        self, scope: Scope, session_id: str
    ) -> list[tuple[str, str, str]]:
        """``(md_path, entry_id, image_id)`` for the owner's pictures of one session.

        The picture id rides along because forgetting the conversation has to
        decide, per file, whether this session held its last live reference —
        and that question is answered from the entries, not from a re-read of
        files that are still being marked.
        """
        found: list[tuple[str, str, str]] = []
        needle = f"**session_id**: {session_id}"
        for path in await self._files_mentioning(scope, needle):
            try:
                parsed = await MarkdownReader.read(path)
            except FileNotFoundError:
                continue
            md_path = _relative(self._root, path)
            for entry in parsed.entries:
                inline = entry.as_structured().inline
                if inline.get("session_id") != session_id:
                    continue
                found.append((md_path, entry.id, inline.get("image_id") or ""))
        return found

    async def read_by_image(
        self, scope: Scope, image_id: str, *, limit: int = _IMAGE_READ_CAP
    ) -> list[Stored[ImageMemory]]:
        """Every entry written about one picture, oldest first, marks included.

        Read from markdown rather than the index on purpose: an undo runs after
        a forget erased the rows through the fence, so the index no longer
        knows the memories the forget marked. The markdown still does — it is
        the truth — and it is also where the mark itself lives.
        """
        found: list[Stored[ImageMemory]] = []
        for path in await self._files_mentioning(scope, f"**image_id**: {image_id}"):
            try:
                parsed = await MarkdownReader.read(path)
            except FileNotFoundError:
                continue
            marks = parsed.frontmatter.get("deprecated_entries") or {}
            md_path = _relative(self._root, path)
            for entry in parsed.entries:
                structured = entry.as_structured()
                if structured.inline.get("image_id") != image_id:
                    continue
                found.append(
                    Stored(
                        ref=MemoryRef(
                            id=f"{scope.owner_id}_{entry.id}",
                            entry_id=entry.id,
                            md_path=md_path,
                        ),
                        scope=scope,
                        item=_image_memory_from(entry, owner_id=scope.owner_id),
                    )
                )
                item = found[-1].item
                item.superseded_by = marks.get(entry.id)
                item.status = "superseded" if item.superseded_by else "current"
                if len(found) >= limit:
                    return found
        return found

    async def read_for_segments(
        self, scope: Scope, memcell_ids: Collection[str]
    ) -> list[Stored[ImageMemory]]:
        """Every picture memory written from one of ``memcell_ids``, marks included."""
        wanted = set(memcell_ids)
        if not wanted:
            return []
        found = await self._entries_where(
            scope,
            [f"**parent_id**: {memcell_id}" for memcell_id in sorted(wanted)],
            lambda e: e.as_structured().inline.get("parent_id") in wanted,
        )
        stored: list[Stored[ImageMemory]] = []
        for md_path, entry, mark in found:
            item = _image_memory_from(entry, owner_id=scope.owner_id)
            item.superseded_by = mark
            item.status = "superseded" if mark else "current"
            stored.append(
                Stored(
                    ref=MemoryRef(
                        id=f"{scope.owner_id}_{entry.id}",
                        entry_id=entry.id,
                        md_path=md_path,
                    ),
                    scope=scope,
                    item=item,
                )
            )
        return stored

    async def read_many(
        self, scope: Scope, entry_ids: Collection[str]
    ) -> list[Stored[ImageMemory]]:
        """The picture memories of ``entry_ids`` still in md, marks included."""
        wanted: dict[Path, set[str]] = defaultdict(set)
        for entry_id in entry_ids:
            path = self._path_of(scope, entry_id)
            if path is not None:
                wanted[path].add(entry_id)
        found: list[Stored[ImageMemory]] = []
        for path, ids in sorted(wanted.items()):
            try:
                parsed = await MarkdownReader.read(path)
            except FileNotFoundError:
                continue
            marks = _marks(parsed.frontmatter)
            md_path = _relative(self._root, path)
            for entry in parsed.entries:
                if entry.id not in ids:
                    continue
                item = _image_memory_from(entry, owner_id=scope.owner_id)
                item.superseded_by = marks.get(entry.id)
                item.status = "superseded" if item.superseded_by else "current"
                found.append(
                    Stored(
                        ref=MemoryRef(
                            id=f"{scope.owner_id}_{entry.id}",
                            entry_id=entry.id,
                            md_path=md_path,
                        ),
                        scope=scope,
                        item=item,
                    )
                )
        return found

    async def remove_session(self, scope: Scope, session_id: str) -> list[MemoryRef]:
        """Drop the owner's picture memories of one session from markdown."""
        return await super().remove_session(scope, session_id)


class MarkdownProfileStore:
    """The single ``users/<id>/user.md`` profile document.

    Args:
        root: Memory root the files live under.
    """

    def __init__(self, root: MemoryRoot) -> None:
        self._root = root
        self._reader = ProfileReader(root)
        self._writer = ProfileWriter(root)

    async def read(self, scope: Scope) -> Profile | None:
        found = await self._reader.read(
            scope.owner_id,
            schema=UserProfileFrontmatter,
            app_id=scope.app_id,
            project_id=scope.project_id,
        )
        if found is None:
            return None
        frontmatter, _body = found
        return Profile.from_stored(
            owner_id=frontmatter.user_id,
            summary=frontmatter.summary,
            timestamp=frontmatter.profile_timestamp_ms,
            explicit_info=list(frontmatter.explicit_info),
            implicit_traits=list(frontmatter.implicit_traits),
            primary_language=_known_language(
                frontmatter.primary_language, owner_id=frontmatter.user_id
            ),
            history=list(frontmatter.history),
        )

    async def write(self, scope: Scope, profile: Profile) -> MemoryRef:
        frontmatter = UserProfileFrontmatter(
            id=f"profile_{scope.owner_id}",
            user_id=scope.owner_id,
            summary=profile.summary,
            explicit_info=[i.model_dump() for i in profile.explicit_info],
            implicit_traits=[i.model_dump() for i in profile.implicit_traits],
            profile_timestamp_ms=profile.timestamp,
            primary_language=profile.primary_language,
            history=list(profile.history),
        )
        async with LOCKS.hold("owner", owner_lock(scope)):
            path = await self._writer.write(
                scope.owner_id,
                frontmatter=frontmatter,
                body=profile.summary,
                app_id=scope.app_id,
                project_id=scope.project_id,
            )
        return MemoryRef(
            id=scope.owner_id,
            entry_id=frontmatter.id,
            md_path=_relative(self._root, path),
        )


class MarkdownSceneStore(_DailyLogStore[Scene]):
    """Scene versions of a group (``groups/<id>/scenes/scene-<date>.md``).

    Append-only, like the picture memories: a new version is a new entry, and
    the version it replaces is listed in its file's ``deprecated_entries``
    against the new one, in the same locked step -- which the cascade turns
    into ``deprecated_by`` on the old row, so only the newest competes. The
    rows live in the episode table (``parent_type = scene``). A scene belongs
    to a group: any other scope is refused.

    Args:
        root: Memory root the files live under.
    """

    _KIND = "scene"

    def __init__(self, root: MemoryRoot) -> None:
        writer = GroupSceneWriter(root)
        super().__init__(root, writer, _scene_entry, group_writer=writer)

    async def latest(self, scope: Scope, scene_id: str) -> Stored[Scene] | None:
        newest: Stored[Scene] | None = None
        for path in await self._files_mentioning(scope, f"**scene_id**: {scene_id}"):
            try:
                parsed = await MarkdownReader.read(path)
            except FileNotFoundError:
                continue
            md_path = _relative(self._root, path)
            for entry in parsed.entries:
                if entry.as_structured().inline.get("scene_id") != scene_id:
                    continue
                scene = _scene_from_entry(entry, owner_id=scope.owner_id)
                if newest is None or scene.version > newest.item.version:
                    newest = Stored(
                        ref=MemoryRef(
                            id=f"{scope.owner_id}_{entry.id}",
                            entry_id=entry.id,
                            md_path=md_path,
                        ),
                        scope=scope,
                        item=scene,
                    )
        return newest

    async def versions(self, scope: Scope) -> list[Stored[Scene]]:
        """Every version of every scene of the group, in file order."""
        found = await self._entries_where(
            scope, ["**scene_id**: "], lambda e: bool(e.as_structured().inline)
        )
        return [
            Stored(
                ref=MemoryRef(
                    id=f"{scope.owner_id}_{entry.id}",
                    entry_id=entry.id,
                    md_path=md_path,
                ),
                scope=scope,
                item=_scene_from_entry(entry, owner_id=scope.owner_id),
            )
            for md_path, entry, _mark in found
        ]

    async def append(
        self,
        scope: Scope,
        scene: Scene,
        *,
        supersedes: Stored[Scene] | None,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None:
        if scope.owner_type != "group":
            raise ValueError("a scene belongs to a group scope")
        writer = self._writer_for(scope)
        async with LOCKS.hold("owner", owner_lock(scope)):
            if not await self._sources_alive(scope, only_if, 1):
                return None
            (eid,) = await writer.append_entries(
                scope.owner_id,
                [_scene_entry(scene)],
                app_id=scope.app_id,
                project_id=scope.project_id,
            )
            if supersedes is not None:
                await writer.patch_frontmatter(
                    self._root.root / supersedes.ref.md_path,
                    {"deprecated_entries": {supersedes.ref.entry_id: eid.format()}},
                )
        return self._ref(scope, eid)


class MarkdownGroupProfileStore(_DailyLogStore[GroupProfileItem]):
    """A group's profile items and its pending batch (``groups/<g>/profile/``).

    Items are daily-log entries (``group_profile-<date>.md``, ``gp_`` ids, the
    format :mod:`._mds.group_profile` documents), append-only: a new version is
    a new entry, and :meth:`append` lists each entry an item ``supersedes`` in
    that entry's own file's ``deprecated_entries`` against the new entry -- in
    the same step, under the group's owner lock, after ``only_if`` passed. A
    forget (:meth:`deprecate` with :data:`FORGOTTEN_MARK`) is final: a later
    item that names a forgotten entry is written, but never turns the mark
    into a supersession, which would bring the entry back into history. A
    named entry that is not in its file (gone, or never this group's) is
    logged and left alone. ``pending.md`` beside the logs holds the batch the
    next update reads; it is no daily log, and no cascade kind matches it.

    Every read and write is the group's: any other scope is refused.

    Args:
        root: Memory root the groups live under.
    """

    _KIND = "group_profile"

    def __init__(self, root: MemoryRoot) -> None:
        writer = GroupProfileWriter(root)
        super().__init__(root, writer, render_group_profile_entry, group_writer=writer)
        self._markdown = MarkdownWriter(root)

    async def items(self, scope: Scope) -> list[Stored[GroupProfileItem]]:
        _require_group(scope)
        found: list[Stored[GroupProfileItem]] = []
        for path in await self._daily_files(scope):
            try:
                parsed = await MarkdownReader.read(path)
            except FileNotFoundError:
                continue
            marks = _marks(parsed.frontmatter)
            md_path = _relative(self._root, path)
            for entry in parsed.entries:
                try:
                    item = group_profile_item_from_entry(entry, owner_id=scope.owner_id)
                except ValueError as exc:  # a hand edit; the rest stays readable
                    logger.warning(
                        "group_profile_entry_unreadable",
                        md_path=md_path,
                        entry_id=entry.id,
                        error=str(exc)[:200],
                    )
                    continue
                item = item.read_back(item_id=entry.id, mark=marks.get(entry.id))
                found.append(
                    Stored(
                        ref=MemoryRef(
                            id=f"{scope.owner_id}_{entry.id}",
                            entry_id=entry.id,
                            md_path=md_path,
                        ),
                        scope=scope,
                        item=item,
                    )
                )
        return found

    async def append(
        self,
        scope: Scope,
        items: Sequence[GroupProfileItem],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        _require_group(scope)
        if not items:
            return []
        writer = self._writer_for(scope)
        async with LOCKS.hold("owner", owner_lock(scope)):
            if not await self._sources_alive(scope, only_if, len(items)):
                return []
            eids = await writer.append_entries(
                scope.owner_id,
                [render_group_profile_entry(item) for item in items],
                app_id=scope.app_id,
                project_id=scope.project_id,
            )
            await self._mark_replaced(
                scope,
                [
                    (item.supersedes, eid.format())
                    for item, eid in zip(items, eids, strict=True)
                    if item.supersedes
                ],
            )
        return [self._ref(scope, eid) for eid in eids]

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        _require_group(scope)
        return await self._deprecate(
            scope, entries=entries, replaced_by=replaced_by, only_if=only_if
        )

    async def pending(self, scope: Scope) -> list[str]:
        return list((await self.pending_batch(scope)).memcell_ids)

    async def pending_batch(self, scope: Scope) -> PendingBatch:
        path = self._pending_path(scope)
        if not await anyio.Path(path).is_file():
            return PendingBatch()
        parsed = await MarkdownReader.read(path)
        frontmatter = GroupProfilePendingFrontmatter.model_validate(parsed.frontmatter)
        return PendingBatch(
            memcell_ids=tuple(frontmatter.pending_memcell_ids),
            failures=frontmatter.failures,
            retry_at=frontmatter.retry_at,
        )

    async def set_pending(
        self,
        scope: Scope,
        memcell_ids: Sequence[str],
        *,
        failures: int = 0,
        retry_at: int = 0,
    ) -> None:
        path = self._pending_path(scope)
        frontmatter = GroupProfilePendingFrontmatter(
            id=f"group_profile_pending_{scope.owner_id}",
            group_id=scope.owner_id,
            pending_memcell_ids=list(memcell_ids),
            failures=failures,
            retry_at=retry_at,
        )
        async with LOCKS.hold("owner", owner_lock(scope)):
            await self._markdown.write_markdown(
                path, frontmatter=frontmatter.model_dump(mode="json")
            )

    async def _mark_replaced(
        self, scope: Scope, replaced: Sequence[tuple[str, str]]
    ) -> None:
        """Mark each ``(old, new)`` in the old entry's file (caller holds the lock)."""
        by_file: dict[Path, dict[str, str]] = defaultdict(dict)
        for old, new in replaced:
            path = self._path_of(scope, old)
            if path is None:
                self._skip_mark(scope, old, reason="not an entry id")
                continue
            by_file[path][old] = new
        writer = self._writer_for(scope)
        for path, marks in by_file.items():
            try:
                parsed = await MarkdownReader.read(path)
            except FileNotFoundError:
                for old in marks:
                    self._skip_mark(scope, old, reason="no such file")
                continue
            present = {entry.id for entry in parsed.entries}
            current = _marks(parsed.frontmatter)
            wanted: dict[str, str] = {}
            for old, new in marks.items():
                if old not in present:
                    self._skip_mark(scope, old, reason="no such entry")
                elif current.get(old) == FORGOTTEN_MARK:
                    self._skip_mark(scope, old, reason="forgotten")
                else:
                    wanted[old] = new
            if wanted:
                await writer.patch_frontmatter(path, {"deprecated_entries": wanted})

    def _skip_mark(self, scope: Scope, entry_id: str, *, reason: str) -> None:
        logger.warning(
            "group_profile_supersede_skipped",
            app_id=scope.app_id,
            project_id=scope.project_id,
            group_id=scope.owner_id,
            entry_id=entry_id,
            reason=reason,
        )

    async def _daily_files(self, scope: Scope) -> list[Path]:
        """The daily logs, oldest first -- never ``pending.md`` beside them."""
        folder = anyio.Path(self._folder(scope))
        if not await folder.is_dir():
            return []
        pattern = f"{GroupProfileDailyFrontmatter.FILE_PREFIX}-*.md"
        return sorted([Path(str(f)) async for f in folder.glob(pattern)])

    def _pending_path(self, scope: Scope) -> Path:
        _require_group(scope)
        return (
            self._root.groups_dir(scope.app_id, scope.project_id)
            / scope.owner_id
            / GroupProfilePendingFrontmatter.DIR_NAME
            / GroupProfilePendingFrontmatter.FILENAME
        )


class MarkdownMemberProfileStore:
    """``groups/<g>/members/<member_id>.md``: a member's profile in one group.

    One file per member, rewritten whole on every update, under the group's
    owner lock like every other markdown write of the group. Reads go to the
    file (the truth): the strategy that updates a profile reads it there, and
    the search side reads the index row the cascade projects from it.

    Args:
        root: Memory root the groups live under.
    """

    def __init__(self, root: MemoryRoot) -> None:
        self._root = root
        self._writer = MarkdownWriter(root)

    async def read(self, scope: Scope, member_id: str) -> MemberProfile | None:
        path = self._path(scope, member_id)
        if not await anyio.Path(path).is_file():
            return None
        parsed = await MarkdownReader.read(path)
        fm = GroupMemberProfileFrontmatter.model_validate(parsed.frontmatter)
        return MemberProfile(
            group_id=scope.owner_id,
            member_id=member_id,
            profile=Profile.from_stored(
                owner_id=member_id,
                summary=fm.summary,
                timestamp=fm.profile_timestamp_ms,
                explicit_info=list(fm.explicit_info),
                implicit_traits=list(fm.implicit_traits),
                primary_language=_known_language(
                    fm.primary_language, owner_id=member_id
                ),
            ),
            pending_memcell_ids=list(fm.pending_memcell_ids),
            history=[dict(item) for item in fm.history if isinstance(item, dict)],
        )

    async def write(self, scope: Scope, profile: MemberProfile) -> MemoryRef:
        path = self._path(scope, profile.member_id)
        document = profile.profile
        frontmatter = GroupMemberProfileFrontmatter(
            id=f"member_profile_{scope.owner_id}_{profile.member_id}",
            group_id=scope.owner_id,
            member_id=profile.member_id,
            summary=document.summary,
            explicit_info=[i.model_dump() for i in document.explicit_info],
            implicit_traits=[i.model_dump() for i in document.implicit_traits],
            profile_timestamp_ms=document.timestamp,
            primary_language=document.primary_language,
            pending_memcell_ids=list(profile.pending_memcell_ids),
            history=list(profile.history),
        )
        body = (
            document.summary
            if document.summary.endswith("\n")
            else (f"{document.summary}\n")
        )
        async with LOCKS.hold("owner", owner_lock(scope)):
            await self._writer.write_markdown(
                path, frontmatter=frontmatter.model_dump(), body=body
            )
        return MemoryRef(
            id=member_profile_row_id(scope.owner_id, profile.member_id),
            entry_id=frontmatter.id,
            md_path=_relative(self._root, path),
        )

    async def remove(
        self, scope: Scope, member_id: str
    ) -> Stored[MemberProfile] | None:
        """Delete the member's profile file; what it held and its row, or ``None``.

        A forget of a member takes the whole document (section 9.4); its undo
        writes the returned profile back.
        """
        path = self._path(scope, member_id)
        async with LOCKS.hold("owner", owner_lock(scope)):
            if not await anyio.Path(path).is_file():
                return None
            held = await self.read(scope, member_id)
            await anyio.Path(path).unlink(missing_ok=True)
        if held is None:
            return None
        return Stored(
            ref=MemoryRef(
                id=member_profile_row_id(scope.owner_id, member_id),
                entry_id=f"member_profile_{scope.owner_id}_{member_id}",
                md_path=_relative(self._root, path),
            ),
            scope=scope,
            item=held,
        )

    async def members(self, scope: Scope) -> list[str]:
        folder = anyio.Path(self._folder(scope))
        if not await folder.is_dir():
            return []
        return sorted([Path(str(f)).stem async for f in folder.glob("*.md")])

    def _folder(self, scope: Scope) -> Path:
        if scope.owner_type != "group":
            raise ValueError("a member profile belongs to a group scope")
        return (
            self._root.groups_dir(scope.app_id, scope.project_id)
            / scope.owner_id
            / GroupMemberProfileFrontmatter.MEMBERS_DIR
        )

    def _path(self, scope: Scope, member_id: str) -> Path:
        return self._folder(scope) / f"{member_id}.md"


class MarkdownAgentSkillStore:
    """``agents/<id>/skills/skill_<name>/SKILL.md`` documents.

    A skill's ref ``entry_id`` is its sanitized name, which is also its
    directory suffix.

    Args:
        root: Memory root the files live under.
    """

    def __init__(self, root: MemoryRoot) -> None:
        self._root = root
        self._reader = AgentSkillReader(root)
        self._writer = AgentSkillWriter(root)

    async def list_by_cluster(
        self, scope: Scope, cluster_id: str
    ) -> list[Stored[AgentSkill]]:
        found = await self._reader.list_by_cluster(
            scope.owner_id,
            cluster_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
        )
        return [
            Stored(
                ref=self._ref(scope, frontmatter.id, frontmatter.name),
                scope=scope,
                item=_skill_from_md(frontmatter, body),
            )
            for frontmatter, body in found
        ]

    async def names_cluster(self, scope: Scope, cluster_id: str) -> bool:
        """Whether any of the agent's ``SKILL.md`` files names ``cluster_id``.

        Reads the raw frontmatter rather than the validated schema
        :meth:`list_by_cluster` returns: a file that names the cluster counts
        even when a later schema revision would reject it, so an operational
        replay never extracts a second skill next to it. A file that does not
        parse at all is logged and does not count.
        """
        skills_dir = (
            self._root.agents_dir(scope.app_id, scope.project_id)
            / scope.owner_id
            / AgentSkillFrontmatter.SKILLS_CONTAINER_NAME
        )
        apath = anyio.Path(skills_dir)
        if not await apath.is_dir():
            return False
        async for skill_dir in apath.iterdir():
            if not await skill_dir.is_dir():
                continue
            if not skill_dir.name.startswith(AgentSkillFrontmatter.SKILL_DIR_PREFIX):
                continue
            md_path = skill_dir / AgentSkillFrontmatter.SKILL_MAIN_FILENAME
            if not await md_path.is_file():
                continue
            try:
                parsed = await MarkdownReader.read(Path(str(md_path)))
            except Exception as exc:
                logger.warning(
                    "cascade_backfill_skill_md_parse_failed",
                    md_path=str(md_path),
                    error=repr(exc),
                )
                continue
            if parsed.frontmatter.get("cluster_id") == cluster_id:
                return True
        return False

    async def write(self, scope: Scope, skill: AgentSkill) -> MemoryRef:
        name = AgentSkillFrontmatter.sanitize_skill_name(skill.name)
        frontmatter = AgentSkillFrontmatter(
            id=f"{scope.owner_id}_{name}",
            agent_id=scope.owner_id,
            name=name,
            description=skill.description,
            confidence=skill.confidence,
            maturity_score=skill.maturity_score,
            source_case_ids=list(skill.source_case_ids),
            cluster_id=skill.cluster_id or None,
        )
        async with LOCKS.hold("owner", owner_lock(scope)):
            await self._writer.write_main(
                scope.owner_id,
                name,
                frontmatter=frontmatter,
                body=skill.content,
                app_id=scope.app_id,
                project_id=scope.project_id,
            )
        return self._ref(scope, frontmatter.id, name)

    async def delete(self, scope: Scope, skill_name: str) -> bool:
        async with LOCKS.hold("owner", owner_lock(scope)):
            return await self._writer.delete_skill(
                scope.owner_id,
                skill_name,
                app_id=scope.app_id,
                project_id=scope.project_id,
            )

    def _ref(self, scope: Scope, skill_id: str, name: str) -> MemoryRef:
        entry_id = AgentSkillFrontmatter.sanitize_skill_name(name)
        path = self._writer.main_path(
            scope.owner_id,
            entry_id,
            app_id=scope.app_id,
            project_id=scope.project_id,
        )
        return MemoryRef(
            id=skill_id, entry_id=entry_id, md_path=_relative(self._root, path)
        )


def _image_memory_from(entry: Entry, *, owner_id: str) -> ImageMemory:
    """The picture memory an entry written by :func:`_image_memory_entry` holds.

    Mirrors :func:`_episode_from_entry`: inline fields carry the identity and
    provenance, the two sections carry the text. An entry without a
    ``timestamp`` cannot be re-written as a memory (there is nothing to date
    it), so the caller treats the parse as a miss rather than inventing one.
    """
    structured = entry.as_structured()
    inline, sections = structured.inline, structured.sections
    return ImageMemory(
        owner_id=inline.get("owner_id") or owner_id,
        description=sections.get("Description", ""),
        timestamp=to_timestamp_ms(from_iso_format(inline["timestamp"])),
        ocr=sections.get("OCR", ""),
        image_id=inline.get("image_id", ""),
        name=inline.get("name") or "",
        original_relpath=inline.get("original_path") or "",
        media_type=inline.get("media_type") or "",
        width=int(inline.get("width") or 0),
        height=int(inline.get("height") or 0),
        session_id=inline.get("session_id") or None,
        parent_id=inline.get("parent_id", ""),
        parent_type=inline.get("parent_type") or "memcell",  # type: ignore[arg-type]
        entry_id=entry.id,
        md_path=None,
        # Group attribution (absent on a one-to-one entry): a forget inside a
        # group reads the poster off the entry to decide who may forget it.
        sender_ids=_list_field(inline.get("sender_ids")),
        subject_ids=_list_field(inline.get("subject_ids")),
        origin=inline.get("origin") or None,  # type: ignore[arg-type]
    )


def _list_field(raw: object) -> list[str]:
    """A list-valued inline field parsed back from its rendered form.

    Inline values round-trip as text (``[a, b, c]``); this is the inverse for
    the one shape the episode writer emits, and it tolerates the field being
    absent (every entry written before it existed) without inventing values.
    """
    if isinstance(raw, list):
        return [str(item) for item in raw]
    if not isinstance(raw, str) or not raw.strip().startswith("["):
        return []
    body = raw.strip()[1:-1]
    return [part.strip() for part in body.split(",") if part.strip()]


def _image_memory_entry(image: ImageMemory) -> EntryBody:
    """Inline audit fields + Description / OCR sections.

    ``ocr`` is omitted when empty: a photograph of a scene has no text in it,
    and writing an empty section would make every such entry look like an entry
    whose text was lost.

    ``name`` is omitted when empty for the same reason: most pictures arrive
    without one, and ``name: `` on every entry would read as a name that was
    lost rather than never sent. It sits beside ``image_id`` because it is
    provenance of this occurrence, while the hash is the identity of the file.
    """
    inline: dict[str, object] = {
        "owner_id": image.owner_id,
        "timestamp": _iso(image.timestamp),
        "parent_type": image.parent_type,
        "parent_id": image.parent_id,
        "image_id": image.image_id,
        "original_path": image.original_relpath,
        "media_type": image.media_type,
        "width": image.width,
        "height": image.height,
    }
    if image.name:
        inline["name"] = image.name
    if image.session_id is not None:
        inline["session_id"] = image.session_id
    # Group attribution; absent on a one-to-one picture memory.
    if image.sender_ids:
        inline["sender_ids"] = list(image.sender_ids)
    if image.subject_ids:
        inline["subject_ids"] = list(image.subject_ids)
    if image.origin is not None:
        inline["origin"] = image.origin
    sections = {"Description": image.description}
    if image.ocr:
        sections["OCR"] = image.ocr
    return inline, sections


# ── Markdown -> domain ───────────────────────────────────────────────────

_OUTPUT_LANGUAGES = frozenset(language.value for language in OutputLanguage)


def _known_language(value: str | None, *, owner_id: str) -> str | None:
    """A stored profile language everalgo accepts, else ``None``.

    ``user.md`` is hand-editable; a value outside ``OutputLanguage`` (case
    included) would make every later extraction fail, so it is ignored.
    """
    if value is None or value in _OUTPUT_LANGUAGES:
        return value
    logger.warning("profile_language_ignored", owner_id=owner_id, value=value)
    return None


# ── Domain -> markdown entry ─────────────────────────────────────────────


def _iso(timestamp_ms: int) -> str:
    rendered = to_iso_format(from_timestamp_ms(timestamp_ms))
    assert rendered is not None  # a concrete timestamp always renders
    return rendered


def _episode_entry(episode: Episode) -> EntryBody:
    """Inline audit fields + Subject / Summary / Content sections.

    ``session_id`` is omitted for an episode without one (a Reflection
    merge has no session); the cascade reads its absence as ``None``.
    ``language`` is omitted when unknown, as in entries written before it.
    """
    inline: dict[str, object] = {"owner_id": episode.owner_id}
    if episode.session_id is not None:
        inline["session_id"] = episode.session_id
    inline["timestamp"] = _iso(episode.timestamp)
    inline["parent_type"] = episode.parent_type
    inline["parent_id"] = episode.parent_id
    if episode.sender_ids:
        inline["sender_ids"] = list(episode.sender_ids)
    if episode.image_ids:
        inline["image_ids"] = list(episode.image_ids)
    if episode.language:
        inline["language"] = episode.language
    # Group attribution (docs/group-memory.md, section 6.1). Absent on a
    # one-to-one episode, whose entries therefore read exactly as before.
    if episode.source_start is not None:
        inline["source_start"] = _iso(episode.source_start)
    if episode.subject_ids:
        inline["subject_ids"] = list(episode.subject_ids)
    if episode.origin is not None:
        inline["origin"] = episode.origin
    sections: dict[str, str] = {}
    if episode.subject:
        sections["Subject"] = episode.subject
    if episode.summary:
        sections["Summary"] = episode.summary
    sections["Content"] = episode.episode
    return inline, sections


def _scene_entry(scene: Scene) -> EntryBody:
    """A scene version in the episode entry contract, plus its own fields.

    ``parent_type = scene`` and ``parent_id`` = the scene id make the
    episode cascade project it into the episode table; ``member_ids`` are
    the episodes it covers and ``version`` counts its versions.
    """
    inline: dict[str, object] = {
        "owner_id": scene.owner_id,
        "scene_id": scene.scene_id,
        "version": scene.version,
        "timestamp": _iso(scene.timestamp),
        "source_start": _iso(scene.source_start),
        "parent_type": "scene",
        "parent_id": scene.scene_id,
        "member_ids": list(scene.member_ids),
    }
    if scene.sender_ids:
        inline["sender_ids"] = list(scene.sender_ids)
    if scene.subject_ids:
        inline["subject_ids"] = list(scene.subject_ids)
    if scene.origin is not None:
        inline["origin"] = scene.origin
    if scene.language:
        inline["language"] = scene.language
    sections: dict[str, str] = {}
    if scene.subject:
        sections["Subject"] = scene.subject
    if scene.summary:
        sections["Summary"] = scene.summary
    sections["Content"] = scene.narrative
    return inline, sections


def _scene_from_entry(entry: Entry, *, owner_id: str) -> Scene:
    """The scene version an entry written by :func:`_scene_entry` holds."""
    structured = entry.as_structured()
    inline, sections = structured.inline, structured.sections
    timestamp = to_timestamp_ms(from_iso_format(inline["timestamp"]))
    return Scene(
        owner_id=inline.get("owner_id") or owner_id,
        scene_id=inline.get("scene_id", ""),
        version=int(inline.get("version") or 1),
        member_ids=_list_field(inline.get("member_ids")),
        subject=sections.get("Subject", ""),
        summary=sections.get("Summary", ""),
        narrative=sections.get("Content", ""),
        timestamp=timestamp,
        source_start=(
            to_timestamp_ms(from_iso_format(inline["source_start"]))
            if inline.get("source_start")
            else timestamp
        ),
        sender_ids=_list_field(inline.get("sender_ids")),
        subject_ids=_list_field(inline.get("subject_ids")),
        origin=inline.get("origin") or None,  # type: ignore[arg-type]
        language=inline.get("language") or None,
    )


def _agent_case_entry(case: AgentCase) -> EntryBody:
    """``quality_score`` rides inline so the cascade hashes it.

    ``source_group_id`` is written only for a case learned in a group, so a
    case from a session bound to no group reads exactly as before.
    """
    inline: dict[str, object] = {
        "owner_id": case.owner_id,
        "session_id": case.session_id,
        "timestamp": _iso(case.timestamp),
        "parent_type": "memcell",
        "parent_id": case.parent_id,
        "quality_score": case.quality_score,
    }
    if case.source_group_id is not None:
        inline["source_group_id"] = case.source_group_id
    sections = {"TaskIntent": case.task_intent, "Approach": case.approach}
    if case.key_insight:
        sections["KeyInsight"] = case.key_insight
    return inline, sections


# ── Markdown -> domain ───────────────────────────────────────────────────


def _skill_from_md(frontmatter: AgentSkillFrontmatter, body: str) -> AgentSkill:
    return AgentSkill(
        id=frontmatter.id,
        cluster_id=frontmatter.cluster_id or "",
        name=frontmatter.name,
        description=frontmatter.description,
        content=body,
        confidence=frontmatter.confidence,
        maturity_score=frontmatter.maturity_score,
        source_case_ids=list(frontmatter.source_case_ids),
    )


def _episode_from_entry(entry: Entry, *, owner_id: str) -> Episode:
    """The episode an entry written by :func:`_episode_entry` holds."""
    structured = entry.as_structured()
    inline, sections = structured.inline, structured.sections
    return Episode(
        owner_id=inline.get("owner_id") or owner_id,
        episode=sections.get("Content", ""),
        summary=sections.get("Summary", ""),
        subject=sections.get("Subject", ""),
        timestamp=to_timestamp_ms(from_iso_format(inline["timestamp"])),
        session_id=inline.get("session_id") or None,
        parent_id=inline.get("parent_id", ""),
        parent_type=inline.get("parent_type") or "memcell",  # type: ignore[arg-type]
        sender_ids=_list_field(inline.get("sender_ids")),
        image_ids=_list_field(inline.get("image_ids")),
        language=inline.get("language") or None,
        subject_ids=_list_field(inline.get("subject_ids")),
        origin=inline.get("origin") or None,
        source_start=(
            to_timestamp_ms(from_iso_format(inline["source_start"]))
            if inline.get("source_start")
            else None
        ),
    )


def _require_group(scope: Scope) -> None:
    if scope.owner_type != "group":
        raise ValueError("a group profile belongs to a group scope")


def _marks(frontmatter: Mapping[str, object]) -> dict[str, str]:
    """A file's ``deprecated_entries``; a hand-broken value reads as none."""
    marks = frontmatter.get("deprecated_entries")
    if not isinstance(marks, dict):
        return {}
    return {str(key): str(value) for key, value in marks.items() if value}


async def _deprecations(path: Path) -> dict[str, str]:
    try:
        parsed = await MarkdownReader.read(path)
    except FileNotFoundError:
        return {}
    marks = parsed.frontmatter.get("deprecated_entries")
    return dict(marks) if isinstance(marks, dict) else {}


def _relative(root: MemoryRoot, path: Path) -> str:
    return path.relative_to(root.root).as_posix()
