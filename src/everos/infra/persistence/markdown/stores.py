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
    "MarkdownAtomicFactStore",
    "MarkdownEpisodeStore",
    "MarkdownForesightStore",
    "MarkdownProfileStore",
]

import datetime as _dt
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from pathlib import Path
from typing import ClassVar

import anyio
from everalgo.user_memory import OutputLanguage

from everos.core.locks import LOCKS, owner_key
from everos.core.observability.logging import get_logger
from everos.core.utils.datetime import (
    from_iso_format,
    from_timestamp,
    to_iso_format,
    to_timestamp_ms,
    today_with_timezone,
)
from everos.infra.persistence.memory_root import MemoryRoot
from everos.memory.models import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    Profile,
)
from everos.memory.ports import (
    MemoryRef,
    NoSourceCheck,
    Scope,
    SourceCheck,
    Stored,
)

from ._base import Entry, EntryId, MarkdownReader
from ._mds import AgentSkillFrontmatter, UserProfileFrontmatter
from ._readers import AgentSkillReader, ProfileReader
from ._writers import (
    AgentCaseWriter,
    AgentSkillWriter,
    AtomicFactWriter,
    BaseDailyWriter,
    EpisodeWriter,
    ForesightWriter,
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
    ) -> None:
        self._root = root
        self._writer = writer
        self._to_entry = to_entry

    async def _append_many(
        self,
        scope: Scope,
        items: Sequence[T],
        only_if: SourceCheck | NoSourceCheck,
        *,
        date: _dt.date | None = None,
    ) -> list[MemoryRef]:
        if not items:
            return []
        async with LOCKS.hold("owner", owner_lock(scope)):
            if not await self._sources_alive(scope, only_if, len(items)):
                return []
            eids = await self._writer.append_entries(
                scope.owner_id,
                [self._to_entry(item) for item in items],
                date=date,
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
                await self._writer.patch_frontmatter(
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
            removed = await self._writer.remove_entries(path, drop)
            if removed:
                LOCKS.bump_delete_count(key)
        md_path = _relative(self._root, path)
        return [
            MemoryRef(id=f"{scope.owner_id}_{eid}", entry_id=eid, md_path=md_path)
            for eid in removed
        ]

    def _folder(self, scope: Scope) -> Path:
        return self._writer.path_for(
            scope.owner_id, app_id=scope.app_id, project_id=scope.project_id
        ).parent

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
        return self._writer.path_for(
            scope.owner_id, day, app_id=scope.app_id, project_id=scope.project_id
        )

    def _ref(self, scope: Scope, eid: EntryId) -> MemoryRef:
        path = self._writer.path_for(
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
    """Episode daily logs (``users/<id>/episodes/``).

    Args:
        root: Memory root the files live under.
    """

    _KIND = "episode"

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root, EpisodeWriter(root), _episode_entry)

    async def append(
        self,
        scope: Scope,
        episode: Episode,
        *,
        only_if: SourceCheck | NoSourceCheck,
        date: _dt.date | None = None,
    ) -> MemoryRef | None:
        refs = await self._append_many(scope, [episode], only_if, date=date)
        return refs[0] if refs else None

    async def append_many(
        self,
        scope: Scope,
        episodes: Sequence[Episode],
        *,
        only_if: SourceCheck | NoSourceCheck,
        date: _dt.date | None = None,
    ) -> list[MemoryRef]:
        return await self._append_many(scope, episodes, only_if, date=date)

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

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        """Drop ``keys`` from every episode file's ``deprecated_entries``."""
        wanted = set(keys)
        for path in await self._daily_files(scope):
            if not (await _deprecations(path)).keys() & wanted:
                continue
            async with LOCKS.hold("owner", owner_lock(scope)):
                await self._writer.drop_deprecations(path, wanted)

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


class MarkdownAtomicFactStore(_DailyLogStore[AtomicFact]):
    """Atomic-fact daily logs (``users/<id>/.atomic_facts/``).

    Args:
        root: Memory root the files live under.
    """

    _KIND = "atomic_fact"

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root, AtomicFactWriter(root), _atomic_fact_entry)

    async def append_many(
        self,
        scope: Scope,
        facts: Sequence[AtomicFact],
        *,
        only_if: SourceCheck | NoSourceCheck,
        date: _dt.date | None = None,
    ) -> list[MemoryRef]:
        if date is None and facts:
            first = facts[0]
            ts_date = (
                from_timestamp(first.timestamp).date()
                if isinstance(first.timestamp, (int, float)) and first.timestamp > 0
                else None
            )
            try:
                ep_id_date = EntryId.parse(first.parent_id).date
            except Exception:
                ep_id_date = None
            if ts_date is not None and ep_id_date is not None and ep_id_date == ts_date:
                date = ts_date
        return await self._append_many(scope, facts, only_if, date=date)

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

    async def remove_by_parents(
        self, scope: Scope, parent_ids: Collection[str]
    ) -> list[MemoryRef]:
        """Remove the owner's facts whose parent episode is one of ``parent_ids``."""
        parents = set(parent_ids)
        if not parents:
            return []
        needles = [f"**parent_id**: {parent}" for parent in parents]
        removed: list[MemoryRef] = []
        for path in await self._files_mentioning(scope, *needles):
            removed += await self._rewrite(
                scope,
                path,
                lambda e: e.as_structured().inline.get("parent_id") in parents,
            )
        return removed


class MarkdownForesightStore(_DailyLogStore[Foresight]):
    """Foresight daily logs (``users/<id>/.foresights/``).

    Args:
        root: Memory root the files live under.
    """

    _KIND = "foresight"

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root, ForesightWriter(root), _foresight_entry)

    async def append_many(
        self,
        scope: Scope,
        items: Sequence[Foresight],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        return await self._append_many(scope, items, only_if)


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
    rendered = to_iso_format(from_timestamp(timestamp_ms))
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
    if episode.language:
        inline["language"] = episode.language
    sections: dict[str, str] = {}
    if episode.subject:
        sections["Subject"] = episode.subject
    if episode.summary:
        sections["Summary"] = episode.summary
    sections["Content"] = episode.episode
    return inline, sections


def _atomic_fact_entry(fact: AtomicFact) -> EntryBody:
    inline: dict[str, object] = {
        "owner_id": fact.owner_id,
        "timestamp": _iso(fact.timestamp),
        "parent_type": "episode",
        "parent_id": fact.parent_id,
    }
    if fact.session_id is not None:
        inline["session_id"] = fact.session_id
    return inline, {"Fact": fact.content}


def _foresight_entry(foresight: Foresight) -> EntryBody:
    """Optional time-window fields are written only when set."""
    inline: dict[str, object] = {
        "owner_id": foresight.owner_id,
        "session_id": foresight.session_id,
        "timestamp": _iso(foresight.timestamp),
        "parent_type": "memcell",
        "parent_id": foresight.parent_id,
    }
    if foresight.start_time:
        inline["start_time"] = foresight.start_time
    if foresight.end_time:
        inline["end_time"] = foresight.end_time
    if foresight.duration_days is not None:
        inline["duration_days"] = foresight.duration_days
    return inline, {"Foresight": foresight.foresight, "Evidence": foresight.evidence}


def _agent_case_entry(case: AgentCase) -> EntryBody:
    """``quality_score`` rides inline so the cascade hashes it."""
    inline: dict[str, object] = {
        "owner_id": case.owner_id,
        "session_id": case.session_id,
        "timestamp": _iso(case.timestamp),
        "parent_type": "memcell",
        "parent_id": case.parent_id,
        "quality_score": case.quality_score,
    }
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
    )


async def _deprecations(path: Path) -> dict[str, str]:
    try:
        parsed = await MarkdownReader.read(path)
    except FileNotFoundError:
        return {}
    marks = parsed.frontmatter.get("deprecated_entries")
    return dict(marks) if isinstance(marks, dict) else {}


def _relative(root: MemoryRoot, path: Path) -> str:
    return path.relative_to(root.root).as_posix()
