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

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from everos.component.utils.datetime import from_timestamp, to_iso_format
from everos.infra.persistence.memory_root import MemoryRoot
from everos.memory.models import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Episode,
    Foresight,
    Profile,
)
from everos.memory.ports import MemoryRef, Scope, Stored

from .base import EntryId
from .mds import AgentSkillFrontmatter, UserProfileFrontmatter
from .readers import AgentSkillReader, ProfileReader
from .writers import (
    AgentCaseWriter,
    AgentSkillWriter,
    AtomicFactWriter,
    BaseDailyWriter,
    EpisodeWriter,
    ForesightWriter,
    ProfileWriter,
)

type EntryBody = tuple[dict[str, object], dict[str, str]]
"""``(inline, sections)`` of one rendered daily-log entry."""


class _DailyLogStore[T]:
    """Append / deprecate for one daily-log kind.

    Args:
        root: Memory root the files live under.
        writer: The kind's daily-log writer.
        to_entry: Renders one domain object as ``(inline, sections)``.
    """

    def __init__(
        self,
        root: MemoryRoot,
        writer: BaseDailyWriter,
        to_entry: Callable[[T], EntryBody],
    ) -> None:
        self._root = root
        self._writer = writer
        self._to_entry = to_entry

    async def _append_many(self, scope: Scope, items: Sequence[T]) -> list[MemoryRef]:
        eids = await self._writer.append_entries(
            scope.owner_id,
            [self._to_entry(item) for item in items],
            app_id=scope.app_id,
            project_id=scope.project_id,
        )
        return [self._ref(scope, eid) for eid in eids]

    async def _deprecate(self, *, replaced: Mapping[str, str], md_path: str) -> None:
        await self._writer.patch_frontmatter(
            self._root.root / md_path, {"deprecated_entries": dict(replaced)}
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

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root, EpisodeWriter(root), _episode_entry)

    async def append(self, scope: Scope, episode: Episode) -> MemoryRef:
        (ref,) = await self._append_many(scope, [episode])
        return ref

    async def append_many(
        self, scope: Scope, episodes: Sequence[Episode]
    ) -> list[MemoryRef]:
        return await self._append_many(scope, episodes)

    async def deprecate(
        self, scope: Scope, *, replaced: Mapping[str, str], md_path: str
    ) -> None:
        await self._deprecate(replaced=replaced, md_path=md_path)


class MarkdownAtomicFactStore(_DailyLogStore[AtomicFact]):
    """Atomic-fact daily logs (``users/<id>/.atomic_facts/``).

    Args:
        root: Memory root the files live under.
    """

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root, AtomicFactWriter(root), _atomic_fact_entry)

    async def append_many(
        self, scope: Scope, facts: Sequence[AtomicFact]
    ) -> list[MemoryRef]:
        return await self._append_many(scope, facts)

    async def deprecate(
        self, scope: Scope, *, replaced: Mapping[str, str], md_path: str
    ) -> None:
        await self._deprecate(replaced=replaced, md_path=md_path)


class MarkdownForesightStore(_DailyLogStore[Foresight]):
    """Foresight daily logs (``users/<id>/.foresights/``).

    Args:
        root: Memory root the files live under.
    """

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root, ForesightWriter(root), _foresight_entry)

    async def append_many(
        self, scope: Scope, items: Sequence[Foresight]
    ) -> list[MemoryRef]:
        return await self._append_many(scope, items)


class MarkdownAgentCaseStore(_DailyLogStore[AgentCase]):
    """Agent-case daily logs (``agents/<id>/.agent_cases/``).

    Args:
        root: Memory root the files live under.
    """

    def __init__(self, root: MemoryRoot) -> None:
        super().__init__(root, AgentCaseWriter(root), _agent_case_entry)

    async def append(self, scope: Scope, case: AgentCase) -> MemoryRef:
        (ref,) = await self._append_many(scope, [case])
        return ref


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
        )

    async def write(self, scope: Scope, profile: Profile) -> MemoryRef:
        frontmatter = UserProfileFrontmatter(
            id=f"profile_{scope.owner_id}",
            user_id=scope.owner_id,
            summary=profile.summary,
            explicit_info=[i.model_dump() for i in profile.explicit_info],
            implicit_traits=[i.model_dump() for i in profile.implicit_traits],
            profile_timestamp_ms=profile.timestamp,
        )
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


# ── Domain -> markdown entry ─────────────────────────────────────────────


def _iso(timestamp_ms: int) -> str:
    rendered = to_iso_format(from_timestamp(timestamp_ms))
    assert rendered is not None  # a concrete timestamp always renders
    return rendered


def _episode_entry(episode: Episode) -> EntryBody:
    """Inline audit fields + Subject / Summary / Content sections.

    ``session_id`` is omitted for an episode without one (a Reflection
    merge has no session); the cascade reads its absence as ``None``.
    """
    inline: dict[str, object] = {"owner_id": episode.owner_id}
    if episode.session_id is not None:
        inline["session_id"] = episode.session_id
    inline["timestamp"] = _iso(episode.timestamp)
    inline["parent_type"] = episode.parent_type
    inline["parent_id"] = episode.parent_id
    if episode.sender_ids:
        inline["sender_ids"] = list(episode.sender_ids)
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


def _relative(root: MemoryRoot, path: Path) -> str:
    return path.relative_to(root.root).as_posix()
