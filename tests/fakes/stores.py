"""In-memory implementations of the write-side store ports.

Each double keeps what was written in plain attributes so a test asserts on
data instead of on calls. Methods no strategy test needs raise
``NotImplementedError`` rather than pretending to work.
"""

from __future__ import annotations

__all__ = [
    "InMemoryAgentCaseStore",
    "InMemoryAgentSkillStore",
    "InMemoryAtomicFactStore",
    "InMemoryClusterStore",
    "InMemoryEpisodeStore",
    "InMemoryForesightStore",
    "InMemoryMemcellStore",
    "InMemoryProfileStore",
    "InMemoryReflectionReportStore",
    "UnusedPort",
]

from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from everos.memory.models import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    Cluster,
    ClusterKind,
    ClusterMemberType,
    Episode,
    Foresight,
    MemCell,
    MemcellRecord,
    Profile,
    ReflectionReport,
)
from everos.memory.ports import MemoryRef, Scope, SessionKey, Stored


class UnusedPort:
    """Stands in for a port the code under test must not touch.

    Args:
        name: The port's name, for the error message.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        raise NotImplementedError(f"{self._name}.{attr} is not faked")


class _DailyLog[T]:
    """Append-only records per scope, with ``<prefix>_<date>_<seq>`` entry ids.

    Args:
        prefix: Entry-id prefix of the kind.
        folder: Kind folder inside the owner directory.
    """

    def __init__(self, prefix: str, folder: str) -> None:
        self._prefix = prefix
        self._folder = folder
        self.records: list[Stored[T]] = []
        self.deprecated: dict[str, dict[str, str]] = defaultdict(dict)

    def items_for(self, owner_id: str) -> list[T]:
        """Everything written under ``owner_id``, in write order."""
        return [s.item for s in self.records if s.scope.owner_id == owner_id]

    async def _append_many(self, scope: Scope, items: Sequence[T]) -> list[MemoryRef]:
        refs = []
        for item in items:
            entry_id = f"{self._prefix}_20260101_{len(self.records) + 1:04d}"
            ref = MemoryRef(
                id=f"{scope.owner_id}_{entry_id}",
                entry_id=entry_id,
                md_path=(
                    f"{scope.owner_type}s/{scope.owner_id}/{self._folder}/2026-01-01.md"
                ),
            )
            self.records.append(Stored(ref=ref, scope=scope, item=item))
            refs.append(ref)
        return refs

    async def _deprecate(
        self, *, entries: Sequence[tuple[str, str]], replaced_by: str
    ) -> None:
        for md_path, entry_id in entries:
            self.deprecated[md_path][entry_id] = replaced_by


class InMemoryEpisodeStore(_DailyLog[Episode]):
    def __init__(self) -> None:
        super().__init__("ep", "episodes")

    async def append(self, scope: Scope, episode: Episode) -> MemoryRef:
        (ref,) = await self._append_many(scope, [episode])
        return ref

    async def append_many(
        self, scope: Scope, episodes: Sequence[Episode]
    ) -> list[MemoryRef]:
        return await self._append_many(scope, episodes)

    async def deprecate(
        self, scope: Scope, *, entries: Sequence[tuple[str, str]], replaced_by: str
    ) -> None:
        await self._deprecate(entries=entries, replaced_by=replaced_by)


class InMemoryAtomicFactStore(_DailyLog[AtomicFact]):
    def __init__(self) -> None:
        super().__init__("af", ".atomic_facts")

    async def append_many(
        self, scope: Scope, facts: Sequence[AtomicFact]
    ) -> list[MemoryRef]:
        return await self._append_many(scope, facts)

    async def deprecate(
        self, scope: Scope, *, entries: Sequence[tuple[str, str]], replaced_by: str
    ) -> None:
        await self._deprecate(entries=entries, replaced_by=replaced_by)


class InMemoryForesightStore(_DailyLog[Foresight]):
    def __init__(self) -> None:
        super().__init__("fs", ".foresights")

    async def append_many(
        self, scope: Scope, items: Sequence[Foresight]
    ) -> list[MemoryRef]:
        return await self._append_many(scope, items)


class InMemoryAgentCaseStore(_DailyLog[AgentCase]):
    def __init__(self) -> None:
        super().__init__("ac", ".agent_cases")

    async def append(self, scope: Scope, case: AgentCase) -> MemoryRef:
        (ref,) = await self._append_many(scope, [case])
        return ref


class InMemoryProfileStore:
    """One profile per scope; ``writes`` records every write in order."""

    def __init__(self) -> None:
        self.profiles: dict[Scope, Profile] = {}
        self.writes: list[tuple[Scope, Profile]] = []

    async def read(self, scope: Scope) -> Profile | None:
        return self.profiles.get(scope)

    async def write(self, scope: Scope, profile: Profile) -> MemoryRef:
        self.profiles[scope] = profile
        self.writes.append((scope, profile))
        return MemoryRef(
            id=scope.owner_id,
            entry_id=f"profile_{scope.owner_id}",
            md_path=f"users/{scope.owner_id}/user.md",
        )


class InMemoryAgentSkillStore:
    """Skills per scope keyed by name; the name doubles as the entry id."""

    def __init__(self) -> None:
        self.skills: dict[Scope, dict[str, AgentSkill]] = defaultdict(dict)
        self.deleted: list[tuple[Scope, str]] = []

    def seed(self, scope: Scope, *skills: AgentSkill) -> None:
        for skill in skills:
            self.skills[scope][skill.name] = skill

    async def list_by_cluster(
        self, scope: Scope, cluster_id: str
    ) -> list[Stored[AgentSkill]]:
        return [
            Stored(ref=self._ref(scope, s), scope=scope, item=s)
            for s in self.skills[scope].values()
            if s.cluster_id == cluster_id
        ]

    async def write(self, scope: Scope, skill: AgentSkill) -> MemoryRef:
        self.skills[scope][skill.name] = skill
        return self._ref(scope, skill)

    async def delete(self, scope: Scope, skill_name: str) -> bool:
        self.deleted.append((scope, skill_name))
        return self.skills[scope].pop(skill_name, None) is not None

    @staticmethod
    def _ref(scope: Scope, skill: AgentSkill) -> MemoryRef:
        return MemoryRef(
            id=skill.id,
            entry_id=skill.name,
            md_path=f"agents/{scope.owner_id}/skills/skill_{skill.name}/SKILL.md",
        )


class InMemoryMemcellStore:
    """Memcells by id."""

    def __init__(self) -> None:
        self.cells: dict[str, MemCell] = {}
        self.case_marks: dict[str, tuple[str | None, dict[str, Any]]] = {}

    def seed(self, memcell_id: str, cell: MemCell) -> None:
        self.cells[memcell_id] = cell

    async def insert_many(
        self, session: SessionKey, records: Sequence[MemcellRecord]
    ) -> None:
        for record in records:
            self.cells[record.memcell_id] = record.cell

    async def find_by_ids(self, memcell_ids: Sequence[str]) -> list[MemCell]:
        return [self.cells[i] for i in memcell_ids if i in self.cells]

    async def mark_case_extracted(
        self,
        memcell_id: str,
        *,
        skip_reason: str | None,
        skip_detail: Mapping[str, Any],
    ) -> None:
        self.case_marks[memcell_id] = (skip_reason, dict(skip_detail))


class InMemoryClusterStore:
    """Clusters with their scope, kind and typed membership."""

    def __init__(self) -> None:
        self.clusters: dict[str, Cluster] = {}
        self.owners: dict[str, tuple[Scope, ClusterKind]] = {}
        self.member_types: dict[str, dict[str, str]] = defaultdict(dict)
        self.upserts: list[Cluster] = []

    def seed(self, scope: Scope, kind: ClusterKind, *clusters: Cluster) -> None:
        for cluster in clusters:
            assert cluster.id is not None
            self.clusters[cluster.id] = cluster
            self.owners[cluster.id] = (scope, kind)

    async def list_for_owner(self, scope: Scope, kind: ClusterKind) -> list[Cluster]:
        return [
            c for cid, c in self.clusters.items() if self.owners[cid] == (scope, kind)
        ]

    async def get(self, cluster_id: str) -> Cluster | None:
        return self.clusters.get(cluster_id)

    async def upsert(
        self,
        scope: Scope,
        kind: ClusterKind,
        cluster: Cluster,
        *,
        member_type: ClusterMemberType,
    ) -> None:
        assert cluster.id is not None
        self.clusters[cluster.id] = cluster
        self.owners[cluster.id] = (scope, kind)
        for member in cluster.members:
            self.member_types[cluster.id].setdefault(member, member_type)
        self.upserts.append(cluster)

    async def find_cluster_for_member(
        self, scope: Scope, member_type: ClusterMemberType, member_id: str
    ) -> str | None:
        raise NotImplementedError

    async def members(self, cluster_id: str) -> list[tuple[str, str]]:
        cluster = self.clusters.get(cluster_id)
        if cluster is None:
            return []
        types = self.member_types[cluster_id]
        return [(m, types.get(m, "episode")) for m in cluster.members]

    async def replace_members(
        self,
        cluster_id: str,
        *,
        removed: Collection[str],
        added: str,
        member_type: ClusterMemberType,
        centroid: Sequence[float],
        count: int,
        last_ts: int,
        preview: Sequence[str],
    ) -> None:
        cluster = self.clusters[cluster_id]
        kept = [m for m in cluster.members if m not in removed]
        self.clusters[cluster_id] = cluster.model_copy(
            update={
                "members": [*kept, added],
                "centroid": tuple(centroid),
                "count": count,
                "last_ts": last_ts,
                "preview": list(preview),
            }
        )
        self.member_types[cluster_id][added] = member_type

    async def list_distinct_owners(self) -> list[Scope]:
        return list(dict.fromkeys(scope for scope, _ in self.owners.values()))


class InMemoryReflectionReportStore:
    """Reflection reports in creation order."""

    def __init__(self) -> None:
        self.reports: list[ReflectionReport] = []

    async def create(self, report: ReflectionReport) -> None:
        self.reports.append(report)

    async def reflected_cluster_ids(self, scope: Scope) -> set[str]:
        return {
            r.cluster_id
            for r in self.reports
            if (r.owner_id, r.app_id, r.project_id)
            == (scope.owner_id, scope.app_id, scope.project_id)
        }
