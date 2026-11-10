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
    "InMemoryForgetLogStore",
    "InMemoryGroupStore",
    "InMemoryImageMemoryStore",
    "InMemoryImageStore",
    "InMemoryMemberProfileStore",
    "InMemoryMemcellStore",
    "InMemoryProfileStore",
    "InMemoryReflectionReportStore",
    "InMemorySceneStore",
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
    Group,
    ImageMemory,
    MemberProfile,
    MemCell,
    MemcellRecord,
    Profile,
    ReflectionReport,
    Roster,
    RosterEvent,
    Scene,
    fold_roster,
)
from everos.memory.ports import (
    ForgetEntry,
    ForgetTarget,
    GroupRecord,
    MemoryRef,
    NoSourceCheck,
    Scope,
    SessionKey,
    SourceCheck,
    Stored,
)


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

    async def _append_many(
        self,
        scope: Scope,
        items: Sequence[T],
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        if not items or not await _alive(only_if):
            return []
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
        self,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        if not await _alive(only_if):
            return False
        for md_path, entry_id in entries:
            self.deprecated[md_path][entry_id] = replaced_by
        return True

    async def deprecations(self, scope: Scope, md_path: str) -> dict[str, str]:
        return dict(self.deprecated.get(md_path, {}))

    async def entries_for_session(
        self, scope: Scope, session_id: str
    ) -> list[tuple[str, str]]:
        return [
            (stored.ref.md_path, stored.ref.entry_id)
            for stored in self.records
            if stored.scope == scope
            and getattr(stored.item, "session_id", None) == session_id
        ]

    async def remove_session(self, scope: Scope, session_id: str) -> list[MemoryRef]:
        return self._remove(
            scope, lambda item: getattr(item, "session_id", None) == session_id
        )

    def _remove(self, scope: Scope, drop: Any) -> list[MemoryRef]:
        gone = [s for s in self.records if s.scope == scope and drop(s.item)]
        self.records = [s for s in self.records if s not in gone]
        return [s.ref for s in gone]


async def _alive(only_if: SourceCheck | NoSourceCheck) -> bool:
    return isinstance(only_if, NoSourceCheck) or await only_if()


class InMemoryEpisodeStore(_DailyLog[Episode]):
    def __init__(self) -> None:
        super().__init__("ep", "episodes")

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
            entries=entries, replaced_by=replaced_by, only_if=only_if
        )

    async def exists(self, scope: Scope, entry_ids: Collection[str]) -> set[str]:
        wanted = set(entry_ids)
        return {
            s.ref.entry_id
            for s in self.records
            if s.scope == scope and s.ref.entry_id in wanted
        }

    async def read_many(
        self, scope: Scope, entry_ids: Collection[str]
    ) -> list[Stored[Episode]]:
        wanted = set(entry_ids)
        return [
            s for s in self.records if s.scope == scope and s.ref.entry_id in wanted
        ]

    async def remove(self, scope: Scope, entry_ids: Collection[str]) -> list[MemoryRef]:
        wanted = set(entry_ids)
        gone = [
            s for s in self.records if s.scope == scope and s.ref.entry_id in wanted
        ]
        self.records = [s for s in self.records if s not in gone]
        return [s.ref for s in gone]

    async def merged_into(self, scope: Scope, targets: Collection[str]) -> set[str]:
        prefix = f"{scope.owner_type}s/{scope.owner_id}/"
        return {
            key
            for md_path, marks in self.deprecated.items()
            if md_path.startswith(prefix)
            for key, merged in marks.items()
            if merged in targets
        }

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        """Drop ``keys`` from every file of this owner's kind.

        The real store knows an entry's file from its id; this fake keys files
        by owner, so it drops the key wherever the owner has it.
        """
        needle = f"/{scope.owner_type}s/{scope.owner_id}/"
        for md_path, marks in self.deprecated.items():
            if needle in f"/{md_path}":
                for key in set(keys) & marks.keys():
                    del marks[key]

    def seed(self, scope: Scope, entry_id: str, episode: Episode) -> None:
        """Put an episode under a chosen entry id (as if read from md)."""
        ref = MemoryRef(
            id=f"{scope.owner_id}_{entry_id}",
            entry_id=entry_id,
            md_path=f"{scope.owner_type}s/{scope.owner_id}/episodes/2026-01-01.md",
        )
        self.records.append(Stored(ref=ref, scope=scope, item=episode))


class InMemoryAtomicFactStore(_DailyLog[AtomicFact]):
    def __init__(self) -> None:
        super().__init__("af", ".atomic_facts")

    async def remove_by_parents(
        self, scope: Scope, parent_ids: Collection[str]
    ) -> list[MemoryRef]:
        parents = set(parent_ids)
        return self._remove(scope, lambda fact: fact.parent_id in parents)

    async def append_many(
        self,
        scope: Scope,
        facts: Sequence[AtomicFact],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        return await self._append_many(scope, facts, only_if)

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        return await self._deprecate(
            entries=entries, replaced_by=replaced_by, only_if=only_if
        )


class InMemoryForesightStore(_DailyLog[Foresight]):
    def __init__(self) -> None:
        super().__init__("fs", ".foresights")

    async def append_many(
        self,
        scope: Scope,
        items: Sequence[Foresight],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        return await self._append_many(scope, items, only_if)


class InMemoryAgentCaseStore(_DailyLog[AgentCase]):
    def __init__(self) -> None:
        super().__init__("ac", ".agent_cases")

    async def append(
        self,
        scope: Scope,
        case: AgentCase,
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None:
        refs = await self._append_many(scope, [case], only_if)
        return refs[0] if refs else None


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


class InMemorySceneStore:
    """Scene versions per group scope, in write order; marks kept beside them."""

    def __init__(self) -> None:
        self.versions: list[Stored[Scene]] = []
        self.superseded: dict[str, str] = {}

    async def latest(self, scope: Scope, scene_id: str) -> Stored[Scene] | None:
        mine = [
            v for v in self.versions if v.scope == scope and v.item.scene_id == scene_id
        ]
        return max(mine, key=lambda v: v.item.version) if mine else None

    async def append(
        self,
        scope: Scope,
        scene: Scene,
        *,
        supersedes: Stored[Scene] | None,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None:
        if not isinstance(only_if, NoSourceCheck) and not await only_if():
            return None
        entry_id = f"gsc_{len(self.versions) + 1:08d}"
        ref = MemoryRef(
            id=f"{scope.owner_id}_{entry_id}",
            entry_id=entry_id,
            md_path=f"groups/{scope.owner_id}/scenes/scene.md",
        )
        self.versions.append(Stored(ref=ref, scope=scope, item=scene))
        if supersedes is not None:
            self.superseded[supersedes.ref.entry_id] = entry_id
        return ref


class InMemoryMemberProfileStore:
    """Member-in-group profiles by ``(group scope, member)``; ``writes`` in order."""

    def __init__(self) -> None:
        self.profiles: dict[tuple[Scope, str], MemberProfile] = {}
        self.writes: list[MemberProfile] = []

    async def read(self, scope: Scope, member_id: str) -> MemberProfile | None:
        return self.profiles.get((scope, member_id))

    async def write(self, scope: Scope, profile: MemberProfile) -> MemoryRef:
        self.profiles[(scope, profile.member_id)] = profile
        self.writes.append(profile)
        return MemoryRef(
            id=f"{scope.owner_id}:{profile.member_id}",
            entry_id=f"member_profile_{scope.owner_id}_{profile.member_id}",
            md_path=f"groups/{scope.owner_id}/members/{profile.member_id}.md",
        )

    async def members(self, scope: Scope) -> list[str]:
        return sorted(member for (s, member) in self.profiles if s == scope)


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

    async def detach(
        self,
        cluster_id: str,
        member_ids: Collection[str],
        *,
        count: int,
        last_ts: int,
        preview: Sequence[str],
    ) -> None:
        cluster = self.clusters[cluster_id]
        self.clusters[cluster_id] = cluster.model_copy(
            update={
                "members": [m for m in cluster.members if m not in member_ids],
                "count": count,
                "last_ts": last_ts,
                "preview": list(preview),
            }
        )

    async def delete(self, cluster_id: str) -> None:
        self.clusters.pop(cluster_id, None)
        self.owners.pop(cluster_id, None)
        self.member_types.pop(cluster_id, None)

    async def delete_for_owner(self, scope: Scope, kind: ClusterKind) -> int:
        ids = [cid for cid, owner in self.owners.items() if owner == (scope, kind)]
        for cid in ids:
            await self.delete(cid)
        return len(ids)

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

    async def delete_for_owner(self, scope: Scope) -> int:
        mine = (scope.owner_id, scope.app_id, scope.project_id)
        kept = [r for r in self.reports if (r.owner_id, r.app_id, r.project_id) != mine]
        removed = len(self.reports) - len(kept)
        self.reports = kept
        return removed


class InMemoryForgetLogStore:
    """The forget log, in memory: entries in append order, undone by a flag."""

    def __init__(self) -> None:
        self.entries: list[ForgetEntry] = []
        self.undone: set[str] = set()

    async def record(
        self,
        *,
        target: ForgetTarget,
        target_id: str,
        app_id: str,
        project_id: str,
        owner_id: str,
        owner_type: str = "user",
        reason: str | None = None,
        entry_ids: tuple[str, ...] = (),
        episode_ids: tuple[tuple[str, str], ...] = (),
        removed_profile_items: tuple[tuple[str, dict[str, object]], ...] = (),
    ) -> ForgetEntry:
        entry = ForgetEntry(
            forget_id=f"fg_{len(self.entries) + 1:016x}",
            target=target,
            target_id=target_id,
            owner_id=owner_id,
            reasons=reason,
            entry_ids=entry_ids,
            episode_ids=episode_ids,
            removed_profile_items=removed_profile_items,
        )
        self.entries.append(entry)
        return entry

    async def latest_live(
        self,
        *,
        target: ForgetTarget,
        target_id: str,
        app_id: str,
        project_id: str,
        owner_id: str,
    ) -> ForgetEntry | None:
        for entry in reversed(self.entries):
            if (
                entry.target == target
                and entry.target_id == target_id
                and entry.owner_id == owner_id
                and entry.forget_id not in self.undone
            ):
                return entry
        return None

    async def mark_undone(self, forget_id: str) -> bool:
        if forget_id in self.undone:
            return False
        self.undone.add(forget_id)
        return True


class InMemoryImageMemoryStore(_DailyLog[ImageMemory]):
    """Image-memory daily log, in memory."""

    def __init__(self) -> None:
        super().__init__("im", "images")

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
        return await self._deprecate(
            entries=entries, replaced_by=replaced_by, only_if=only_if
        )

    async def undeprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        if not await _alive(only_if):
            return False
        for md_path, entry_id in entries:
            self.deprecated[md_path].pop(entry_id, None)
        return True

    async def entries_for_session(
        self, scope: Scope, session_id: str
    ) -> list[tuple[str, str, str]]:
        return [
            (stored.ref.md_path, stored.ref.entry_id, stored.item.image_id)
            for stored in self.records
            if stored.scope == scope and stored.item.session_id == session_id
        ]

    async def read_by_image(
        self, scope: Scope, image_id: str, *, limit: int = 1_000
    ) -> list[Stored[ImageMemory]]:
        found: list[Stored[ImageMemory]] = []
        for stored in self.records:
            if stored.scope != scope or stored.item.image_id != image_id:
                continue
            mark = self.deprecated[stored.ref.md_path].get(stored.ref.entry_id)
            found.append(
                Stored(
                    ref=stored.ref,
                    scope=stored.scope,
                    item=stored.item.model_copy(
                        update={
                            "superseded_by": mark,
                            "status": "superseded" if mark else "current",
                        }
                    ),
                )
            )
            if len(found) >= limit:
                break
        return found


class InMemoryImageStore:
    """Pictures held as bytes in a dict, keyed by content hash.

    Mirrors the real store's contract rather than its filesystem: storing the
    same bytes twice keeps one entry and counts a second reference, so a test
    can assert the deduplication that the design rests on.
    """

    def __init__(self) -> None:
        self.images: dict[str, tuple[bytes, str]] = {}
        self.refs: dict[str, int] = {}
        self.forgotten: set[str] = set()

    async def store(self, data: bytes, *, limits: Any) -> Any:
        from everos.component.images import decode_image

        stored = decode_image(data, limits=limits)
        self.images.setdefault(stored.image_id, (data, stored.media_type))
        self.refs[stored.image_id] = self.refs.get(stored.image_id, 0) + 1
        return stored

    async def keep_unreferenced(self, data: bytes, *, limits: Any) -> Any:
        """Store the bytes without counting a reference (the question picture)."""
        from everos.component.images import decode_image

        stored = decode_image(data, limits=limits)
        self.images.setdefault(stored.image_id, (data, stored.media_type))
        return stored

    async def read(self, image_id: str) -> bytes:
        # Mirrors the real store: an unknown id is ``FileNotFoundError``, which
        # is what the serving route turns into a 404.
        if image_id not in self.images:
            raise FileNotFoundError(image_id)
        return self.images[image_id][0]

    async def read_derived(self, image_id: str, *, edge: int, quality: int) -> bytes:
        return await self.read(image_id)

    async def relpath_of(self, image_id: str) -> str:
        if image_id not in self.images:
            raise FileNotFoundError(image_id)
        return f"figures/{image_id[:2]}/{image_id}.{self.images[image_id][1]}"

    async def path_of(self, image_id: str) -> Any:
        from pathlib import Path

        return Path("/nonexistent") / await self.relpath_of(image_id)

    async def metadata(self, image_id: str) -> Any:
        from everos.component.images import StoredImageInfo

        data, media_type = self.images[image_id]
        return StoredImageInfo(
            image_id=image_id,
            media_type=media_type,
            byte_size=len(data),
            width=1,
            height=1,
            original_path=await self.relpath_of(image_id),
            ref_count=self.refs.get(image_id, 0),
        )

    async def release(self, image_id: str) -> int:
        self.refs[image_id] = max(0, self.refs.get(image_id, 1) - 1)
        return self.refs[image_id]

    async def exists(self, image_id: str) -> bool:
        return image_id in self.images

    async def forget(self, image_id: str) -> bool:
        self.forgotten.add(image_id)
        return image_id in self.images

    async def restore(self, image_id: str) -> bool:
        self.forgotten.discard(image_id)
        return image_id in self.images


class InMemoryGroupStore:
    """:class:`~everos.memory.ports.GroupStore` over dicts, with the real fold.

    ``events`` holds every group's log in append order and ``bindings`` the
    session bindings, so a test asserts on what the store was asked to keep.
    ``saves`` counts writes (a conflict writes nothing).
    """

    def __init__(self) -> None:
        self.groups: dict[tuple[str, str, str], Group] = {}
        self.events: dict[tuple[str, str, str], list[RosterEvent]] = defaultdict(list)
        self.bindings: dict[tuple[str, str, str], str] = {}
        self.saves = 0

    @staticmethod
    def _key(scope: Scope) -> tuple[str, str, str]:
        return (scope.app_id, scope.project_id, scope.owner_id)

    def _record(self, key: tuple[str, str, str]) -> GroupRecord:
        sessions = tuple(
            session
            for (app, project, session), group in self.bindings.items()
            if (app, project, group) == key
        )
        return GroupRecord(
            group=self.groups[key],
            roster=Roster(intervals=fold_roster(self.events[key])),
            sessions=sessions,
        )

    async def get(self, scope: Scope) -> GroupRecord | None:
        key = self._key(scope)
        return self._record(key) if key in self.groups else None

    async def save(
        self, scope: Scope, *, group: Group, events: Sequence[RosterEvent]
    ) -> GroupRecord:
        key = self._key(scope)
        fold_roster([*self.events[key], *events])  # raises before keeping anything
        self.groups[key] = group
        self.events[key].extend(events)
        self.saves += 1
        return self._record(key)

    async def binding(self, session: SessionKey) -> str | None:
        return self.bindings.get(
            (session.app_id, session.project_id, session.session_id)
        )

    async def bind(self, session: SessionKey, group_id: str, *, at: int) -> str:
        key = (session.app_id, session.project_id, session.session_id)
        return self.bindings.setdefault(key, group_id)
