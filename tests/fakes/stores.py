"""In-memory implementations of the write-side store ports.

Each double keeps what was written in plain attributes so a test asserts on
data instead of on calls. Methods no strategy test needs raise
``NotImplementedError`` rather than pretending to work.
"""

from __future__ import annotations

__all__ = [
    "InMemoryClusterStore",
    "InMemoryEpisodeStore",
    "InMemoryForgetLogStore",
    "InMemoryGroupStore",
    "InMemoryImageMemoryStore",
    "InMemoryImageStore",
    "InMemoryMemcellStore",
    "InMemoryMessageForgetStore",
    "InMemoryProfileStore",
    "InMemorySceneStore",
    "UnusedPort",
]

import dataclasses
import json
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from everos.memory.models import (
    FORGOTTEN_MARK,
    Cluster,
    ClusterKind,
    ClusterMemberType,
    Episode,
    Group,
    GroupProfileItem,
    ImageMemory,
    MemCell,
    MemcellRecord,
    Profile,
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
    OwnerType,
    PendingBatch,
    Scope,
    SessionKey,
    SourceCheck,
    Stored,
    Withdrawal,
    WithdrawnEntry,
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
        self.withdrawn: dict[str, Stored[T]] = {}

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

    async def withdraw(self, scope: Scope, entry_ids: Collection[str]) -> Withdrawal:
        """Take records out, keeping each whole (as JSON) for :meth:`reinstate`."""
        wanted = set(entry_ids)
        gone = [
            s for s in self.records if s.scope == scope and s.ref.entry_id in wanted
        ]
        self.records = [s for s in self.records if s not in gone]
        self.withdrawn.update({s.ref.entry_id: s for s in gone})
        return Withdrawal(
            entries=tuple(
                WithdrawnEntry(
                    entry_id=s.ref.entry_id,
                    md_path=s.ref.md_path,
                    text=s.item.model_dump_json(),  # type: ignore[attr-defined]
                    mark=self.deprecated.get(s.ref.md_path, {}).get(s.ref.entry_id),
                )
                for s in gone
            )
        )

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        present = {s.ref.entry_id for s in self.records}
        refs: list[MemoryRef] = []
        for entry in withdrawal.entries:
            stored = self.withdrawn.get(entry.entry_id)
            if stored is None or entry.entry_id in present:
                continue
            self.records.append(stored)
            refs.append(stored.ref)
        return refs


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

    async def read_for_segments(
        self, scope: Scope, memcell_ids: Collection[str]
    ) -> list[Stored[Episode]]:
        wanted = set(memcell_ids)
        marks = {k for m in self.deprecated.values() for k in m}
        return [
            s
            for s in self.records
            if s.scope == scope
            and s.item.parent_id in wanted
            and s.ref.entry_id not in marks
        ]

    async def read_referring(
        self, scope: Scope, image_id: str
    ) -> list[Stored[Episode]]:
        return [
            s
            for s in self.records
            if s.scope == scope
            and (image_id in s.item.episode or image_id in s.item.image_ids)
            and not self.deprecated.get(s.ref.md_path, {}).get(s.ref.entry_id)
        ]

    async def read_all(self, scope: Scope) -> list[tuple[Stored[Episode], str | None]]:
        return [
            (s, self.deprecated.get(s.ref.md_path, {}).get(s.ref.entry_id))
            for s in self.records
            if s.scope == scope
        ]

    def seed(self, scope: Scope, entry_id: str, episode: Episode) -> None:
        """Put an episode under a chosen entry id (as if read from md)."""
        ref = MemoryRef(
            id=f"{scope.owner_id}_{entry_id}",
            entry_id=entry_id,
            md_path=f"{scope.owner_type}s/{scope.owner_id}/episodes/2026-01-01.md",
        )
        self.records.append(Stored(ref=ref, scope=scope, item=episode))


class InMemoryProfileStore:
    """One profile per person and space, by ``(scope, person)``.

    A user's own profile is keyed by their scope and their id, a member's by
    the group's scope and the member. ``writes`` records every write in order.
    """

    def __init__(self) -> None:
        self.profiles: dict[tuple[Scope, str], Profile] = {}
        self.writes: list[tuple[Scope, Profile]] = []

    async def read(self, scope: Scope, person_id: str | None = None) -> Profile | None:
        return self.profiles.get((scope, _person(scope, person_id)))

    async def write(self, scope: Scope, profile: Profile) -> MemoryRef:
        person = _person(
            scope, profile.owner_id if scope.owner_type == "group" else None
        )
        self.profiles[(scope, person)] = profile
        self.writes.append((scope, profile))
        return _profile_ref(scope, person)

    async def people(self, scope: Scope) -> list[str]:
        return sorted(person for (s, person) in self.profiles if s == scope)

    async def remove(self, scope: Scope, person_id: str) -> Stored[Profile] | None:
        held = self.profiles.pop((scope, person_id), None)
        if held is None:
            return None
        return Stored(ref=_profile_ref(scope, person_id), scope=scope, item=held)


def _person(scope: Scope, person_id: str | None) -> str:
    if scope.owner_type == "group":
        if person_id is None:
            raise ValueError("a profile in a group belongs to one member")
        return person_id
    return scope.owner_id


def _profile_ref(scope: Scope, person: str) -> MemoryRef:
    if scope.owner_type == "group":
        return MemoryRef(
            id=f"{scope.owner_id}:{person}",
            entry_id=f"member_profile_{scope.owner_id}_{person}",
            md_path=f"groups/{scope.owner_id}/members/{person}.md",
        )
    return MemoryRef(
        id=person, entry_id=f"profile_{person}", md_path=f"users/{person}/user.md"
    )


class InMemorySceneStore:
    """Scene versions per scope (a user's or a group's), in write order.

    ``written`` holds every version; ``superseded`` the marks beside them,
    keyed by the version they retire -- a successor's entry id, or
    ``FORGOTTEN_MARK``. A user's versions get ``sc_`` ids, a group's ``gsc_``.
    """

    def __init__(self) -> None:
        self.written: list[Stored[Scene]] = []
        self.superseded: dict[str, str] = {}

    @staticmethod
    def _md_path(scope: Scope) -> str:
        return f"{scope.owner_type}s/{scope.owner_id}/scenes/scene.md"

    def current(self, scope: Scope) -> list[Stored[Scene]]:
        """The scope's versions no mark retires, in write order."""
        return [
            v
            for v in self.written
            if v.scope == scope and v.ref.entry_id not in self.superseded
        ]

    async def latest(self, scope: Scope, scene_id: str) -> Stored[Scene] | None:
        mine = [
            v for v in self.written if v.scope == scope and v.item.scene_id == scene_id
        ]
        return max(mine, key=lambda v: v.item.version) if mine else None

    async def versions(self, scope: Scope) -> list[Stored[Scene]]:
        return [v for v in self.written if v.scope == scope]

    async def read_all(self, scope: Scope) -> list[tuple[Stored[Scene], str | None]]:
        return [
            (v, self.superseded.get(v.ref.entry_id))
            for v in self.written
            if v.scope == scope
        ]

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
        prefix = "gsc" if scope.owner_type == "group" else "sc"
        entry_id = f"{prefix}_{len(self.written) + 1:08d}"
        ref = MemoryRef(
            id=f"{scope.owner_id}_{entry_id}",
            entry_id=entry_id,
            md_path=self._md_path(scope),
        )
        self.written.append(Stored(ref=ref, scope=scope, item=scene))
        if supersedes is not None:
            self.superseded[supersedes.ref.entry_id] = entry_id
        return ref

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        if not await _alive(only_if):
            return False
        for _md_path, entry_id in entries:
            self.superseded[entry_id] = replaced_by
        return True

    async def deprecations(self, scope: Scope, md_path: str) -> dict[str, str]:
        mine = {v.ref.entry_id for v in self.written if v.ref.md_path == md_path}
        return {k: v for k, v in self.superseded.items() if k in mine}

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        mine = {v.ref.entry_id for v in self.written if v.scope == scope}
        for key in set(keys) & mine:
            self.superseded.pop(key, None)


class InMemoryGroupProfileStore:
    """Group-profile items per group scope, in write order, with entry marks.

    Mirrors the markdown store's read-back: ``items()`` sets ``item_id`` and
    the status a mark gives (``FORGOTTEN_MARK`` -> forgotten, another entry id
    -> superseded), and a supersede never overwrites a forgotten mark.
    """

    def __init__(self) -> None:
        self.rows: list[Stored[GroupProfileItem]] = []
        self.marks: dict[str, str] = {}
        self.queued: dict[Scope, PendingBatch] = {}

    @staticmethod
    def _check(scope: Scope) -> None:
        if scope.owner_type != "group":
            raise ValueError("a group profile belongs to a group scope")

    async def items(self, scope: Scope) -> list[Stored[GroupProfileItem]]:
        self._check(scope)
        return [
            Stored(
                ref=row.ref,
                scope=row.scope,
                item=row.item.read_back(
                    item_id=row.ref.entry_id, mark=self.marks.get(row.ref.entry_id)
                ),
            )
            for row in self.rows
            if row.scope == scope
        ]

    async def append(
        self,
        scope: Scope,
        items: Sequence[GroupProfileItem],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        self._check(scope)
        if not items:
            return []
        if not isinstance(only_if, NoSourceCheck) and not await only_if():
            return []
        refs: list[MemoryRef] = []
        for item in items:
            entry_id = f"gp_{len(self.rows) + 1:08d}"
            ref = MemoryRef(
                id=f"{scope.owner_id}_{entry_id}",
                entry_id=entry_id,
                md_path=f"groups/{scope.owner_id}/profile/group_profile.md",
            )
            self.rows.append(Stored(ref=ref, scope=scope, item=item))
            old = item.supersedes
            if old is not None and self.marks.get(old) != FORGOTTEN_MARK:
                self.marks[old] = entry_id
            refs.append(ref)
        return refs

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        self._check(scope)
        if not isinstance(only_if, NoSourceCheck) and not await only_if():
            return False
        for _md_path, entry_id in entries:
            self.marks[entry_id] = replaced_by
        return True

    async def deprecations(self, scope: Scope, md_path: str) -> dict[str, str]:
        mine = {r.ref.entry_id for r in self.rows if r.ref.md_path == md_path}
        return {k: v for k, v in self.marks.items() if k in mine}

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        mine = {r.ref.entry_id for r in self.rows if r.scope == scope}
        for key in set(keys) & mine:
            self.marks.pop(key, None)

    async def pending(self, scope: Scope) -> list[str]:
        return list((await self.pending_batch(scope)).memcell_ids)

    async def pending_batch(self, scope: Scope) -> PendingBatch:
        self._check(scope)
        return self.queued.get(scope, PendingBatch())

    async def set_pending(
        self,
        scope: Scope,
        memcell_ids: Sequence[str],
        *,
        failures: int = 0,
        retry_at: int = 0,
    ) -> None:
        self._check(scope)
        self.queued[scope] = PendingBatch(
            memcell_ids=tuple(memcell_ids), failures=failures, retry_at=retry_at
        )


class InMemoryMemcellStore:
    """Memcells by id, with the session each was inserted under."""

    def __init__(self) -> None:
        self.cells: dict[str, MemCell] = {}
        self.sessions: dict[str, SessionKey] = {}
        self.case_marks: dict[str, tuple[str | None, dict[str, Any]]] = {}

    def seed(
        self, memcell_id: str, cell: MemCell, session: SessionKey | None = None
    ) -> None:
        self.cells[memcell_id] = cell
        if session is not None:
            self.sessions[memcell_id] = session

    async def insert_many(
        self, session: SessionKey, records: Sequence[MemcellRecord]
    ) -> None:
        for record in records:
            self.cells[record.memcell_id] = record.cell
            self.sessions[record.memcell_id] = session

    async def find_by_ids(self, memcell_ids: Sequence[str]) -> list[MemCell]:
        return [self.cells[i] for i in memcell_ids if i in self.cells]

    async def cells_by_id(self, memcell_ids: Sequence[str]) -> dict[str, MemCell]:
        return {i: self.cells[i] for i in memcell_ids if i in self.cells}

    async def cells_in_session(self, session: SessionKey) -> dict[str, MemCell]:
        return {
            memcell_id: cell
            for memcell_id, cell in sorted(
                self.cells.items(), key=lambda pair: pair[1].timestamp
            )
            if self.sessions.get(memcell_id) == session
        }

    async def recent_in_session(
        self,
        session: SessionKey,
        *,
        until_ms: int,
        limit: int,
        exclude: str | None = None,
    ) -> list[MemCell]:
        found = sorted(
            (
                (cell.timestamp, memcell_id)
                for memcell_id, cell in self.cells.items()
                if self.sessions.get(memcell_id) == session
                and cell.timestamp <= until_ms
                and memcell_id != exclude
            ),
            reverse=True,
        )
        return [self.cells[memcell_id] for _, memcell_id in found[:limit]]


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
        self,
        scope: Scope,
        member_type: ClusterMemberType,
        member_id: str,
        *,
        kind: ClusterKind | None = None,
    ) -> str | None:
        raise NotImplementedError

    async def delete(self, cluster_id: str) -> None:
        self.clusters.pop(cluster_id, None)
        self.owners.pop(cluster_id, None)
        self.member_types.pop(cluster_id, None)


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
        requested_by: str | None = None,
        undo: Mapping[str, Any] | None = None,
        forget_id: str | None = None,
    ) -> ForgetEntry:
        entry = ForgetEntry(
            forget_id=forget_id or f"fg_{len(self.entries) + 1:016x}",
            target=target,
            target_id=target_id,
            owner_id=owner_id,
            reasons=reason,
            entry_ids=entry_ids,
            episode_ids=episode_ids,
            removed_profile_items=removed_profile_items,
            owner_type=owner_type,
            requested_by=requested_by,
            undo=undo,
            app_id=app_id,
            project_id=project_id,
            created_at=len(self.entries) + 1,
        )
        self.entries.append(entry)
        return entry

    async def get(self, forget_id: str) -> ForgetEntry | None:
        for entry in self.entries:
            if entry.forget_id == forget_id:
                return dataclasses.replace(entry, undone=forget_id in self.undone)
        return None

    def _live(
        self, app_id: str, project_id: str, owner_id: str, owner_type: OwnerType
    ) -> list[ForgetEntry]:
        """The owner's live entries, oldest first; keyed on id **and** type."""
        return [
            entry
            for entry in self.entries
            if (entry.app_id, entry.project_id, entry.owner_id, entry.owner_type)
            == (app_id, project_id, owner_id, owner_type)
            and entry.forget_id not in self.undone
        ]

    async def live_entries(
        self, *, app_id: str, project_id: str, owner_id: str, owner_type: OwnerType
    ) -> list[ForgetEntry]:
        return list(reversed(self._live(app_id, project_id, owner_id, owner_type)))

    async def live_targets(
        self,
        *,
        target: ForgetTarget,
        app_id: str,
        project_id: str,
        owner_id: str,
        owner_type: OwnerType,
    ) -> list[str]:
        return list(
            dict.fromkeys(
                entry.target_id
                for entry in self._live(app_id, project_id, owner_id, owner_type)
                if entry.target == target
            )
        )

    async def holds_picture(self, image_id: str) -> bool:
        return any(
            entry.owner_type == "group"
            and entry.forget_id not in self.undone
            and f'"image_id": "{image_id}"' in json.dumps(entry.undo or {})
            for entry in self.entries
        )

    async def latest_live(
        self,
        *,
        target: ForgetTarget,
        target_id: str,
        app_id: str,
        project_id: str,
        owner_id: str,
        owner_type: OwnerType,
    ) -> ForgetEntry | None:
        for entry in reversed(self._live(app_id, project_id, owner_id, owner_type)):
            if entry.target == target and entry.target_id == target_id:
                return entry
        return None

    async def undone_entries(
        self, *, app_id: str, project_id: str, owner_id: str, owner_type: OwnerType
    ) -> list[ForgetEntry]:
        return [
            dataclasses.replace(entry, undone=True)
            for entry in reversed(self.entries)
            if (entry.app_id, entry.project_id, entry.owner_id, entry.owner_type)
            == (app_id, project_id, owner_id, owner_type)
            and entry.forget_id in self.undone
        ]

    async def mark_undone(
        self, forget_id: str, *, restored: Mapping[str, Any] | None = None
    ) -> bool:
        if forget_id in self.undone:
            return False
        self.undone.add(forget_id)
        if restored:
            self.entries = [
                dataclasses.replace(
                    entry, undo={**(entry.undo or {}), "restored": dict(restored)}
                )
                if entry.forget_id == forget_id and entry.undo is not None
                else entry
                for entry in self.entries
            ]
        return True


class InMemoryMessageForgetStore:
    """Forgotten messages, in memory: ``(session, message_id, forget_id)`` marks.

    ``segments`` maps a memcell id to its session, as the memcell rows do.
    """

    def __init__(self) -> None:
        self.marks: list[tuple[SessionKey, str, str]] = []
        self.undone: set[str] = set()
        self.segments: dict[str, SessionKey] = {}

    async def record(
        self, session: SessionKey, message_ids: Collection[str], *, forget_id: str
    ) -> None:
        for message_id in dict.fromkeys(message_ids):
            mark = (session, message_id, forget_id)
            if mark not in self.marks:
                self.marks.append(mark)

    async def undo(self, forget_id: str) -> int:
        if forget_id in self.undone:
            return 0
        self.undone.add(forget_id)
        return sum(1 for _, _, f in self.marks if f == forget_id)

    async def forgotten(self, session: SessionKey) -> frozenset[str]:
        return frozenset(
            message_id
            for key, message_id, forget_id in self.marks
            if key == session and forget_id not in self.undone
        )

    async def forgotten_in_segments(
        self, memcell_ids: Collection[str]
    ) -> dict[str, frozenset[str]]:
        return {
            memcell_id: await self.forgotten(self.segments[memcell_id])
            for memcell_id in memcell_ids
            if memcell_id in self.segments
        }


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

    async def read_many(
        self, scope: Scope, entry_ids: Collection[str]
    ) -> list[Stored[ImageMemory]]:
        wanted = set(entry_ids)
        return [s for s in await self.read_all(scope) if s.ref.entry_id in wanted]

    async def read_all(self, scope: Scope) -> list[Stored[ImageMemory]]:
        found: list[Stored[ImageMemory]] = []
        for stored in self.records:
            if stored.scope != scope:
                continue
            mark = self.deprecated[stored.ref.md_path].get(stored.ref.entry_id)
            status = (
                "forgotten" if mark == "forgotten" else "superseded" if mark else None
            )
            found.append(
                Stored(
                    ref=stored.ref,
                    scope=stored.scope,
                    item=stored.item.model_copy(
                        update={"superseded_by": mark, "status": status or "current"}
                    ),
                )
            )
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

    async def ensure_derived(self, image_id: str, *, edge: int, quality: int) -> str:
        await self.read(image_id)  # an unknown id fails as the real store's does
        return f"figures/.cache/{image_id}-{edge}.jpeg"

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
            forgotten=image_id in self.forgotten,
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

    async def group_ids(self, app_id: str, project_id: str) -> list[str]:
        return sorted(g for a, p, g in self.groups if (a, p) == (app_id, project_id))

    async def member_groups(
        self, app_id: str, project_id: str, member_id: str
    ) -> list[str]:
        return sorted(
            g
            for a, p, g in self.groups
            if (a, p) == (app_id, project_id)
            and self._record((a, p, g)).roster.is_current_member(member_id)
        )

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
