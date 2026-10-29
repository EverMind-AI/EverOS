"""Write-side storage ports: the truth (markdown) and the system state (SQLite).

Each protocol is what the domain needs from one kind of record, in domain
types only. Infra adapters implement them; the domain never sees a file
format, a row model, or a path.
"""

from __future__ import annotations

__all__ = [
    "LOST_STATUS_WRITE",
    "REPLAYS_EXHAUSTED_PREFIX",
    "AgentCaseStore",
    "AgentSkillStore",
    "AtomicFactStore",
    "ClusterStore",
    "EpisodeStore",
    "ForesightStore",
    "MemcellStore",
    "MemorizeInboxStore",
    "ProfileStore",
    "ReflectionReportStore",
    "SessionBufferStore",
]

import datetime as dt
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol, runtime_checkable

from everos.memory.models import (
    AgentCase,
    AgentSkill,
    AtomicFact,
    CanonicalMessage,
    Cluster,
    ClusterKind,
    ClusterMemberType,
    Episode,
    Foresight,
    MemCell,
    MemcellRecord,
    MemorizeTask,
    Profile,
    ReflectionReport,
)
from everos.memory.ports.query import (
    MemoryRef,
    PageResult,
    Scope,
    SessionKey,
    Stored,
)

# ── User-track daily logs ────────────────────────────────────────────────


@runtime_checkable
class EpisodeStore(Protocol):
    """Episode truth: append-only daily logs per owner."""

    async def append(self, scope: Scope, episode: Episode) -> MemoryRef: ...

    async def append_many(
        self, scope: Scope, episodes: Sequence[Episode]
    ) -> list[MemoryRef]: ...

    async def deprecate(
        self, scope: Scope, *, replaced: Mapping[str, str], md_path: str
    ) -> None:
        """Record ``entry_id -> replacing entry_id`` on one daily-log file."""
        ...


@runtime_checkable
class AtomicFactStore(Protocol):
    """Atomic-fact truth: append-only daily logs per owner."""

    async def append_many(
        self, scope: Scope, facts: Sequence[AtomicFact]
    ) -> list[MemoryRef]: ...

    async def deprecate(
        self, scope: Scope, *, replaced: Mapping[str, str], md_path: str
    ) -> None:
        """Record ``entry_id -> replacing entry_id`` on one daily-log file."""
        ...


@runtime_checkable
class ForesightStore(Protocol):
    """Foresight truth: append-only daily logs per owner."""

    async def append_many(
        self, scope: Scope, items: Sequence[Foresight]
    ) -> list[MemoryRef]: ...


@runtime_checkable
class ProfileStore(Protocol):
    """The single profile document of one user."""

    async def read(self, scope: Scope) -> Profile | None: ...

    async def write(self, scope: Scope, profile: Profile) -> MemoryRef: ...


# ── Agent track ──────────────────────────────────────────────────────────


@runtime_checkable
class AgentCaseStore(Protocol):
    """Agent-case truth: append-only daily logs per agent."""

    async def append(self, scope: Scope, case: AgentCase) -> MemoryRef: ...


@runtime_checkable
class AgentSkillStore(Protocol):
    """One document per agent skill, addressed by its name.

    A skill's ``MemoryRef.entry_id`` is the storage-safe form of its name.
    """

    async def list_by_cluster(
        self, scope: Scope, cluster_id: str
    ) -> list[Stored[AgentSkill]]: ...

    async def write(self, scope: Scope, skill: AgentSkill) -> MemoryRef: ...

    async def delete(self, scope: Scope, skill_name: str) -> bool: ...


# ── Conversation state ───────────────────────────────────────────────────


@runtime_checkable
class SessionBufferStore(Protocol):
    """Messages received but not yet cut into memcells, per session."""

    async def list(self, session: SessionKey) -> list[CanonicalMessage]: ...

    async def replace(
        self, session: SessionKey, messages: Sequence[CanonicalMessage]
    ) -> None: ...

    async def quarantine(
        self, session: SessionKey, extra: Sequence[CanonicalMessage]
    ) -> int:
        """Set the buffer (plus ``extra``) aside; return how many moved."""
        ...

    async def touch_last_message(
        self, session: SessionKey, ts: dt.datetime
    ) -> None: ...

    async def touch_last_memcell(
        self, session: SessionKey, ts: dt.datetime
    ) -> None: ...


@runtime_checkable
class MemcellStore(Protocol):
    """Boundary-detected conversation slices."""

    async def insert_many(
        self, session: SessionKey, records: Sequence[MemcellRecord]
    ) -> None: ...

    async def find_by_ids(self, memcell_ids: Sequence[str]) -> list[MemCell]: ...


# ── Clusters ─────────────────────────────────────────────────────────────


@runtime_checkable
class ClusterStore(Protocol):
    """Clusters of related memories and their membership."""

    async def list_for_owner(
        self, scope: Scope, kind: ClusterKind
    ) -> list[Cluster]: ...

    async def get(self, cluster_id: str) -> Cluster | None: ...

    async def upsert(
        self,
        scope: Scope,
        kind: ClusterKind,
        cluster: Cluster,
        *,
        member_type: ClusterMemberType,
    ) -> None:
        """Persist the snapshot and add any members not yet recorded."""
        ...

    async def find_cluster_for_member(
        self, scope: Scope, member_type: ClusterMemberType, member_id: str
    ) -> str | None: ...

    async def members(self, cluster_id: str) -> list[tuple[str, str]]:
        """``(member_id, member_type)`` pairs in insertion order."""
        ...

    async def add_member(
        self, cluster_id: str, member_id: str, member_type: ClusterMemberType
    ) -> None: ...

    async def remove_members(
        self, cluster_id: str, member_ids: Collection[str]
    ) -> None: ...

    async def update_metadata(
        self,
        cluster_id: str,
        *,
        centroid: Sequence[float],
        count: int,
        last_ts: int,
        preview: Sequence[str],
    ) -> None: ...

    async def list_ids_and_member_counts(
        self, scope: Scope, kind: ClusterKind
    ) -> list[tuple[str, int]]: ...

    async def list_distinct_owners(self) -> list[Scope]: ...

    async def count(self) -> int: ...


# ── Operational state ────────────────────────────────────────────────────


LOST_STATUS_WRITE = "status write lost; outcome unknown"
"""``last_error`` an inbox store writes on a row whose status write was lost."""
REPLAYS_EXHAUSTED_PREFIX = "replayed "
"""``last_error`` prefix an inbox store writes on a row it stopped replaying."""


@runtime_checkable
class MemorizeInboxStore(Protocol):
    """Durable queue of accepted asynchronous memorize requests."""

    async def enqueue(
        self,
        session: SessionKey,
        *,
        request_id: str,
        is_final: bool,
        defer_extraction: bool,
        payload_json: str,
        message_count: int,
    ) -> MemorizeTask: ...

    async def claim_next(
        self, *, exclude_sessions: Collection[SessionKey] = ()
    ) -> MemorizeTask | None:
        """Move the oldest queued task outside ``exclude_sessions`` to processing."""
        ...

    async def mark(
        self,
        task_id: int,
        *,
        status: str,
        attempt: int | None = None,
        last_error: str | None = None,
        payload_json: str | None = None,
    ) -> None: ...

    async def count_pending(self, session: SessionKey | None = None) -> int: ...

    async def count_status(self, status: str) -> int: ...

    async def fail_abandoned(self, *, exclude_ids: Collection[int] = ()) -> int: ...

    async def requeue_processing(self, *, max_replays: int = 3) -> tuple[int, int]:
        """Requeue tasks left processing; return ``(requeued, given_up)``."""
        ...

    async def oldest_pending_at(self) -> dt.datetime | None: ...

    async def list_tasks(
        self,
        *,
        session: SessionKey | None = None,
        status: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> PageResult[MemorizeTask]: ...

    async def get_task(self, request_id: str) -> MemorizeTask | None: ...

    async def purge_expired(self, *, before: dt.datetime) -> tuple[int, int]:
        """Delete finished and stale tasks; return ``(deleted, never_delivered)``."""
        ...


@runtime_checkable
class ReflectionReportStore(Protocol):
    """Audit trail of Reflection merges."""

    async def create(self, report: ReflectionReport) -> None: ...

    async def reflected_cluster_ids(self, scope: Scope) -> set[str]: ...
