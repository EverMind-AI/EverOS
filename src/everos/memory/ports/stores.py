"""Write-side storage ports: the truth (markdown) and the system state (SQLite).

Each protocol is what the domain needs from one kind of record, in domain
types only. Infra adapters implement them; the domain never sees a file
format, a row model, or a path.
"""

from __future__ import annotations

__all__ = [
    "LOST_STATUS_WRITE",
    "NO_SOURCE_CHECK",
    "REPLAYS_EXHAUSTED_PREFIX",
    "AgentCaseStore",
    "AgentSkillStore",
    "AtomicFactStore",
    "ClusterStore",
    "EpisodeStore",
    "ForesightStore",
    "GroupProfileStore",
    "ImageMemoryStore",
    "KnowledgeCatalog",
    "KnowledgeStore",
    "MemberProfileStore",
    "MemcellStore",
    "MemorizeInboxStore",
    "NoSourceCheck",
    "OwnerDirDeletion",
    "OwnerDirStore",
    "ProfileStore",
    "ReflectionReportStore",
    "SceneStore",
    "SessionBufferStore",
    "SessionPurgeStore",
    "SourceCheck",
    "Withdrawal",
    "WithdrawnEntry",
    "WrittenDocument",
]

import datetime as dt
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Protocol, final, runtime_checkable

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
    GroupProfileItem,
    ImageMemory,
    KnowledgeCategory,
    KnowledgeDocument,
    KnowledgeMemory,
    KnowledgeTopic,
    MemberProfile,
    MemCell,
    MemcellRecord,
    MemorizeTask,
    Profile,
    ReflectionReport,
    Scene,
)
from everos.memory.ports.query import (
    MemoryRef,
    Page,
    PageResult,
    Scope,
    SessionKey,
    Stored,
)

# ── Source checks ────────────────────────────────────────────────────────

type SourceCheck = Callable[[], Awaitable[bool]]
"""Whether everything a derived write was made from still exists.

The daily-log stores run it under the owner's write lock, right before the
write, and skip the write when it returns ``False``: a deletion that removed
a source first and then took the same lock cannot be followed by a write
derived from that source. It may read markdown and SQLite; it must not take
the owner lock again (not reentrant), call a model or touch an index.
"""


@final
class NoSourceCheck:
    """The ``only_if`` of a write that has no deletable source to check."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "NO_SOURCE_CHECK"


NO_SOURCE_CHECK: Final = NoSourceCheck()


# ── Withdrawal (a forget inside a group) ─────────────────────────────────


@dataclass(frozen=True)
class WithdrawnEntry:
    """One daily-log entry taken out of the files, exactly as it was.

    ``text`` is the entry as its file held it -- opaque to the domain, which
    only keeps it and hands it back: a reinstatement puts it back under its
    own id, so an undo restores the previous version byte for byte. ``mark``
    is the entry's own supersession mark (``None`` while it was current).
    """

    entry_id: str
    md_path: str
    text: str
    mark: str | None = None


@dataclass(frozen=True)
class Withdrawal:
    """What one withdrawal took out of one owner's files of one kind.

    Attributes:
        entries: The entries removed, with their text and marks.
        released: ``(md_path, entry_id, mark)`` of the entries whose mark named
            a removed entry: the mark was dropped, so the version a removed
            entry had replaced is current again. A reinstatement marks them
            again.
    """

    entries: tuple[WithdrawnEntry, ...] = ()
    released: tuple[tuple[str, str, str], ...] = ()


# ── User-track daily logs ────────────────────────────────────────────────


@runtime_checkable
class EpisodeStore(Protocol):
    """Episode truth: append-only daily logs per owner.

    Every write takes ``only_if`` (no default, so no writer can forget it):
    a skipped write returns ``None`` / an empty list / ``False``.
    """

    async def append(
        self,
        scope: Scope,
        episode: Episode,
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None: ...

    async def append_many(
        self,
        scope: Scope,
        episodes: Sequence[Episode],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]: ...

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        """Retire ordered ``(md_path, entry_id)`` references in favor of one entry.

        The adapter groups references for storage; failures propagate after any
        earlier writes, without an all-or-nothing guarantee.

        Returns:
            Whether the references were marked (``False``: ``only_if`` failed).
        """
        ...

    async def exists(self, scope: Scope, entry_ids: Collection[str]) -> set[str]:
        """The ``entry_ids`` still in the owner's files, merged or not.

        Reads the files without the owner lock, so a source check running
        under it may call this.
        """
        ...

    async def read_many(
        self, scope: Scope, entry_ids: Collection[str]
    ) -> list[Stored[Episode]]:
        """The episodes of ``entry_ids`` still in md (no lock)."""
        ...

    async def remove_session(self, scope: Scope, session_id: str) -> list[MemoryRef]:
        """Remove the owner's entries of ``session_id``; their refs.

        Takes the owner lock once first (a barrier), then rewrites each file
        under it. The caller deletes the session's memcells before.
        """
        ...

    async def remove(self, scope: Scope, entry_ids: Collection[str]) -> list[MemoryRef]:
        """Remove these entries, each file under the owner lock; their refs."""
        ...

    async def merged_into(self, scope: Scope, targets: Collection[str]) -> set[str]:
        """Entries the owner's files mark as merged into one of ``targets``."""
        ...

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        """Drop ``keys`` from the owner's ``deprecated_entries`` marks."""
        ...

    async def read_for_segments(
        self, scope: Scope, memcell_ids: Collection[str]
    ) -> list[Stored[Episode]]:
        """The owner's current episodes written from one of ``memcell_ids``."""
        ...

    async def read_referring(
        self, scope: Scope, image_id: str
    ) -> list[Stored[Episode]]:
        """The owner's current episodes that refer to the picture ``image_id``."""
        ...

    async def withdraw(self, scope: Scope, entry_ids: Collection[str]) -> Withdrawal:
        """Take entries out of the owner's files, keeping their exact text.

        A forget inside a group (docs/group-memory.md, 9.4) takes a forgotten
        version out of markdown rather than marking it, so what a forgotten
        message said is nowhere under the memory root; the forget keeps the
        :class:`Withdrawal` for its undo. An entry whose mark named a removed
        one is current again.
        """
        ...

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        """Put a withdrawal's entries back under their ids, marks included."""
        ...


@runtime_checkable
class ImageMemoryStore(Protocol):
    """Picture-memory truth: append-only daily logs per owner.

    A picture memory is never edited in place. A changed description, or the
    version a forget re-writes without the picture, is a new entry that
    ``deprecate``s the old one -- which is why ``deprecate`` is the only way to
    retire one and there is no ``update``.

    Every write takes ``only_if`` (no default, so no writer can forget it): a
    write whose source was deleted while it ran returns ``None`` / an empty
    list / ``False``.
    """

    async def append(
        self,
        scope: Scope,
        image: ImageMemory,
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None: ...

    async def append_many(
        self,
        scope: Scope,
        images: Sequence[ImageMemory],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]: ...

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        """Retire ordered ``(md_path, entry_id)`` references in favor of one entry."""
        ...

    async def undeprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        """Drop the retirement marks of ``(md_path, entry_id)``, making them current.

        The inverse of :meth:`deprecate`, and the one edit a supersession mark
        ever gets: undo-of-a-forget re-writes the memory as a new entry and
        clears the mark of the version it restores, so the chain of versions
        keeps saying which one is current.
        """
        ...

    async def read_by_image(
        self, scope: Scope, image_id: str, *, limit: int = 1_000
    ) -> list[Stored[ImageMemory]]:
        """Every entry about one picture, marks included.

        An undo reads this instead of the index because a forget erased the
        rows: markdown is the only place the occurrences still exist, and it is
        also where the mark the undo has to clear lives.
        """
        ...

    async def entries_for_session(
        self, scope: Scope, session_id: str
    ) -> list[tuple[str, str, str]]:
        """``(md_path, entry_id, image_id)`` for the owner's pictures of a session.

        Read from markdown, for the same reason as :meth:`read_by_image`: a
        forget of a conversation must act on what the files say, not on what the
        index has caught up to.
        """
        ...

    async def remove_session(self, scope: Scope, session_id: str) -> list[MemoryRef]:
        """Remove the owner's picture memories that came from one session."""
        ...

    async def read_for_segments(
        self, scope: Scope, memcell_ids: Collection[str]
    ) -> list[Stored[ImageMemory]]:
        """Every picture memory written from one of ``memcell_ids``, marks included."""
        ...

    async def read_many(
        self, scope: Scope, entry_ids: Collection[str]
    ) -> list[Stored[ImageMemory]]:
        """The picture memories of ``entry_ids`` still in md, marks included."""
        ...

    async def withdraw(self, scope: Scope, entry_ids: Collection[str]) -> Withdrawal:
        """Take entries out of the owner's files, keeping their exact text.

        A forget inside a group (docs/group-memory.md, 9.4) takes a forgotten
        version out of markdown rather than marking it, so what a forgotten
        message said is nowhere under the memory root; the forget keeps the
        :class:`Withdrawal` for its undo. An entry whose mark named a removed
        one is current again.
        """
        ...

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        """Put a withdrawal's entries back under their ids, marks included."""
        ...


@runtime_checkable
class AtomicFactStore(Protocol):
    """Atomic-fact truth: append-only daily logs per owner (``only_if``: above)."""

    async def append_many(
        self,
        scope: Scope,
        facts: Sequence[AtomicFact],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]: ...

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        """Retire ordered ``(md_path, entry_id)`` references in favor of one entry.

        The adapter groups references for storage; failures propagate after any
        earlier writes, without an all-or-nothing guarantee.

        Returns:
            Whether the references were marked (``False``: ``only_if`` failed).
        """
        ...

    async def remove_session(self, scope: Scope, session_id: str) -> list[MemoryRef]:
        """Remove the owner's entries of ``session_id``; their refs.

        Takes the owner lock once first (a barrier), then rewrites each file
        under it. The caller deletes the session's memcells before.
        """
        ...

    async def remove_by_parents(
        self, scope: Scope, parent_ids: Collection[str]
    ) -> list[MemoryRef]:
        """Remove the owner's facts of these parent episodes; their refs."""
        ...


@runtime_checkable
class ForesightStore(Protocol):
    """Foresight truth: append-only daily logs per owner (``only_if``: above)."""

    async def append_many(
        self,
        scope: Scope,
        items: Sequence[Foresight],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]: ...

    async def remove_session(self, scope: Scope, session_id: str) -> list[MemoryRef]:
        """Remove the owner's entries of ``session_id``; their refs.

        Takes the owner lock once first (a barrier), then rewrites each file
        under it. The caller deletes the session's memcells before.
        """
        ...


@runtime_checkable
class ProfileStore(Protocol):
    """The single profile document of one user."""

    async def read(self, scope: Scope) -> Profile | None: ...

    async def write(self, scope: Scope, profile: Profile) -> MemoryRef: ...


@runtime_checkable
class SceneStore(Protocol):
    """A group's scene versions (``groups/<g>/scenes/``), append-only."""

    async def latest(self, scope: Scope, scene_id: str) -> Stored[Scene] | None:
        """The newest version of one scene, or ``None`` when it has none."""
        ...

    async def append(
        self,
        scope: Scope,
        scene: Scene,
        *,
        supersedes: Stored[Scene] | None,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None:
        """Write a version; mark ``supersedes`` replaced by it, in one step."""
        ...

    async def versions(self, scope: Scope) -> list[Stored[Scene]]:
        """Every version of every scene of the group."""
        ...

    async def withdraw(self, scope: Scope, entry_ids: Collection[str]) -> Withdrawal:
        """Take entries out of the owner's files, keeping their exact text.

        A forget inside a group (docs/group-memory.md, 9.4) takes a forgotten
        version out of markdown rather than marking it, so what a forgotten
        message said is nowhere under the memory root; the forget keeps the
        :class:`Withdrawal` for its undo. An entry whose mark named a removed
        one is current again.
        """
        ...

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        """Put a withdrawal's entries back under their ids, marks included."""
        ...


@runtime_checkable
class MemberProfileStore(Protocol):
    """The member-in-group profiles of a group (``groups/<g>/members/<m>.md``).

    ``scope`` is the group's (``owner_type = group``). A member's profile in a
    group is a document of its own: it is never the member's personal profile
    and never another group's (docs/group-memory.md, D6).
    """

    async def read(self, scope: Scope, member_id: str) -> MemberProfile | None: ...

    async def write(self, scope: Scope, profile: MemberProfile) -> MemoryRef: ...

    async def members(self, scope: Scope) -> list[str]:
        """The members of the group that have a profile document, sorted."""
        ...

    async def remove(
        self, scope: Scope, member_id: str
    ) -> Stored[MemberProfile] | None:
        """Delete the member's profile document; what it held, or ``None``.

        The ref is the document's index row, which the caller erases.
        """
        ...


@runtime_checkable
class GroupProfileStore(Protocol):
    """A group's profile items and its pending batch (docs/group-memory.md, 7.5).

    Items live in ``groups/<g>/profile/group_profile-<YYYY-MM-DD>.md``, one
    entry per item, append-only (D7): a new version is a new entry, and the
    entry it supersedes is marked against it in the same locked write. The
    pending batch -- the segments the next update reads -- is state beside
    them. ``scope`` is the group's (``owner_type = group``); any other scope
    is refused.
    """

    async def items(self, scope: Scope) -> list[Stored[GroupProfileItem]]:
        """Every item of the group, from markdown, in the order written.

        Each comes back with ``item_id`` (its entry id), ``superseded_by``
        (the entry that replaced it, ``None`` while it is the newest version)
        and ``status``: ``forgotten`` when a forget marked it, ``superseded``
        when a later item replaced it, else :meth:`GroupProfileItem.own_status`.
        """
        ...

    async def append(
        self,
        scope: Scope,
        items: Sequence[GroupProfileItem],
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> list[MemoryRef]:
        """Append ``items`` in one locked write; their refs in order.

        An item whose ``supersedes`` names an entry marks that entry as
        replaced by the new one in the same write. ``[]`` when ``only_if``
        finds the sources gone.
        """
        ...

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        """Mark ``(md_path, entry_id)`` entries as replaced by ``replaced_by``."""
        ...

    async def pending(self, scope: Scope) -> list[str]:
        """The segments queued for the next update, oldest first."""
        ...

    async def set_pending(self, scope: Scope, memcell_ids: Sequence[str]) -> None:
        """Replace the queued segments."""
        ...

    async def withdraw(self, scope: Scope, entry_ids: Collection[str]) -> Withdrawal:
        """Take entries out of the owner's files, keeping their exact text.

        A forget inside a group (docs/group-memory.md, 9.4) takes a forgotten
        version out of markdown rather than marking it, so what a forgotten
        message said is nowhere under the memory root; the forget keeps the
        :class:`Withdrawal` for its undo. An entry whose mark named a removed
        one is current again.
        """
        ...

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        """Put a withdrawal's entries back under their ids, marks included."""
        ...


# ── Agent track ──────────────────────────────────────────────────────────


@runtime_checkable
class AgentCaseStore(Protocol):
    """Agent-case truth: append-only daily logs per agent (``only_if``: above)."""

    async def append(
        self,
        scope: Scope,
        case: AgentCase,
        *,
        only_if: SourceCheck | NoSourceCheck,
    ) -> MemoryRef | None: ...

    async def remove_session(self, scope: Scope, session_id: str) -> list[MemoryRef]:
        """Remove the owner's entries of ``session_id``; their refs.

        Takes the owner lock once first (a barrier), then rewrites each file
        under it. The caller deletes the session's memcells before.
        """
        ...


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


@dataclass(frozen=True)
class OwnerDirDeletion:
    """What deleting one owner's directory removed.

    Attributes:
        refs: The daily-log entries, by index table (``episode``,
            ``atomic_fact``, ``foresight``, ``agent_case``).
        count: Entries, plus one per skill directory, plus one for
            ``user.md``.
    """

    refs: Mapping[str, Sequence[MemoryRef]]
    count: int


@runtime_checkable
class OwnerDirStore(Protocol):
    """Whole owner directories: deletion and its crash recovery."""

    async def delete(self, scope: Scope) -> OwnerDirDeletion:
        """Remove the owner's directory, keeping today's id counters."""
        ...

    async def sweep_trash(self) -> int:
        """Finish deletions a crash interrupted; how many it finished."""
        ...

    async def owners_with_session(
        self, app_id: str, project_id: str, session_id: str
    ) -> list[Scope]:
        """Every user and agent whose files mention ``session_id`` (slow path)."""
        ...


# ── Conversation state ───────────────────────────────────────────────────


@runtime_checkable
class SessionBufferStore(Protocol):
    """Messages received but not yet cut into memcells, per session, and the
    cut that turns some of them into memcells."""

    async def list(self, session: SessionKey) -> list[CanonicalMessage]: ...

    async def keep(
        self,
        session: SessionKey,
        messages: Sequence[CanonicalMessage],
        *,
        last_message_ts: dt.datetime,
    ) -> None:
        """Make ``messages`` the whole buffer and stamp ``last_message_ts``,
        in one transaction."""
        ...

    async def cut(
        self,
        session: SessionKey,
        records: Sequence[MemcellRecord],
        *,
        tail: Sequence[CanonicalMessage],
        last_memcell_ts: dt.datetime | None,
    ) -> None:
        """Record the memcells the buffer was cut into, stamp
        ``last_memcell_ts`` (when given) and leave only ``tail`` buffered,
        in one transaction: a failure writes none of it."""
        ...

    async def quarantine(
        self, session: SessionKey, extra: Sequence[CanonicalMessage]
    ) -> int:
        """Set the buffer (plus ``extra``) aside; return how many moved."""
        ...


@runtime_checkable
class MemcellStore(Protocol):
    """Boundary-detected conversation slices."""

    async def insert_many(
        self, session: SessionKey, records: Sequence[MemcellRecord]
    ) -> None: ...

    async def find_by_ids(self, memcell_ids: Sequence[str]) -> list[MemCell]: ...

    async def cells_by_id(self, memcell_ids: Sequence[str]) -> dict[str, MemCell]:
        """The cells of ``memcell_ids`` that exist, keyed by id, in one query.

        :meth:`find_by_ids` returns bare cells, which carry no id: when some of
        the ids are gone the caller cannot tell which cell is which. This is
        the read for a caller that holds several ids and needs each answer.
        """
        ...

    async def cells_in_session(self, session: SessionKey) -> dict[str, MemCell]:
        """Every cell of the session, keyed by memcell id, oldest first.

        What a forget inside a group reads to find the segments holding the
        messages it names (docs/group-memory.md, 9.4).
        """
        ...

    async def recent_in_session(
        self,
        session: SessionKey,
        *,
        until_ms: int,
        limit: int,
        exclude: str | None = None,
    ) -> list[MemCell]:
        """The session's latest cells up to ``until_ms`` (inclusive), newest first.

        At most ``limit`` cells; ``exclude`` leaves one memcell id out (the
        cell the caller already holds). What a cell with too little text of
        its own reads its conversation's language from.
        """
        ...

    async def mark_case_extracted(
        self,
        memcell_id: str,
        *,
        skip_reason: str | None,
        skip_detail: Mapping[str, Any],
    ) -> None:
        """Record that agent-case extraction ran for a memcell, and why it
        produced nothing if it did not.

        "Extracted" means the extraction ran to completion, not that it
        produced a case: ``skip_reason`` is ``None`` exactly when a case came
        out. A gated trajectory is a successful run with no output, so this
        stamp is the only place its reason survives. An unknown
        ``memcell_id`` is a no-op.
        """
        ...


# ── Clusters ─────────────────────────────────────────────────────────────


@runtime_checkable
class ClusterStore(Protocol):
    """Clusters of related memories and their membership."""

    async def list_for_owner(self, scope: Scope, kind: ClusterKind) -> list[Cluster]:
        """Every cluster of one kind the owner has, each with its members."""
        ...

    async def get(self, cluster_id: str) -> Cluster | None:
        """One cluster with its members, or ``None`` when it does not exist."""
        ...

    async def upsert(
        self,
        scope: Scope,
        kind: ClusterKind,
        cluster: Cluster,
        *,
        member_type: ClusterMemberType,
    ) -> None:
        """Record a clustering decision: the snapshot plus any new members."""
        ...

    async def find_cluster_for_member(
        self, scope: Scope, member_type: ClusterMemberType, member_id: str
    ) -> str | None:
        """The cluster a memory already belongs to, or ``None``."""
        ...

    async def members(self, cluster_id: str) -> list[tuple[str, str]]:
        """``(member_id, member_type)`` pairs in the order they joined."""
        ...

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
        """Fold ``removed`` members into the merged memory ``added``.

        The cluster's snapshot (centroid, count, latest timestamp, preview)
        is restamped in the same step, so membership and snapshot never
        disagree.
        """
        ...

    async def detach(
        self,
        cluster_id: str,
        member_ids: Collection[str],
        *,
        count: int,
        last_ts: int,
        preview: Sequence[str],
    ) -> None:
        """Remove members and restamp count / last_ts / preview; the
        centroid stays."""
        ...

    async def delete(self, cluster_id: str) -> None:
        """Delete a cluster with its members."""
        ...

    async def delete_for_owner(self, scope: Scope, kind: ClusterKind) -> int:
        """Delete every cluster of the owner of this kind; how many."""
        ...

    async def list_distinct_owners(self) -> list[Scope]:
        """Every owner that has at least one cluster."""
        ...


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

    async def delete_for_owner(self, scope: Scope) -> int:
        """Delete every report of the owner; how many."""
        ...


@runtime_checkable
class SessionPurgeStore(Protocol):
    """A session's conversation state, for deleting the session."""

    async def purge(self, session: SessionKey) -> list[MemcellRecord]:
        """Delete the session's memcells, buffers and status in one
        transaction, and blank its failed queued requests' bodies.

        Returns the deleted memcells (their senders name the owners whose
        memories came from the session).
        """
        ...


# ── Knowledge base ───────────────────────────────────────────────────────
#
# Knowledge is project-level: records are addressed by ``(app_id,
# project_id)`` and their own ids, never by an owner ``Scope``.


@dataclass(frozen=True)
class WrittenDocument:
    """Where a knowledge document landed -- the storage side of its identity.

    The domain ``KnowledgeDocument`` carries no path; the store finds a
    document by its id, and this value object hands the location to whoever
    keeps a record of it (the catalog), as ``MemoryRef`` does for owner
    memories.

    Attributes:
        location: The document's directory, as an absolute path.
        md_path: The document's ``index.md``, relative to the memory root.
        original_file_path: The stored copy of the uploaded file, when one
            exists.
    """

    location: str
    md_path: str
    original_file_path: str | None


@runtime_checkable
class KnowledgeStore(Protocol):
    """Knowledge truth: one markdown directory per document, plus the taxonomy.

    Every file-system detail of the knowledge tree lives behind this port:
    directory layout, frontmatter, the uploaded originals, the backup a
    replacement keeps until it succeeds.
    """

    async def categories(self, app_id: str, project_id: str) -> list[KnowledgeCategory]:
        """The project's taxonomy, created with its defaults on first use."""
        ...

    async def write_document(
        self,
        app_id: str,
        project_id: str,
        memories: Sequence[KnowledgeMemory],
        *,
        source_name: str | None,
        source_type: str | None,
        original: bytes | None,
    ) -> WrittenDocument:
        """Write a document's topic tree, keeping ``original`` under its name.

        Raises:
            InvalidInputError: ``source_name`` cannot hold ``original``.
            PathTraversalError: ``source_name`` would escape the document.
        """
        ...

    async def replace_document(
        self,
        previous: KnowledgeDocument,
        memories: Sequence[KnowledgeMemory],
        *,
        source_name: str | None,
        source_type: str | None,
        original: bytes | None,
    ) -> WrittenDocument:
        """Write over ``previous``; on any failure its files come back."""
        ...

    async def delete_document(self, app_id: str, project_id: str, doc_id: str) -> int:
        """Remove every directory of ``doc_id``; returns the topic files removed."""
        ...

    async def read_document(
        self, app_id: str, project_id: str, doc_id: str
    ) -> KnowledgeDocument | None:
        """The document as its ``index.md`` records it; ``None`` when absent."""
        ...

    async def update_document(
        self, document: KnowledgeDocument, *, previous_category_id: str
    ) -> WrittenDocument:
        """Write ``document``'s title / category to its markdown.

        A category other than ``previous_category_id`` moves the directory.

        Returns:
            Where the document is afterwards.

        Raises:
            DocumentNotFoundError: The document is not on disk.
            PathTraversalError: The category would leave the knowledge tree.
        """
        ...

    async def original_file(self, document: KnowledgeDocument) -> str | None:
        """Absolute path of the stored upload, when it exists."""
        ...


@runtime_checkable
class KnowledgeCatalog(Protocol):
    """Knowledge state: the document and topic records the cascade keeps.

    Reads serve the document / topic APIs and search; the one write records a
    metadata patch at once instead of waiting for the cascade.
    """

    async def get_document(self, doc_id: str) -> KnowledgeDocument | None: ...

    async def get_documents(
        self, doc_ids: Collection[str]
    ) -> list[KnowledgeDocument]: ...

    async def document_exists(self, doc_id: str) -> bool: ...

    async def list_documents(
        self,
        app_id: str,
        project_id: str,
        *,
        category_id: str | None,
        page: Page,
    ) -> PageResult[KnowledgeDocument]:
        """``page.sort_by`` is ``created_at``, ``updated_at`` or ``title``."""
        ...

    async def count_by_category(
        self, app_id: str, project_id: str
    ) -> dict[str, int]: ...

    async def record_document(
        self, document: KnowledgeDocument, *, md_path: str
    ) -> None:
        """Insert or update ``document``, stored at ``md_path`` (its ``index.md``)."""
        ...

    async def get_topics(self, topic_ids: Sequence[str]) -> list[KnowledgeTopic]: ...

    async def topics_of(self, doc_id: str) -> list[KnowledgeTopic]: ...

    async def count_topics(self, doc_id: str) -> int: ...
