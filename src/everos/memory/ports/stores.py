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
    "ClusterStore",
    "EpisodeStore",
    "GroupProfileStore",
    "ImageMemoryStore",
    "MemcellStore",
    "MemorizeInboxStore",
    "NoSourceCheck",
    "OwnerDirStore",
    "ProfileStore",
    "SceneStore",
    "SessionBufferStore",
    "SessionPurgeStore",
    "SourceCheck",
    "Withdrawal",
    "WithdrawnEntry",
]

import datetime as dt
from collections.abc import Awaitable, Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, final, runtime_checkable

from everos.memory.models import (
    CanonicalMessage,
    Cluster,
    ClusterKind,
    ClusterMemberType,
    Episode,
    GroupProfileItem,
    ImageMemory,
    MemCell,
    MemcellRecord,
    MemorizeTask,
    Profile,
    Scene,
)
from everos.memory.ports.query import (
    MemoryRef,
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


# ── A group profile's pending batch ──────────────────────────────────────


@dataclass(frozen=True)
class PendingBatch:
    """The segments a group's next profile update reads, and its failed tries.

    Attributes:
        memcell_ids: The queued segments, oldest first.
        failures: Updates of this batch in a row whose call yielded nothing
            usable (docs/group-memory.md, section 14, Step 2 notes).
        retry_at: Epoch ms before which no update tries the batch again;
            ``0`` when nothing is waited for.
    """

    memcell_ids: tuple[str, ...] = ()
    failures: int = 0
    retry_at: int = 0


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

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        """Drop ``keys`` from the owner's ``deprecated_entries`` marks."""
        ...

    async def deprecations(self, scope: Scope, md_path: str) -> dict[str, str]:
        """One file's marks, keyed by the entry they retire: a successor's
        entry id, or the forget marker."""
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

    async def read_all(self, scope: Scope) -> list[tuple[Stored[Episode], str | None]]:
        """Every entry of the owner's episode files with its mark (``None`` while
        current: a successor's entry id, or the forget marker otherwise).

        What a forget by condition or by owner, and a version history, read:
        markdown, not the index, so the answer does not wait for the cascade.
        """
        ...

    async def withdraw(self, scope: Scope, entry_ids: Collection[str]) -> Withdrawal:
        """Take entries out of the owner's files, keeping their exact text.

        Only the undo of a group forget logged when forgets took records out
        of markdown uses it: it takes out the versions that forget wrote. An
        entry whose mark named a removed one is current again.
        """
        ...

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        """Put a withdrawal's entries back under their ids, marks included
        (the undo of a group forget logged when forgets took records out)."""
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

    async def read_all(self, scope: Scope) -> list[Stored[ImageMemory]]:
        """Every picture memory of the owner, marks included (``superseded_by``,
        and ``status`` ``forgotten`` for the forget marker)."""
        ...

    async def deprecations(self, scope: Scope, md_path: str) -> dict[str, str]:
        """One file's marks, keyed by the entry they retire."""
        ...

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        """Drop ``keys`` from the owner's marks (an undo of a forget)."""
        ...

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        """Put a withdrawal's entries back under their ids, marks included
        (the undo of a group forget logged when forgets took records out)."""
        ...


@runtime_checkable
class ProfileStore(Protocol):
    """A person's profile in a space: one document per person and space.

    ``scope`` is the space: a user's own (``users/<u>/user.md``, the person is
    the user) or a group's (``groups/<g>/members/<m>.md``, the person is a
    member). A member's profile in a group is a document of its own: never
    the member's personal profile, never another group's
    (docs/group-memory.md, D6).
    """

    async def read(self, scope: Scope, person_id: str | None = None) -> Profile | None:
        """The person's profile, or ``None``; ``person_id`` is required in a
        group and is the user in their own space."""
        ...

    async def write(self, scope: Scope, profile: Profile) -> MemoryRef:
        """Write the whole document of ``profile.owner_id`` in ``scope``."""
        ...

    async def people(self, scope: Scope) -> list[str]:
        """The people of the space that have a profile document, sorted."""
        ...

    async def remove(self, scope: Scope, person_id: str) -> Stored[Profile] | None:
        """Delete the person's document; what it held, or ``None``.

        The ref is the document's index row, which the caller erases.
        """
        ...


@runtime_checkable
class SceneStore(Protocol):
    """Scene versions, append-only: a user's (``users/<u>/scenes/``) or a
    group's (``groups/<g>/scenes/``)."""

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
        """Every version of every scene of the owner."""
        ...

    async def read_all(self, scope: Scope) -> list[tuple[Stored[Scene], str | None]]:
        """Every version of every scene of the owner with its mark (``None``
        while current)."""
        ...

    async def deprecate(
        self,
        scope: Scope,
        *,
        entries: Sequence[tuple[str, str]],
        replaced_by: str,
        only_if: SourceCheck | NoSourceCheck,
    ) -> bool:
        """Mark ``(md_path, entry_id)`` entries replaced by ``replaced_by``
        (a successor's entry id, or the forget marker)."""
        ...

    async def deprecations(self, scope: Scope, md_path: str) -> dict[str, str]:
        """One file's marks, keyed by the entry they retire."""
        ...

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        """Drop ``keys`` from the owner's marks (an undo of a forget)."""
        ...

    async def withdraw(self, scope: Scope, entry_ids: Collection[str]) -> Withdrawal:
        """Take entries out of the owner's files, keeping their exact text.

        Only the undo of a group forget logged when forgets took records out
        of markdown uses it: it takes out the versions that forget wrote. An
        entry whose mark named a removed one is current again.
        """
        ...

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        """Put a withdrawal's entries back under their ids, marks included
        (the undo of a group forget logged when forgets took records out)."""
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

    async def pending_batch(self, scope: Scope) -> PendingBatch:
        """The queued segments with the record of their failed updates."""
        ...

    async def set_pending(
        self,
        scope: Scope,
        memcell_ids: Sequence[str],
        *,
        failures: int = 0,
        retry_at: int = 0,
    ) -> None:
        """Replace the queued segments and their failure record (none by default)."""
        ...

    async def deprecations(self, scope: Scope, md_path: str) -> dict[str, str]:
        """One file's marks, keyed by the entry they retire."""
        ...

    async def clear_deprecations(self, scope: Scope, keys: Collection[str]) -> None:
        """Drop ``keys`` from the owner's marks (an undo of a forget)."""
        ...

    async def reinstate(self, scope: Scope, withdrawal: Withdrawal) -> list[MemoryRef]:
        """Put a withdrawal's entries back under their ids, marks included
        (the undo of a group forget logged when forgets took records out)."""
        ...


# ── Owner directories ────────────────────────────────────────────────────


@runtime_checkable
class OwnerDirStore(Protocol):
    """Whole owner directories: who holds a conversation, and the trash of a
    whole-owner deletion an earlier build left half done (2.0 deletes none)."""

    async def sweep_trash(self) -> int:
        """Finish deletions a crash interrupted; how many it finished."""
        ...

    async def owners_with_session(
        self, app_id: str, project_id: str, session_id: str
    ) -> list[Scope]:
        """Every user whose files mention ``session_id`` (slow path)."""
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
        self,
        scope: Scope,
        member_type: ClusterMemberType,
        member_id: str,
        *,
        kind: ClusterKind | None = None,
    ) -> str | None:
        """The cluster a memory already belongs to, or ``None``.

        ``kind`` narrows the lookup to one cluster kind (an older store can
        still hold clusters of kinds nothing reads any more).
        """
        ...

    async def delete(self, cluster_id: str) -> None:
        """Delete a cluster with its members."""
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
class SessionPurgeStore(Protocol):
    """A session's conversation state, for deleting the session."""

    async def purge(self, session: SessionKey) -> list[MemcellRecord]:
        """Delete the session's memcells, buffers and status in one
        transaction, and blank its failed queued requests' bodies.

        Returns the deleted memcells (their senders name the owners whose
        memories came from the session).
        """
        ...
