"""Forget log — the record an undo of a forget reads.

Forgetting is a mark, not a delete, so most of it is reversible from markdown
alone. Two parts are not:

- the picture file's status, which the picture store clears on restore, and
- the **profile items** a forget removed, which ``user.md`` cannot remember
  because everalgo replaces that document wholesale.

This port is that memory. One entry per forget, scoped to the owner the forget
was for, keyed by what it addressed: a picture hash or a session id. An undo
asks for the newest entry of its target that no undo has reversed yet — never
an older one, because a forget that a later forget superseded must not come
back.

A forget inside a group (docs/group-memory.md, section 9.4) records more: who
asked (``requested_by``) and, in ``undo``, everything the forget took out of
markdown and wrote in its place -- the group's memory of a forgotten message
must leave the files, so the log is the only place the previous versions are
kept. Its undo names the entry by ``forget_id``.

Entries are appended after the changes they describe are in place, so an entry
exists only for a forget that completed. Undoing one stamps it; nothing is ever
deleted, which keeps the log readable as an audit trail.

:class:`MessageForgetStore` is the per-message half of a group forget: which
messages of a session are forgotten (``message_forget``), so that every reader
of a segment -- re-extraction, the member and group profiles, quotes -- skips
them while the segment itself (the memcell payload) keeps them for an undo.
"""

from __future__ import annotations

__all__ = ["ForgetEntry", "ForgetLogStore", "ForgetTarget", "MessageForgetStore"]

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from .query import SessionKey

type ForgetTarget = Literal[
    "image", "session", "message", "member", "record", "owner", "condition"
]
"""What a forget was addressed by: one picture's hash or one conversation; a
user's records by id, everything of the user, or a kind and a time range; and
inside a group also messages, a member, or records by id."""


@dataclass(frozen=True)
class ForgetEntry:
    """One recorded forget, as the log stores it.

    Attributes:
        forget_id: The entry's id (``fg_<hex>``); what an explicit undo names.
        target: What the forget addressed (:data:`ForgetTarget`).
        target_id: The picture's content hash, the session id, the member id,
            or a short label of the messages or records.
        owner_id: Whose memories the forget touched (the group, inside one).
        reasons: The caller's free text, for the audit trail only.
        entry_ids: Picture-memory entry ids the forget marked.
        episode_ids: ``(superseded, replacement)`` episode entry-id pairs the
            forget wrote, in the order it wrote them.
        removed_profile_items: ``(bucket, item)`` pairs the forget took out of
            the profile, kept whole so an undo can put them back.
        undone: Whether an undo has already reversed this forget.
        owner_type: ``user``, ``agent`` or ``group``.
        requested_by: Who asked for a group forget; ``None`` elsewhere.
        undo: What a group forget's undo restores (the material it took out
            of markdown, what it wrote, what it queued), as the forget
            recorded it. A user's forget by records, owner or condition
            keeps the exact entries it marked here. ``None`` otherwise.
        app_id / project_id: The space of ``owner_id``.
        created_at: Epoch ms the forget was recorded; ``None`` when unknown.
    """

    forget_id: str
    target: ForgetTarget
    target_id: str
    owner_id: str
    reasons: str | None = None
    entry_ids: tuple[str, ...] = ()
    episode_ids: tuple[tuple[str, str], ...] = ()
    removed_profile_items: tuple[tuple[str, dict[str, object]], ...] = ()
    undone: bool = False
    owner_type: str = "user"
    requested_by: str | None = None
    undo: Mapping[str, Any] | None = None
    app_id: str = "default"
    project_id: str = "default"
    created_at: int | None = None


@runtime_checkable
class ForgetLogStore(Protocol):
    """The append-only log of forgets, per owner."""

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
        """Append one forget; returns it with the id an undo can name.

        ``forget_id`` is minted when ``None``; a group forget mints its own
        first, because its messages are marked under that id before the log
        entry is written.
        """
        ...

    async def latest_live(
        self,
        *,
        target: ForgetTarget,
        target_id: str,
        app_id: str,
        project_id: str,
        owner_id: str,
    ) -> ForgetEntry | None:
        """The newest forget of this target that no undo has reversed."""
        ...

    async def get(self, forget_id: str) -> ForgetEntry | None:
        """One forget by id, undone or not; ``None`` when there is none."""
        ...

    async def live_targets(
        self, *, target: ForgetTarget, app_id: str, project_id: str, owner_id: str
    ) -> list[str]:
        """The target ids of the owner's forgets of one kind no undo reversed.

        A group's live picture forgets, for instance: every later
        re-extraction of a segment that showed the picture leaves it out.
        """
        ...

    async def live_entries(
        self, *, app_id: str, project_id: str, owner_id: str
    ) -> list[ForgetEntry]:
        """The owner's forgets no undo reversed, newest first.

        What an undo of one forget asks before it brings an entry back: an
        entry another live forget also names stays forgotten.
        """
        ...

    async def holds_picture(self, image_id: str) -> bool:
        """Whether a group forget no undo reversed took out a memory of the picture.

        What the picture-bytes check asks about a hash no live memory refers
        to any more: one a group forgot is served to nobody, while a hash
        that only ever had personal memories behaves as before groups.
        """
        ...

    async def mark_undone(self, forget_id: str) -> bool:
        """Stamp ``undone``; ``False`` when it already was (nothing restores twice)."""
        ...


@runtime_checkable
class MessageForgetStore(Protocol):
    """Which messages of a group's sessions are forgotten (``message_forget``).

    A message is addressed by its session and its id inside the session: the
    client's ``message_id`` when the batch sent one, the generated id
    otherwise -- the id a segment's message carries. A message stays
    forgotten while any forget of it is live (not undone).
    """

    async def record(
        self, session: SessionKey, message_ids: Collection[str], *, forget_id: str
    ) -> None:
        """Mark ``message_ids`` of ``session`` forgotten by ``forget_id``."""
        ...

    async def undo(self, forget_id: str) -> int:
        """Stamp every mark ``forget_id`` wrote undone; how many it stamped."""
        ...

    async def forgotten(self, session: SessionKey) -> frozenset[str]:
        """The ids of the session's messages a live forget names."""
        ...

    async def forgotten_in_segments(
        self, memcell_ids: Collection[str]
    ) -> dict[str, frozenset[str]]:
        """Per segment, the forgotten message ids of its session.

        Keyed by every memcell id of ``memcell_ids`` that exists; the ids are
        the session's (a superset of the segment's own), which is what
        :func:`everos.memory.groups.without_forgotten` takes.
        """
        ...
