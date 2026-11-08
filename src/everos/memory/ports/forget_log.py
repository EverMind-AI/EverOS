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

Entries are appended after the changes they describe are in place, so an entry
exists only for a forget that completed. Undoing one stamps it; nothing is ever
deleted, which keeps the log readable as an audit trail.
"""

from __future__ import annotations

__all__ = ["ForgetEntry", "ForgetLogStore", "ForgetTarget"]

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

type ForgetTarget = Literal["image", "session"]
"""What a forget was addressed by: one picture's hash, or one conversation."""


@dataclass(frozen=True)
class ForgetEntry:
    """One recorded forget, as the log stores it.

    Attributes:
        forget_id: The entry's id (``fg_<hex>``); what an explicit undo names.
        target: ``image`` or ``session``.
        target_id: The picture's content hash, or the session id.
        owner_id: Whose memories the forget touched.
        reasons: The caller's free text, for the audit trail only.
        entry_ids: Picture-memory entry ids the forget marked.
        episode_ids: ``(superseded, replacement)`` episode entry-id pairs the
            forget wrote, in the order it wrote them.
        removed_profile_items: ``(bucket, item)`` pairs the forget took out of
            the profile, kept whole so an undo can put them back.
        undone: Whether an undo has already reversed this forget.
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
    ) -> ForgetEntry:
        """Append one forget; returns it with the id an undo can name."""
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

    async def mark_undone(self, forget_id: str) -> bool:
        """Stamp ``undone``; ``False`` when it already was (nothing restores twice)."""
        ...
