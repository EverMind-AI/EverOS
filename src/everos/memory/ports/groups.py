"""Group store port — a group's meta, its roster log and the sessions bound to it.

``groups/<group_id>/group.md`` is the truth: the group's meta in the
frontmatter and the roster as an append-only log of :class:`RosterEvent`
entries (docs/group-memory.md, section 6.1). The adapter mirrors it into the
system database (``group_meta`` / ``group_member`` / ``session_binding``) in
the same locked operation, so a member removed from a group loses access the
moment the write returns, not when the cascade catches up (section 6.2). Every
read below is served from that mirror.

Invariants a caller relies on:

* :meth:`GroupStore.save` folds the whole log *with* the new events before it
  writes anything; a log that does not fold raises
  :class:`~everos.core.errors.RosterConflictError` and leaves both stores as
  they were.
* A session is bound to at most one group (D18). :meth:`GroupStore.bind` never
  rebinds: it answers the group the session is already bound to, and the caller
  decides whether that is a conflict.

:class:`GroupMessageStore` is the group's dedupe ledger (section 5.2): which
messages the group already holds, by key, in whichever of its sessions. It is
system state, not a mirror of ``group.md``.
"""

from __future__ import annotations

__all__ = ["GroupMessageStore", "GroupRecord", "GroupStore"]

from collections.abc import Collection, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from everos.memory.models import Group, Roster, RosterEvent

from .query import Scope, SessionKey


@dataclass(frozen=True)
class GroupRecord:
    """One group as the mirror holds it.

    Attributes:
        group: The meta.
        roster: Every membership interval, open and closed, and the group's
            deliverer, which the record copies from the meta whatever store
            built it (the access checks read it there).
        sessions: Sessions bound to the group, in binding order.
    """

    group: Group
    roster: Roster
    sessions: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.roster.deliverer_id != self.group.deliverer_id:
            roster = self.roster.model_copy(
                update={"deliverer_id": self.group.deliverer_id}
            )
            object.__setattr__(self, "roster", roster)


@runtime_checkable
class GroupStore(Protocol):
    """The groups of one memory root, addressed by ``Scope(owner_type="group")``."""

    async def get(self, scope: Scope) -> GroupRecord | None:
        """The group, or ``None`` when it does not exist."""
        ...

    async def save(
        self, scope: Scope, *, group: Group, events: Sequence[RosterEvent]
    ) -> GroupRecord:
        """Write the meta and append ``events``, creating the group if needed.

        ``group.md`` and the mirror change together, under the group's lock.

        Raises:
            RosterConflictError: The log with ``events`` does not fold.
        """
        ...

    async def group_ids(self, app_id: str, project_id: str) -> list[str]:
        """Every group of one ``(app_id, project_id)``, sorted by id."""
        ...

    async def member_groups(
        self, app_id: str, project_id: str, member_id: str
    ) -> list[str]:
        """The groups of one ``(app_id, project_id)`` that ``member_id`` is a
        current member of (an open interval), sorted by id.

        One indexed read of the mirror: a read over the asker's own groups
        (a project scope, ``include_groups``) asks it first and then reads
        only those rosters, whatever the number of groups in the space.
        """
        ...

    async def binding(self, session: SessionKey) -> str | None:
        """The group ``session`` is bound to, or ``None``."""
        ...

    async def bind(self, session: SessionKey, group_id: str, *, at: int) -> str:
        """Bind ``session`` to ``group_id`` unless it is already bound.

        Returns:
            The group the session is bound to afterwards: ``group_id``, or the
            group an earlier batch bound it to.
        """
        ...


@runtime_checkable
class GroupMessageStore(Protocol):
    """Which messages each group already holds: its dedupe ledger.

    One entry per message a group took, under the message's dedupe key
    (:func:`everos.memory.groups.message_key`), naming the session that holds
    it. An entry is recorded once its message is stored (buffered or cut into
    a segment), and a forget never removes one: a forgotten message is still
    held.
    """

    async def held(self, scope: Scope, keys: Collection[str]) -> dict[str, str]:
        """Of ``keys``, those the group ``scope`` holds, each with its session."""
        ...

    async def record(
        self, scope: Scope, session: SessionKey, keys: Collection[str]
    ) -> None:
        """Note that ``session`` holds ``keys`` for the group ``scope``.

        A key the group held before moves to ``session`` (the message was
        taken again after a ``/memory/delete`` released its old session).
        """
        ...
