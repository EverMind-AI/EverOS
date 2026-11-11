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
"""

from __future__ import annotations

__all__ = ["GroupRecord", "GroupStore"]

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from everos.memory.models import Group, Roster, RosterEvent

from .query import Scope, SessionKey


@dataclass(frozen=True)
class GroupRecord:
    """One group as the mirror holds it.

    Attributes:
        group: The meta.
        roster: Every membership interval, open and closed.
        sessions: Sessions bound to the group, in binding order.
    """

    group: Group
    roster: Roster
    sessions: tuple[str, ...] = field(default_factory=tuple)


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
