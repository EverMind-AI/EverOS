"""The current group profile: the fold of its append-only items (section 7.5).

The newest version of each topic, the decisions no later item replaced, the
dissent still open and the roles still active. An item's ``status`` is what
its read gave it (:meth:`GroupProfileItem.read_back`): a superseded or
forgotten item is never current. Pure: no I/O, so the update strategy, the
search attachments and the project roll-up fold the same way.
"""

from __future__ import annotations

__all__ = ["GroupProfileView", "current_profile", "is_current"]

from collections.abc import Sequence
from dataclasses import dataclass, field

from everos.memory.models import GroupProfileItem

_GONE = frozenset({"superseded", "forgotten"})


@dataclass(frozen=True)
class GroupProfileView:
    """The current items by kind.

    Attributes:
        topics: The newest version of each topic, most recently active first
            (a topic's activity is its own span's end, or that of its newest
            current decision when later).
        decisions: Current decisions, newest first.
        dissent: Open dissent, newest first.
        roles: Active roles, newest first.
    """

    topics: list[GroupProfileItem] = field(default_factory=list)
    decisions: list[GroupProfileItem] = field(default_factory=list)
    dissent: list[GroupProfileItem] = field(default_factory=list)
    roles: list[GroupProfileItem] = field(default_factory=list)

    def ids(self) -> set[str]:
        """Every id the view shows: item ids, and topic ids for topics."""
        shown = {i.item_id for i in self.all() if i.item_id}
        return shown | {t.topic_id for t in self.topics if t.topic_id}

    def all(self) -> list[GroupProfileItem]:
        return [*self.topics, *self.decisions, *self.dissent, *self.roles]


def is_current(item: GroupProfileItem) -> bool:
    """Whether ``item`` is the newest version of what it states, still in force."""
    if item.status in _GONE:
        return False
    if item.kind == "dissent":
        return not item.resolved
    if item.kind == "role":
        return not item.retired
    return True


def current_profile(items: Sequence[GroupProfileItem]) -> GroupProfileView:
    """Fold ``items`` (read back, with their statuses) into the current profile."""
    current = [i for i in items if is_current(i)]

    def newest_first(kind: str) -> list[GroupProfileItem]:
        return sorted(
            (i for i in current if i.kind == kind),
            key=lambda i: (i.source_end, i.observed_at, i.item_id or ""),
            reverse=True,
        )

    decisions = newest_first("decision")
    activity: dict[str, int] = {}
    for decision in decisions:
        if decision.topic_id:
            activity[decision.topic_id] = max(
                activity.get(decision.topic_id, 0), decision.source_end
            )
    topics = sorted(
        (i for i in current if i.kind == "topic"),
        key=lambda t: (
            max(t.source_end, activity.get(t.topic_id or "", 0)),
            t.item_id or "",
        ),
        reverse=True,
    )
    return GroupProfileView(
        topics=topics,
        decisions=decisions,
        dissent=newest_first("dissent"),
        roles=newest_first("role"),
    )
