"""Implicit groups: three or more people talking without naming a group.

docs/group-memory.md, section 7.9. A
session sent without a ``group_id`` whose messages come from three or more
people (role ``user`` senders) is not a conversation of one or two people:
it is remembered once, in the group of those people, and every one of them
reads it as a member (D3, D9). Two people are not a group: each keeps a copy
of every segment in their own memory, as one person does (D14, amended
2026-11-13). The group is implicit -- nobody declared it -- and fully
determined by its people:

* **Its id** is :func:`implicit_group_id` of the people's ids, the same for
  every session between the same people, so the same people share one group
  across their conversations. The prefix (:data:`IMPLICIT_GROUP_PREFIX`) is
  reserved: a client never names a group with it.
* **Its members** are the people and the assistants of the messages that
  created it, all from its creation (its first message), with the group's
  history visible from creation; it has no owner or admin. An assistant
  first seen later joins; a person first seen later is refused, because the
  people are the group's identity: a new set of people is another group,
  which a new session between them gets.
"""

from __future__ import annotations

__all__ = [
    "IMPLICIT_GROUP_MIN_PEOPLE",
    "IMPLICIT_GROUP_PREFIX",
    "implicit_group_id",
    "is_implicit_group",
]

import hashlib
from collections.abc import Collection

IMPLICIT_GROUP_PREFIX = "implicit_"
"""Every implicit group's id starts with it, and no declared group's may."""

IMPLICIT_GROUP_MIN_PEOPLE = 3
"""The people a session needs, without a ``group_id``, to be their group; one
or two people keep their own memories (D14, amended 2026-11-13)."""


def implicit_group_id(people: Collection[str]) -> str:
    """The id of the implicit group of ``people`` (their sender ids).

    Deterministic and order-free: the sorted, distinct ids, hashed. Path-safe
    and of fixed length whatever the ids, so it fits every id limit.

    Raises:
        ValueError: Fewer than :data:`IMPLICIT_GROUP_MIN_PEOPLE` people.
    """
    distinct = sorted(set(people))
    if len(distinct) < IMPLICIT_GROUP_MIN_PEOPLE:
        raise ValueError(
            f"an implicit group needs {IMPLICIT_GROUP_MIN_PEOPLE} or more people"
        )
    digest = hashlib.sha256("\n".join(distinct).encode("utf-8")).hexdigest()
    return f"{IMPLICIT_GROUP_PREFIX}{digest[:32]}"


def is_implicit_group(group_id: str) -> bool:
    """Whether ``group_id`` names an implicit group."""
    return group_id.startswith(IMPLICIT_GROUP_PREFIX)
