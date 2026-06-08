"""Shared StrEnum types used across EverOS layers.

These enums live at the ``everos.core`` level so both ``infra`` and
``memory`` can import them without violating the layered-architecture
contract enforced by import-linter.
"""

from __future__ import annotations

from enum import StrEnum


class ChangeKind(StrEnum):
    """Registered cascade handler kinds.

    Each value corresponds to a :class:`Handler` subclass's ``kind``
    class attribute in :mod:`everos.memory.cascade.handlers`.
    """

    EPISODE = "episode"
    ATOMIC_FACT = "atomic_fact"
    FORESIGHT = "foresight"
    AGENT_CASE = "agent_case"
    AGENT_SKILL = "agent_skill"
    USER_PROFILE = "user_profile"


class ChangeType(StrEnum):
    """Lifecycle hint for a single md path's work-queue row.

    The handler re-derives truth from the actual file state at run
    time (DD-3 in 12 doc); this field is a dispatch hint only.
    """

    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"


class ChangeStatus(StrEnum):
    """Work-queue row lifecycle.

    ``PROCESSING`` is an internal claim state used by
    :meth:`MdChangeStateRepo.claim_one`; CLI output rolls it back
    into ``PENDING`` for display (16 doc §4.2 — DD-12).
    """

    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
