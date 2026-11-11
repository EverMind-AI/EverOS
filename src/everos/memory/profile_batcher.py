"""ProfileBatcher — which episodes a profile run folded ahead of their own runs.

``extract_user_profile`` folds an owner's waiting episodes along with the one
its event names; this records them so their own runs, queued behind the same
owner's lock, end without an LLM call. One instance lives for the process,
built by the bootstrap; every access for an owner happens under that owner's
partition lock, so it needs no lock of its own.

The marks are in memory only: a crash loses them, and those queued runs fold
their episode a second time -- the at-least-once repeat the strategy already
accepts. An owner holds at most ``_MAX_MARKS_PER_OWNER`` marks, oldest dropped
first, so marks whose run never comes cannot grow without bound; dropping a
live one only costs that episode a second fold.
"""

from __future__ import annotations

__all__ = ["ProfileBatcher"]

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from everos.memory.ports import Scope

_MAX_MARKS_PER_OWNER = 1_000


class ProfileBatcher:
    """The per-call episode cap, and the marks of episodes folded ahead.

    Args:
        max_episodes: Most episodes one extraction call folds, the run's own
            included.

    Raises:
        ValueError: If ``max_episodes`` is below 1.
    """

    def __init__(self, max_episodes: int) -> None:
        if max_episodes < 1:
            raise ValueError(f"max_episodes must be >= 1, got {max_episodes}")
        self.max_episodes = max_episodes
        self._marks: dict[Scope, dict[str, None]] = {}

    def mark(self, scope: Scope, entry_ids: Iterable[str]) -> None:
        """Record that ``entry_ids`` of ``scope`` were folded ahead of their runs."""
        marks = self._marks.setdefault(scope, {})
        marks.update(dict.fromkeys(entry_ids))
        for stale in list(marks)[: max(len(marks) - _MAX_MARKS_PER_OWNER, 0)]:
            del marks[stale]
        if not marks:
            del self._marks[scope]

    def is_marked(self, scope: Scope, entry_id: str) -> bool:
        """Whether ``entry_id`` of ``scope`` was folded ahead of its run."""
        return entry_id in self._marks.get(scope, {})

    def take(self, scope: Scope, entry_id: str) -> bool:
        """Remove ``entry_id``'s mark and return whether it had one."""
        marks = self._marks.get(scope)
        if marks is None or entry_id not in marks:
            return False
        del marks[entry_id]
        if not marks:
            del self._marks[scope]
        return True
