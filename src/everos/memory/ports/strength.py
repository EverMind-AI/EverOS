"""Memory strength -- the forgetting curve's state, and the queue that grows it.

The forgetting curve (``[search] forgetting_curve``,
:mod:`everos.memory.search.forgetting`) gives every memory a strength ``S``
in days and the time it was last reinforced. Both are derived state, kept in
``system.db``: nothing in markdown records them, and a store rebuilt from
markdown starts every memory over at the initial strength.

* :class:`StrengthStore` reads and writes the rows. One row per memory, not
  per version: a memory's versions share its :class:`StrengthKey` (an
  episode's segment, a scene's id, a picture's hash and segment), so a
  person's edit or a scene update keeps what the memory earned.
* :class:`ReinforcementSink` is where the read paths hand what they used --
  the decider's core of a search, the records a ``get`` returned. It queues
  and never blocks a read; a read-only server has none, so it never
  reinforces.
"""

from __future__ import annotations

__all__ = [
    "Reinforcement",
    "ReinforcementSink",
    "Strength",
    "StrengthKey",
    "StrengthStore",
]

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .query import OwnerType


@dataclass(frozen=True, order=True)
class StrengthKey:
    """One memory, across its versions.

    Attributes:
        app_id / project_id: The space.
        owner_type / owner_id: Whose memory (a user, or a group).
        memory: The memory inside the owner's space, e.g. ``ep:memcell:mc_..``
            (an episode, by its segment), ``sc:sc_..`` (a scene, by its id),
            ``im:<hash>:mc_..`` (a picture memory, by picture and segment).
    """

    app_id: str
    project_id: str
    owner_type: OwnerType
    owner_id: str
    memory: str


@dataclass(frozen=True)
class Strength:
    """A memory's stored strength.

    Attributes:
        days: ``S``, in days: the time its retention takes to fall to 0.9.
        reinforced_at: When it was last reinforced, epoch ms.
        reinforcements: How many times it was.
    """

    days: float
    reinforced_at: int
    reinforcements: int


@dataclass(frozen=True)
class Reinforcement:
    """One use of a memory that reinforces it.

    Attributes:
        key: The memory.
        since: The memory's own time, epoch ms (an episode's, a picture's:
            when it happened; a scene version's: its newest member's). A
            memory with no stored strength yet has been forgetting since then.
        at: When it was used, epoch ms.
    """

    key: StrengthKey
    since: int
    at: int


@runtime_checkable
class StrengthStore(Protocol):
    """The stored strengths (``system.db``)."""

    async def read(self, keys: Collection[StrengthKey]) -> dict[StrengthKey, Strength]:
        """The rows of ``keys`` that exist; a key without one is absent."""
        ...

    async def write(self, rows: Mapping[StrengthKey, Strength]) -> None:
        """Insert or replace ``rows``, in one transaction."""
        ...


@runtime_checkable
class ReinforcementSink(Protocol):
    """Where a read hands the memories it used."""

    def reinforce(self, events: Sequence[Reinforcement]) -> None:
        """Queue ``events``; never blocks and never raises."""
        ...
