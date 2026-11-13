"""``memory_strength`` -- each memory's strength under the forgetting curve.

One row per memory a read has reinforced (``[search] forgetting_curve``): its
strength ``S`` in days, when it was last reinforced and how many times. A
memory without a row has never been reinforced and reads as the initial
strength. Derived state only: markdown records none of it, and a store
rebuilt from markdown starts every memory over. Rows are written by the
server's one reinforcement writer and never deleted.

PK ``(app_id, project_id, owner_type, owner_id, memory_key)``: ``memory_key``
names the memory inside its owner's space and is shared by its versions
(:func:`everos.memory.search.forgetting.strength_key`).
"""

from __future__ import annotations

from sqlmodel import Field

from everos.infra.persistence.sqlite._base.base import BaseTable


class MemoryStrength(BaseTable, table=True):
    """One memory's strength."""

    __tablename__ = "memory_strength"  # type: ignore[assignment]

    app_id: str = Field(primary_key=True)
    project_id: str = Field(primary_key=True)
    owner_type: str = Field(primary_key=True)
    """``user`` or ``group``."""
    owner_id: str = Field(primary_key=True)
    memory_key: str = Field(primary_key=True)
    strength_days: float
    """``S``: the days its retention takes to fall to 0.9."""
    reinforced_at_ms: int
    """When it was last reinforced, epoch ms."""
    reinforcements: int
    """How many times it was."""
