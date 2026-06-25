"""``ParentType`` — provenance label for memory records linked back to a source.

Three values cover the current provenance graph:

* :attr:`ParentType.MEMCELL` — the original ingestion unit; every
  business row (episode / foresight / atomic_fact / agent_case) points
  back to a source MemCell by default.
* :attr:`ParentType.EPISODE` — used by atomic facts that are extracted
  from an episode rather than directly from a MemCell.
* :attr:`ParentType.CLUSTER` — used by merged episodes produced by the
  Reflection consolidation mechanism.

LanceDB's pydantic-to-arrow conversion does not accept ``Enum`` field
annotations, so table schemas declare
``parent_type: str = ParentType.MEMCELL.value`` and reference the enum
only at the default-value level.
"""

from __future__ import annotations

from enum import StrEnum


class ParentType(StrEnum):
    """Provenance label of a memory record's parent."""

    MEMCELL = "memcell"
    EPISODE = "episode"
    CLUSTER = "cluster"
