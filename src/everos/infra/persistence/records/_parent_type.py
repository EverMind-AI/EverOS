"""``ParentType`` — provenance label for memory records linked back to a source.

Four values cover the current provenance graph:

* :attr:`ParentType.MEMCELL` — the original ingestion unit; every
  business row (episode / foresight / atomic_fact / agent_case) points
  back to a source MemCell by default.
* :attr:`ParentType.EPISODE` — used by atomic facts that are extracted
  from an episode rather than directly from a MemCell.
* :attr:`ParentType.CLUSTER` — used by merged episodes produced by the
  Reflection consolidation mechanism.
* :attr:`ParentType.SCENE` — a group scene version in the episode table; its
  ``parent_id`` is the scene id (docs/group-memory.md, section 7.7).

The backend-neutral index schema (``schema_for``) has no ``Enum`` kind,
so records declare
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
    SCENE = "scene"
