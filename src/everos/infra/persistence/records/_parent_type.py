"""``ParentType`` — provenance label for memory records linked back to a source.

Two values cover the current provenance graph:

* :attr:`ParentType.MEMCELL` — the original ingestion unit; every
  business row (episode / image_memory) points back to a source MemCell
  by default.
* :attr:`ParentType.SCENE` — a scene version in the episode table; its
  ``parent_id`` is the scene id (docs/scenes.md).

An episode the Reflection merge of an earlier 2.0 build wrote carries
``cluster``, its ``parent_id`` the cluster's id; nothing writes one now.

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
    SCENE = "scene"
