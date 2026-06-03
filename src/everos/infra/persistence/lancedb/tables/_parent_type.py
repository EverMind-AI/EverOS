"""``ParentType`` — provenance label for memory records linked back to a source.

Currently the only value is :attr:`ParentType.MEMCELL`: every business row
(episode / foresight / atomic_fact / agent_case) points back to a source
MemCell. The earlier opensource design enumerated ``"episode"`` as an
alternative parent but the production path never wrote that value, so the
new framework collapses the enum to its single in-use member.

Kept as an :class:`enum.Enum` (rather than a bare string constant) so that
adding a future parent kind stays a non-breaking enum extension. LanceDB's
pydantic-to-arrow conversion does not accept ``Enum`` field annotations,
so table schemas declare ``parent_type: str = ParentType.MEMCELL.value``
and reference the enum only at the default-value level.
"""

from __future__ import annotations

from enum import StrEnum


class ParentType(StrEnum):
    """Provenance label of a memory record's parent."""

    MEMCELL = "memcell"
