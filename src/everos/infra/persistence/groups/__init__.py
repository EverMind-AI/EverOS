"""Group persistence: ``group.md`` plus the roster mirror in ``system.db``.

:class:`GroupRosterStore` implements :class:`everos.memory.ports.GroupStore`;
the projection functions are shared with the cascade, which re-derives the
mirror from ``group.md`` (docs/group-memory.md, section 6.2).
:class:`PartitionProjectLinkStore` implements
:class:`everos.memory.ports.ProjectLinkStore` by the partition rule: a
project's groups are the groups stored in its space (section 4.4).

External usage::

    from everos.infra.persistence.groups import (
        GroupRosterStore,
        PartitionProjectLinkStore,
    )

    store = GroupRosterStore(root, sqlite.group_roster)
    links = PartitionProjectLinkStore(store)
"""

from __future__ import annotations

from .project_links import PartitionProjectLinkStore as PartitionProjectLinkStore
from .store import GroupRosterStore as GroupRosterStore

__all__ = ["GroupRosterStore", "PartitionProjectLinkStore"]
