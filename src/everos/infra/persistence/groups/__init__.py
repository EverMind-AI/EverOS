"""Group persistence: ``group.md`` plus the roster mirror in ``system.db``.

:class:`GroupRosterStore` implements :class:`everos.memory.ports.GroupStore`;
the projection functions are shared with the cascade, which re-derives the
mirror from ``group.md`` (docs/group-memory.md, section 6.2).

External usage::

    from everos.infra.persistence.groups import GroupRosterStore

    store = GroupRosterStore(root, sqlite.group_roster)
"""

from __future__ import annotations

from .store import GroupRosterStore as GroupRosterStore

__all__ = ["GroupRosterStore"]
