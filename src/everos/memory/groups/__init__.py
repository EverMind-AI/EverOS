"""Groups as memory owners: meta, roster and the group side of a batch.

:class:`RosterManager` authorises and applies roster changes and admits a
``/memory/add`` batch into its group (session binding, inline roster events,
implicit joins). The roster itself -- events folded into membership intervals
-- is :func:`everos.memory.models.fold_roster`; who may read what is
:mod:`everos.memory.access`. :func:`named_members` finds the members a text
names by their roster names and aliases (attribution, and the members a query
names).

External usage::

    from everos.memory.groups import RosterManager

    rosters = RosterManager(stores.groups)
    change = await rosters.change(scope, asker_id="u_chen", events=[...])
"""

from __future__ import annotations

from .manager import BatchAdmission as BatchAdmission
from .manager import BatchMessage as BatchMessage
from .manager import GroupMetaUpdate as GroupMetaUpdate
from .manager import RosterChange as RosterChange
from .manager import RosterManager as RosterManager
from .manager import implicit_joins as implicit_joins
from .names import NamedMember as NamedMember
from .names import named_members as named_members
from .names import names_of as names_of
from .names import roster_names as roster_names

__all__ = [
    "BatchAdmission",
    "BatchMessage",
    "GroupMetaUpdate",
    "NamedMember",
    "RosterChange",
    "RosterManager",
    "implicit_joins",
    "named_members",
    "names_of",
    "roster_names",
]
