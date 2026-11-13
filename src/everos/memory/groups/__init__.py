"""Groups as memory owners: meta, roster and the group side of a batch.

:class:`RosterManager` authorises and applies roster changes and admits a
``/memory/add`` batch into its group (session binding, inline roster events,
implicit joins) -- or into the implicit group of a session's people
(:func:`implicit_group_id`, section 7.9). The roster itself -- events
folded into membership intervals -- is
:func:`everos.memory.models.fold_roster`; who may read what is
:mod:`everos.memory.access`. :func:`named_members` finds the members a text
names by their roster names and aliases (attribution, and the members a query
names); :func:`record_audience` who was there when a record's sources were
said, and who joined since (section 8.6). :func:`without_forgotten` reads a
segment without the messages a live forget names -- what every reader of a
group segment does (section 9.4). :func:`verbatim_part` is the one test of
whether a model's quote copies a group message (sections 7.5 and 8.4).
:class:`GroupDedupe` decides which messages of a batch the group already
holds (:func:`message_key`), so a group takes each message once (section
5.2).

External usage::

    from everos.memory.groups import RosterManager

    rosters = RosterManager(stores.groups)
    change = await rosters.change(scope, asker_id="u_chen", events=[...])
"""

from __future__ import annotations

from .audience import RecordAudience as RecordAudience
from .audience import record_audience as record_audience
from .dedupe import Claim as Claim
from .dedupe import GroupDedupe as GroupDedupe
from .dedupe import message_key as message_key
from .forgotten import readable_segments as readable_segments
from .forgotten import without_forgotten as without_forgotten
from .implicit import IMPLICIT_GROUP_MIN_PEOPLE as IMPLICIT_GROUP_MIN_PEOPLE
from .implicit import IMPLICIT_GROUP_PREFIX as IMPLICIT_GROUP_PREFIX
from .implicit import implicit_group_id as implicit_group_id
from .implicit import is_implicit_group as is_implicit_group
from .manager import BatchAdmission as BatchAdmission
from .manager import BatchMessage as BatchMessage
from .manager import GroupMetaUpdate as GroupMetaUpdate
from .manager import RosterChange as RosterChange
from .manager import RosterManager as RosterManager
from .manager import founding_joins as founding_joins
from .manager import implicit_joins as implicit_joins
from .names import NamedMember as NamedMember
from .names import named_members as named_members
from .names import names_of as names_of
from .names import roster_names as roster_names
from .quotes import normalise_space as normalise_space
from .quotes import verbatim_part as verbatim_part

__all__ = [
    "IMPLICIT_GROUP_MIN_PEOPLE",
    "IMPLICIT_GROUP_PREFIX",
    "BatchAdmission",
    "BatchMessage",
    "Claim",
    "GroupDedupe",
    "GroupMetaUpdate",
    "NamedMember",
    "RecordAudience",
    "RosterChange",
    "RosterManager",
    "founding_joins",
    "implicit_group_id",
    "implicit_joins",
    "is_implicit_group",
    "message_key",
    "named_members",
    "names_of",
    "normalise_space",
    "readable_segments",
    "record_audience",
    "roster_names",
    "verbatim_part",
    "without_forgotten",
]
