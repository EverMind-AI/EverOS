"""The access policy every read path goes through (docs/group-memory.md, D9).

``read_scope`` compiles who may see what into the predicate every recall puts
into its ``where``; ``visible`` answers the same for one record read outside
the index; ``buffered_messages`` filters a session's in-flight messages; the
``authorize_*`` functions decide roster changes, meta updates and forgets.
Invisible records are absent, never "forbidden": a read that names a record
the asker cannot see gets the same answer as for one that does not exist.

External usage::

    from everos.memory.access import AccessPolicy, ScopeRequest, resolve_identity

    policy = AccessPolicy(stores.groups, projects=stores.project_links)
    scope = await policy.read_scope(
        resolve_identity(header=None, body_asker="u_betty"),
        ScopeRequest(project_id="spring-offsite", group_ids=("trip-team",)),
    )
    where = scope.predicate
"""

from __future__ import annotations

from .identity import Identity as Identity
from .identity import resolve_identity as resolve_identity
from .policy import AccessPolicy as AccessPolicy
from .policy import Audience as Audience
from .policy import GroupView as GroupView
from .policy import ReadScope as ReadScope
from .policy import ScopeRequest as ScopeRequest
from .policy import Window as Window
from .policy import admits as admits
from .policy import authorize_delivery as authorize_delivery
from .policy import authorize_edit as authorize_edit
from .policy import authorize_forget as authorize_forget
from .policy import authorize_manage as authorize_manage
from .policy import authorize_roster_change as authorize_roster_change
from .policy import intersect_windows as intersect_windows
from .policy import is_deliverer as is_deliverer
from .policy import owner_predicate as owner_predicate
from .policy import visible as visible
from .policy import visible_profile as visible_profile

__all__ = [
    "AccessPolicy",
    "Audience",
    "GroupView",
    "Identity",
    "ReadScope",
    "ScopeRequest",
    "Window",
    "admits",
    "authorize_delivery",
    "authorize_edit",
    "authorize_forget",
    "authorize_manage",
    "authorize_roster_change",
    "intersect_windows",
    "is_deliverer",
    "owner_predicate",
    "resolve_identity",
    "visible",
    "visible_profile",
]
