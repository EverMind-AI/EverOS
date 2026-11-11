"""The group profile: topics, decisions, dissent and roles (docs/group-memory.md, 7.5).

An append-only set of items, updated by one model call per batch of segments
(``chat.group_profile_update``, :mod:`.prompt`), whose operations code checks
and turns into items (:mod:`.verify`) and whose current state is a fold of the
items (:mod:`.fold`). The prompt lives in EverOS until an everalgo operator
takes over the same registry id (section 11).

External usage::

    from everos.memory.extract.group_profile import (
        GROUP_PROFILE_UPDATE, Segment, build_messages, current_profile, verify,
    )
"""

from __future__ import annotations

from .fold import GroupProfileView as GroupProfileView
from .fold import current_profile as current_profile
from .fold import is_current as is_current
from .prompt import GROUP_PROFILE_UPDATE as GROUP_PROFILE_UPDATE
from .prompt import MAX_TOPICS as MAX_TOPICS
from .prompt import Segment as Segment
from .prompt import build_messages as build_messages
from .reply import GroupProfileReply as GroupProfileReply
from .reply import Operation as Operation
from .verify import DROP_REASONS as DROP_REASONS
from .verify import Verified as Verified
from .verify import drop_summary as drop_summary
from .verify import verify as verify

__all__ = [
    "DROP_REASONS",
    "GROUP_PROFILE_UPDATE",
    "MAX_TOPICS",
    "GroupProfileReply",
    "GroupProfileView",
    "Operation",
    "Segment",
    "Verified",
    "build_messages",
    "current_profile",
    "drop_summary",
    "is_current",
    "verify",
]
