"""Cascade handlers — one per kind, sharing the :class:`Handler` chassis.

Three daily-log handlers (episode / agent_case / image_memory) inherit
:class:`BaseDailyLogHandler` for the shared
read / diff / embed / upsert / delete loop; the per-kind subclass only
declares its repo binding, the texts to embed (``_embed_texts``) and
the ``_build_row`` mapping. ``agent_skill``,
``user_profile``, ``group_member_profile``, ``knowledge_topic`` and
``group_roster`` stand alone — they're
single-file kinds (no entries, no per-entry diff), so they implement
:class:`Handler` directly and own their reconcile loop.
"""

from .agent_case import AgentCaseHandler as AgentCaseHandler
from .agent_skill import AgentSkillHandler as AgentSkillHandler
from .base import Handler as Handler
from .base import HandlerDeps as HandlerDeps
from .episode import EpisodeHandler as EpisodeHandler
from .group_member_profile import (
    GroupMemberProfileHandler as GroupMemberProfileHandler,
)
from .group_roster import GroupRosterHandler as GroupRosterHandler
from .image_memory import ImageMemoryHandler as ImageMemoryHandler
from .knowledge_document import KnowledgeDocumentHandler as KnowledgeDocumentHandler
from .knowledge_topic import KnowledgeTopicHandler as KnowledgeTopicHandler
from .user_profile import UserProfileHandler as UserProfileHandler

__all__ = [
    "AgentCaseHandler",
    "AgentSkillHandler",
    "EpisodeHandler",
    "GroupMemberProfileHandler",
    "GroupRosterHandler",
    "Handler",
    "HandlerDeps",
    "ImageMemoryHandler",
    "KnowledgeDocumentHandler",
    "KnowledgeTopicHandler",
    "UserProfileHandler",
]
