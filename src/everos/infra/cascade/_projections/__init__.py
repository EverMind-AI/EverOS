"""Cascade handlers — one per kind, sharing the :class:`Handler` chassis.

Four daily-log handlers (episode / agent_case / image_memory / group_profile)
inherit :class:`BaseDailyLogHandler` for the shared read / diff / embed /
upsert / delete loop; the per-kind subclass only declares its repo binding,
the texts to embed (``_embed_texts``, none for group_profile) and the
``_build_row`` mapping. ``agent_skill``, the profiles (``user_profile`` and
``group_member_profile``, one :class:`ProfileHandler`), ``knowledge_topic``
and ``group_roster`` stand alone — they're single-file kinds (no entries, no
per-entry diff), so they implement :class:`Handler` directly and own their
reconcile loop.
"""

from .agent_case import AgentCaseHandler as AgentCaseHandler
from .agent_skill import AgentSkillHandler as AgentSkillHandler
from .base import Handler as Handler
from .base import HandlerDeps as HandlerDeps
from .episode import EpisodeHandler as EpisodeHandler
from .group_profile import GroupProfileHandler as GroupProfileHandler
from .group_roster import GroupRosterHandler as GroupRosterHandler
from .image_memory import ImageMemoryHandler as ImageMemoryHandler
from .knowledge_document import KnowledgeDocumentHandler as KnowledgeDocumentHandler
from .knowledge_topic import KnowledgeTopicHandler as KnowledgeTopicHandler
from .profile import ProfileHandler as ProfileHandler

__all__ = [
    "AgentCaseHandler",
    "AgentSkillHandler",
    "EpisodeHandler",
    "GroupProfileHandler",
    "GroupRosterHandler",
    "Handler",
    "HandlerDeps",
    "ImageMemoryHandler",
    "KnowledgeDocumentHandler",
    "KnowledgeTopicHandler",
    "ProfileHandler",
]
