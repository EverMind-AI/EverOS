"""Per-kind recall layer.

External usage::

    from everos.memory.search.recall import (
        RecallerDeps,
        RecallResult,
        EpisodeRecaller,
        ImageMemoryRecaller,
        ProfileRecaller,
        KnowledgeTopicRecaller,
    )
"""

from .base import RecallerDeps as RecallerDeps
from .base import RecallResult as RecallResult
from .episode import EpisodeRecaller as EpisodeRecaller
from .image_memory import ImageMemoryRecaller as ImageMemoryRecaller
from .knowledge_topic import KnowledgeMethod as KnowledgeMethod
from .knowledge_topic import KnowledgeTopicRecaller as KnowledgeTopicRecaller
from .knowledge_topic import TopicRecall as TopicRecall
from .knowledge_topic import TopicRetriever as TopicRetriever
from .profile import ProfileRecaller as ProfileRecaller
from .profile import ProfileView as ProfileView
from .profile import fetch_member_profile as fetch_member_profile
from .profile import fetch_user_profile as fetch_user_profile

__all__ = [
    "EpisodeRecaller",
    "ImageMemoryRecaller",
    "KnowledgeMethod",
    "KnowledgeTopicRecaller",
    "ProfileRecaller",
    "ProfileView",
    "RecallResult",
    "RecallerDeps",
    "TopicRecall",
    "TopicRetriever",
    "fetch_member_profile",
    "fetch_user_profile",
]
