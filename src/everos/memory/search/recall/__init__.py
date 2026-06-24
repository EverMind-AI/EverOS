"""Per-kind recall layer.

External usage::

    from everos.memory.search.recall import (
        KindRecaller,
        RecallerDeps,
        EpisodeRecaller,
        AtomicFactRecaller,
        AgentCaseRecaller,
        AgentSkillRecaller,
        ProfileRecaller,
        KnowledgeTopicRecaller,
    )
"""

from .agent_case import AgentCaseRecaller as AgentCaseRecaller
from .agent_skill import AgentSkillRecaller as AgentSkillRecaller
from .atomic_fact import AtomicFactRecaller as AtomicFactRecaller
from .base import KindRecaller as KindRecaller
from .base import RecallerDeps as RecallerDeps
from .base import cosine_score_from_distance as cosine_score_from_distance
from .base import row_to_candidate as row_to_candidate
from .episode import EpisodeRecaller as EpisodeRecaller
from .knowledge_topic import KnowledgeTopicRecaller as KnowledgeTopicRecaller
from .profile import ProfileRecaller as ProfileRecaller

__all__ = [
    "AgentCaseRecaller",
    "AgentSkillRecaller",
    "AtomicFactRecaller",
    "EpisodeRecaller",
    "KindRecaller",
    "KnowledgeTopicRecaller",
    "ProfileRecaller",
    "RecallerDeps",
    "cosine_score_from_distance",
    "row_to_candidate",
]
