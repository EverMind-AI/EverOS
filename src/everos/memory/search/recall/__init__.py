"""Per-kind recall layer.

External usage::

    from everos.memory.search.recall import (
        KindRecaller,
        RecallerDeps,
        RecallResult,
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
from .atomic_fact import FactsByEpisode as FactsByEpisode
from .base import KindRecaller as KindRecaller
from .base import RecallerDeps as RecallerDeps
from .base import RecallResult as RecallResult
from .base import merge_by_id_max as merge_by_id_max
from .base import merge_stored as merge_stored
from .base import rank_input as rank_input
from .episode import EpisodeRecaller as EpisodeRecaller
from .knowledge_topic import KnowledgeMethod as KnowledgeMethod
from .knowledge_topic import KnowledgeTopicRecaller as KnowledgeTopicRecaller
from .knowledge_topic import TopicRecall as TopicRecall
from .knowledge_topic import TopicRetriever as TopicRetriever
from .maxsim import maxsim_episode_recall as maxsim_episode_recall
from .profile import ProfileRecaller as ProfileRecaller

__all__ = [
    "AgentCaseRecaller",
    "AgentSkillRecaller",
    "AtomicFactRecaller",
    "EpisodeRecaller",
    "FactsByEpisode",
    "KindRecaller",
    "KnowledgeMethod",
    "KnowledgeTopicRecaller",
    "ProfileRecaller",
    "RecallResult",
    "RecallerDeps",
    "TopicRecall",
    "TopicRetriever",
    "maxsim_episode_recall",
    "merge_by_id_max",
    "merge_stored",
    "rank_input",
]
