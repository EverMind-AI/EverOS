"""LanceDB table schemas (one ``BaseLanceTable`` subclass per business table).

Schemas live here; cascade-daemon-driven row population is wired
through the matching repo singletons in :mod:`..repos`.

External usage::

    from everos.infra.persistence.lancedb.tables import (
        Episode,
        AtomicFact,
        Foresight,
        AgentCase,
        AgentSkill,
        UserProfile,
        KnowledgeTopic,
        ParentType,
    )
"""

from ._parent_type import ParentType as ParentType
from .agent_case import AgentCase as AgentCase
from .agent_skill import AgentSkill as AgentSkill
from .atomic_fact import AtomicFact as AtomicFact
from .episode import Episode as Episode
from .foresight import Foresight as Foresight
from .knowledge_topic import KnowledgeTopic as KnowledgeTopic
from .user_profile import UserProfile as UserProfile

__all__ = [
    "AgentCase",
    "AgentSkill",
    "AtomicFact",
    "Episode",
    "Foresight",
    "KnowledgeTopic",
    "ParentType",
    "UserProfile",
]
