"""Derived-index record declarations, one model per business table.

Every index backend reads the same ``TABLE_NAME`` / ``BM25_FIELDS`` /
``PRIMARY_KEY`` declarations from these models (``IndexRecordMeta``). The
models are still ``BaseLanceTable`` subclasses with LanceDB ``Vector``
columns — decoupling them from the lancedb library is issue #476. Row
population is driven by the cascade through the repo singletons in
:mod:`everos.infra.persistence.lancedb.repos` and
:mod:`everos.infra.persistence.milvus.repos`.

External usage::

    from everos.infra.persistence.records import (
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
from ._provenance_key import EMBEDDING_MODEL_KEY as EMBEDDING_MODEL_KEY
from .agent_case import AgentCase as AgentCase
from .agent_skill import AgentSkill as AgentSkill
from .atomic_fact import AtomicFact as AtomicFact
from .episode import Episode as Episode
from .foresight import Foresight as Foresight
from .knowledge_topic import KnowledgeTopic as KnowledgeTopic
from .user_profile import UserProfile as UserProfile

__all__ = [
    "EMBEDDING_MODEL_KEY",
    "AgentCase",
    "AgentSkill",
    "AtomicFact",
    "Episode",
    "Foresight",
    "KnowledgeTopic",
    "ParentType",
    "UserProfile",
]
