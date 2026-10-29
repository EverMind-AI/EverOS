"""Derived-index record declarations, one model per business table.

Every index backend reads the same ``TABLE_NAME`` / ``BM25_FIELDS`` /
``PRIMARY_KEY`` declarations from these models (``IndexRecordMeta``) and the
same columns (``IndexRecord`` subclasses, embedding columns marked
``DenseVector``). The records name no backend and no vector width: each
backend derives its physical schema from
:func:`everos.infra.persistence.index.schema_for` at the width the
application configures. Row population is driven by the cascade through the
repositories each backend builds.

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
from .base import IndexRecord as IndexRecord
from .episode import Episode as Episode
from .foresight import Foresight as Foresight
from .index_record import IndexRecordMeta as IndexRecordMeta
from .knowledge_topic import KnowledgeTopic as KnowledgeTopic
from .user_profile import UserProfile as UserProfile
from .vector import DenseVector as DenseVector
from .vector import VectorColumn as VectorColumn
from .vector import is_vector_field as is_vector_field

__all__ = [
    "EMBEDDING_MODEL_KEY",
    "AgentCase",
    "AgentSkill",
    "AtomicFact",
    "DenseVector",
    "Episode",
    "Foresight",
    "IndexRecord",
    "IndexRecordMeta",
    "KnowledgeTopic",
    "ParentType",
    "UserProfile",
    "VectorColumn",
    "is_vector_field",
]
