"""LanceDB business persistence layer.

Sits on top of :mod:`.base` (connection factory + ``LanceRepoBase``)
and :mod:`everos.infra.persistence.records` (``BaseLanceTable``) and provides:

    * :class:`LanceDbConnection` -- one memory root's async connection and
      table cache (:mod:`.lancedb_manager`)
    * concrete repositories under :mod:`.repos`
    * :class:`LanceIndexBackend` -- the derived-index backend that builds
      those repositories over a connection and owns schema migration,
      verification and the embedding-model attestation (:mod:`.backend`)

External usage::

    from everos.infra.persistence.lancedb import LanceDbConnection, LanceIndexBackend

    backend = LanceIndexBackend(LanceDbConnection(root, settings.lancedb))
    await backend.repositories.episode.upsert([...])

Three index kinds: scalar / BM25 / vector. Tables are created lazily on
first access; row population is the cascade daemon's job (see
``docs/cascade_runbook.md``).
"""

from ..records import EMBEDDING_MODEL_KEY as EMBEDDING_MODEL_KEY
from ..records import AgentCase as AgentCase
from ..records import AgentSkill as AgentSkill
from ..records import AtomicFact as AtomicFact
from ..records import Episode as Episode
from ..records import Foresight as Foresight
from ..records import KnowledgeTopic as KnowledgeTopic
from ..records import ParentType as ParentType
from ..records import UserProfile as UserProfile
from .backend import BUSINESS_SCHEMAS_WITH_VECTOR as BUSINESS_SCHEMAS_WITH_VECTOR
from .backend import LanceDBMigrationError as LanceDBMigrationError
from .backend import LanceDBSchemaMismatchError as LanceDBSchemaMismatchError
from .backend import LanceIndexBackend as LanceIndexBackend
from .base import LanceRepoBase as LanceRepoBase
from .base import LanceTimeouts as LanceTimeouts
from .base import TableSource as TableSource
from .base import open_lancedb_connection as open_lancedb_connection
from .lancedb_manager import LanceDbConnection as LanceDbConnection
from .repos import LanceIndexContract as LanceIndexContract
from .repos import build_or_query as build_or_query
from .repos import render_predicate as render_predicate

__all__ = [
    "BUSINESS_SCHEMAS_WITH_VECTOR",
    "EMBEDDING_MODEL_KEY",
    "AgentCase",
    "AgentSkill",
    "AtomicFact",
    "Episode",
    "Foresight",
    "KnowledgeTopic",
    "LanceDBMigrationError",
    "LanceDBSchemaMismatchError",
    "LanceDbConnection",
    "LanceIndexBackend",
    "LanceIndexContract",
    "LanceRepoBase",
    "LanceTimeouts",
    "ParentType",
    "TableSource",
    "UserProfile",
    "build_or_query",
    "open_lancedb_connection",
    "render_predicate",
]
