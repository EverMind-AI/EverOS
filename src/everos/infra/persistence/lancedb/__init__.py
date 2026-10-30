"""LanceDB business persistence layer.

Sits on top of :mod:`._base` (connection factory, ``LanceRepoBase``, the
Arrow schema derivation and the FTS / vector index upkeep) and the neutral
record declarations of :mod:`everos.infra.persistence.records`, and provides:

    * :class:`LanceDbConnection` -- one memory root's async connection and
      table cache (:mod:`.lancedb_manager`)
    * concrete repositories under :mod:`._repos`
    * :class:`LanceIndexBackend` -- the derived-index backend that builds
      those repositories over a connection and owns schema migration,
      verification and the embedding-model attestation (:mod:`.backend`)

External usage::

    from everos.infra.persistence.lancedb import LanceDbConnection, LanceIndexBackend

    backend = LanceIndexBackend(
        LanceDbConnection(root, settings.lancedb), dim=settings.embedding.dim
    )
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
from ._base import LanceRepoBase as LanceRepoBase
from ._base import LanceTimeouts as LanceTimeouts
from ._base import TableSource as TableSource
from ._base import open_lancedb_connection as open_lancedb_connection
from ._repos import LanceIndexContract as LanceIndexContract
from ._repos import build_or_query as build_or_query
from ._repos import render_predicate as render_predicate
from .backend import BUSINESS_SCHEMAS_WITH_VECTOR as BUSINESS_SCHEMAS_WITH_VECTOR
from .backend import LanceDBMigrationError as LanceDBMigrationError
from .backend import LanceDBSchemaMismatchError as LanceDBSchemaMismatchError
from .backend import LanceIndexBackend as LanceIndexBackend
from .lancedb_manager import LanceDbConnection as LanceDbConnection

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
