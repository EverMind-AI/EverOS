"""Backend-neutral facade for the rebuildable BM25/vector index.

Markdown remains the source of truth and SQLite remains the system-state
store. This boundary owns only the derived business indexes used by cascade,
search, get and vector backfill. LanceDB and Milvus implement the same typed
ports, so callers do not branch on physical storage: the application builds one
:class:`DerivedIndex` over the configured engine's backend and hands its
:class:`IndexRepositories` to whoever reads or writes the index.
"""

from __future__ import annotations

from ._provenance import EmbeddingProvenance as EmbeddingProvenance
from .derived_index import DerivedIndex as DerivedIndex
from .derived_index import check_primary_keys as check_primary_keys
from .maintenance import IndexMaintenance as IndexMaintenance
from .predicate import All as All
from .predicate import AnyOf as AnyOf
from .predicate import Comparison as Comparison
from .predicate import Contains as Contains
from .predicate import Filter as Filter
from .predicate import In as In
from .predicate import IsNull as IsNull
from .predicate import Predicate as Predicate
from .predicate import Scalar as Scalar
from .predicate import all_of as all_of
from .predicate import any_of as any_of
from .predicate import compare as compare
from .predicate import contains as contains
from .predicate import eq as eq
from .predicate import gt as gt
from .predicate import gte as gte
from .predicate import is_null as is_null
from .predicate import lt as lt
from .predicate import lte as lte
from .predicate import ne as ne
from .predicate import one_of as one_of
from .predicate import primary_key as primary_key
from .protocols import AgentSkillIndexRepository as AgentSkillIndexRepository
from .protocols import EpisodeIndexRepository as EpisodeIndexRepository
from .protocols import IndexBackend as IndexBackend
from .protocols import IndexRepository as IndexRepository
from .repositories import BUSINESS_SCHEMAS as BUSINESS_SCHEMAS
from .repositories import IndexRepositories as IndexRepositories
from .schema import IndexField as IndexField
from .schema import IndexFieldKind as IndexFieldKind
from .schema import IndexSchema as IndexSchema
from .schema import schema_for as schema_for
from .vector_backfill import ProgressCallback as ProgressCallback
from .vector_backfill import ProjectionVectorBackfill as ProjectionVectorBackfill
from .vector_backfill import VectorBackfill as VectorBackfill
from .vector_backfill import VectorBackfillCounts as VectorBackfillCounts
from .vector_backfill import VectorBacklog as VectorBacklog
from .vector_projection import VECTOR_PROJECTIONS as VECTOR_PROJECTIONS
from .vector_projection import VectorProjection as VectorProjection

__all__ = [
    "BUSINESS_SCHEMAS",
    "VECTOR_PROJECTIONS",
    "AgentSkillIndexRepository",
    "All",
    "AnyOf",
    "Comparison",
    "Contains",
    "DerivedIndex",
    "EmbeddingProvenance",
    "EpisodeIndexRepository",
    "Filter",
    "In",
    "IndexBackend",
    "IndexField",
    "IndexFieldKind",
    "IndexMaintenance",
    "IndexRepositories",
    "IndexRepository",
    "IndexSchema",
    "IsNull",
    "Predicate",
    "ProgressCallback",
    "ProjectionVectorBackfill",
    "Scalar",
    "VectorBackfill",
    "VectorBackfillCounts",
    "VectorBacklog",
    "VectorProjection",
    "all_of",
    "any_of",
    "check_primary_keys",
    "compare",
    "contains",
    "eq",
    "gt",
    "gte",
    "is_null",
    "lt",
    "lte",
    "ne",
    "one_of",
    "primary_key",
    "schema_for",
]
