"""LanceDB async persistence primitives.

External usage (connection):
    from everos.infra.persistence.lancedb import open_lancedb_connection

External usage (generic CRUD repository base):
    from everos.infra.persistence.lancedb import LanceRepoBase

External usage (FTS / vector index upkeep over a table declaration):
    from everos.infra.persistence.lancedb._base import ensure_fts_indexes

The record declarations (``IndexRecord`` and its subclasses) live in
:mod:`everos.infra.persistence.records` and name no backend. Turning their
neutral schema into an Arrow schema (:mod:`.arrow`) and building LanceDB's
indexes over it (:mod:`.indexes`) is this package's job.
"""

from .arrow import arrow_schema_for as arrow_schema_for
from .connection import open_lancedb_connection as open_lancedb_connection
from .indexes import VECTOR_INDEX_MAX_DELTAS as VECTOR_INDEX_MAX_DELTAS
from .indexes import (
    VECTOR_INDEX_ROWS_PER_PARTITION as VECTOR_INDEX_ROWS_PER_PARTITION,
)
from .indexes import VECTOR_QUERY_NPROBES as VECTOR_QUERY_NPROBES
from .indexes import ensure_fts_indexes as ensure_fts_indexes
from .indexes import ensure_vector_indexes as ensure_vector_indexes
from .repository import LanceDailyLogRepoBase as LanceDailyLogRepoBase
from .repository import LanceRepoBase as LanceRepoBase
from .repository import LanceTimeouts as LanceTimeouts
from .repository import TableSource as TableSource

__all__ = [
    "VECTOR_INDEX_MAX_DELTAS",
    "VECTOR_INDEX_ROWS_PER_PARTITION",
    "VECTOR_QUERY_NPROBES",
    "LanceDailyLogRepoBase",
    "LanceRepoBase",
    "LanceTimeouts",
    "TableSource",
    "arrow_schema_for",
    "ensure_fts_indexes",
    "ensure_vector_indexes",
    "open_lancedb_connection",
]
