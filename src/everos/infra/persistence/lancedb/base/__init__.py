"""LanceDB async persistence primitives.

External usage (connection):
    from everos.infra.persistence.lancedb import open_lancedb_connection

External usage (generic CRUD repository base):
    from everos.infra.persistence.lancedb import LanceRepoBase

The table chassis (``BaseLanceTable``) lives with the table declarations in
:mod:`everos.infra.persistence.records`.
"""

from .connection import open_lancedb_connection as open_lancedb_connection
from .repository import LanceDailyLogRepoBase as LanceDailyLogRepoBase
from .repository import LanceRepoBase as LanceRepoBase
from .repository import LanceTimeouts as LanceTimeouts
from .repository import TableSource as TableSource

__all__ = [
    "LanceDailyLogRepoBase",
    "LanceRepoBase",
    "LanceTimeouts",
    "TableSource",
    "open_lancedb_connection",
]
