"""LanceDB async persistence.

External usage (connection):
    from everos.core.persistence.lancedb import open_lancedb_connection

External usage (ORM model basics — re-exported from lancedb.pydantic):
    from everos.core.persistence.lancedb import (
        LanceModel, Vector, BaseLanceTable, touch,
    )

External usage (generic CRUD repository base):
    from everos.core.persistence.lancedb import LanceRepoBase
"""

# Re-export the LanceDB-flavoured Pydantic primitives so business code has a
# single canonical entry point for table schemas.
from lancedb.pydantic import LanceModel as LanceModel
from lancedb.pydantic import Vector as Vector

from .base import BaseLanceTable as BaseLanceTable
from .base import touch as touch
from .connection import open_lancedb_connection as open_lancedb_connection
from .repository import LanceDailyLogRepoBase as LanceDailyLogRepoBase
from .repository import LanceRepoBase as LanceRepoBase

__all__ = [
    "BaseLanceTable",
    "LanceDailyLogRepoBase",
    "LanceModel",
    "LanceRepoBase",
    "Vector",
    "open_lancedb_connection",
    "touch",
]
