"""LanceDB repo singleton for the ``principle`` table."""

from __future__ import annotations

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceRepoBase

from ..lancedb_manager import get_table
from ..tables.principle import Principle


class _PrincipleRepo(LanceRepoBase[Principle]):
    schema = Principle

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)


principle_repo = _PrincipleRepo()
