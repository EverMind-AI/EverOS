"""LanceDB repo singleton for the ``decision`` table."""

from __future__ import annotations

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceDailyLogRepoBase

from ..lancedb_manager import get_table
from ..tables.decision import Decision


class _DecisionRepo(LanceDailyLogRepoBase[Decision]):
    schema = Decision

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)


decision_repo = _DecisionRepo()
