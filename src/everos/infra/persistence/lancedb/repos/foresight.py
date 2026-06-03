"""LanceDB repo singleton for the ``foresight`` table."""

from __future__ import annotations

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceDailyLogRepoBase

from ..lancedb_manager import get_table
from ..tables.foresight import Foresight


class _ForesightRepo(LanceDailyLogRepoBase[Foresight]):
    schema = Foresight

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)


foresight_repo = _ForesightRepo()
