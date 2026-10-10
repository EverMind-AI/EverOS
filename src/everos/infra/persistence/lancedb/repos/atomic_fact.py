"""LanceDB repo singleton for the ``atomic_fact`` table."""

from __future__ import annotations

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceDailyLogRepoBase

from ...records.atomic_fact import AtomicFact
from ..lancedb_manager import get_table
from ._contract import LanceIndexContract


class _AtomicFactRepo(
    LanceIndexContract[AtomicFact], LanceDailyLogRepoBase[AtomicFact]
):
    schema = AtomicFact

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)


atomic_fact_repo = _AtomicFactRepo()
