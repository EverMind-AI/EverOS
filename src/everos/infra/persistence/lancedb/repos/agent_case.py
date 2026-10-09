"""LanceDB repo singleton for the ``agent_case`` table."""

from __future__ import annotations

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceDailyLogRepoBase

from ...records.agent_case import AgentCase
from ..lancedb_manager import get_table


class _AgentCaseRepo(LanceDailyLogRepoBase[AgentCase]):
    schema = AgentCase

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)


agent_case_repo = _AgentCaseRepo()
