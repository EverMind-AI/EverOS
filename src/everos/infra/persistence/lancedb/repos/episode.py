"""LanceDB repo singleton for the ``episode`` table."""

from __future__ import annotations

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceDailyLogRepoBase

from ..lancedb_manager import get_table
from ..tables.episode import Episode


class _EpisodeRepo(LanceDailyLogRepoBase[Episode]):
    schema = Episode

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)


episode_repo = _EpisodeRepo()
