"""LanceDB repo singleton for the ``user_profile`` table."""

from __future__ import annotations

from lancedb import AsyncTable

from everos.core.persistence.lancedb import LanceRepoBase

from ...records.user_profile import UserProfile
from ..lancedb_manager import get_table
from ._contract import LanceIndexContract


class _UserProfileRepo(LanceIndexContract[UserProfile], LanceRepoBase[UserProfile]):
    schema = UserProfile

    async def _table_lookup(self) -> AsyncTable:
        return await get_table(self.schema.TABLE_NAME, self.schema)


user_profile_repo = _UserProfileRepo()
