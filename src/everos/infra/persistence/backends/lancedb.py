"""LanceDB lifecycle adapter for the derived-index backend port.

The per-table repositories in :mod:`everos.infra.persistence.lancedb` speak
the neutral contract themselves (:class:`LanceIndexContract`); this module
only owns the connection, schema and index lifecycle the port asks for.
"""

from __future__ import annotations

from typing import Any, ClassVar

from everos.infra.persistence import lancedb as _lancedb

ALL_REPOS = (
    _lancedb.episode_repo,
    _lancedb.atomic_fact_repo,
    _lancedb.foresight_repo,
    _lancedb.agent_case_repo,
    _lancedb.agent_skill_repo,
    _lancedb.user_profile_repo,
    _lancedb.knowledge_topic_repo,
)


class LanceIndexBackend:
    """Own LanceDB connection, schema, index, and repository lifecycle."""

    name: ClassVar[str] = "lancedb"
    repositories = ALL_REPOS

    async def connect(self) -> Any:
        return await _lancedb.get_connection()

    async def startup(self) -> Any:
        connection = await self.connect()
        await self.verify_business_schemas()
        await self.ensure_business_indexes()
        return connection

    async def shutdown(self) -> None:
        await _lancedb.dispose_connection()

    async def ensure_business_indexes(self) -> None:
        await _lancedb.ensure_business_indexes()

    async def verify_business_schemas(self) -> None:
        await _lancedb.verify_business_schemas()

    async def drop_business_tables(self) -> list[str]:
        return await _lancedb.drop_business_tables()


lance_index_backend = LanceIndexBackend()

__all__ = ["ALL_REPOS", "LanceIndexBackend", "lance_index_backend"]
