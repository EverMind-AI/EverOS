"""Milvus lifecycle adapter for the derived-index backend port."""

from __future__ import annotations

from types import ModuleType
from typing import Any, ClassVar

from .protocols import IndexRepository


class MilvusIndexBackend:
    """Own remote Milvus connection, collection, and repository lifecycle."""

    name: ClassVar[str] = "milvus"

    @property
    def repositories(self) -> tuple[IndexRepository[Any], ...]:
        return tuple(_milvus().ALL_REPOS)

    async def connect(self) -> Any:
        return await _milvus().get_client()

    async def startup(self) -> Any:
        client = await self.connect()
        # ensure_collection creates a missing collection and drift-checks an
        # existing one in a single round trip, so unlike the LanceDB backend
        # there is no separate verify step to run first.
        await self.ensure_business_indexes()
        return client

    async def shutdown(self) -> None:
        await _milvus().dispose_connection()

    async def ensure_business_indexes(self) -> None:
        await _milvus().ensure_business_indexes()

    async def verify_business_schemas(self) -> None:
        # Same call as ensure_business_indexes: ensure_collection creates a
        # missing collection and drift-checks an existing one. Kept distinct
        # because the port and the CLI both address verification by name.
        await self.ensure_business_indexes()

    async def drop_business_tables(self) -> list[str]:
        return await _milvus().drop_business_tables()


def _milvus() -> ModuleType:
    # Keep pymilvus genuinely optional for default LanceDB installations.
    from everos.infra.persistence import milvus

    return milvus


milvus_index_backend = MilvusIndexBackend()

__all__ = ["MilvusIndexBackend", "milvus_index_backend"]
