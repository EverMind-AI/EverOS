"""Milvus lifecycle adapter for the derived-index backend port."""

from __future__ import annotations

from typing import Any, ClassVar


class MilvusIndexBackend:
    """Own remote Milvus connection, collection, and repository lifecycle."""

    name: ClassVar[str] = "milvus"

    @property
    def repositories(self):  # type: ignore[no-untyped-def]
        return _milvus().ALL_REPOS

    async def connect(self) -> Any:
        return await _milvus().get_client()

    async def startup(self) -> Any:
        client = await self.connect()
        await self.verify_business_schemas()
        await self.ensure_business_indexes()
        return client

    async def shutdown(self) -> None:
        await _milvus().dispose_connection()

    async def ensure_business_indexes(self) -> None:
        await _milvus().ensure_business_indexes()

    async def verify_business_schemas(self) -> None:
        # Verification tolerates absent collections; ensure_collection creates
        # them immediately afterwards and verifies every existing collection.
        for repo in self.repositories:
            await repo.ensure_collection()

    async def drop_business_tables(self) -> list[str]:
        return await _milvus().drop_business_tables()


def _milvus():  # type: ignore[no-untyped-def]
    # Keep pymilvus genuinely optional for default LanceDB installations.
    from everos.infra.persistence import milvus

    return milvus


milvus_index_backend = MilvusIndexBackend()

__all__ = ["MilvusIndexBackend", "milvus_index_backend"]
