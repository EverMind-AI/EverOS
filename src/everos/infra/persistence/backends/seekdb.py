"""SeekDB lifecycle adapter for the derived-index backend port."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ..index.protocols import IndexRepository


class SeekdbIndexBackend:
    """Own embedded or remote SeekDB table and connection lifecycle."""

    name: ClassVar[str] = "seekdb"

    @property
    def repositories(self) -> tuple[IndexRepository[Any], ...]:
        return tuple(_seekdb().ALL_REPOS)

    async def connect(self) -> Any:
        return await _seekdb().get_session()

    async def startup(self) -> Any:
        session = await self.connect()
        await self.ensure_business_indexes()
        return session

    async def shutdown(self) -> None:
        await _seekdb().dispose_connection()

    async def ensure_business_indexes(self) -> None:
        await _seekdb().ensure_business_indexes()

    async def verify_business_schemas(self) -> None:
        await self.ensure_business_indexes()

    async def drop_business_tables(self) -> list[str]:
        return await _seekdb().drop_business_tables()


def _seekdb() -> ModuleType:
    from everos.infra.persistence import seekdb

    return seekdb


seekdb_index_backend = SeekdbIndexBackend()

__all__ = ["SeekdbIndexBackend", "seekdb_index_backend"]
