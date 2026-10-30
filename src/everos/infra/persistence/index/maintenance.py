"""Maintenance contract of one derived-index table.

The domain only reads indexes through :mod:`everos.memory.ports`; writing
projected rows, compacting, pruning and rebuilding are the storage's own
business, used by the cascade engine and by operational backfill. Every
derived-index repository of either engine satisfies this protocol as is.
"""

from __future__ import annotations

__all__ = ["IndexMaintenance"]

import datetime as dt
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel

from everos.memory.ports import Filter


@runtime_checkable
class IndexMaintenance(Protocol):
    """Write and housekeeping operations on one derived-index table."""

    schema: type[Any]

    @property
    def table_name(self) -> str: ...

    async def scan(self, where: Filter | None = None) -> Sequence[BaseModel]: ...

    async def upsert(self, records: Sequence[Any]) -> None: ...

    async def update(self, updates: dict[str, Any], *, where: Filter) -> None: ...

    async def delete(self, predicate: Filter) -> None: ...

    async def delete_by_md_path(self, md_path: str) -> int: ...

    async def count(self) -> int: ...

    async def count_where(self, where: Filter | None = None) -> int: ...

    async def optimize(self) -> None: ...

    async def prune(self, older_than: dt.timedelta) -> None: ...

    async def rebuild_indexes(self) -> None: ...

    async def ensure_vector_indexes(self) -> None: ...
