"""System-DB schema creation, including forward-only column additions.

``SQLModel.metadata.create_all`` creates missing tables but never alters an
existing one, so a column added to a table model would be absent from every
``system.db`` created before it. :func:`create_schema` runs ``create_all``
and then adds each column listed in :data:`_ADDED_COLUMNS` that the live
table lacks (``PRAGMA table_info`` → ``ALTER TABLE ... ADD COLUMN``).

Invariants:

* Forward-only and idempotent: columns are only ever added, and a second run
  finds nothing to do.
* Only nullable columns may be listed -- SQLite cannot add a ``NOT NULL``
  column without a default, and existing rows get ``NULL``.
* The column type is compiled from the table model, so the DDL cannot drift
  from what the ORM reads and writes.
"""

from __future__ import annotations

__all__ = ["create_schema"]

from collections.abc import Mapping

from sqlalchemy import Connection, Table, text
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import SQLModel

from everos.core.observability.logging import get_logger

logger = get_logger(__name__)

_ADDED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "memcell": ("case_extracted_at", "case_skip_reason", "case_skip_detail_json"),
}
"""Columns added to a table after its first release, by table name."""


async def create_schema(engine: AsyncEngine) -> int:
    """Create every registered table and add columns older databases lack.

    Returns:
        The number of registered tables.
    """
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
        await conn.run_sync(_add_missing_columns)
    return len(SQLModel.metadata.tables)


def _add_missing_columns(conn: Connection) -> None:
    for table_name, column_names in _ADDED_COLUMNS.items():
        table = SQLModel.metadata.tables[table_name]
        present = _existing_columns(conn, table_name)
        for name in column_names:
            if name not in present:
                conn.execute(text(_add_column_ddl(conn, table, name)))
                logger.info("sqlite_column_added", table=table_name, column=name)


def _existing_columns(conn: Connection, table_name: str) -> set[str]:
    rows = conn.execute(text(f'PRAGMA table_info("{table_name}")')).fetchall()
    return {row[1] for row in rows}


def _add_column_ddl(conn: Connection, table: Table, name: str) -> str:
    column = table.columns[name]
    if not column.nullable:
        raise ValueError(f"{table.name}.{name} must be nullable to be added")
    column_type = column.type.compile(dialect=conn.dialect)
    return f'ALTER TABLE "{table.name}" ADD COLUMN "{name}" {column_type}'
