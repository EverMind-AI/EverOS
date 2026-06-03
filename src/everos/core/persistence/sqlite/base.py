"""Common SQLModel base for everos tables.

:class:`BaseTable` adds ``created_at`` / ``updated_at`` columns. The
``updated_at`` column auto-refreshes on UPDATE through SA's ``onupdate``
hook (no explicit assignment needed in business code).

The **two-zone storage-UTC discipline** is enforced by a SQLAlchemy
:class:`TypeDecorator` (:class:`UtcDateTimeColumn`) used as the SQL
column type for every datetime field:

* **on write** — ``process_bind_param`` converts every datetime to
  aware UTC before SQLAlchemy emits the bound parameter. This covers
  *every* SQLAlchemy write path uniformly:

  - ORM ``session.add()`` / ``session.merge()`` (unit-of-work flush)
  - Core ``session.execute(insert(...).values(...))``
  - Core ``session.execute(update(...).values(...))``
  - Bulk ``bulk_insert_mappings`` / ``bulk_save_objects``
  - Raw SQL with bound parameters

  Reaching into the column type is the only place SQLAlchemy guarantees
  *every* write path passes through. Mapper events (``before_insert`` /
  ``before_update``) only fire on the ORM unit-of-work path and would
  silently miss Core statements — which :mod:`everos.infra.persistence
  .sqlite.repos.md_change_state` uses heavily.

* **on read** — ``process_result_value`` re-attaches ``tzinfo=UTC`` to
  every naive datetime returned from SQLite (which has no native tz
  storage and always returns naive). Callers therefore never observe a
  naive datetime regardless of which read API they use.

Subclass with ``table=True`` to declare a real SQLite table::

    from sqlmodel import Field

    class Sender(BaseTable, table=True):
        id: int | None = Field(default=None, primary_key=True)
        name: str
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy import types as sa_types
from sqlmodel import Field, SQLModel

from everos.component.utils.datetime import UtcDatetime, ensure_utc, get_utc_now


class UtcDateTimeColumn(sa_types.TypeDecorator[_dt.datetime]):
    """SQLAlchemy column type enforcing storage-UTC on every read/write.

    Implementation:

    * ``impl = DateTime`` — uses the dialect's standard DateTime SQL type
      (TEXT ISO-8601 on SQLite; ``TIMESTAMP`` on Postgres etc.).
    * ``process_bind_param`` — write hook. Awares → ``astimezone(UTC)``;
      naives → assumed already UTC (storage-boundary convention; see
      :func:`ensure_utc` docstring); ``None`` passes through.
    * ``process_result_value`` — read hook. Naive ``datetime`` →
      ``replace(tzinfo=UTC)``; aware passes through unchanged.

    ``cache_ok = True`` — SQLAlchemy can safely cache statement
    compilations using this type (no per-instance mutable state).
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(
        self, value: _dt.datetime | None, _dialect: Any
    ) -> _dt.datetime | None:
        if value is None:
            return None
        if not isinstance(value, _dt.datetime):
            return value
        return ensure_utc(value)

    def process_result_value(
        self, value: _dt.datetime | None, _dialect: Any
    ) -> _dt.datetime | None:
        if value is None:
            return None
        if isinstance(value, _dt.datetime) and value.tzinfo is None:
            return value.replace(tzinfo=_dt.UTC)
        return value


class BaseTable(SQLModel):
    """Mixin providing ``created_at`` / ``updated_at`` columns.

    Both default to :func:`get_utc_now` on INSERT.
    ``updated_at`` is auto-refreshed by SQLAlchemy on every UPDATE via the
    ``onupdate`` hook — do not set it manually unless overriding intentionally.

    Both columns use :class:`UtcDateTimeColumn` as the SQL column type
    so storage-UTC is enforced **at the SQLAlchemy bind layer** on every
    write path (ORM + Core + bulk + raw bound params).
    """

    created_at: UtcDatetime = Field(
        default_factory=get_utc_now,
        sa_type=UtcDateTimeColumn,
    )
    updated_at: UtcDatetime = Field(
        default_factory=get_utc_now,
        sa_type=UtcDateTimeColumn,
        sa_column_kwargs={"onupdate": get_utc_now},
    )
