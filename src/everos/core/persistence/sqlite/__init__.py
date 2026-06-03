"""SQLite async persistence (SQLModel + SQLAlchemy 2.0 + aiosqlite).

External usage (engine + sessions):
    from everos.core.persistence.sqlite import (
        create_system_engine, create_session_factory, session_scope,
    )

External usage (ORM model basics — re-exported from sqlmodel):
    from everos.core.persistence.sqlite import (
        SQLModel, Field, Relationship, BaseTable,
    )

External usage (generic CRUD repository base):
    from everos.core.persistence.sqlite import RepoBase

The ``system_db`` is the everos
``<memory_root>/.index/sqlite/system.db`` SQLite file holding system
state, audit log, task queue, LSN watermark, and other metadata.
"""

# Re-export key sqlmodel symbols so business code has a single canonical
# entry point (``everos.core.persistence.sqlite``) for ORM authoring.
from sqlmodel import Field as Field
from sqlmodel import Relationship as Relationship
from sqlmodel import SQLModel as SQLModel

from .base import BaseTable as BaseTable
from .engine import create_system_engine as create_system_engine
from .repository import RepoBase as RepoBase
from .session import create_session_factory as create_session_factory
from .session import session_scope as session_scope

__all__ = [
    "BaseTable",
    "Field",
    "Relationship",
    "RepoBase",
    "SQLModel",
    "create_session_factory",
    "create_system_engine",
    "session_scope",
]
