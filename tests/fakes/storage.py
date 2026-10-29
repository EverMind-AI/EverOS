"""Real storage over a test's memory root, built the way the bootstrap builds it.

:func:`make_storage` gives a test the system database, its repositories and
the derived index over one root -- what used to be reached through the
process-wide engine / connection singletons::

    storage = make_storage(tmp_path)
    await storage.create_schema()
    ...
    await storage.aclose()

:func:`handler_deps` and :func:`index_repositories` build the cascade
handler bundle with only the collaborators a test cares about; every other
slot is an :class:`UnusedPort` that fails loudly when touched.

:func:`open_storage_view` is for tests that drive a whole application and
then assert on what it stored: the application owns its connections, so the
test reads back through a second set over the same root.
"""

from __future__ import annotations

__all__ = [
    "TEST_DIM",
    "Storage",
    "StorageView",
    "arrow_schema",
    "handler_deps",
    "index_repositories",
    "make_storage",
    "open_storage_view",
]

import dataclasses
from pathlib import Path
from typing import Any

import pyarrow as pa

from everos.config import Settings, load_settings
from everos.infra.cascade.projections import HandlerDeps
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_derived_index
from everos.infra.persistence.index import (
    DerivedIndex,
    IndexRepositories,
    schema_for,
)
from everos.infra.persistence.lancedb import LanceDbConnection, LanceIndexBackend
from everos.infra.persistence.lancedb.base import arrow_schema_for
from everos.infra.persistence.sqlite import SqliteDatabase, SqliteRepos

from .stores import UnusedPort

TEST_DIM = 1024
"""Vector width of the business tables in tests that create them by hand
(the ``[embedding] dim`` default)."""


def arrow_schema(model: type[Any], *, dim: int = TEST_DIM) -> pa.Schema:
    """The Arrow schema the LanceDB backend creates ``model``'s table with."""
    return arrow_schema_for(schema_for(model, dim=dim))


@dataclasses.dataclass(frozen=True)
class Storage:
    """The system database, its repositories and the LanceDB index of one root."""

    root: MemoryRoot
    database: SqliteDatabase
    sqlite: SqliteRepos
    lance: LanceDbConnection
    backend: LanceIndexBackend
    index: DerivedIndex

    @property
    def repos(self) -> IndexRepositories:
        """The derived-index repositories."""
        return self.index.repositories

    async def create_schema(self) -> None:
        """Create every SQLite table, as ``Application.start`` does."""
        await self.database.create_schema()

    async def aclose(self) -> None:
        """Close the index connection and dispose the database."""
        await self.index.shutdown()
        await self.database.dispose()


def make_storage(
    root: MemoryRoot | Path | str, settings: Settings | None = None
) -> Storage:
    """LanceDB-backed storage over ``root``; settings default to the configured ones."""
    root = root if isinstance(root, MemoryRoot) else MemoryRoot(root)
    settings = settings or load_settings()
    database = SqliteDatabase(root, settings.sqlite)
    lance = LanceDbConnection(root, settings.lancedb)
    backend = LanceIndexBackend(lance, dim=settings.embedding.dim)
    return Storage(
        root=root,
        database=database,
        sqlite=SqliteRepos.over(database),
        lance=lance,
        backend=backend,
        index=DerivedIndex(backend, embedding_dim=settings.embedding.dim),
    )


@dataclasses.dataclass(frozen=True)
class StorageView:
    """A second handle on the storage an application writes, for assertions."""

    root: MemoryRoot
    database: SqliteDatabase
    sqlite: SqliteRepos
    index: DerivedIndex

    @property
    def repos(self) -> IndexRepositories:
        """The derived-index repositories."""
        return self.index.repositories

    async def create_schema(self) -> None:
        """Create every SQLite table, as ``Application.start`` does."""
        await self.database.create_schema()

    async def aclose(self) -> None:
        """Close the index connection and dispose the database."""
        await self.index.shutdown()
        await self.database.dispose()


def open_storage_view(
    root: MemoryRoot | Path | str, settings: Settings | None = None
) -> StorageView:
    """Storage over ``root`` on the configured backend, reading the latest data.

    LanceDB is opened with a zero read-consistency interval: every read
    re-checks the table version, so rows another connection commits -- the
    application's -- are visible at once instead of after the default
    never-refresh snapshot.
    """
    root = root if isinstance(root, MemoryRoot) else MemoryRoot(root)
    settings = settings or load_settings()
    settings = settings.model_copy(
        update={
            "lancedb": settings.lancedb.model_copy(
                update={"read_consistency_seconds": 0.0}
            )
        }
    )
    database = SqliteDatabase(root, settings.sqlite)
    return StorageView(
        root=root,
        database=database,
        sqlite=SqliteRepos.over(database),
        index=build_derived_index(settings, root),
    )


def index_repositories(**repos: Any) -> IndexRepositories:
    """An :class:`IndexRepositories` with ``repos`` and unused ports elsewhere."""
    names = [field.name for field in dataclasses.fields(IndexRepositories)]
    return IndexRepositories(
        **{name: repos.get(name, UnusedPort(f"{name} index")) for name in names}
    )


def handler_deps(
    *,
    memory_root: MemoryRoot,
    tokenizer: Any,
    index: IndexRepositories | None = None,
    knowledge_documents: Any = None,
    knowledge_topics: Any = None,
    embedding: Any = None,
) -> HandlerDeps:
    """:class:`HandlerDeps` whose unset collaborators fail loudly when touched."""
    return HandlerDeps(
        memory_root=memory_root,
        tokenizer=tokenizer,
        index=index or index_repositories(),
        knowledge_documents=knowledge_documents or UnusedPort("knowledge_documents"),
        knowledge_topics=knowledge_topics or UnusedPort("knowledge_topics"),
        embedding=embedding,
    )
