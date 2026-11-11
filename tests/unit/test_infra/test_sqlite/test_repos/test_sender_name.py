"""``SqliteSenderNameStore`` over a real ``sender_name`` table.

Pins: an upsert overwrites a sender's name, rows are keyed by
``(app_id, project_id, sender_id)``, and a deletion removes one sender only.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteSenderNameStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite._repos import SenderNameRepo


@pytest.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteSenderNameStore]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield SqliteSenderNameStore(SenderNameRepo(create_session_factory(engine)))
    await engine.dispose()


async def test_an_upsert_overwrites_the_senders_name(
    store: SqliteSenderNameStore,
) -> None:
    await store.upsert_many("a", "p", {"u1": "Ali", "u2": "Bob"})
    await store.upsert_many("a", "p", {"u1": "Alice"})
    assert await store.get_many("a", "p", ["u1", "u2", "u3"]) == {
        "u1": "Alice",
        "u2": "Bob",
    }


async def test_rows_are_scoped_by_app_and_project(
    store: SqliteSenderNameStore,
) -> None:
    await store.upsert_many("a", "p", {"u1": "Alice"})
    await store.upsert_many("a", "q", {"u1": "Ally"})
    assert await store.get_many("a", "p", ["u1"]) == {"u1": "Alice"}
    assert await store.get_many("a", "q", ["u1"]) == {"u1": "Ally"}
    assert await store.get_many("b", "p", ["u1"]) == {}


async def test_delete_removes_one_sender_in_one_space(
    store: SqliteSenderNameStore,
) -> None:
    await store.upsert_many("a", "p", {"u1": "Alice", "u2": "Bob"})
    await store.upsert_many("a", "q", {"u1": "Ally"})
    await store.delete("a", "p", "u1")
    await store.delete("a", "p", "nobody")
    assert await store.get_many("a", "p", ["u1", "u2"]) == {"u2": "Bob"}
    assert await store.get_many("a", "q", ["u1"]) == {"u1": "Ally"}


async def test_empty_requests_touch_nothing(store: SqliteSenderNameStore) -> None:
    await store.upsert_many("a", "p", {})
    assert await store.get_many("a", "p", []) == {}
