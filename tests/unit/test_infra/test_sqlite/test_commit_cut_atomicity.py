"""``commit_cut`` keeps memcells and the buffer in one SQLite transaction (#539).

A failure after the memcell insert used to leave messages in a memcell *and*
still in the buffer; the next add duplicated them. With ``commit_cut``, any
error before commit rolls the whole cut back.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest import mock

import pytest
from everalgo.types import ChatMessage, MemCell
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel, select

from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteSessionBufferStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite import stores as stores_mod
from everos.infra.persistence.sqlite._repos.conversation_status import (
    ConversationStatusRepo,
)
from everos.infra.persistence.sqlite._repos.memcell import MemcellRepo
from everos.infra.persistence.sqlite._repos.unprocessed_buffer import (
    UnprocessedBufferRepo,
)
from everos.infra.persistence.sqlite._tables import Memcell, UnprocessedBuffer
from everos.memory.models import CanonicalMessage, MemcellRecord
from everos.memory.ports import SessionKey


@pytest.fixture
async def buffer_store(
    tmp_path: Path,
) -> AsyncIterator[tuple[SqliteSessionBufferStore, async_sessionmaker[AsyncSession]]]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    factory = create_session_factory(engine)
    store = SqliteSessionBufferStore(
        UnprocessedBufferRepo(session_factory=factory),
        ConversationStatusRepo(session_factory=factory),
        MemcellRepo(session_factory=factory),
    )
    yield store, factory
    await engine.dispose()


def _session() -> SessionKey:
    return SessionKey(app_id="default", project_id="default", session_id="s1")


def _msg(n: int) -> CanonicalMessage:
    return CanonicalMessage(
        message_id=f"m{n}",
        session_id="s1",
        sender_id="u",
        sender_name="u",
        role="user",
        timestamp=dt.datetime(2025, 1, 1, tzinfo=dt.UTC) + dt.timedelta(seconds=n),
        content_items=[{"type": "text", "text": f"m{n}"}],
        text=f"m{n}",
    )


def _record(*ids: str) -> MemcellRecord:
    items = [
        ChatMessage(
            id=mid,
            role="user",
            content=mid,
            sender_id="u",
            timestamp=1_700_000_000 + i,
        )
        for i, mid in enumerate(ids)
    ]
    return MemcellRecord(
        memcell_id=f"mc_{''.join(ids)}",
        cell=MemCell(items=items, timestamp=items[-1].timestamp),
        raw_type="conversation",
        message_ids=list(ids),
        sender_ids=["u"],
    )


@pytest.mark.asyncio
async def test_commit_cut_persists_memcell_and_clears_buffer(
    buffer_store: tuple[SqliteSessionBufferStore, async_sessionmaker[AsyncSession]],
) -> None:
    store, factory = buffer_store
    session = _session()
    await store.replace(session, [_msg(1), _msg(2)])
    await store.commit_cut(
        session,
        [_record("m1", "m2")],
        tail=[],
        last_memcell_ts=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
    )
    assert [m.message_id for m in await store.list(session)] == []
    async with factory() as s:
        cells = list((await s.execute(select(Memcell))).scalars().all())
    assert len(cells) == 1
    assert "m1" in cells[0].message_ids_json


@pytest.mark.asyncio
async def test_commit_cut_rolls_back_when_commit_fails(
    buffer_store: tuple[SqliteSessionBufferStore, async_sessionmaker[AsyncSession]],
) -> None:
    store, factory = buffer_store
    session = _session()
    await store.replace(session, [_msg(1), _msg(2)])

    original = stores_mod.session_scope

    class _BoomSession:
        def __init__(self, inner: AsyncSession) -> None:
            self._inner = inner

        async def commit(self) -> None:
            raise RuntimeError("database is locked")

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    @asynccontextmanager
    async def boom_scope(session_factory):  # type: ignore[no-untyped-def]
        async with original(session_factory) as s:
            yield _BoomSession(s)

    with (
        mock.patch.object(stores_mod, "session_scope", boom_scope),
        pytest.raises(RuntimeError, match="database is locked"),
    ):
        await store.commit_cut(
            session,
            [_record("m1", "m2")],
            tail=[],
            last_memcell_ts=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        )

    assert [m.message_id for m in await store.list(session)] == ["m1", "m2"]
    async with factory() as s:
        cells = list((await s.execute(select(Memcell))).scalars().all())
        buf = list((await s.execute(select(UnprocessedBuffer))).scalars().all())
    assert cells == []
    assert {r.message_id for r in buf} == {"m1", "m2"}
