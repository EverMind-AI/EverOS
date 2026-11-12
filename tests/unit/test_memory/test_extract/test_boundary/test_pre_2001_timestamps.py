"""A conversation dated before 2001-09-09 must survive boundary detection.

A memcell's timestamp is epoch milliseconds. Before 2001-09-09 that value is
under the 1e12 threshold ``from_timestamp`` uses to tell seconds from
milliseconds, so reading it through that heuristic lands in year ~32,700 and the
cut raised ``ValueError: year ... is out of range`` -- measured on MemLens-32K,
whose sessions dated 2000-10 .. 2001-08 could not be ingested at all. The add
API's own ``timestamp`` keeps the heuristic (its contract tolerates seconds), so
a client sends such an instant in seconds; every value EverOS computes itself is
milliseconds and must be read back as milliseconds.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from everalgo.testing import FakeLLMClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel

from everos.config import BoundaryDetectionSettings, SqliteSettings
from everos.core.utils.datetime import ensure_utc
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteSessionBufferStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite._repos.session_cut import SessionCutRepo
from everos.infra.persistence.sqlite._repos.unprocessed_buffer import (
    UnprocessedBufferRepo,
)
from everos.infra.persistence.sqlite._tables import ConversationStatus, Memcell
from everos.memory.extract.boundary import BoundaryDetection
from everos.memory.extract.ingest import service as ingest
from tests.fakes import boundary_reply, bundled_instructions, unused_raw_file_builder

# 2000-10-06T00:00:00Z in seconds: the unit the add API's sec/ms heuristic reads
# correctly for an instant before 2001-09-09.
_SENT_SECONDS = 970_790_400
_NO_FORCED_CUTS = BoundaryDetectionSettings(
    hard_token_limit=10**9, hard_msg_limit=10**6
)


@pytest.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


async def test_a_conversation_from_2000_is_cut_and_keeps_its_date(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    boundary = BoundaryDetection(
        llm=FakeLLMClient(handler=lambda *_args, **_kwargs: boundary_reply([2])),
        instructions=bundled_instructions(),
        session_buffer=SqliteSessionBufferStore(
            UnprocessedBufferRepo(session_factory=factory),
            SessionCutRepo(session_factory=factory),
        ),
        limits=_NO_FORCED_CUTS,
    )
    payload = {
        "session_id": "s1",
        "messages": [
            {
                "content": text,
                "timestamp": _SENT_SECONDS + 30 * i,
                "sender_id": "u",
                "role": "user",
            }
            for i, text in enumerate(("hello", "goodbye"))
        ],
    }
    ingested = await ingest.process(
        payload, multimodal_llm=None, raw_file_builder=unused_raw_file_builder
    )

    outcome = await boundary.detect(ingested, is_final=True)

    assert outcome.status == "extracted"
    async with factory() as s:
        cell_ts = (await s.execute(select(Memcell.timestamp))).scalar_one()
        status_ts = (
            await s.execute(select(ConversationStatus.last_memcell_ts))
        ).scalar_one()
    for stamp in (cell_ts, status_ts):
        aware = ensure_utc(stamp)
        assert aware is not None
        assert aware.date() == dt.date(2000, 10, 6)
