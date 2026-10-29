"""A session whose buffer the model cannot segment is set aside, not stuck.

Boundary detection tries an unusable reply three times per call. A message
the model can never segment would otherwise stay in the buffer, and every
later call on the session would merge it back in and fail the same way. The
third consecutive call that ends this way moves the session's buffer (and
that call's own messages) to the ``quarantine`` track, raises an error the
client can tell apart, and the next call starts from an empty buffer.
Input the operator cannot take at all -- a message everalgo refuses, or a
row that maps to no conversation item (an orphan tool result) -- counts the
same way but is never retried. Infrastructure failures (no quota, network)
are not counted: they propagate on the first attempt, so the buffer waits
for the service to come back.

Real: ingest, :class:`BoundaryDetection`, everalgo's boundary operator and
the SQLite stores on a temp file. Scripted: the model's replies.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from everalgo.llm import LLMOutputError
from everalgo.testing import FakeLLMClient
from sqlmodel import SQLModel

from everos.config import BoundaryDetectionSettings, SqliteSettings
from everos.core.errors import LLMServiceError, SessionBufferQuarantinedError
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.sqlite import (
    SqliteMemcellStore,
    SqliteSessionBufferStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.conversation_status import (
    ConversationStatusRepo,
)
from everos.infra.persistence.sqlite.repos.memcell import MemcellRepo
from everos.infra.persistence.sqlite.repos.unprocessed_buffer import (
    UnprocessedBufferRepo,
)
from everos.memory.extract.boundary import BoundaryDetection, BoundaryOutcome
from everos.memory.extract.boundary.quarantine import POISON_CALLS
from everos.memory.extract.ingest import service as ingest
from everos.memory.extract.llm_retry import OUTPUT_ATTEMPTS
from tests.fakes import (
    UNUSABLE_REPLY,
    boundary_reply,
    bundled_instructions,
    unused_raw_file_builder,
)

BASE_TS = 1_700_000_000_000
GARBAGE: str = UNUSABLE_REPLY
OK = boundary_reply()


class _Stage:
    """Boundary detection on a temp database with a scripted model."""

    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="quarantine_"))
        root = MemoryRoot(self.tmp)
        root.ensure()
        self.engine = create_system_engine(root.system_db, SqliteSettings())
        self.factory = create_session_factory(self.engine)
        self.buffer = UnprocessedBufferRepo(session_factory=self.factory)
        self.replies: list[str | Exception] = []
        self.llm = FakeLLMClient(handler=self._reply)
        self.boundary = BoundaryDetection(
            llm=self.llm,
            instructions=bundled_instructions(),
            session_buffer=SqliteSessionBufferStore(
                self.buffer, ConversationStatusRepo(session_factory=self.factory)
            ),
            memcells=SqliteMemcellStore(MemcellRepo(session_factory=self.factory)),
            limits=BoundaryDetectionSettings(
                hard_token_limit=10_000, hard_msg_limit=100
            ),
        )

    def _reply(self, _messages: Any, **_kwargs: Any) -> str:
        reply = self.replies.pop(0) if self.replies else OK
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def __aenter__(self) -> _Stage:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.engine.dispose()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def defer(self, session: str, start: int) -> BoundaryOutcome:
        """A deferred add: the message lands in the buffer with no model call."""
        ingested = await ingest.process(
            _payload(session, start),
            multimodal_llm=None,
            raw_file_builder=unused_raw_file_builder,
        )
        return await self.boundary.buffer(ingested, mode="chat")

    async def add(
        self, session: str, start: int, *, final: bool = False
    ) -> BoundaryOutcome:
        ingested = await ingest.process(
            _payload(session, start),
            multimodal_llm=None,
            raw_file_builder=unused_raw_file_builder,
        )
        return await self.boundary.detect(ingested, mode="chat", is_final=final)

    async def add_orphan_tool_row(self, session: str, start: int) -> BoundaryOutcome:
        """An agent-mode add whose tool row answers no tool call."""
        payload = _payload(session, start)
        payload["messages"].append(
            {
                "content": "tool output",
                "timestamp": BASE_TS + start,
                "sender_id": "tool",
                "role": "tool",
            }
        )
        ingested = await ingest.process(
            payload, multimodal_llm=None, raw_file_builder=unused_raw_file_builder
        )
        return await self.boundary.detect(ingested, mode="agent", is_final=False)

    async def tracks(self, session: str) -> dict[str, list[str]]:
        return {
            track: [
                r.message_id for r in await self.buffer.list_for_track(session, track)
            ]
            for track in ("memorize", "quarantine")
        }


def _payload(session: str, start: int) -> dict[str, Any]:
    return {
        "session_id": session,
        "messages": [
            {
                "content": f"m{start}",
                "timestamp": BASE_TS + start,
                "sender_id": "u",
                "role": "user",
            }
        ],
    }


async def test_third_consecutive_unusable_call_quarantines_the_buffer() -> None:
    async with _Stage() as stage:
        # The poison arrives through a deferred add and sits in the buffer;
        # every later call merges it back in and the model chokes on it.
        await stage.defer("s1", 1)
        stage.replies = [GARBAGE] * (OUTPUT_ATTEMPTS * POISON_CALLS)
        with pytest.raises(LLMOutputError):
            await stage.add("s1", 2)
        with pytest.raises(LLMOutputError):
            await stage.add("s1", 3)
        with pytest.raises(SessionBufferQuarantinedError):
            await stage.add("s1", 4)
        tracks = await stage.tracks("s1")
        assert tracks["memorize"] == [], "the session starts over"
        # The buffered poison and the quarantining call's own message are
        # kept; the two earlier calls' messages were never persisted.
        assert sorted(tracks["quarantine"]) == [
            "m_s1_1700000000001_000",
            "m_s1_1700000000004_000",
        ]
        assert stage.llm.call_count == OUTPUT_ATTEMPTS * POISON_CALLS
        # The session goes on: a healthy flush cuts a cell from an empty buffer.
        outcome = await stage.add("s1", 5, final=True)
        assert outcome.status == "extracted"
        assert outcome.per_cell_message_ids == [["m_s1_1700000000005_000"]]
        assert (await stage.tracks("s1"))["memorize"] == []
        assert len((await stage.tracks("s1"))["quarantine"]) == 2


async def test_a_usable_reply_resets_the_count() -> None:
    async with _Stage() as stage:
        stage.replies = [GARBAGE] * OUTPUT_ATTEMPTS + [OK] + [GARBAGE] * OUTPUT_ATTEMPTS
        with pytest.raises(LLMOutputError):
            await stage.add("s1", 1)
        assert (await stage.add("s1", 2)).status == "accumulated"
        with pytest.raises(LLMOutputError):  # the count starts again at one
            await stage.add("s1", 3)
        assert (await stage.tracks("s1"))["quarantine"] == []


async def test_infrastructure_failures_are_neither_retried_nor_counted() -> None:
    async with _Stage() as stage:
        stage.replies = [LLMServiceError("insufficient quota") for _ in range(5)]
        for i in range(1, 6):
            with pytest.raises(LLMServiceError):
                await stage.add("s1", i)
        assert stage.llm.call_count == 5
        tracks = await stage.tracks("s1")
        assert tracks["quarantine"] == []
        assert tracks["memorize"] == [], (
            "an outage keeps nothing and loses nothing: the client retries"
        )


async def test_sessions_are_counted_apart() -> None:
    async with _Stage() as stage:
        stage.replies = [GARBAGE] * (OUTPUT_ATTEMPTS * 2) + [OK]
        with pytest.raises(LLMOutputError):
            await stage.add("s1", 1)
        with pytest.raises(LLMOutputError):
            await stage.add("s1", 2)
        assert (await stage.add("s2", 3)).status == "accumulated"
        assert (await stage.tracks("s1"))["quarantine"] == []


async def test_an_orphan_tool_row_counts_without_reaching_the_model() -> None:
    """A row that maps to no conversation item poisons the session too."""
    async with _Stage() as stage:
        for start in (1, 2):
            with pytest.raises(ValueError, match="cannot map"):
                await stage.add_orphan_tool_row("s1", start)
        with pytest.raises(SessionBufferQuarantinedError):
            await stage.add_orphan_tool_row("s1", 3)
        assert stage.llm.call_count == 0, "never retried, never sent"
        tracks = await stage.tracks("s1")
        assert tracks["memorize"] == []
        assert sorted(tracks["quarantine"]) == [
            "m_s1_1700000000003_000",
            "m_s1_1700000000003_001",
        ]
