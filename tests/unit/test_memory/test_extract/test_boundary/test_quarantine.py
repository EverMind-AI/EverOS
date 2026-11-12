"""A session whose buffer the model cannot segment is set aside, not stuck.

Boundary detection tries an unusable reply three times per call. A message
the model can never segment would otherwise stay in the buffer, and every
later call on the session would merge it back in and fail the same way. The
third consecutive call that ends this way moves the session's buffer (and
that call's own messages) to the ``quarantine`` track, raises an error the
client can tell apart, and the next call starts from an empty buffer.
Input the operator cannot take at all -- a message everalgo refuses --
counts the same way but is never retried. Infrastructure failures (no quota,
network) are not counted: they propagate on the first attempt, so the buffer
waits for the service to come back. A tool row is not chat: it is dropped
before anything is buffered or cut, so it never poisons a session -- not
even one an older build left in the buffer.

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
    SqliteSessionBufferStore,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite._repos.session_cut import SessionCutRepo
from everos.infra.persistence.sqlite._repos.unprocessed_buffer import (
    UnprocessedBufferRepo,
)
from everos.memory.extract.boundary import BoundaryDetection, BoundaryOutcome
from everos.memory.extract.boundary.quarantine import POISON_CALLS
from everos.memory.extract.ingest import service as ingest
from everos.memory.extract.llm_retry import OUTPUT_ATTEMPTS
from everos.memory.ports import SessionKey
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
        self.session_buffer = SqliteSessionBufferStore(
            self.buffer, SessionCutRepo(session_factory=self.factory)
        )
        self.boundary = BoundaryDetection(
            llm=self.llm,
            instructions=bundled_instructions(),
            session_buffer=self.session_buffer,
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
        return await self.boundary.buffer(ingested)

    async def add(
        self, session: str, start: int, *, final: bool = False
    ) -> BoundaryOutcome:
        ingested = await ingest.process(
            _payload(session, start),
            multimodal_llm=None,
            raw_file_builder=unused_raw_file_builder,
        )
        return await self.boundary.detect(ingested, is_final=final)

    async def add_refused(self, session: str, start: int) -> BoundaryOutcome:
        """An add with a message everalgo refuses: text without a timestamp."""
        payload = _payload(session, start)
        payload["messages"].append(
            {"content": "undated", "timestamp": 0, "sender_id": "u", "role": "user"}
        )
        ingested = await ingest.process(
            payload, multimodal_llm=None, raw_file_builder=unused_raw_file_builder
        )
        return await self.boundary.detect(ingested, is_final=False)

    async def add_with_tool_row(
        self, session: str, start: int, *, defer: bool = False
    ) -> BoundaryOutcome:
        """An add whose messages include a tool row answering no tool call."""
        ingested = await ingest.process(
            _with_tool_row(_payload(session, start), start),
            multimodal_llm=None,
            raw_file_builder=unused_raw_file_builder,
        )
        if defer:
            return await self.boundary.buffer(ingested)
        return await self.boundary.detect(ingested, is_final=False)

    async def leave_tool_row_buffered(self, session: str, start: int) -> None:
        """A tool row in the buffer, as an agent-mode build left one there."""
        ingested = await ingest.process(
            _with_tool_row({"session_id": session, "messages": []}, start),
            multimodal_llm=None,
            raw_file_builder=unused_raw_file_builder,
        )
        [row] = ingested.messages
        await self.session_buffer.keep(
            SessionKey(
                app_id=ingested.app_id,
                project_id=ingested.project_id,
                session_id=session,
            ),
            [row],
            last_message_ts=row.timestamp,
        )

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


def _with_tool_row(payload: dict[str, Any], start: int) -> dict[str, Any]:
    payload["messages"].append(
        {
            "content": "tool output",
            "timestamp": BASE_TS + start,
            "sender_id": "tool",
            "role": "tool",
        }
    )
    return payload


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


async def test_a_refused_message_counts_without_reaching_the_model() -> None:
    """A message everalgo refuses poisons the session too, and is not retried."""
    async with _Stage() as stage:
        for start in (1, 2):
            with pytest.raises(ValueError, match="timestamp"):
                await stage.add_refused("s1", start)
        with pytest.raises(SessionBufferQuarantinedError):
            await stage.add_refused("s1", 3)
        assert stage.llm.call_count == 0, "never retried, never sent"
        tracks = await stage.tracks("s1")
        assert tracks["memorize"] == []
        assert sorted(tracks["quarantine"]) == [
            "m_s1_0_001",
            "m_s1_1700000000003_000",
        ]


async def test_a_tool_row_is_dropped_and_never_poisons_the_session() -> None:
    """The agent-mode add that once poisoned a session: its tool row is not
    chat, so it is dropped before the buffer and the chat rows go on."""
    async with _Stage() as stage:
        deferred = await stage.add_with_tool_row("s1", 1, defer=True)
        assert (deferred.status, deferred.message_count) == ("accumulated", 1)
        for start in (2, 3, 4):
            outcome = await stage.add_with_tool_row("s1", start)
            assert (outcome.status, outcome.message_count) == ("accumulated", 1)
        assert stage.llm.call_count == 3
        assert await stage.tracks("s1") == {
            "memorize": [f"m_s1_170000000000{start}_000" for start in (1, 2, 3, 4)],
            "quarantine": [],
        }


async def test_a_tool_row_an_older_build_buffered_is_dropped_too() -> None:
    async with _Stage() as stage:
        await stage.leave_tool_row_buffered("s1", 1)
        assert (await stage.tracks("s1"))["memorize"] == ["m_s1_1700000000001_000"]

        outcome = await stage.add("s1", 2, final=True)

        assert outcome.status == "extracted"
        assert outcome.per_cell_message_ids == [["m_s1_1700000000002_000"]]
        assert await stage.tracks("s1") == {"memorize": [], "quarantine": []}
