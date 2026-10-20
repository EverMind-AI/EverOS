"""A session whose buffer the model cannot segment is set aside, not stuck.

Boundary detection retries an unusable reply three times per call. Before
this change a message the model could never segment stayed in the buffer
and every later call on the session merged it back in and failed the same
way: the session was dead until someone cleared the buffer by hand. Now the
third consecutive call that ends this way moves the session's buffer (and
that call's own messages) to the ``quarantine`` track, raises an error the
client can tell apart, and the next call starts from an empty buffer.
Infrastructure failures (no quota, network) are not counted: they are not
``ValueError`` and propagate unchanged, so the buffer waits for the service
to come back.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
from everalgo.boundary.chat import DetectionResult
from everalgo.types import MemCell
from sqlmodel import SQLModel

from everos.config import SqliteSettings
from everos.core.errors import LLMServiceError, SessionBufferQuarantinedError
from everos.core.persistence import (
    MemoryRoot,
    create_session_factory,
    create_system_engine,
)
from everos.infra.persistence.sqlite.repos.conversation_status import (
    _ConversationStatusRepo,
)
from everos.infra.persistence.sqlite.repos.memcell import _MemcellRepo
from everos.infra.persistence.sqlite.repos.unprocessed_buffer import (
    _UnprocessedBufferRepo,
)
from everos.memory.extract.ingest import service as ingest
from everos.service import _boundary as boundary_mod

BASE_TS = 1_700_000_000_000


class _Stage:
    """The boundary stage on a temp database with a scripted model."""

    def __init__(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="quarantine_"))
        root = MemoryRoot(self.tmp)
        root.ensure()
        self.engine = create_system_engine(root.system_db, SqliteSettings())
        self.factory = create_session_factory(self.engine)
        self.buffer = _UnprocessedBufferRepo(session_factory=self.factory)
        self.replies: list[
            str
        ] = []  # "ok" | "garbage" | "outage", consumed per attempt
        self.detect_calls = 0
        self.patches = [
            mock.patch.object(boundary_mod, "unprocessed_buffer_repo", self.buffer),
            mock.patch.object(
                boundary_mod, "memcell_repo", _MemcellRepo(session_factory=self.factory)
            ),
            mock.patch.object(
                boundary_mod,
                "conversation_status_repo",
                _ConversationStatusRepo(session_factory=self.factory),
            ),
            mock.patch.object(boundary_mod, "detect_boundaries", self.detect),
        ]

    async def __aenter__(self) -> _Stage:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        for p in self.patches:
            p.start()
        boundary_mod._unusable_replies.clear()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        for p in self.patches:
            p.stop()
        await self.engine.dispose()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def detect(self, messages: Any, **kw: Any) -> DetectionResult:
        self.detect_calls += 1
        reply = self.replies.pop(0) if self.replies else "ok"
        if reply == "garbage":
            raise ValueError("No JSON object found in boundary LLM response")
        if reply == "outage":
            raise LLMServiceError("insufficient quota")
        cells = [MemCell(items=list(messages), timestamp=messages[-1].timestamp)]
        return DetectionResult(cells=cells, tail=[])

    async def defer(self, session: str, n: int, start: int) -> Any:
        """A deferred add: the messages land in the buffer with no model call."""
        ingested = await ingest.process(self._payload(session, n, start))
        return await boundary_mod.buffer_messages(ingested, mode="chat")

    @staticmethod
    def _payload(session: str, n: int, start: int) -> dict[str, Any]:
        return {
            "session_id": session,
            "messages": [
                {
                    "content": f"m{start + i}",
                    "timestamp": BASE_TS + start + i,
                    "sender_id": "u",
                    "role": "user",
                }
                for i in range(n)
            ],
        }

    async def add(self, session: str, n: int, start: int) -> Any:
        payload = {
            "session_id": session,
            "messages": [
                {
                    "content": f"m{start + i}",
                    "timestamp": BASE_TS + start + i,
                    "sender_id": "u",
                    "role": "user",
                }
                for i in range(n)
            ],
        }
        ingested = await ingest.process(payload)
        return await boundary_mod.prepare_cells(
            ingested,
            mode="chat",
            is_final=False,
            llm_client=object(),
            prompt_loader=SimpleNamespace(load=lambda name: "p"),  # type: ignore[arg-type]
            hard_token_limit=10_000,
            hard_msg_limit=100,
        )

    async def tracks(self, session: str) -> dict[str, list[str]]:
        return {
            track: [
                r.message_id for r in await self.buffer.list_for_track(session, track)
            ]
            for track in ("memorize", "quarantine")
        }


async def test_third_consecutive_unusable_call_quarantines_the_buffer() -> None:
    async with _Stage() as stage:
        attempts = boundary_mod._BOUNDARY_MAX_ATTEMPTS
        # The poison arrives through a deferred add and sits in the buffer;
        # every later call merges it back in and the model chokes on it.
        await stage.defer("s1", 1, 1)
        stage.replies = ["garbage"] * (attempts * boundary_mod._POISON_CALLS)
        with pytest.raises(ValueError):
            await stage.add("s1", 1, 2)
        with pytest.raises(ValueError):
            await stage.add("s1", 1, 3)
        with pytest.raises(SessionBufferQuarantinedError):
            await stage.add("s1", 1, 4)
        tracks = await stage.tracks("s1")
        assert tracks["memorize"] == [], "the session starts over"
        # The buffered poison and the quarantining call's own message are
        # kept; the two earlier calls' messages were never persisted (the
        # calls failed, as they did before this change).
        assert sorted(tracks["quarantine"]) == sorted(
            ["m_s1_1700000000001_000", "m_s1_1700000000004_000"]
        )
        assert stage.detect_calls == attempts * boundary_mod._POISON_CALLS
        # The session goes on: a healthy call is extracted from an empty buffer.
        outcome = await stage.add("s1", 1, 5)
        assert outcome.status == "extracted"
        assert (await stage.tracks("s1"))["memorize"] == []
        assert len((await stage.tracks("s1"))["quarantine"]) == 2


async def test_a_usable_reply_resets_the_count() -> None:
    async with _Stage() as stage:
        attempts = boundary_mod._BOUNDARY_MAX_ATTEMPTS
        stage.replies = ["garbage"] * attempts + ["ok"] + ["garbage"] * attempts
        with pytest.raises(ValueError):
            await stage.add("s1", 1, 1)
        assert (await stage.add("s1", 1, 2)).status == "extracted"
        with pytest.raises(ValueError):  # the count starts again at one
            await stage.add("s1", 1, 3)
        assert (await stage.tracks("s1"))["quarantine"] == []


async def test_infrastructure_failures_are_not_counted() -> None:
    async with _Stage() as stage:
        stage.replies = ["outage"] * 5
        for i in range(1, 6):
            with pytest.raises(LLMServiceError):
                await stage.add("s1", 1, i)
        tracks = await stage.tracks("s1")
        assert tracks["quarantine"] == []
        assert tracks["memorize"] == [], (
            "an outage keeps nothing and loses nothing: the client retries"
        )


async def test_sessions_are_counted_apart() -> None:
    async with _Stage() as stage:
        attempts = boundary_mod._BOUNDARY_MAX_ATTEMPTS
        stage.replies = ["garbage"] * (attempts * 2) + ["ok"]
        with pytest.raises(ValueError):
            await stage.add("s1", 1, 1)
        with pytest.raises(ValueError):
            await stage.add("s1", 1, 2)
        assert (await stage.add("s2", 1, 3)).status == "extracted"
        assert (await stage.tracks("s1"))["quarantine"] == []
