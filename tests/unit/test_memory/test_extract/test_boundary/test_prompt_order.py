"""Boundary requests carry the window's count after the records.

Every add re-sends the session's uncut buffer, so consecutive boundary requests of
one session repeat their first records. everalgo puts the count -- the one part
that changes -- in front of them (``<messages count="N">``), and a provider's prefix
cache then matches only the system prompt (measured 2026-11-11: 1,664 tokens shared
per call). With the count after the records, each call's records are a prefix of the
next call's.

The requests below are rendered by the pinned everalgo itself, so an everalgo
release that renders them differently fails here instead of silently losing the
reorder.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from everalgo.boundary import adetect_boundaries
from everalgo.llm import LLMMessage
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import SQLModel

from everos.config import BoundaryDetectionSettings, SqliteSettings
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
from everos.memory.extract.boundary import BoundaryDetection
from everos.memory.extract.boundary.prompt_order import (
    CountAfterMessages,
    count_after_messages,
)
from everos.memory.extract.ingest import service as ingest
from tests.fakes import bundled_instructions, unused_raw_file_builder

_NO_CUT = json.dumps(
    {"reasoning": "Question unanswered", "boundaries": [], "tail_complete": False}
)


class _Reply(BaseModel):
    ok: bool


def _chat(i: int) -> ChatMessage:
    return ChatMessage(
        id=f"m{i}",
        role="user",
        content=f"message {i} <b>with</b> a tag\nand a second line",
        timestamp=1_767_690_000_000 + i * 60_000,
        sender_id="u1",
        sender_name="Alice",
    )


async def _requests(
    windows: list[int], *, reorder: bool = True
) -> list[list[LLMMessage]]:
    """What reaches the model when everalgo segments each window."""
    seen: list[list[LLMMessage]] = []

    def handler(messages: list[LLMMessage], **_kwargs: Any) -> str:
        seen.append(messages)
        return _NO_CUT

    fake = FakeLLMClient(handler=handler)
    llm = CountAfterMessages(fake) if reorder else fake
    for n in windows:
        await adetect_boundaries([_chat(i) for i in range(n)], llm=llm)
    return seen


async def test_the_count_follows_the_records_and_nothing_else_changes() -> None:
    (sent,) = await _requests([3])
    (original,) = await _requests([3], reorder=False)
    system, user = sent
    assert system == original[0]  # the system message is untouched
    head = '<messages count="3">\n'
    assert original[1].content.startswith(head)
    records = original[1].content[len(head) : -len("\n</messages>")]
    assert user.content == f"<messages>\n{records}\n</messages>\n<count>3</count>"


async def test_each_request_repeats_the_previous_records_as_a_prefix() -> None:
    """What a provider's prefix cache needs: call k is a prefix of call k+1 up to
    the end of its last record, so only the new records are uncached."""
    first, second, third = await _requests([2, 3, 5])
    for earlier, later in ((first, second), (second, third)):
        assert earlier[0] == later[0]
        shared = earlier[1].content.removesuffix(
            earlier[1].content[earlier[1].content.index("\n</messages>") :]
        )
        assert later[1].content.startswith(shared)
        assert len(shared) > len("<messages>\n")


async def test_the_reply_and_the_call_keywords_pass_through() -> None:
    calls: list[dict[str, Any]] = []

    def handler(messages: list[LLMMessage], **kwargs: Any) -> str:
        calls.append(kwargs)
        return "reply"

    llm = CountAfterMessages(FakeLLMClient(handler=handler))
    got = await llm.achat(
        [LLMMessage(role="user", content="plain")],
        response_format=_Reply,
        temperature=0.4,
        max_tokens=9,
        seed=1,
    )
    assert got == "reply"
    assert (calls[0]["temperature"], calls[0]["max_tokens"], calls[0]["seed"]) == (
        0.4,
        9,
        1,
    )


@pytest.mark.parametrize(
    "content",
    [
        "plain text",
        '<messages count="2">\n[1] a\n[2] b\n</messages>\ntrailing',
        '<messages count="x">\n[1] a\n</messages>',
        'prefix <messages count="2">\n[1] a\n</messages>',
        "",
    ],
)
def test_any_other_shape_is_sent_unchanged(content: str) -> None:
    assert count_after_messages(content) == content


# ── on the product path: two adds of one session ────────────────────────────


@pytest.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    root = MemoryRoot(tmp_path)
    root.ensure()
    engine = create_system_engine(root.system_db, SqliteSettings())
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


async def test_a_second_add_repeats_the_first_adds_records_as_a_prefix(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    """The uncut buffer is re-sent with the new messages; the count no longer
    stands between the system prompt and the records the two requests share."""
    sent: list[list[LLMMessage]] = []

    def handler(messages: list[LLMMessage], **_kwargs: Any) -> str:
        sent.append(messages)
        return _NO_CUT

    boundary = BoundaryDetection(
        llm=FakeLLMClient(handler=handler),
        instructions=bundled_instructions(),
        session_buffer=SqliteSessionBufferStore(
            UnprocessedBufferRepo(session_factory=factory),
            SessionCutRepo(session_factory=factory),
        ),
        limits=BoundaryDetectionSettings(hard_token_limit=10**9, hard_msg_limit=10**6),
    )
    for start in (0, 2):
        payload = {
            "session_id": "s1",
            "messages": [
                {
                    "content": f"message {i}",
                    "timestamp": 1_767_690_000 + 60 * i,
                    "sender_id": "u",
                    "role": "user",
                }
                for i in range(start, start + 2)
            ],
        }
        ingested = await ingest.process(
            payload, multimodal_llm=None, raw_file_builder=unused_raw_file_builder
        )
        outcome = await boundary.detect(ingested, is_final=False)
        assert outcome.status == "accumulated"

    first, second = (request[-1].content for request in sent)
    assert first.endswith("\n</messages>\n<count>2</count>")
    assert second.endswith("\n</messages>\n<count>4</count>")
    assert second.startswith(first.removesuffix("\n</messages>\n<count>2</count>"))
