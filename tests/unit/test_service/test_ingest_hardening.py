"""The one-to-one write path: message ids, timestamps and failed segments.

* Two different messages with one timestamp, each sent alone, are both kept:
  a message's id is its sender and content, not its place in its batch, and a
  true re-send is one message, counted in ``duplicates``.
* A millisecond timestamp before 2001-09-09 is milliseconds, not seconds past
  year 9999.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from pydantic import ValidationError

from everos.core.utils.datetime import MAX_EPOCH_MS, to_timestamp_ms
from everos.entrypoints.api.schemas.memorize import MessageItemDTO
from everos.memory.ports import Scope
from tests.helpers.groups import APP, PROJECT, GroupWorld

T0 = 1_772_323_200_000  # 2026-03-01 00:00:00 UTC


class _Model:
    """Boundary never cuts (a flush closes the segment); episodes scripted."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fail_episode = 0

    def __call__(self, messages: Any, *, response_format: type, **_kw: Any) -> str:
        name = response_format.__name__
        self.calls.append(name)
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "EpisodeReply":
            if self.fail_episode:
                self.fail_episode -= 1
                raise RuntimeError("episode model down")
            return json.dumps({"title": "T", "summary": "S", "content": "C"})
        return json.dumps({"operations": []})


def _message(sender: str, at: int, text: str, *, role: str = "user") -> dict[str, Any]:
    return {"sender_id": sender, "role": role, "timestamp": at, "content": text}


async def _add(
    world: GroupWorld, session: str, *messages: dict[str, Any], defer: bool = False
) -> Any:
    return await world.memory.add(
        {
            "session_id": session,
            "app_id": APP,
            "project_id": PROJECT,
            "messages": list(messages),
        },
        request_id="r",
        async_mode=False,
        defer_extraction=defer,
    )


async def _flush(world: GroupWorld, session: str) -> Any:
    return await world.memory.flush(
        session_id=session, request_id="f", app_id=APP, project_id=PROJECT
    )


async def _buffered(world: GroupWorld, session: str) -> list[str]:
    return [
        m.text for m in await world.stores.session_buffer.list(world.session(session))
    ]


async def _episodes(world: GroupWorld, user: str) -> list[Any]:
    scope = Scope(app_id=APP, project_id=PROJECT, owner_id=user, owner_type="user")
    return await world.stores.episode.read_all(scope)


# ── A-1: one timestamp, two messages ─────────────────────────────────────


async def test_two_messages_with_one_timestamp_sent_alone_are_both_kept(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        first = await _add(world, "s", _message("amy", T0, "Where is my passport?"))
        second = await _add(
            world, "s", _message("bot", T0, "In the drawer.", role="assistant")
        )
        assert (first.duplicates, second.duplicates) == (0, 0)
        assert await _buffered(world, "s") == [
            "Where is my passport?",
            "In the drawer.",
        ]


async def test_seconds_timestamps_in_one_second_are_both_kept(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        at = T0 // 1000  # seconds
        await _add(world, "s", _message("amy", at, "Book the 9:40 train"))
        await _add(world, "s", _message("bot", at, "Booked.", role="assistant"))
        assert await _buffered(world, "s") == ["Book the 9:40 train", "Booked."]


async def test_a_resend_is_one_message_and_is_counted(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        batch = [_message("amy", T0, "ok"), _message("amy", T0 + 5, "see you")]
        assert (await _add(world, "s", *batch)).duplicates == 0
        # Re-sent alone, and in another order: the same two messages.
        assert (await _add(world, "s", batch[1])).duplicates == 1
        assert (await _add(world, "s", *reversed(batch))).duplicates == 2
        assert await _buffered(world, "s") == ["ok", "see you"]


async def test_two_equal_messages_in_one_batch_are_both_kept(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        twice = [_message("amy", T0, "ok"), _message("amy", T0, "ok")]
        assert (await _add(world, "s", *twice)).duplicates == 0
        assert await _buffered(world, "s") == ["ok", "ok"]
        assert (await _add(world, "s", *twice)).duplicates == 2


# ── A-4: a millisecond timestamp before 2001-09-09 ───────────────────────


async def test_a_millisecond_timestamp_before_2001_is_milliseconds(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_Model())) as world:
        new_year_2001 = 978_307_200_000
        await _add(world, "ms", _message("amy", new_year_2001, "Happy new year"))
        await _add(world, "s", _message("amy", new_year_2001 // 1000, "Happy new year"))
        for session in ("ms", "s"):
            [row] = await world.stores.session_buffer.list(world.session(session))
            assert to_timestamp_ms(row.timestamp) == new_year_2001


def test_a_timestamp_past_the_calendar_is_refused_not_a_500() -> None:
    fields = {"sender_id": "amy", "role": "user", "content": "hi"}
    assert MessageItemDTO(timestamp=MAX_EPOCH_MS, **fields).timestamp == MAX_EPOCH_MS
    with pytest.raises(ValidationError, match="less than or equal"):
        MessageItemDTO(timestamp=MAX_EPOCH_MS + 1, **fields)
