"""The one-to-one write path: message ids, timestamps and failed segments.

* Two different messages with one timestamp, each sent alone, are both kept:
  a message's id is its sender and content, not its place in its batch, and a
  true re-send is one message, counted in ``duplicates``.
* A millisecond timestamp before 2001-09-09 is milliseconds, not seconds past
  year 9999.
* A segment whose extraction failed is extracted again by the session's next
  add or flush -- once: its re-send is a duplicate, an add waits
  ``RETRY_AFTER_MS``, and after ``MAX_EXTRACTION_FAILURES`` it is left.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from pydantic import ValidationError

from everos.core.utils.datetime import MAX_EPOCH_MS, to_timestamp_ms
from everos.entrypoints.api.schemas.memorize import MessageItemDTO
from everos.memory.models import MemcellRecord
from everos.memory.ports import Scope
from everos.service.memory._retry import MAX_EXTRACTION_FAILURES
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


# ── B-7 / A-3: a segment whose extraction failed ─────────────────────────


async def test_a_failed_segment_is_extracted_by_the_next_flush_once(
    tmp_path: Path,
) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        said = [
            _message("amy", T0, "My door code is 4417"),
            _message("bot", T0 + 1, "Noted.", role="assistant"),
        ]
        await _add(world, "s", *said)
        model.fail_episode = 10
        with pytest.raises(RuntimeError, match="episode model down"):
            await _flush(world, "s")
        model.fail_episode = 0
        assert await _buffered(world, "s") == []
        assert await _episodes(world, "amy") == []

        model.calls.clear()
        assert (await _flush(world, "s")).status == "extracted"
        assert model.calls == ["EpisodeReply"]
        assert len(await _episodes(world, "amy")) == 1

        # Extracted: nothing is left to try, and nothing is held any more.
        model.calls.clear()
        assert (await _flush(world, "s")).status == "no_extraction"
        assert model.calls == []
        assert await world.stores.memcell.failed_segments(world.session("s")) == []


async def test_a_resend_of_a_failed_segment_is_its_retry(tmp_path: Path) -> None:
    """The documented recovery -- send the batch again -- extracts the segment
    in that call, once: its messages are not taken again, nor duplicates."""
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        said = [
            _message("amy", T0, "My door code is 4417"),
            _message("bot", T0 + 1, "Noted.", role="assistant"),
        ]
        await _add(world, "s", *said)
        model.fail_episode = 10
        with pytest.raises(RuntimeError, match="episode model down"):
            await _flush(world, "s")
        model.fail_episode = 0
        model.calls.clear()
        resent = await _add(world, "s", *said)
        assert (resent.duplicates, resent.status) == (0, "extracted")
        assert model.calls == ["EpisodeReply"]
        assert await _buffered(world, "s") == []
        assert len(await _episodes(world, "amy")) == 1
        assert (await _flush(world, "s")).status == "no_extraction"
        assert len(await _episodes(world, "amy")) == 1


async def test_an_add_waits_before_it_retries_a_flush_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _add(world, "s", _message("amy", T0, "Trip to Porto"))
        model.fail_episode = 1
        with pytest.raises(RuntimeError):
            await _flush(world, "s")
        model.calls.clear()
        await _add(world, "s", _message("amy", T0 + 10, "in May"))
        assert "EpisodeReply" not in model.calls, "within RETRY_AFTER_MS"

        monkeypatch.setattr("everos.service.memory._retry.RETRY_AFTER_MS", 0)
        model.calls.clear()
        later = await _add(world, "s", _message("amy", T0 + 20, "with Ben"))
        assert model.calls.count("EpisodeReply") == 1
        assert later.status == "extracted"
        assert len(await _episodes(world, "amy")) == 1


async def test_a_segment_that_keeps_failing_is_left(tmp_path: Path) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _add(world, "s", _message("amy", T0, "Trip to Porto"))
        model.fail_episode = 100
        with pytest.raises(RuntimeError):
            await _flush(world, "s")
        for _ in range(MAX_EXTRACTION_FAILURES + 2):
            await _flush(world, "s")
        assert model.calls.count("EpisodeReply") == MAX_EXTRACTION_FAILURES
        [left] = await world.stores.memcell.failed_segments(world.session("s"))
        assert left.failures == MAX_EXTRACTION_FAILURES
        # Left: its messages are no longer held, so the client can send them.
        model.fail_episode = 0
        resent = await _add(world, "s", _message("amy", T0, "Trip to Porto"))
        assert resent.duplicates == 0


async def test_a_forget_of_the_conversation_holds_the_retry(tmp_path: Path) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await _add(world, "s", _message("amy", T0, "My door code is 4417"))
        model.fail_episode = 1
        with pytest.raises(RuntimeError):
            await _flush(world, "s")
        await world.memory.delete(
            app_id=APP, project_id=PROJECT, user_id="amy", session_id="s"
        )
        model.calls.clear()
        assert (await _flush(world, "s")).status == "no_extraction"
        assert "EpisodeReply" not in model.calls
        assert await _episodes(world, "amy") == []
        # Its messages are not held: sent again after the delete, they are new.
        again = await _add(world, "s", _message("amy", T0, "My door code is 4417"))
        assert again.duplicates == 0


async def test_a_segment_cut_before_a_crash_is_extracted_after_it(
    tmp_path: Path,
) -> None:
    """Every memcell is cut marked "to extract": one whose extraction never ran
    (the process died after the cut) is extracted by the session's next call."""
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        text = "My door code is 4417"
        cell = MemCell(
            items=[
                ChatMessage(
                    id="m1", role="user", content=text, sender_id="amy", timestamp=T0
                )
            ],
            timestamp=T0,
        )
        await world.stores.session_buffer.cut(
            world.session("s"),
            [
                MemcellRecord(
                    memcell_id="mc_crash",
                    cell=cell,
                    raw_type="Conversation",
                    message_ids=["m1"],
                    sender_ids=["amy"],
                    to_extract=True,
                )
            ],
            tail=[],
            last_memcell_ts=None,
        )
        assert (await _flush(world, "s")).status == "extracted"
        assert model.calls == ["EpisodeReply"]
        assert [e.item.episode for e, _ in await _episodes(world, "amy")] == ["C"]
        assert (await _flush(world, "s")).status == "no_extraction"
        assert model.calls == ["EpisodeReply"]
