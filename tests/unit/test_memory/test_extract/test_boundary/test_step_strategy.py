"""The ``step`` boundary strategy replays the EverAlgo LoCoMo reference loop.

The buffer is the history already judged. Every fresh message is handed to the
operator's ``adetect_step`` together with that history, and the next history
starts where the returned ``Cuts`` say; the cells are the rows at the returned
positions. A flush closes whatever history is left. Tool rows and messages
older than the history are refused rather than guessed at.

Under everalgo's own operator the reference behaviour follows: the first two
messages of a history are taken without the model, every later message costs
one step call, and a cut after a history of more than five messages carries
its last message into the next history (bridging), so that message lands in
two cells.

Real: ingest, :class:`BoundaryDetection`, everalgo's ``BoundaryDetector`` (or
a scripted operator) and the SQLite stores on a temp file. Scripted: the
model's step replies.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from datetime import UTC, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any, Literal

import pytest
from everalgo.boundary import BoundaryDetector, Cuts
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, ConversationItem
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
from everos.memory.extract.boundary import (
    BoundaryDetection,
    BoundaryOperator,
    BoundaryOutcome,
)
from everos.memory.extract.ingest import service as ingest
from tests.fakes import (
    InMemorySenderNameStore,
    bundled_instructions,
    unused_raw_file_builder,
)

BASE_TS = 1_700_000_000_000


def _step_reply(should_end: bool) -> str:
    return json.dumps({"reasoning": "scripted", "should_end": should_end})


class _ScriptedOperator:
    """Answers each step with the next scripted ``Cuts`` and records the windows."""

    def __init__(self, cuts: list[Cuts]) -> None:
        self.cuts = cuts
        self.windows: list[list[str]] = []
        self.zones: list[tzinfo] = []

    async def adetect_batch(
        self,
        items: Sequence[ConversationItem],
        *,
        is_final: bool,
        custom_instructions: str | None,
        tz: tzinfo,
        hard_token_limit: int,
        hard_message_limit: int,
    ) -> Cuts:
        raise AssertionError("the step strategy never asks for a batch")

    async def adetect_step(
        self,
        items: Sequence[ChatMessage],
        *,
        custom_instructions: str | None,
        tz: tzinfo,
    ) -> Cuts:
        self.windows.append([item.content for item in items])
        self.zones.append(tz)
        return self.cuts.pop(0) if self.cuts else Cuts(ends=[], tail_start=0)


class _Stage:
    """``step`` boundary detection on a temp database with scripted replies."""

    def __init__(
        self,
        *,
        detector: BoundaryOperator | None = None,
        tz: tzinfo = UTC,
        strategy: Literal["batch", "step"] = "step",
    ) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="step_boundary_"))
        root = MemoryRoot(self.tmp)
        root.ensure()
        self.engine = create_system_engine(root.system_db, SqliteSettings())
        self.factory = create_session_factory(self.engine)
        self.buffer = UnprocessedBufferRepo(session_factory=self.factory)
        self.cuts: list[bool] = []
        self.requests: list[str] = []
        self.llm = FakeLLMClient(handler=self._reply)
        self.boundary = BoundaryDetection(
            detector=detector or BoundaryDetector(llm=self.llm),
            instructions=bundled_instructions(),
            session_buffer=SqliteSessionBufferStore(
                self.buffer, SessionCutRepo(session_factory=self.factory)
            ),
            settings=BoundaryDetectionSettings(strategy=strategy),
            tz=tz,
        )

    def _reply(self, messages: Any, **_kwargs: Any) -> str:
        self.requests.append(messages[-1].content)
        return _step_reply(self.cuts.pop(0) if self.cuts else False)

    async def __aenter__(self) -> _Stage:
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.engine.dispose()
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def add(
        self,
        starts: list[int],
        *,
        final: bool = False,
        extra: list[dict[str, Any]] | None = None,
        mode: Literal["chat", "agent"] = "chat",
    ) -> BoundaryOutcome:
        payload = {
            "session_id": "s1",
            "messages": [
                {
                    "content": f"m{n}",
                    "timestamp": BASE_TS + n,
                    "sender_id": "u",
                    "role": "user",
                }
                for n in starts
            ]
            + (extra or []),
        }
        ingested = await ingest.process(
            payload,
            multimodal_llm=None,
            raw_file_builder=unused_raw_file_builder,
            sender_names=InMemorySenderNameStore(),
        )
        return await self.boundary.detect(ingested, mode=mode, is_final=final)

    async def buffered(self) -> list[str]:
        rows = await self.buffer.list_for_track("s1", "memorize")
        return [r.message_id for r in rows]


def _texts(outcome: BoundaryOutcome) -> list[list[str]]:
    return [[item.content for item in cell.items] for cell in outcome.cells]


async def test_the_first_two_messages_never_reach_the_model() -> None:
    async with _Stage() as stage:
        outcome = await stage.add([1, 2])
        assert outcome.status == "accumulated"
        assert stage.llm.call_count == 0
        assert len(await stage.buffered()) == 2


async def test_every_later_message_is_judged_once() -> None:
    async with _Stage() as stage:
        outcome = await stage.add([1, 2, 3, 4, 5])
        assert outcome.status == "accumulated"
        assert stage.llm.call_count == 3


async def test_a_cut_after_a_short_history_starts_clean() -> None:
    async with _Stage() as stage:
        stage.cuts = [False, True]  # judging m3, then m4
        outcome = await stage.add([1, 2, 3, 4])
        assert _texts(outcome) == [["m1", "m2", "m3"]]
        assert len(outcome.per_cell_message_ids[0]) == 3
        assert len(await stage.buffered()) == 1, "the history restarts at m4"


async def test_a_cut_after_a_long_history_bridges_its_last_message() -> None:
    async with _Stage() as stage:
        stage.cuts = [False, False, False, False, True]  # m3..m6 stay, m7 cuts
        first = await stage.add([1, 2, 3, 4, 5, 6, 7])
        assert _texts(first) == [["m1", "m2", "m3", "m4", "m5", "m6"]]
        assert len(await stage.buffered()) == 2, "the history is [m6, m7]"

        last = await stage.add([8], final=True)
        assert _texts(last) == [["m6", "m7", "m8"]]
        bridged = set(first.per_cell_message_ids[0]) & set(last.per_cell_message_ids[0])
        assert len(bridged) == 1, "m6 sits in both cells, by id"
        assert await stage.buffered() == []


async def test_the_history_carries_over_between_calls() -> None:
    async with _Stage() as stage:
        await stage.add([1, 2])
        await stage.add([3])
        assert stage.llm.call_count == 1
        assert "m1" in stage.requests[0] and "m2" in stage.requests[0]


async def test_a_flush_closes_the_history_into_a_last_cell() -> None:
    async with _Stage() as stage:
        await stage.add([1, 2, 3])
        outcome = await stage.add([4], final=True)
        assert _texts(outcome) == [["m1", "m2", "m3", "m4"]]
        assert outcome.status == "extracted"
        assert await stage.buffered() == []


async def test_a_tool_row_fails_the_call() -> None:
    tool_call = {
        "content": "",
        "timestamp": BASE_TS + 3,
        "sender_id": "agent",
        "role": "assistant",
        "tool_calls": [
            {
                "id": "tc1",
                "type": "function",
                "function": {"name": "f", "arguments": "{}"},
            }
        ],
    }
    async with _Stage() as stage:
        with pytest.raises(ValueError, match="chat messages only"):
            await stage.add([1, 2], extra=[tool_call], mode="agent")
        assert stage.llm.call_count == 0


async def test_a_message_older_than_the_history_fails_the_call() -> None:
    async with _Stage() as stage:
        await stage.add([10, 11])
        with pytest.raises(ValueError, match="older than the buffered history"):
            await stage.add([5])


async def test_every_fresh_message_reaches_the_operator_after_its_history() -> None:
    operator = _ScriptedOperator([])
    async with _Stage(detector=operator) as stage:
        await stage.add([1, 2])
        await stage.add([3])
        assert operator.windows == [["m1"], ["m1", "m2"], ["m1", "m2", "m3"]]


async def test_the_next_history_starts_at_the_returned_tail_start() -> None:
    no_cut = Cuts(ends=[], tail_start=0)
    operator = _ScriptedOperator([no_cut, no_cut, Cuts(ends=[2], tail_start=1)])
    async with _Stage(detector=operator) as stage:
        outcome = await stage.add([1, 2, 3])
        assert _texts(outcome) == [["m1", "m2"]]
        assert operator.windows[-1] == ["m1", "m2", "m3"]
        assert len(await stage.buffered()) == 2, "the history is [m2, m3]"

        await stage.add([4])
        assert operator.windows[-1] == ["m2", "m3", "m4"]


async def test_the_configured_zone_reaches_every_step_judgement() -> None:
    zone = timezone(timedelta(hours=8))
    operator = _ScriptedOperator([])
    async with _Stage(detector=operator, tz=zone) as stage:
        await stage.add([1, 2, 3])
        assert operator.zones and all(z is zone for z in operator.zones)


async def test_the_step_request_shows_message_times_in_the_configured_zone() -> None:
    async with _Stage(tz=timezone(timedelta(hours=8))) as stage:
        await stage.add([1, 2, 3])
        assert stage.requests
        assert all("UTC+08:00" in request for request in stage.requests)


class _BatchZoneRecorder:
    """Records the zone and hard limits each batch call is given; never cuts."""

    def __init__(self) -> None:
        self.calls: list[tuple[tzinfo, int, int]] = []

    async def adetect_batch(
        self,
        items: Sequence[ConversationItem],
        *,
        is_final: bool,
        custom_instructions: str | None,
        tz: tzinfo,
        hard_token_limit: int,
        hard_message_limit: int,
    ) -> Cuts:
        self.calls.append((tz, hard_token_limit, hard_message_limit))
        return Cuts(ends=[], tail_start=0)

    async def adetect_step(
        self,
        items: Sequence[ChatMessage],
        *,
        custom_instructions: str | None,
        tz: tzinfo,
    ) -> Cuts:
        raise AssertionError("the batch strategy never asks for a step")


async def test_a_batch_call_gets_the_configured_zone_and_hard_limits() -> None:
    zone = timezone(timedelta(hours=8))
    recorder = _BatchZoneRecorder()
    async with _Stage(detector=recorder, tz=zone, strategy="batch") as stage:
        await stage.add([1, 2])
    defaults = BoundaryDetectionSettings()
    assert recorder.calls == [
        (zone, defaults.hard_token_limit, defaults.hard_msg_limit)
    ]
