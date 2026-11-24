"""Preserve parsed boundary decisions from buffer input to memcell dependencies."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, tzinfo
from pathlib import Path
from types import SimpleNamespace

import pytest
from everalgo.boundary import Cuts
from everalgo.types import ConversationItem

from everos.config import BoundaryDetectionSettings
from everos.core.utils.datetime import from_timestamp
from everos.infra.memtrace.tracker import MemTraceTracker
from everos.memory import CanonicalMessage
from everos.memory.extract.boundary import BoundaryDetection
from everos.memory.extract.trace import collect_construction_trace

from .test_memtrace_tracker import _TS_MS, _message, _payload


@pytest.fixture(autouse=True)
def _memtrace_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing is opt-in; these tests exercise the enabled tracker."""
    monkeypatch.setenv("EVEROS_MEMTRACE_ENABLED", "1")


class _ScriptedOperator:
    def __init__(self, boundaries: list[int]) -> None:
        self.boundaries = boundaries

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
        ends = list(self.boundaries)
        if is_final and (not ends or ends[-1] != len(items)):
            ends.append(len(items))
        return Cuts(ends=ends, tail_start=ends[-1] if ends else 0)


class _Buffer:
    async def list(self, session: object) -> list[CanonicalMessage]:
        return []

    async def keep(self, *args: object, **kwargs: object) -> None:
        return None

    async def cut(self, *args: object, **kwargs: object) -> None:
        return None

    async def held_by(self, session: object, message_ids: object) -> dict[str, str]:
        return {}


@pytest.mark.parametrize("mode", ["chat", "agent"])
@pytest.mark.parametrize(
    ("boundaries", "is_final", "cell_count"),
    [([2], False, 1), ([1, 2], True, 3), ([], False, 0), ([], True, 1)],
)
async def test_model_decision_to_memcell_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
    boundaries: list[int],
    is_final: bool,
    cell_count: int,
) -> None:
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    messages = [
        CanonicalMessage(
            message_id=f"m_sess_a_{_TS_MS + i}_000",
            session_id="sess_a",
            sender_id="alice",
            role="user",
            timestamp=from_timestamp((_TS_MS + i) / 1000),
            text=f"Part {i}",
        )
        for i in range(4)
    ]
    ingested = SimpleNamespace(
        app_id="default",
        project_id="default",
        session_id="sess_a",
        messages=messages,
    )
    boundary = BoundaryDetection(
        detector=_ScriptedOperator(boundaries),
        instructions=SimpleNamespace(load=lambda _: "Messages: {messages}"),
        session_buffer=_Buffer(),
        settings=BoundaryDetectionSettings(strategy="batch"),
        tz=UTC,
    )

    outcome = await boundary.detect(ingested, mode=mode, is_final=is_final)

    assert len(outcome.cells) == cell_count
    detection = outcome.detection
    assert detection is not None
    assert detection["detection_performed"] is True
    assert detection["model_result"]["cell_message_ids"] == (
        outcome.per_cell_message_ids
    )
    assert detection["message_ids"] == [m.message_id for m in messages]
    assert "raw_response" not in detection

    tracker = MemTraceTracker()
    for i, message in enumerate(messages):
        await tracker.on_memorize(
            _payload(_message(message.text, _TS_MS + i)),
            is_final=False,
            defer_extraction=True,
            result=None,
            request_id=f"add-{i}",
        )
    event_cells = [
        {
            "id": memcell_id,
            "message_ids": message_ids,
            "sender_ids": sender_ids,
            "content": cell.model_dump(mode="json"),
        }
        for memcell_id, message_ids, sender_ids, cell in zip(
            outcome.memcell_ids,
            outcome.per_cell_message_ids,
            outcome.per_cell_all_senders,
            outcome.cells,
            strict=True,
        )
    ]
    await tracker.on_memorize(
        _payload(),
        is_final=is_final,
        defer_extraction=False,
        result=None,
        request_id="detect",
        construction_events=[
            {
                "phase": "boundary",
                "app_id": "default",
                "project_id": "default",
                "session_id": "sess_a",
                "observed_at": "2026-01-01T00:00:00Z",
                "detection_performed": True,
                "detection": detection,
                "cells": event_cells,
            }
        ],
    )
    graph = tracker._graphs[("default", "default", "alice", "sess_a")]
    result = graph.search_variables(class_name="boundary_detection_result")[0]
    assert not graph.search_variables(class_name="raw_boundary_detection_response")
    assert json.loads(result.raw_value)["model_result"] == detection["model_result"]
    assert json.loads(result.raw_value)["message_ids"] == detection["message_ids"]
    assert result.category == "llm_response"
    edges = graph.get_all_edges()

    def parents(node_id: str) -> set[str]:
        return {
            e.source_full_node_id for e in edges if e.target_full_node_id == node_id
        }

    (buffer_id,) = parents(result.full_node_id)
    assert buffer_id == "buffer-sess_a-alice@5"
    for cell in event_cells:
        assert parents(cell["id"] + "@1") == {result.full_node_id}
        assert parents("cell-input-" + cell["id"] + "@1") == {cell["id"] + "@1"}
    if not outcome.cells:
        marker = graph.search_variables(class_name="no_split_marker")[0]
        assert parents(marker.full_node_id) == {result.full_node_id}


class _AllCutBuffer(_Buffer):
    """Buffer whose memcells already hold every fresh id (timeout resend)."""

    def __init__(self, held: list[CanonicalMessage]) -> None:
        self._held = held

    async def list(self, session: object) -> list[CanonicalMessage]:
        return list(self._held)

    async def held_by(
        self, session: object, message_ids: Sequence[str]
    ) -> dict[str, str]:
        return dict.fromkeys(message_ids, "mc_held")


@pytest.mark.parametrize("call", ["buffer", "detect"])
async def test_fully_filtered_resend_records_unchanged_buffer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    call: str,
) -> None:
    """A resend whose rows were all cut already must trace the buffer as-is.

    Without the authoritative ``buffer_loaded`` snapshot the tracker falls
    back to appending the resent request messages to the graph buffer,
    although the real session buffer never changed.
    """
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    held = CanonicalMessage(
        message_id=f"m_sess_a_{_TS_MS}_000",
        session_id="sess_a",
        sender_id="alice",
        role="user",
        timestamp=from_timestamp(_TS_MS / 1000),
        text="Kept tail",
    )
    resent = CanonicalMessage(
        message_id=f"m_sess_a_{_TS_MS + 1}_000",
        session_id="sess_a",
        sender_id="alice",
        role="user",
        timestamp=from_timestamp((_TS_MS + 1) / 1000),
        text="Resent after a client timeout",
    )
    ingested = SimpleNamespace(
        app_id="default",
        project_id="default",
        session_id="sess_a",
        messages=[resent],
    )
    boundary = BoundaryDetection(
        detector=_ScriptedOperator([]),
        instructions=SimpleNamespace(load=lambda _: "Messages: {messages}"),
        session_buffer=_AllCutBuffer([held]),
        settings=BoundaryDetectionSettings(strategy="batch"),
        tz=UTC,
    )

    with collect_construction_trace() as events:
        if call == "buffer":
            outcome = await boundary.buffer(ingested, mode="chat")
        else:
            outcome = await boundary.detect(ingested, mode="chat", is_final=False)

    assert outcome.status == "skipped"
    loaded = [e for e in events if e["phase"] == "buffer_loaded"]
    assert len(loaded) == 1
    assert loaded[0]["before_message_ids"] == [held.message_id]
    assert loaded[0]["message_ids"] == [held.message_id]
