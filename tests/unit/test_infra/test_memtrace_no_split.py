"""Keep deferred and empty boundary outcomes connected across repeated adds."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from smartcomment.runtime import ExecNetwork

from everos.infra.memtrace.tracker import MemTraceTracker
from everos.memory.extract.boundary import BoundaryOutcome
from everos.memory.extract.trace import collect_construction_trace

from .test_memtrace_tracker import _TS_MS, _load_export, _message, _payload


@pytest.fixture(autouse=True)
def _memtrace_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing is opt-in; these tests exercise the enabled tracker."""
    monkeypatch.setenv("EVEROS_MEMTRACE_ENABLED", "1")


mm = importlib.import_module("everos.service.memory._memorize")


@pytest.mark.parametrize("deferred", [False, True])
@pytest.mark.parametrize("restart", [False, True])
async def test_each_add_without_cells_has_a_connected_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    deferred: bool,
    restart: bool,
) -> None:
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    tracker = MemTraceTracker()
    ingested = SimpleNamespace(
        app_id="default", project_id="default", session_id="sess_a"
    )
    monkeypatch.setattr(mm, "ingest_process", AsyncMock(return_value=ingested))
    empty = BoundaryOutcome([], [], [], [], "accumulated", 1)
    buffer = AsyncMock(return_value=empty)
    detect = AsyncMock(return_value=empty)
    memorizer = mm.Memorizer.__new__(mm.Memorizer)
    memorizer._boundary = SimpleNamespace(buffer=buffer, detect=detect)
    memorizer._multimodal_llm = None
    memorizer._raw_file_builder = None
    memorizer._sender_names = None

    for index in range(2):
        if restart and index:
            tracker = MemTraceTracker()
        payload = _payload(_message(f"Conversation part {index}", _TS_MS + index))
        with collect_construction_trace() as events:
            result = await memorizer._cut(
                payload,
                mode="chat",
                is_final=False,
                defer_extraction=deferred,
            )
        assert len(events) == 1 and events[0]["phase"] == "boundary"
        await tracker.on_memorize(
            payload,
            is_final=False,
            defer_extraction=deferred,
            result=result,
            request_id=f"add-{index}",
            construction_events=events,
        )
    graph = ExecNetwork.import_graph(
        _load_export(tracker, ("default", "default", "alice", "sess_a"))
    )
    outcomes = graph.search_variables(class_name="boundary_detection_result")
    assert len(outcomes) == 2
    assert len(graph.search_variables(class_name="no_split_marker")) == 2
    assert not graph.search_variables(category="construction_input")
    for index, outcome in enumerate(outcomes):
        value = json.loads(outcome.raw_value)
        assert value["cell_ids"] == []
        assert value["decision"] == ("deferred" if deferred else "no_split")
        assert len(value["message_ids"]) == index + 1
        if deferred:
            assert value["detection_performed"] is False
        edges = graph.get_all_edges()
        incoming = [e for e in edges if e.target_full_node_id == outcome.full_node_id]
        assert len(incoming) == 1
        assert incoming[0].source_full_node_id == f"buffer-sess_a-alice@{index + 2}"
        outgoing = [e for e in edges if e.source_full_node_id == outcome.full_node_id]
        assert len(outgoing) == 1
        assert graph.get_variable(outgoing[0].target_full_node_id).class_name == (
            "no_split_marker"
        )
    assert (buffer.await_count, detect.await_count) == ((2, 0) if deferred else (0, 2))
