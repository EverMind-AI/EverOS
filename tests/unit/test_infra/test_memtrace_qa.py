"""Unit tests for post-hoc QA injection into exported MemTrace graphs.

Pins :func:`everos.infra.memtrace.qa.inject_qa_dir`: QA records are stitched
onto the recorded query node with the same identity / category / operation
vocabulary as the live tracker hooks (``prompt`` / ``prediction`` /
``golden_answers`` / ``judge_response`` nodes, ``response`` / ``evaluation``
operations), the exported graph round-trips through ``ExecNetwork``, and
records whose query node cannot be resolved are skipped without failing the
batch.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any

import pytest
from smartcomment.runtime import ExecNetwork

from everos.infra.memtrace.qa import inject_qa_dir
from everos.infra.memtrace.tracker import MemTraceTracker
from everos.memory.search.dto import (
    SearchData,
    SearchEpisodeItem,
    SearchMethod,
    SearchRequest,
    SearchResponse,
)


@pytest.fixture(autouse=True)
def _memtrace_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing is opt-in; these tests exercise the enabled tracker."""
    monkeypatch.setenv("EVEROS_MEMTRACE_ENABLED", "1")


_TS = _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MemTraceTracker:
    """A tracker exporting into an isolated per-test directory."""
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    return MemTraceTracker()


def _req() -> SearchRequest:
    return SearchRequest(
        user_id="alice",
        query="camping",
        method=SearchMethod.HYBRID,
        top_k=5,
    )


def _resp() -> SearchResponse:
    return SearchResponse(
        request_id="r" * 32,
        data=SearchData(
            episodes=[
                SearchEpisodeItem(
                    id="ep_1",
                    user_id="alice",
                    session_id="sess_a",
                    timestamp=_TS,
                    sender_ids=["alice"],
                    subject="s",
                    summary="s",
                    episode="body ep_1",
                    type="Conversation",
                    score=0.9,
                )
            ],
        ),
    )


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "user_id": "alice",
        "request_id": "req-1",
        "query": "camping",
        "prompt": "answer prompt text",
        "answer": "She went camping in July.",
        "golden_answers": ["July"],
        "judge_response": "VERDICT: 1 correct",
        "judge_score": 1.0,
    }
    record.update(overrides)
    return record


async def test_inject_qa_round_trip(tracker: MemTraceTracker, tmp_path: Path) -> None:
    """Injection adds the full QA chain and the graph still imports cleanly."""
    await tracker.on_search(_req(), _resp(), request_id="req-1")
    graph_dir = tmp_path / "memtrace"

    injected = inject_qa_dir(graph_dir, [_record()])

    assert sum(injected.values()) == 1
    path = next(graph_dir.glob("*.json"))
    data = json.loads(path.read_text(encoding="utf-8"))
    nodes = {node["name"]: node for node in data["data"]["nodes"]}
    query_id = "query-alice-req-1"
    assert nodes[f"prompt-{query_id}"]["category"] == "prompt"
    prediction = nodes[f"prediction-{query_id}"]
    assert prediction["category"] == "llm_response"
    assert prediction["class_name"] == "prediction"
    golden = nodes[f"golden-{query_id}"]
    assert golden["category"] == "golden_answers"
    assert nodes[f"judge-{query_id}"]["class_name"] == "judge_response"

    ops = {op["op_name"]: op for op in data["data"]["operations"]}
    assert ops[f"answer_generation_{query_id}"]["category"] == "response"
    assert ops[f"inject_golden_answers_{query_id}"]["category"] == "evaluation"
    assert ops[f"judge_answer_{query_id}"]["category"] == "evaluation"
    assert ops[f"answer_generation_{query_id}"]["metadata"]["source"] == "qa_injection"

    # The injected graph still round-trips through the smartcomment importer.
    ExecNetwork.import_graph(data)


async def test_inject_qa_skips_unknown_request(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """A record with no matching query node is skipped, not an error."""
    await tracker.on_search(_req(), _resp(), request_id="req-1")
    graph_dir = tmp_path / "memtrace"

    injected = inject_qa_dir(graph_dir, [_record(request_id="req-nope")])

    assert injected == {}
    data = json.loads(next(graph_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert not any(
        node["name"].startswith(("prediction-", "golden-", "judge-", "prompt-"))
        for node in data["data"]["nodes"]
    )


async def test_inject_qa_without_golden_records_answer_only(
    tracker: MemTraceTracker, tmp_path: Path
) -> None:
    """Without golden answers only the answer stage is injected."""
    await tracker.on_search(_req(), _resp(), request_id="req-1")
    graph_dir = tmp_path / "memtrace"

    injected = inject_qa_dir(
        graph_dir,
        [_record(golden_answers=[], judge_response=None, judge_score=None)],
    )

    assert sum(injected.values()) == 1
    data = json.loads(next(graph_dir.glob("*.json")).read_text(encoding="utf-8"))
    names = {node["name"] for node in data["data"]["nodes"]}
    query_id = "query-alice-req-1"
    assert f"prediction-{query_id}" in names
    assert f"golden-{query_id}" not in names
    assert f"judge-{query_id}" not in names
    ops = {op["op_name"] for op in data["data"]["operations"]}
    assert f"answer_generation_{query_id}" in ops
    assert f"inject_golden_answers_{query_id}" not in ops
