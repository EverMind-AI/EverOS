"""Pin actual revisions, request isolation, construction evidence and QA inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from smartcomment.runtime import ExecNetwork

from everos.infra.memtrace.qa import inject_qa_dir
from everos.infra.memtrace.tracker import MemTraceTracker, resolve_query_node
from everos.memory.search.trace import collect_search_trace, record_stage

from .test_memtrace_tracker import (
    _EPISODE_MD,
    _hit_node,
    _load_export,
    _req,
    _resp,
)


@pytest.fixture(autouse=True)
def _memtrace_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tracing is opt-in; these tests exercise the enabled tracker."""
    monkeypatch.setenv("EVEROS_MEMTRACE_ENABLED", "1")


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> MemTraceTracker:
    monkeypatch.setenv("EVEROS_MEMTRACE_DIR", str(tmp_path / "memtrace"))
    return MemTraceTracker()


def _path_exists(payload: dict, source: str, target: str) -> bool:
    adjacency: dict[str, list[str]] = {}
    for edge in payload["data"]["edges"]:
        adjacency.setdefault(edge["source_full_node_id"], []).append(
            edge["target_full_node_id"]
        )
    pending, visited = [source], {source}
    for node in pending:
        for child in adjacency.get(node, []):
            if child not in visited:
                visited.add(child)
                pending.append(child)
    return target in visited


async def test_updated_revision_is_not_renumbered_in_search_graph(
    tracker: MemTraceTracker,
    tmp_path: Path,
) -> None:
    from examples.memtrace.merge import merge_graphs

    path = tmp_path / "users/alice/episodes/2026-01-01.md"
    path.parent.mkdir(parents=True)
    path.write_text(_EPISODE_MD.replace("Alice went camping in July.", "Friday"))
    await tracker._sync_files()
    path.write_text(_EPISODE_MD.replace("Alice went camping in July.", "Monday"))
    await tracker._sync_files()
    response = _resp("ep_1")
    response.data.episodes[0].episode = "Monday"
    await tracker.on_search(_req(), response, request_id="after-update")
    search = _load_export(tracker)
    hit = _hit_node(search, "alice_ep_1")
    assert json.loads(hit["value"])["source_revision"] == "alice_ep_1@2"
    canonical = [n for n in search["data"]["nodes"] if n["name"] == "alice_ep_1"]
    assert [n["version"] for n in canonical] == [2]
    assert json.loads(canonical[0]["value"])["content"] == "Monday"
    construction = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    merged = merge_graphs([construction, search])
    ExecNetwork.import_graph(merged)
    memories = [n for n in merged["data"]["nodes"] if n["name"].endswith("/alice_ep_1")]
    assert sorted(n["version"] for n in memories) == [1, 2]
    merged_hit = _hit_node(merged, "alice_ep_1")
    revision = json.loads(merged_hit["value"])["source_revision"]
    assert revision in {n["full_node_id"] for n in memories}
    assert revision.endswith("@2")


async def test_disk_state_does_not_replace_actual_stale_search_text(
    tracker: MemTraceTracker,
    tmp_path: Path,
) -> None:
    path = tmp_path / "users/alice/episodes/2026-01-01.md"
    path.parent.mkdir(parents=True)
    path.write_text(_EPISODE_MD.replace("Alice went camping in July.", "Monday"))
    response = _resp("ep_1")
    response.data.episodes[0].episode = "Friday"
    await tracker.on_search(_req(), response)
    hit = _hit_node(_load_export(tracker), "alice_ep_1")
    snapshot = json.loads(hit["value"])
    assert snapshot["content"] == "Friday"
    assert snapshot["source_revision"] is None
    assert hit["metadata"]["revision_resolved"] is False


async def test_shared_memory_does_not_connect_queries_to_other_contexts(
    tracker: MemTraceTracker,
    tmp_path: Path,
) -> None:
    path = tmp_path / "users/alice/episodes/2026-01-01.md"
    path.parent.mkdir(parents=True)
    path.write_text(_EPISODE_MD.replace("Alice went camping in July.", "body ep_1"))
    tracker._root = tmp_path
    await tracker.on_search(_req(), _resp("ep_1"), request_id="first")
    await tracker.on_search(_req(), _resp("ep_1"), request_id="second")
    payload = _load_export(tracker)
    assert _hit_node(payload, "alice_ep_1", "query-alice-first")["metadata"][
        "revision_resolved"
    ]
    assert _hit_node(payload, "alice_ep_1", "query-alice-second")["metadata"][
        "revision_resolved"
    ]
    assert not _path_exists(
        payload, "query:query-alice-second@1", "context:context-query-alice-first@1"
    )
    assert _path_exists(
        payload, "query:query-alice-second@1", "context:context-query-alice-second@1"
    )


async def test_skill_bridge_inputs_match_actual_computation(
    tracker: MemTraceTracker,
) -> None:
    from types import SimpleNamespace

    candidates = [SimpleNamespace(id="skill", score=0.8, source="dense")]
    with collect_search_trace() as stages:
        record_stage("fusion", "agent_case", candidates)
        record_stage("sparse", "agent_skill", candidates)
        record_stage("dense", "agent_skill", candidates)
        record_stage(
            "bridged_dense",
            "agent_skill",
            candidates,
            inputs=[("agent_skill", "dense"), ("agent_case", "fusion")],
        )
        record_stage(
            "fusion",
            "agent_skill",
            candidates,
            inputs=[("agent_skill", "sparse"), ("agent_skill", "bridged_dense")],
        )
    await tracker.on_search(_req(), _resp(), stages=stages, request_id="bridge")
    graph = ExecNetwork.import_graph(_load_export(tracker))
    for phase, expected in (
        ("bridged_dense", {"agent_skill_dense", "agent_case_fusion"}),
        ("fusion", {"agent_skill_sparse", "agent_skill_bridged_dense"}),
    ):
        op = graph.search_operations(name_pattern=f"^agent_skill_{phase}_")[0]
        inputs = {
            e.source_full_node_id for e in graph.filter_by_operation(op.op_id).edges
        }
        assert len(inputs) == 2
        assert all(
            any(f"candidates_{name}-" in node for node in inputs) for name in expected
        )


async def test_restart_query_ids_and_explicit_qa_resolution(
    tracker: MemTraceTracker,
) -> None:
    await tracker.on_search(_req(), _resp(), request_id="repeat")
    restarted = MemTraceTracker()
    await restarted.on_search(_req(), _resp(), request_id="repeat")
    graph = restarted._graphs[("default", "default", "alice", "global")]
    assert len(graph.get_versions("query:query-alice-repeat")) == 1
    assert resolve_query_node(graph, owner="alice", request_id="repeat") is None
    queries = graph.search_variables(class_name="query")
    assert len(queries) == 2
    for query in queries:
        resolved = resolve_query_node(
            graph,
            owner="alice",
            request_id="repeat",
            query_full_node_id=query.full_node_id,
        )
        assert resolved is not None and resolved[1].full_node_id == query.full_node_id


async def test_failed_add_messages_reach_the_error_operation(
    tracker: MemTraceTracker,
) -> None:
    payload = {
        "session_id": "s",
        "messages": [
            {
                "sender_id": "alice",
                "content": "Friday",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ],
    }
    await tracker.on_memorize_error(
        payload,
        is_final=False,
        defer_extraction=False,
        error=ValueError("invented"),
        request_id="failed",
    )
    graph = tracker._graphs[("default", "default", "alice", "s")]
    message = graph.search_variables(category="message")[0]
    op = graph.search_operations(name_pattern="^memory_add_")[0]
    subgraph = graph.filter_by_operation(op.op_id)
    assert message.full_node_id in {e.source_full_node_id for e in subgraph.edges}
    assert op.metadata["status"] == "error"
    assert message.metadata["acceptance"] == "attempted"


async def test_actual_construction_values_exist_without_markdown_outputs(
    tracker: MemTraceTracker,
) -> None:
    common = {
        "app_id": "default",
        "project_id": "default",
        "session_id": "s",
        "observed_at": "2026-01-01T00:00:00Z",
    }
    events = [
        {
            **common,
            "phase": "boundary",
            "cells": [
                {
                    "id": "mc",
                    "message_ids": [],
                    "sender_ids": ["alice"],
                    "content": {"messages": ["Friday"]},
                }
            ],
        },
        {
            **common,
            "phase": "extraction_result",
            "memcell_id": "mc",
            "owners": ["alice"],
            "content": "",
        },
    ]
    await tracker.on_memorize(
        {"session_id": "s", "messages": []},
        is_final=False,
        defer_extraction=False,
        result=None,
        construction_events=events,
    )
    graph = tracker._graphs[("default", "default", "alice", "s")]
    result = graph.search_variables(category="extraction_result")[0]
    assert json.loads(result.value)["content"] == ""
    op = graph.search_operations(name_pattern="^extraction_result_")[0]
    assert op.metadata["source"] == "construction_event"
    assert not op.metadata.get("observation_only")
    assert "Friday" in graph.search_variables(category="construction_input")[0].value


async def test_qa_uses_actual_answer_context_and_injection_is_idempotent(
    tracker: MemTraceTracker,
    tmp_path: Path,
) -> None:
    await tracker.on_search(_req(), _resp("ep_1"), request_id="qa")
    record = {
        "user_id": "alice",
        "request_id": "qa",
        "answer_context": "episode only",
        "answer": "Unknown",
        "golden_answers": ["Friday"],
        "judge_response": "VERDICT: 0",
        "judge_score": 0,
    }
    inject_qa_dir(tmp_path / "memtrace", [record])
    payload = _load_export(tracker)
    inject_qa_dir(tmp_path / "memtrace", [record])
    second = _load_export(tracker)
    assert len(payload["data"]["operations"]) == len(second["data"]["operations"])
    graph = ExecNetwork.import_graph(second)
    answer = graph.search_operations(category="response")[0]
    nodes = graph.filter_by_operation(answer.op_id).nodes
    context = next(n for n in nodes if n.class_name == "answer_context")
    assert json.loads(context.value)["content"] == "episode only"
    assert not any(n.class_name == "context" for n in nodes)


@pytest.mark.parametrize("answer_only", [False, True])
async def test_qa_injection_is_idempotent_for_partial_records(
    tracker: MemTraceTracker, answer_only: bool
) -> None:
    await tracker.on_search(_req(), _resp(), request_id="once")
    record = {"user_id": "alice", "request_id": "once"}
    if answer_only:
        record.update(answer="Friday", answer_context="Friday")
    else:
        record.update(golden_answers=["Friday"])
    assert sum(inject_qa_dir(tracker.out_dir, [record]).values()) == 1
    path = next(tracker.out_dir.glob("*.json"))
    before = path.read_bytes()
    assert inject_qa_dir(tracker.out_dir, [record]) == {}
    assert path.read_bytes() == before


async def test_qa_injection_requires_unique_tenant_scope(
    tracker: MemTraceTracker,
) -> None:
    for app in ("a", "b"):
        await tracker.on_search(
            _req().model_copy(update={"app_id": app}), _resp(), request_id="same"
        )
    record = {"user_id": "alice", "request_id": "same", "answer": "Friday"}
    assert inject_qa_dir(tracker.out_dir, [record]) == {}
    assert (
        sum(inject_qa_dir(tracker.out_dir, [{**record, "app_id": "b"}]).values()) == 1
    )
    for path in tracker.out_dir.glob("*.json"):
        graph = ExecNetwork.import_graph(json.loads(path.read_text()))
        assert bool(graph.search_variables(class_name="prediction")) == (
            graph.metadata["app_id"] == "b"
        )


@pytest.mark.parametrize("stage", ["apply", "export"])
async def test_failed_scan_application_retries_unchanged_files(
    tracker: MemTraceTracker,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    path = tmp_path / "users/alice/episodes/2026-01-01.md"
    path.parent.mkdir(parents=True)
    path.write_text(_EPISODE_MD)
    tracker._root = tmp_path
    name = "_apply_snapshot" if stage == "apply" else "_export"
    original = getattr(tracker, name)
    calls = 0

    def fail_once(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(tracker, name, fail_once)
    with pytest.raises(RuntimeError, match="injected failure"):
        await tracker._sync_files()
    assert not tracker._file_state
    await tracker._sync_files()
    data = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    revisions = [n for n in data["data"]["nodes"] if n["name"] == "alice_ep_1"]
    assert len(revisions) == 1
    assert json.loads(revisions[0]["value"])["content"] == "Alice went camping in July."


@pytest.mark.parametrize("rewritten_before_scan", [False, True])
async def test_persisted_episode_links_only_matching_actual_extractor_output(
    tracker: MemTraceTracker, tmp_path: Path, rewritten_before_scan: bool
) -> None:
    path = tmp_path / "users/alice/episodes/2026-01-01.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        _EPISODE_MD.replace("Alice went camping in July.", "Monday")
        if rewritten_before_scan
        else _EPISODE_MD
    )
    tracker._root = tmp_path
    common = {
        "app_id": "default",
        "project_id": "default",
        "session_id": "sess_a",
        "observed_at": "2026-01-01T00:00:00Z",
    }
    events = [
        {
            **common,
            "phase": "boundary",
            "cells": [
                {
                    "id": "mc_1",
                    "message_ids": [],
                    "sender_ids": ["alice"],
                    "content": "Camping in July",
                }
            ],
        },
        {
            **common,
            "phase": "extraction_result",
            "memcell_id": "mc_1",
            "owners": ["alice"],
            "content": "Alice went camping in July.",
        },
        {
            **common,
            "phase": "persisted_episode",
            "memcell_id": "mc_1",
            "owners": ["alice"],
            "entry_id": "ep_1",
        },
    ]
    await tracker.on_memorize(
        {"session_id": "sess_a", "messages": []},
        is_final=True,
        defer_extraction=False,
        result=None,
        construction_events=events,
        request_id="add",
    )
    payload = _load_export(tracker, ("default", "default", "alice", "sess_a"))
    graph = ExecNetwork.import_graph(payload)
    extracted = graph.search_variables(category="extraction_result")[0]
    assert _path_exists(payload, extracted.full_node_id, "alice_ep_1@1") is (
        not rewritten_before_scan
    )
    op = graph.search_operations(name_pattern="^episode_extraction_ep_1$")[0]
    assert op.metadata["source"] == (
        "file_diff" if rewritten_before_scan else "construction_event"
    )
    assert bool(op.metadata.get("observation_only")) is rewritten_before_scan
