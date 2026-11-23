"""Pin graph-tool state, QA decoding and source-candidate ranking contracts."""

from __future__ import annotations

import json

import pytest
from smartcomment.runtime import ExecNetwork

from examples.memtrace.graphtrace import (
    AttributionResult,
    GraphTraceNotebook,
    GraphTraceState,
    collect_failed_queries,
)
from examples.memtrace.graphtrace.models import GraphTraceNodeRecord
from examples.memtrace.graphtrace.notebook import DUMMY_ROOT, paginate
from examples.memtrace.graphtrace.retrieval import bm25_scores, retrieve_starting_nodes

from .conftest import make_graph


def test_qa_decodes_stringified_reference_list(graph: ExecNetwork) -> None:
    cases, diagnostics = collect_failed_queries(graph)
    assert diagnostics == []
    assert len(cases) == 1
    assert cases[0].query_full_node_id == "query:query@1"
    assert cases[0].golden_answers == ["Friday"]
    assert cases[0].prediction == "Unknown"


def test_native_judge_score_node_is_supported(graph: ExecNetwork) -> None:
    payload = graph.export_graph()
    for op in payload["data"]["operations"]:
        if op["op_id"] == "judge-op":
            op["op_name"] = "llm-judge"
    judge = next(
        node
        for node in payload["data"]["nodes"]
        if node["class_name"] == "judge_response"
    )
    judge["value"] = "VERDICT: 0"
    score = {**judge, "name": "score", "class_name": "judge_score", "value": "0.0"}
    payload["data"]["nodes"].append(score)
    edge = next(
        edge for edge in payload["data"]["edges"] if edge["op_id"] == "judge-op"
    )
    payload["data"]["edges"].append(
        {
            **edge,
            "edge_id": "judge-score-edge",
            "target_full_node_id": "judge_score:score@1",
        }
    )
    cases, diagnostics = collect_failed_queries(ExecNetwork.import_graph(payload))
    assert diagnostics == []
    assert cases[0].judge_response == "VERDICT: 0"


@pytest.mark.parametrize("score", [1, "1.0", 0.5])
def test_successful_or_partial_judges_are_excluded(score: object) -> None:
    assert collect_failed_queries(make_graph(score=score)) == ([], [])


@pytest.mark.parametrize(
    "overrides",
    [
        {"score": "nan"},
        {"score": None},
        {"score": "bad"},
        {"omit_prediction": True},
        {"answers": []},
        {"answers": [None]},
        {"mismatch": True},
    ],
)
def test_malformed_qa_chains_have_diagnostics(overrides: dict) -> None:
    cases, diagnostics = collect_failed_queries(make_graph(**overrides))
    assert cases == []
    assert len(diagnostics) == 1
    assert "judge-op" in diagnostics[0]


def test_frontier_order_active_stability_and_capacity() -> None:
    state = GraphTraceState()
    for node_id, time in [("b@1", "02"), ("a@1", "02")]:
        state.add(GraphTraceNodeRecord(full_node_id=node_id, created_at=time), 2)
    assert state.start_next().full_node_id == "a@1"
    state.add(GraphTraceNodeRecord(full_node_id="older@1", created_at="01"), 2)
    assert state.start_next().full_node_id == "a@1"
    with pytest.raises(ValueError, match="full"):
        state.add(GraphTraceNodeRecord(full_node_id="extra@1", created_at="03"), 2)
    with pytest.raises(ValueError, match="Only a queued"):
        state.delete("a@1")
    state.finish_node("locally correct")
    assert state.start_next().full_node_id == "older@1"
    with pytest.raises(ValueError, match="already"):
        state.add(GraphTraceNodeRecord(full_node_id="a@1", created_at="02"), 2)
    assert state.exploration_history["a@1"].outcome == "locally correct"


def test_initialize_is_atomic_and_restarts_without_mutating_graph(
    graph: ExecNetwork,
) -> None:
    original = graph.export_graph()
    notebook = GraphTraceNotebook(graph, max_trace_nodes=1)
    with pytest.raises(ValueError, match="full"):
        notebook.initialize_execution_graph(["message@1", "query:query@1"])
    assert notebook.state is None
    notebook.initialize_execution_graph()
    assert notebook.state.start_next().full_node_id == DUMMY_ROOT
    preview = notebook.explore_graph_trace_node(include_variable_value=False)
    assert "message@1" in preview and "query:query@1" in preview
    notebook.finish_graph_exploration("restart from query")
    assert notebook.final_state.outcome == "restart from query"
    notebook.initialize_execution_graph(["query:query@1"])
    notebook.explore_graph_trace_node(include_variable_value=False)
    assert notebook.state.exploring_node.full_node_id == "query:query@1"
    assert graph.export_graph() == original


def test_operation_preview_values_and_provenance(graph: ExecNetwork) -> None:
    notebook = GraphTraceNotebook(graph)
    notebook.initialize_execution_graph(["memory@1"])
    preview = notebook.explore_graph_trace_node(
        include_variable_value=False, offset=1, limit=16000
    )
    assert preview.index("# Operation `extract-op`") < preview.index(
        "# Operation `retrieve-op`"
    )
    assert "Produced variables" in preview and "Direct successors" in preview
    assert "Final output variables" in preview and "event_seq" in preview
    assert "adoption meeting Friday" not in preview
    notebook.update_to_explore_nodes("add", "context:context@1", "retrieve-op")
    record = notebook.state.to_explore_nodes["context:context@1"]
    assert record.discovered_from == "memory@1"
    assert record.discovered_by_op_id == "retrieve-op"
    with pytest.raises(ValueError, match="does not contain"):
        notebook.update_to_explore_nodes("add", "judge@1", "retrieve-op")
    with pytest.raises(ValueError):
        notebook.update_to_explore_nodes("add", "missing@1")
    assert "adoption meeting Friday" in notebook.view_node_value("memory@1")


def test_pagination_and_regex_search(graph: ExecNetwork) -> None:
    assert "abc\n" in paginate("abcdef", 1, 3)
    assert "next offset=4" in paginate("abcdef", 1, 3)
    assert "def\n" in paginate("abcdef", 4, 3)
    assert "No content" in paginate("abcdef", 100, 3)
    with pytest.raises(ValueError):
        paginate("a", 0, 1)
    notebook = GraphTraceNotebook(graph)
    notebook.initialize_execution_graph()
    result = notebook.view_node_value(
        "message@1", pattern="Friday", max_surrounding_chars=0
    )
    assert "Friday" in result and "Match at" in result
    assert "No matches" in notebook.view_node_value("message@1", pattern="Tuesday")
    with pytest.raises(ValueError, match="Invalid regex"):
        notebook.view_node_value("message@1", pattern="[")


def test_regex_pagination_resumes_at_next_match(graph: ExecNetwork) -> None:
    notebook = GraphTraceNotebook(graph)
    notebook.initialize_execution_graph()
    first = notebook.view_node_value(
        "message@1", pattern=r"[A-Za-z]+", max_surrounding_chars=0, limit=25
    )
    assert "next offset=" in first
    offset = int(first.split("next offset=")[1].split(".")[0])
    second = notebook.view_node_value(
        "message@1",
        pattern=r"[A-Za-z]+",
        max_surrounding_chars=0,
        limit=25,
        offset=offset,
    )
    assert first.splitlines()[0] != second.splitlines()[0]


def test_bm25_scores_and_candidates(graph: ExecNetwork) -> None:
    scores = bm25_scores(["adoption meeting Friday", "meeting", "cycling"], "Friday")
    assert scores[0] > scores[1] == scores[2]
    assert bm25_scores(["", ""], "Friday") == [0, 0]
    assert bm25_scores([], "Friday") == []
    case = collect_failed_queries(graph)[0][0]
    assert retrieve_starting_nodes(graph, case, 1) == ["message@1"]


def test_node_and_attribution_validation() -> None:
    with pytest.raises(ValueError):
        GraphTraceNodeRecord(full_node_id="unversioned", created_at="00")
    with pytest.raises(ValueError):
        AttributionResult(error_type="RetrievalError", op_id="op", reason="  ")


def test_duplicate_event_sequences_fall_back_to_timestamp(graph: ExecNetwork) -> None:
    payload = graph.export_graph()
    for op in payload["data"]["operations"]:
        if op["op_id"] == "extract-op":
            op["metadata"]["event_seq"] = 2
            op["created_at"] = "2026-01-01 00:00:02.000"
    notebook = GraphTraceNotebook(ExecNetwork.import_graph(payload))
    assert not notebook.unique_event_sequences
    notebook.initialize_execution_graph(["memory@1"])
    preview = notebook.explore_graph_trace_node(include_variable_value=False)
    assert preview.index("# Operation `retrieve-op`") < preview.index(
        "# Operation `extract-op`"
    )


def test_completed_history_remains_in_fresh_hints(graph: ExecNetwork) -> None:
    notebook = GraphTraceNotebook(graph)
    notebook.initialize_execution_graph(["memory@1"])
    notebook.explore_graph_trace_node(include_variable_value=False)
    notebook.finish_node_exploration("Evidence entered memory.")
    assert "Evidence entered memory." in notebook.hint()
    history = json.loads(
        notebook.view_exploration_history().split("\n\nCharacters", 1)[0]
    )
    assert history["memory@1"]["state"] == "explored"
