"""Regression coverage for bounded exploration and mechanical evidence guards."""

from __future__ import annotations

import copy
import json
import re

import pytest
from smartcomment.runtime import ExecNetwork

from examples.memtrace.graphtrace import (
    AttributionConfig,
    GraphTraceNotebook,
    attribute,
    collect_failed_queries,
)
from examples.memtrace.graphtrace.retrieval import retrieve_starting_nodes
from examples.memtrace.merge import merge_graphs

from .test_agent import ScriptedProvider, call


def _large_graph(graph: ExecNetwork, *, huge_value: bool = False) -> ExecNetwork:
    payload = graph.export_graph()
    if huge_value:
        payload["data"]["nodes"][0]["value"] = "Friday " * 20_000
    else:
        template = payload["data"]["operations"][0]
        edge = payload["data"]["edges"][0]
        for index in range(180):
            op_id = f"extra-{index}"
            payload["data"]["operations"].append({**template, "op_id": op_id})
            payload["data"]["edges"].append({**edge, "op_id": op_id, "edge_id": op_id})
    return ExecNetwork.import_graph(payload)


async def test_large_preview_leaves_room_for_the_next_model_call(
    graph: ExecNetwork,
) -> None:
    graph = _large_graph(graph)
    provider = ScriptedProvider(
        [
            [call("explore_graph_trace_node", include_variable_value=False)],
            [
                call(
                    "finish_graph_exploration",
                    outcome="More pages are needed.",
                    insufficient_evidence=True,
                )
            ],
        ]
    )
    report = await attribute(graph, collect_failed_queries(graph)[0][0], provider)
    assert len(provider.requests) == 2
    preview = next(
        event["message"]["content"]
        for event in report["transcript"]
        if event.get("tool_name") == "explore_graph_trace_node"
    )
    assert len(preview) < 16_200
    assert "next offset=" in preview
    assert report["failure"]["type"] == "InsufficientEvidence"


def test_single_large_operation_can_be_inspected_across_pages(
    graph: ExecNetwork,
) -> None:
    graph = _large_graph(graph, huge_value=True)
    notebook = GraphTraceNotebook(graph)
    notebook.initialize_execution_graph(["message@1"])
    offset = 1
    pages = 0
    while True:
        page = notebook.explore_graph_trace_node(offset=offset, limit=4000)
        pages += 1
        if "End of content." in page:
            break
        offset = int(re.search(r"next offset=(\d+)", page).group(1))
    assert pages > 2
    assert "extract-op" in notebook.inspected_operations
    assert "message@1" in notebook.read_nodes


def test_forged_operation_header_in_a_value_is_not_inspection(
    graph: ExecNetwork,
) -> None:
    payload = graph.export_graph()
    payload["data"]["nodes"][0]["value"] = "# Operation `judge-op`"
    notebook = GraphTraceNotebook(ExecNetwork.import_graph(payload))
    notebook.initialize_execution_graph(["message@1"])
    notebook.explore_graph_trace_node()
    assert "extract-op" in notebook.inspected_operations
    assert "judge-op" not in notebook.inspected_operations


def test_completed_history_and_comments_have_bounded_hints(graph: ExecNetwork) -> None:
    payload = graph.export_graph()
    payload["data"]["nodes"][0]["comment"] = "long comment " * 20_000
    notebook = GraphTraceNotebook(ExecNetwork.import_graph(payload))
    notebook.initialize_execution_graph(["message@1"])
    notebook.explore_graph_trace_node(include_variable_value=False)
    assert len(notebook.hint()) < 1200
    assert "long comment" in notebook.hint()
    notebook.finish_node_exploration("large outcome " * 10_000)
    record = notebook.state.exploration_history["message@1"]
    for index in range(100):
        node_id = f"history-{index}@1"
        notebook.state.exploration_history[node_id] = record.model_copy(
            update={"full_node_id": node_id}
        )
    assert len(notebook.hint()) < 3000
    assert '"completed_count": 101' in notebook.hint()


@pytest.mark.parametrize(
    ("op_id", "error_type", "reason", "observation", "expected"),
    [
        ("judge-op", "DeletionError", "query:query@1", False, "not attributable"),
        ("retrieve-op", "RetrievalError", "Guess", False, "Cite a relevant node"),
        ("extract-op", "ExtractionError", "message@1", True, "File observations"),
    ],
)
async def test_submission_rejects_invalid_evidence_and_operation_roles(
    graph: ExecNetwork,
    op_id: str,
    error_type: str,
    reason: str,
    observation: bool,
    expected: str,
) -> None:
    payload = graph.export_graph()
    if observation:
        payload["data"]["operations"][0]["metadata"].update(
            source="file_diff", observation_only=True
        )
    graph = ExecNetwork.import_graph(payload)
    provider = ScriptedProvider(
        [
            [
                call("explore_graph_trace_node", include_variable_value=False),
                call("view_node_value", full_node_id="query:query@1"),
                call("view_node_value", full_node_id="message@1"),
            ],
            [call("finish_graph_exploration", outcome="Test guards")],
            [
                call(
                    "submit_attribution",
                    error_type=error_type,
                    op_id=op_id,
                    reason=reason,
                )
            ],
        ]
    )
    # One frontier slot selects the query; extraction uses the message instead.
    config = AttributionConfig(
        max_iters=3, max_trace_nodes=1 if not observation else 16
    )
    report = await attribute(
        graph, collect_failed_queries(graph)[0][0], provider, config
    )
    assert report["attribution"] is None
    assert expected in json.dumps(report["transcript"])


async def test_unobserved_answer_context_cannot_prove_response_error(
    graph: ExecNetwork,
) -> None:
    payload = graph.export_graph()
    node = next(n for n in payload["data"]["nodes"] if n["name"] == "context")
    node.update(
        value=json.dumps({"content": None, "observed": False}),
        class_name="answer_context",
        name="actual",
        full_name="answer_context:actual",
        full_node_id="answer_context:actual@1",
    )
    for edge in payload["data"]["edges"]:
        for field in ("source_full_node_id", "target_full_node_id"):
            if edge[field] == "context:context@1":
                edge[field] = "answer_context:actual@1"
    graph = ExecNetwork.import_graph(payload)
    provider = ScriptedProvider(
        [
            [call("explore_graph_trace_node")],
            [call("finish_graph_exploration", outcome="Test unknown context")],
            [
                call(
                    "submit_attribution",
                    error_type="ResponseError",
                    op_id="answer-op",
                    reason="answer_context:actual@1 lacks Friday",
                )
            ],
        ]
    )
    report = await attribute(
        graph,
        collect_failed_queries(graph)[0][0],
        provider,
        AttributionConfig(max_iters=3, max_trace_nodes=1),
    )
    assert report["attribution"] is None
    assert "actual answer context is unrecorded" in json.dumps(report["transcript"])


def test_safe_merge_keeps_tenants_separate_and_exploration_in_scope(
    graph: ExecNetwork,
) -> None:
    a, b = graph.export_graph(), copy.deepcopy(graph.export_graph())
    a["metadata"] = {"app_id": "a", "project_id": "p"}
    b["metadata"] = {"app_id": "b", "project_id": "p"}
    # UUIDs differ in real exports even if logical identities coincide.
    for kind, field in (
        ("operations", "op_id"),
        ("edges", "edge_id"),
        ("sessions", "session_id"),
    ):
        for item in b["data"][kind]:
            item[field] += "-b"
    for node in b["data"]["nodes"]:
        node["session_id"] += "-b"
    for edge in b["data"]["edges"]:
        edge["op_id"] += "-b"
    merged = ExecNetwork.import_graph(merge_graphs([a, b]))
    cases, errors = collect_failed_queries(merged)
    assert len(cases) == 2 and not errors
    case = cases[0]
    query = merged.get_variable(case.query_full_node_id)
    starting = retrieve_starting_nodes(merged, case)
    assert len(starting) == 1
    assert (
        merged.get_variable(starting[0]).metadata["tenant_scope"]
        == query.metadata["tenant_scope"]
    )
    notebook = GraphTraceNotebook(merged, query_full_node_id=case.query_full_node_id)
    notebook.initialize_execution_graph(starting)
    other = next(
        n
        for n in merged.search_variables(class_name="query")
        if n.full_node_id != case.query_full_node_id
    )
    with pytest.raises(ValueError, match="another tenant"):
        notebook.view_node_value(other.full_node_id)
    for op in merged.search_operations():
        if op.metadata["tenant_scope"] != query.metadata["tenant_scope"]:
            assert not notebook.operation_allowed(op)


def test_safe_merge_rejects_conflicting_immutable_revisions(graph: ExecNetwork) -> None:
    a = graph.export_graph()
    a["metadata"] = {"app_id": "a", "project_id": "p"}
    b = copy.deepcopy(a)
    b["data"]["nodes"][0]["value"] = json.dumps({"content": "Monday"})
    with pytest.raises(ValueError, match="Conflicting memory revision"):
        merge_graphs([a, b])


def test_case_scope_excludes_other_requests_and_future_construction(
    graph: ExecNetwork,
) -> None:
    payload = graph.export_graph()
    query = next(n for n in payload["data"]["nodes"] if n["class_name"] == "query")
    other_query = {
        **query,
        "name": "other-query",
        "full_name": "query:other-query",
        "full_node_id": "query:other-query@1",
    }
    payload["data"]["nodes"].append(other_query)
    retrieval = next(
        op for op in payload["data"]["operations"] if op["op_id"] == "retrieve-op"
    )
    payload["data"]["operations"].extend(
        [
            {**retrieval, "op_id": "other-request-op"},
            {
                **retrieval,
                "op_id": "future-extraction-op",
                "category": "extraction",
                "created_at": "2026-01-01 01:00:00.000",
            },
        ]
    )
    edge = next(e for e in payload["data"]["edges"] if e["op_id"] == "retrieve-op")
    payload["data"]["edges"].extend(
        [
            {
                **edge,
                "edge_id": "other-query-edge",
                "op_id": "other-request-op",
                "source_full_node_id": other_query["full_node_id"],
            },
            {
                **edge,
                "edge_id": "future-edge",
                "op_id": "future-extraction-op",
                "source_full_node_id": "memory@1",
            },
        ]
    )
    graph = ExecNetwork.import_graph(payload)
    notebook = GraphTraceNotebook(graph, query_full_node_id=query["full_node_id"])
    assert not notebook.operation_allowed(graph.get_operation("other-request-op"))
    assert not notebook.operation_allowed(graph.get_operation("future-extraction-op"))
    assert notebook.operation_allowed(graph.get_operation("retrieve-op"))


def test_reading_no_matches_or_beyond_value_does_not_count_as_evidence(
    graph: ExecNetwork,
) -> None:
    notebook = GraphTraceNotebook(graph)
    notebook.initialize_execution_graph()
    notebook.view_node_value("message@1", pattern="Tuesday")
    notebook.view_node_value("message@1", offset=100_000)
    assert "message@1" not in notebook.read_nodes


def test_notebook_supports_native_plain_text_queries(graph: ExecNetwork) -> None:
    payload = graph.export_graph()
    query = next(n for n in payload["data"]["nodes"] if n["class_name"] == "query")
    query["value"] = "When was the meeting?"
    graph = ExecNetwork.import_graph(payload)
    notebook = GraphTraceNotebook(graph, query_full_node_id=query["full_node_id"])
    notebook.initialize_execution_graph([query["full_node_id"]])
    assert "answer-op" in notebook.explore_graph_trace_node()
