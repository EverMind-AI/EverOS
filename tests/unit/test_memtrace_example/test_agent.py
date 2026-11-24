"""Pin end-to-end tool calling, budget integrity, and honest failure reports."""

from __future__ import annotations

import json
from typing import Any

import pytest
from smartcomment.runtime import ExecNetwork

from examples.memtrace.attribute import load_credentials
from examples.memtrace.graphtrace import (
    AttributionConfig,
    attribute,
    collect_failed_queries,
)
from examples.memtrace.graphtrace.agent import ContextBudgetError, trim_history


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "id": "call-" + name,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


class ScriptedProvider:
    def __init__(self, batches: list[list[dict[str, Any]]]) -> None:
        self.batches = iter(batches)
        self.requests: list[list[dict[str, Any]]] = []

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.requests.append(messages)
        return {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": next(self.batches),
            },
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        }


async def test_attribution_inspects_graph_and_corrects_invalid_submission(
    graph: ExecNetwork,
) -> None:
    provider = ScriptedProvider(
        [
            [call("explore_graph_trace_node", include_variable_value=False)],
            [call("view_node_value", full_node_id="message@1", pattern="Friday")],
            [
                call(
                    "update_to_explore_nodes",
                    action="add",
                    full_node_id="memory@1",
                    discovered_by_op_id="extract-op",
                ),
                call("finish_node_exploration", outcome="Extraction preserved Friday."),
            ],
            [call("explore_graph_trace_node", include_variable_value=False)],
            [call("view_node_value", full_node_id="context:context@1")],
            [
                call(
                    "finish_graph_exploration",
                    outcome="Memory holds Friday; context omits it.",
                )
            ],
            [
                call(
                    "submit_attribution",
                    error_type="RetrievalError",
                    op_id="missing-op",
                    reason="Memory has Friday; final context does not.",
                )
            ],
            [
                call(
                    "submit_attribution",
                    error_type="RetrievalError",
                    op_id="retrieve-op",
                    reason=(
                        "memory@1 preserves Friday but context:context@1 omits it. "
                        "Extraction is correct; fixing retrieve-op supplies evidence "
                        "to response."
                    ),
                )
            ],
        ]
    )
    original = graph.export_graph()
    report = await attribute(graph, collect_failed_queries(graph)[0][0], provider)
    assert report["status"] == "success"
    assert report["attribution"]["op_id"] == "retrieve-op"
    assert report["attribution"]["error_type"] == "RetrievalError"
    assert report["iterations"] == 8
    assert report["usage"]["total_tokens"] == 72
    assert graph.export_graph() == original
    errors = [
        event["message"]["content"]
        for event in report["transcript"]
        if event.get("message", {}).get("role") == "tool"
        and event["message"]["content"].startswith("Tool error:")
    ]
    assert len(errors) == 1 and "does not exist" in errors[0]
    for request in provider.requests:
        assert sum(message["role"] == "system" for message in request) == 2
    assert "Memory holds Friday" in report["final_state"]["outcome"]


async def test_uninspected_operation_cannot_be_submitted(graph: ExecNetwork) -> None:
    provider = ScriptedProvider(
        [
            [call("finish_graph_exploration", outcome="No inspection")],
            [
                call(
                    "submit_attribution",
                    error_type="RetrievalError",
                    op_id="retrieve-op",
                    reason="Guess",
                )
            ],
        ]
    )
    report = await attribute(
        graph,
        collect_failed_queries(graph)[0][0],
        provider,
        AttributionConfig(max_iters=2),
    )
    assert report["attribution"] is None
    assert report["failure"]["type"] == "IterationLimit"
    assert "Inspect the submitted operation" in json.dumps(report["transcript"])


async def test_iteration_and_context_limits_preserve_state(graph: ExecNetwork) -> None:
    case = collect_failed_queries(graph)[0][0]
    report = await attribute(
        graph,
        case,
        ScriptedProvider(
            [
                [call("explore_graph_trace_node", include_variable_value=False)],
            ]
        ),
        AttributionConfig(max_iters=1),
    )
    assert report["failure"]["type"] == "IterationLimit"
    assert report["attribution"] is None
    assert report["final_state"]["exploring_node"]["full_node_id"] == "message@1"
    provider = ScriptedProvider([])
    report = await attribute(graph, case, provider, AttributionConfig(context_chars=1))
    assert report["failure"]["type"] == "ContextBudgetError"
    assert not provider.requests
    assert report["final_state"] is not None


async def test_missing_source_evidence_ends_without_fabricated_attribution(
    graph: ExecNetwork,
) -> None:
    report = await attribute(
        graph,
        collect_failed_queries(graph)[0][0],
        ScriptedProvider(
            [
                [call("explore_graph_trace_node", include_variable_value=False)],
                [
                    call(
                        "finish_graph_exploration",
                        outcome="Source event is not recorded.",
                        insufficient_evidence=True,
                    )
                ],
            ]
        ),
    )
    assert report["failure"] == {
        "type": "InsufficientEvidence",
        "message": "Source event is not recorded.",
    }
    assert report["attribution"] is None
    assert report["final_state"]["finished_at"]


async def test_provider_errors_never_serialize_credentials(graph: ExecNetwork) -> None:
    class FailingProvider:
        async def complete(self, messages: list, tools: list) -> dict:
            raise RuntimeError("sensitive-test-placeholder")

    report = await attribute(
        graph, collect_failed_queries(graph)[0][0], FailingProvider()
    )
    assert report["failure"]["type"] == "RuntimeError"
    assert "sensitive-test-placeholder" not in json.dumps(report)
    assert report["attribution"] is None


async def test_no_message_candidates_use_synthetic_root(graph: ExecNetwork) -> None:
    payload = graph.export_graph()
    for node in payload["data"]["nodes"]:
        if node["category"] == "message":
            node["category"] = "other"
    graph = ExecNetwork.import_graph(payload)
    report = await attribute(
        graph,
        collect_failed_queries(graph)[0][0],
        ScriptedProvider(
            [
                [call("explore_graph_trace_node", include_variable_value=False)],
            ]
        ),
        AttributionConfig(max_iters=1),
    )
    assert report["pseudo_evidence_nodes"] == []
    assert report["final_state"]["exploring_node"]["category"] == "synthetic_root"


def test_context_trimming_keeps_entire_multi_tool_batch() -> None:
    history = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "task"},
    ]
    for index, length in [(0, 3000), (1, 300)]:
        calls = [call("a"), call("b")]
        for c in calls:
            c["id"] += str(index)
        history += [
            {"role": "assistant", "content": None, "tool_calls": calls},
            *[
                {"role": "tool", "tool_call_id": c["id"], "content": "x" * length}
                for c in calls
            ],
        ]
    trimmed, removed = trim_history(history, "state", [], 1400)
    assert removed == 1
    assert [m["role"] for m in trimmed] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
    ]
    ids = {c["id"] for c in trimmed[2]["tool_calls"]}
    assert ids == {m["tool_call_id"] for m in trimmed[3:]}
    with pytest.raises(ContextBudgetError):
        trim_history(history, "state", [], 10)


def test_credentials_validate_pairs_and_environment(
    tmp_path: Any, monkeypatch: Any
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {"api_keys": ["fake-unit-key"], "base_urls": ["https://example.invalid/v1"]}
        ),
        encoding="utf-8",
    )
    assert load_credentials(path) == ("fake-unit-key", "https://example.invalid/v1")
    path.write_text(
        json.dumps({"api_keys": ["fake-unit-key"], "base_urls": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_credentials(path)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-env-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    assert load_credentials(None)[0] == "fake-env-key"
