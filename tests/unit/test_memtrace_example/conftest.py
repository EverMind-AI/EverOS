"""Small invented QA graphs; no benchmark conversations or live credentials."""

from __future__ import annotations

import json
from typing import Any

import pytest
from smartcomment.runtime import ExecNetwork


def make_graph(
    *,
    score: Any = "0.0",
    answers: Any = "['Friday']",
    omit_prediction: bool = False,
    mismatch: bool = False,
) -> ExecNetwork:
    """An extraction -> retrieval -> response graph with an EverOS QA chain."""
    common = {
        "filename": "invented.py",
        "lineno": 1,
        "graph_id": "test-graph",
        "session_id": "test-session",
        "created_at": "2026-01-01 00:00:00.000",
    }
    specs = [
        ("message", "message", None, {"content": "The adoption meeting was Friday."}),
        ("memory", "memory_entry", None, {"content": "adoption meeting Friday"}),
        (
            "query",
            "message & query",
            "query",
            {"query": "When was the adoption meeting?"},
        ),
        ("context", "memory_context", "context", {"content": "unrelated"}),
        ("prediction", "llm_response", "prediction", {"answer": "Unknown"}),
        ("golden", "golden_answers", "golden_answers", {"answers": answers}),
        (
            "judge",
            "judge_response",
            "judge_response",
            {
                "query_id": "other" if mismatch else "query",
                "score": score,
                "response": "VERDICT: 0",
            },
        ),
    ]
    nodes = [
        {
            **common,
            "name": name,
            "version": 1,
            "category": category,
            "class_name": class_name,
            "value": json.dumps(value),
        }
        for name, category, class_name, value in specs
    ]
    ids = {
        name: f"{class_name + ':' if class_name else ''}{name}@1"
        for name, _category, class_name, _value in specs
    }
    operations = []
    edges = []
    for op_id, name, category, inputs, output, seq in [
        ("extract-op", "extract", "extraction", ["message"], "memory", 1),
        ("retrieve-op", "search", "retrieval", ["query", "memory"], "context", 2),
        ("answer-op", "answer", "response", ["query", "context"], "prediction", 3),
        (
            "judge-op",
            "judge_answer_query",
            "evaluation",
            ["query", "golden"]
            if omit_prediction
            else ["query", "prediction", "golden"],
            "judge",
            None,
        ),
    ]:
        operations.append(
            {
                **common,
                "op_id": op_id,
                "op_name": name,
                "category": category,
                "metadata": {"event_seq": seq} if seq else {},
            }
        )
        for source in inputs:
            edges.append(
                {
                    **common,
                    "edge_id": f"{op_id}-{source}",
                    "op_id": op_id,
                    "source_full_node_id": ids[source],
                    "target_full_node_id": ids[output],
                }
            )
    return ExecNetwork.import_graph(
        {
            "graph_id": "test-graph",
            "strict": False,
            "data": {
                "nodes": nodes,
                "edges": edges,
                "operations": operations,
                "sessions": [{**common, "session_name": "synthetic-test"}],
            },
        }
    )


@pytest.fixture
def graph() -> ExecNetwork:
    return make_graph()
