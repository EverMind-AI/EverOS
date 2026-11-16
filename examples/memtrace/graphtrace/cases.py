"""Read EverOS QA snapshots without changing execution graphs or annotations."""

from __future__ import annotations

import ast
import json
import math
from typing import TYPE_CHECKING, Any

from .models import FailedQueryCase

if TYPE_CHECKING:
    from smartcomment.runtime import ExecNetwork


def decode_snapshot(value: str) -> Any:
    """Decode JSON when present, preserving ordinary text values."""
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return value


def _text(value: str, key: str) -> str:
    decoded = decode_snapshot(value)
    if isinstance(decoded, dict):
        decoded = decoded.get(key)
    if not isinstance(decoded, str) or not decoded.strip():
        raise ValueError(f"Missing or invalid {key} text.")
    return decoded


def _answers(value: str) -> list[str]:
    decoded = decode_snapshot(value)
    if isinstance(decoded, dict):
        decoded = decoded.get("answers")
    if isinstance(decoded, str):
        # The existing tracer stores lists as Python repr inside JSON snapshots.
        try:
            parsed = ast.literal_eval(decoded)
        except (ValueError, SyntaxError):
            parsed = decoded
        decoded = parsed if isinstance(parsed, list) else [decoded]
    if not isinstance(decoded, list) or not decoded:
        raise ValueError("Missing reference answers.")
    if any(not isinstance(item, str) or not item.strip() for item in decoded):
        raise ValueError("Reference answers must be nonempty strings.")
    return decoded


def collect_failed_queries(
    graph: ExecNetwork,
) -> tuple[list[FailedQueryCase], list[str]]:
    """Return complete failed cases and diagnostics for malformed QA chains."""
    cases: list[FailedQueryCase] = []
    diagnostics: list[str] = []
    operations = graph.search_operations(
        name_pattern=r"^(judge_answer_.*|llm-judge)$", category="evaluation"
    )
    for op in sorted(operations, key=lambda item: (item.created_at, item.op_id)):
        try:
            nodes: dict[str, Any] = {}
            for node in graph.filter_by_operation(op.op_id).nodes:
                if node.class_name in (
                    "query",
                    "prediction",
                    "golden_answers",
                    "judge_response",
                    "judge_score",
                ):
                    if node.class_name in nodes:
                        raise ValueError(f"Ambiguous {node.class_name} nodes.")
                    nodes[node.class_name] = node
            judge = nodes["judge_response"]
            snapshot = decode_snapshot(judge.value)
            score = (
                nodes["judge_score"].value
                if "judge_score" in nodes
                else snapshot.get("score")
                if isinstance(snapshot, dict)
                else None
            )
            score = float(score)
            if not math.isfinite(score):
                raise ValueError("Judge score is not finite.")
            if score != 0.0:
                continue
            query = nodes["query"]
            query_snapshot = decode_snapshot(query.value)
            original_name = query.full_name.removeprefix("query:").removeprefix(
                query.metadata.get("tenant_scope", "")
            )
            query_identity = (
                query_snapshot.get("id", original_name)
                if isinstance(query_snapshot, dict)
                else original_name
            )
            for node in nodes.values():
                decoded = decode_snapshot(node.value)
                if (
                    isinstance(decoded, dict)
                    and decoded.get("query_id") is not None
                    and decoded["query_id"] != query_identity
                ):
                    raise ValueError("QA snapshots refer to different queries.")
            cases.append(
                FailedQueryCase(
                    query_full_node_id=query.full_node_id,
                    query=_text(query.value, "query"),
                    golden_answers=_answers(nodes["golden_answers"].value),
                    prediction=_text(nodes["prediction"].value, "answer"),
                    judge_response=_text(judge.value, "response"),
                    judge_op_id=op.op_id,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            diagnostics.append(f"Skipped judge operation {op.op_id}: {exc}")
    return cases, diagnostics
