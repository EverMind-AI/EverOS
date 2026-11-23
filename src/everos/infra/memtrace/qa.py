"""Post-hoc QA-stage injection into exported MemTrace execution graphs.

The file-tracking tracer (:mod:`everos.infra.memtrace.tracker`) records the
memory-side pipeline (messages -> entries -> retrieval -> context), but the
QA answer stage runs outside the EverOS server process — e.g. in the LoCoMo
drive script, which talks to the server over HTTP. This module closes that
gap offline: it loads an already-exported graph, appends the missing
``prompt`` / ``prediction`` / ``golden_answers`` / ``judge_response`` chain
onto the recorded query node, and re-exports the graph atomically.

The injection uses exactly the same identity / category / operation
vocabulary as the live tracker hooks (:meth:`MemTraceTracker.on_answer` /
:meth:`MemTraceTracker.on_evaluation`), so a graph is coherent whether the
QA stage was recorded in-process or injected afterwards. Records whose
query node cannot be resolved are skipped with a log line, never an error.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import portalocker
from smartcomment import (
    comment_graph,
    comment_op,
    comment_session,
    comment_variable,
    enable_tracing,
)
from smartcomment.runtime import ExecNetwork

from everos.core.observability.logging import get_logger

from .answer_context import record_answer_context
from .descriptions import describe_sentinels
from .tracker import (
    _by_id,
    _encode,
    _latest_handle,
    _qa_golden_snapshot,
    _qa_judge_snapshot,
    _qa_prediction_snapshot,
    _qa_prompt_snapshot,
    resolve_query_node,
)

logger = get_logger(__name__)

# One QA record per searched query. ``user_id`` + ``request_id`` locate the
# query node (``query-{user_id}-{request_id}``); the rest is the QA payload.
QaRecord = dict[str, Any]


def inject_qa_dir(graph_dir: Path, records: list[QaRecord]) -> dict[str, int]:
    """Inject QA records into every exported graph under ``graph_dir``.

    Each graph is loaded once. A record must uniquely resolve across all files.
    Use app/project scope and an exact query id when request ids are reused.
    Identical records are idempotent; only changed graphs are re-exported.
    The server must be stopped so its live graph cannot overwrite the injection.

    Args:
        graph_dir: Directory holding the exported ``*.json`` graphs.
        records: The QA records to inject (see :data:`QaRecord`).

    Returns:
        `dict[str, int]`:
            Per graph file name, how many records were injected into it.
    """
    enable_tracing()
    loaded: dict[Path, tuple[dict[str, Any], ExecNetwork]] = {}
    injected: dict[str, int] = {}
    for path in sorted(Path(graph_dir).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            graph = ExecNetwork.import_graph(data)
        except Exception as exc:  # an unreadable graph must not stop the batch
            logger.warning(
                "memtrace_inject_graph_unreadable", path=str(path), error=str(exc)
            )
            continue
        loaded[path] = (data, graph)
    matched: dict[Path, list[tuple[QaRecord, str, Any]]] = {}
    for record in records:
        owner = str(record.get("user_id") or "")
        request_id = str(record.get("request_id") or "")
        candidates = []
        for path, (_data, graph) in loaded.items():
            if (
                not owner
                or not request_id
                or any(
                    record.get(field) is not None
                    and record[field] != graph.metadata.get(field)
                    for field in ("app_id", "project_id")
                )
            ):
                continue
            resolved = resolve_query_node(
                graph,
                owner=owner,
                request_id=request_id,
                query_full_node_id=record.get("query_full_node_id"),
            )
            if resolved is not None:
                candidates.append((path, *resolved))
        if len(candidates) != 1:
            logger.warning(
                "memtrace_inject_query_not_unique",
                user_id=owner,
                request_id=request_id,
                matches=len(candidates),
            )
            continue
        path, query_id, query_handle = candidates[0]
        matched.setdefault(path, []).append((record, query_id, query_handle))
    for path, batch in matched.items():
        data, graph = loaded[path]
        count = 0
        with (
            comment_graph(graph=graph),
            comment_session(session_name="qa_injection", category="memory_retrieval"),
        ):
            for record, query_id, query_handle in batch:
                count += _inject_one(
                    graph, record, query_id=query_id, query_handle=query_handle
                )
        if count:
            _export(path, data, graph)
            injected[path.name] = count
    return injected


def _inject_one(
    graph: ExecNetwork, record: QaRecord, *, query_id: str, query_handle: Any
) -> bool:
    """Inject one QA record's nodes and operations into the graph context."""
    request_id = str(record.get("request_id") or "")
    op_metadata = {
        "request_id": request_id,
        "query_id": query_id,
        "source": "qa_injection",
    }
    signature = hashlib.sha256(_encode(record).encode()).hexdigest()
    prior = graph.search_operations(
        name_pattern=(
            f"^(answer_generation|inject_golden_answers|judge_answer)_"
            f"{re.escape(query_id)}$"
        )
    )
    if any(op.metadata.get("qa_record_digest") == signature for op in prior):
        return False
    op_metadata["qa_record_digest"] = signature
    answer = record.get("answer")
    if answer is not None:
        prompt = record.get("prompt")
        prompt_handle = (
            comment_variable(
                _qa_prompt_snapshot(query_id, str(prompt)),
                to_runtime=True,
                id_strategy=_by_id,
                encoding_fn=_encode,
                category="prompt",
                comment="The answer-generation prompt for this query.",
            )
            if prompt is not None
            else None
        )
        prediction_handle = comment_variable(
            _qa_prediction_snapshot(query_id, str(answer)),
            to_runtime=True,
            id_strategy=_by_id,
            encoding_fn=_encode,
            category="llm_response",
            class_name="prediction",
            comment="The answer generated from the retrieved memory context.",
        )
        context_handle = record_answer_context(
            query_id=query_id,
            query_handle=query_handle,
            retrieved=_latest_handle(graph, f"context:context-{query_id}"),
            answer_context=record.get("answer_context"),
            request_id=request_id,
        )
        comment_op(
            inputs=[
                handle
                for handle in (query_handle, context_handle, prompt_handle)
                if handle is not None
            ],
            outputs=[prediction_handle],
            op_name=f"answer_generation_{query_id}",
            category="response",
            comment="Generate the final answer from the retrieved memory context.",
            metadata=dict(op_metadata),
        )
    golden_answers = record.get("golden_answers")
    if golden_answers:
        golden_handle = comment_variable(
            _qa_golden_snapshot(query_id, golden_answers),
            to_runtime=True,
            id_strategy=_by_id,
            encoding_fn=_encode,
            category="golden_answers",
            class_name="golden_answers",
            comment="Reference answers injected by the evaluation batch.",
        )
        comment_op(
            inputs=[query_handle],
            outputs=[golden_handle],
            op_name=f"inject_golden_answers_{query_id}",
            category="evaluation",
            comment="Reference answers injected by the evaluation batch.",
            metadata=dict(op_metadata),
        )
        judge_response = record.get("judge_response")
        judge_score = record.get("judge_score")
        if judge_response is not None or judge_score is not None:
            judge_handle = comment_variable(
                _qa_judge_snapshot(query_id, judge_response, judge_score),
                to_runtime=True,
                id_strategy=_by_id,
                encoding_fn=_encode,
                category="judge_response",
                class_name="judge_response",
                comment="The judge verdict for this query's answer.",
            )
            prediction_handle = _latest_handle(
                graph, f"prediction:prediction-{query_id}"
            )
            comment_op(
                inputs=[
                    handle
                    for handle in (prediction_handle, golden_handle, query_handle)
                    if handle is not None
                ],
                outputs=[judge_handle],
                op_name=f"judge_answer_{query_id}",
                category="evaluation",
                comment="Judge the predicted answer against the reference answers.",
                metadata=dict(op_metadata),
            )
    return answer is not None or bool(golden_answers)


def _export(path: Path, original: dict[str, Any], graph: ExecNetwork) -> None:
    """Re-export the graph atomically, preserving bench annotations.

    ``ExecNetwork.export_graph`` drops keys the schema does not know (the
    MemTraceBench ``data.annotations`` block), so the annotations from the
    file we just read are re-attached before the atomic tmp-write + replace,
    under the same inter-process lock the tracker uses.
    """
    data = graph.export_graph()
    describe_sentinels(data)
    annotations = original.get("data", {}).get("annotations") or []
    data.setdefault("data", {})["annotations"] = annotations
    lock_path = path.with_suffix(".json.lock")
    with portalocker.Lock(str(lock_path), mode="w", timeout=30):
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp_path.replace(path)
