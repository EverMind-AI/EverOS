"""Record the exact answer input separately from the retrieved response."""

from __future__ import annotations

import json
from typing import Any

from smartcomment import comment_op, comment_variable


def record_answer_context(
    *,
    query_id: str,
    query_handle: Any,
    retrieved: Any,
    answer_context: str | None,
    request_id: str,
) -> Any:
    """Unknown answer input stays unknown; it is never inferred from retrieval."""
    snapshot = {
        "id": f"answer-context-{query_id}",
        "query_id": query_id,
        "content": answer_context,
        "observed": answer_context is not None,
    }
    handle = comment_variable(
        snapshot,
        to_runtime=True,
        id_strategy=lambda value: value["id"],
        encoding_fn=lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True),
        category="memory_context",
        class_name="answer_context",
        comment="This is the exact answer-model context, or null when unrecorded.",
    )
    comment_op(
        inputs=[h for h in (query_handle, retrieved) if h is not None],
        outputs=[handle],
        op_name=f"answer_context_assembly_{query_id}",
        category="retrieval",
        comment="Record the exact context supplied to the answer model for this query.",
        metadata={
            "request_id": request_id,
            "query_id": query_id,
            "context_observed": answer_context is not None,
        },
    )
    return handle
