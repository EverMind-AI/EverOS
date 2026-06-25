"""Ingest pipeline entry — normalise external input into canonical form.

Input shape (received from the service layer, decoupled from any
specific DTO module):

    {
      "session_id": "...",
      "messages": [
        {
          "sender_id": "...",
          "sender_name": "...",        # optional
          "role": "user" | "assistant" | "tool",
          "timestamp": 1740564000000,  # unix ms
          "content": "..." | [ContentItem dicts],
          "tool_calls": [...] | None,  # OpenAI shape
          "tool_call_id": "..." | None,
        },
        ...
      ]
    }

Output: :class:`everos.memory.IngestResult`.
"""

from __future__ import annotations

from typing import Any

from everos.component.utils.datetime import from_timestamp
from everos.config import load_settings
from everos.memory import CanonicalMessage, IngestResult, ToolCall
from everos.memory.extract.parser import (
    enrich_content_items,
    has_unparsed_multimodal,
    require_multimodal,
)

from .id_gen import gen_message_id
from .multimodal import coerce_items, derive_text


async def process(payload: dict[str, Any]) -> IngestResult:
    """Normalise the raw add payload into an :class:`IngestResult`.

    The function is ``async`` for symmetry with the rest of the pipeline,
    even though current logic is pure CPU.
    """
    session_id: str = payload["session_id"]
    app_id: str = payload.get("app_id") or "default"
    project_id: str = payload.get("project_id") or "default"
    raw_messages: list[dict[str, Any]] = payload["messages"]

    canonical: list[CanonicalMessage] = []
    non_text_total = 0
    for idx, m in enumerate(raw_messages):
        content_items = coerce_items(m["content"])
        if has_unparsed_multimodal(content_items):
            require_multimodal()
            await enrich_content_items(
                content_items,
                max_concurrency=load_settings().multimodal.max_concurrency,
            )
        text, non_text = derive_text(content_items)
        non_text_total += non_text

        ts_ms: int = int(m["timestamp"])
        message_id = gen_message_id(session_id, ts_ms, idx)
        ts = from_timestamp(ts_ms)

        canonical.append(
            CanonicalMessage(
                message_id=message_id,
                session_id=session_id,
                sender_id=m["sender_id"],
                sender_name=m.get("sender_name"),
                role=m["role"],
                timestamp=ts,
                content_items=content_items,
                text=text,
                tool_calls=_coerce_tool_calls(m.get("tool_calls")),
                tool_call_id=m.get("tool_call_id"),
            )
        )

    return IngestResult(
        session_id=session_id,
        app_id=app_id,
        project_id=project_id,
        messages=canonical,
        unparsed_non_text_count=non_text_total,
    )


def _coerce_tool_calls(
    raw: list[dict[str, Any]] | list[Any] | None,
) -> list[ToolCall] | None:
    """Convert raw tool-call dicts to ToolCall models."""
    if not raw:
        return None
    out: list[ToolCall] = []
    for tc in raw:
        if isinstance(tc, ToolCall):
            out.append(tc)
        elif hasattr(tc, "model_dump"):
            out.append(ToolCall.model_validate(tc.model_dump()))
        else:
            out.append(ToolCall.model_validate(tc))
    return out
