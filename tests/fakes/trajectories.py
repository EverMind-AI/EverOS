"""A small agent trajectory the agent-track operators accept.

Three tool-call rounds (``AgentCaseExtractor``'s default minimum) closed by a
final assistant answer, so the case extractor reaches its model calls. The
same trajectory is offered as a :class:`~everalgo.types.MemCell` for the
operators and strategies, and as ``/add`` message dicts for the HTTP path.
"""

from __future__ import annotations

__all__ = ["TRAJECTORY_START_MS", "agent_trajectory", "agent_trajectory_messages"]

from typing import Any

from everalgo.types import (
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCall,
    ToolCallFunction,
    ToolCallRequest,
    ToolCallResult,
)

TRAJECTORY_START_MS = 1_775_116_800_000  # 2026-04-02 08:00 UTC

_STEPS: list[tuple[str, str, str]] = [
    ("pytest", '{"path": "tests/test_billing.py"}', "FAILED: can't compare naive"),
    ("read_file", '{"path": "billing/dates.py"}', "def parse(s): return strptime"),
    ("pytest", '{"path": "tests"}', "42 passed"),
]


def agent_trajectory(
    *, user_id: str = "u_alice", agent_id: str = "agent_42"
) -> MemCell:
    """The trajectory as one memcell."""
    items = [_to_item(m) for m in agent_trajectory_messages(user_id, agent_id)]
    return MemCell(items=items, timestamp=items[-1].timestamp)


def agent_trajectory_messages(
    user_id: str = "u_alice", agent_id: str = "agent_42"
) -> list[dict[str, Any]]:
    """The trajectory as ``/add`` message dicts, one minute apart."""
    messages: list[dict[str, Any]] = [
        _message(user_id, "user", "The billing date test is flaky, please fix it.")
    ]
    for i, (tool, arguments, output) in enumerate(_STEPS, start=1):
        call_id = f"call_{i}"
        messages.append(
            _message(
                agent_id,
                "assistant",
                f"Step {i}: running {tool}.",
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool, "arguments": arguments},
                    }
                ],
            )
        )
        messages.append(_message("tool", "tool", output, tool_call_id=call_id))
    messages.append(
        _message(agent_id, "assistant", "Fixed: dates are parsed as UTC-aware now.")
    )
    for minute, message in enumerate(messages):
        message["timestamp"] = TRAJECTORY_START_MS + minute * 60_000
    return messages


def _message(sender_id: str, role: str, content: str, **extra: Any) -> dict[str, Any]:
    return {"sender_id": sender_id, "role": role, "content": content, **extra}


def _to_item(message: dict[str, Any]) -> ConversationItem:
    timestamp = message["timestamp"]
    if message["role"] == "tool":
        return ToolCallResult(
            tool_call_id=message["tool_call_id"],
            content=message["content"],
            timestamp=timestamp,
        )
    if "tool_calls" in message:
        return ToolCallRequest(
            tool_calls=[
                ToolCall(id=c["id"], function=ToolCallFunction(**c["function"]))
                for c in message["tool_calls"]
            ],
            timestamp=timestamp,
            content=message["content"],
            sender_id=message["sender_id"],
        )
    return ChatMessage(
        id=f"m{timestamp}",
        role=message["role"],
        sender_id=message["sender_id"],
        content=message["content"],
        timestamp=timestamp,
    )
