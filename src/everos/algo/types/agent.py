"""Agent trajectory wire types (``ToolCallRequest``, ``ToolCallResult``) and agent-memory output types."""

from __future__ import annotations

__all__ = [
    "AgentCase",
    "AgentProfilePatch",
    "AgentProfileSignal",
    "AgentProfileUpdate",
    "AgentSkill",
    "ToolCall",
    "ToolCallFunction",
    "ToolCallRequest",
    "ToolCallResult",
]

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Trajectory wire types — agent-specific items in a ConversationItem union
# ---------------------------------------------------------------------------


class ToolCallFunction(BaseModel):
    """Function descriptor inside a single :class:`ToolCall` (OpenAI Chat Completion shape)."""

    name: str
    arguments: (
        str  # Raw JSON string per OpenAI spec; algorithm library does not pre-parse.
    )


class ToolCall(BaseModel):
    """A single tool invocation declared by the assistant within :class:`ToolCallRequest`."""

    id: str  # Unique identifier — referenced back by ToolCallResult.tool_call_id.
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ToolCallRequest(BaseModel):
    """Assistant-emitted request to invoke one or more tools.

    A single turn may issue several tool calls in parallel; ``tool_calls`` is the ordered list.
    ``content`` carries the optional reasoning text the assistant produced alongside the calls.
    """

    kind: Literal["tool_call"] = "tool_call"
    tool_calls: list[ToolCall]
    timestamp: int  # Unix epoch milliseconds
    content: str | None = None
    sender_id: str = Field(description="Assistant identity")
    sender_name: str | None = Field(default=None, description="Assistant display name")

    model_config = ConfigDict(extra="ignore")


class ToolCallResult(BaseModel):
    """Result returned by tool execution; ``tool_call_id`` points back to the originating :class:`ToolCall`."""

    kind: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    timestamp: int  # Unix epoch milliseconds

    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------------------
# Agent-memory output types — produced by the agent-memory extractors
# ---------------------------------------------------------------------------


class AgentCase(BaseModel):
    """One agent-trajectory experience distilled from a single MemCell — at most one case per cell.

    ``quality_score`` ∈ [0.0, 1.0] gates success-vs-failure prompt selection in ``AgentSkillExtractor``.
    Caller-managed fields (``vector``, ``user_id``, …) are accepted via ``extra="allow"``.
    """

    id: str
    timestamp: int  # Unix epoch milliseconds, mirrors source MemCell
    task_intent: str
    approach: str = ""
    quality_score: float = 0.5
    key_insight: str = ""

    model_config = ConfigDict(extra="allow")


class AgentSkill(BaseModel):
    """Reusable skill aggregated from clustered ``AgentCase`` instances.

    ``confidence`` ∈ [0.0, 1.0] gates soft-retire; ``maturity_score`` ∈ [0.0, 1.0] is LLM readiness.
    Embedding / caller-managed fields persist via ``extra="allow"``.

    ``cluster_id`` is caller-stamped after extraction (algo emits empty); caller fills it from its own cluster identity mapping.
    """

    id: str
    cluster_id: str = ""

    name: str = ""
    description: str = ""
    content: str = ""
    confidence: float = 0.0
    maturity_score: float = (
        0.6  # populated only when skip_maturity_scoring=False is passed to aextract
    )

    source_case_ids: list[str] = []

    model_config = ConfigDict(extra="allow")


class AgentProfileSignal(BaseModel):
    """One implicit configuration signal observed in a trajectory but still below the recurrence gate.

    ``AgentProfileExtractor`` requires implicit signals (user corrections without an explicit "from now
    on" marker) to recur across sessions before a patch is proposed. The library is stateless, so the
    caller persists these between runs and feeds them back via ``pending_signals``; ``key`` is the
    cross-session identity used to match recurrences and bump ``occurrences``.
    """

    key: str  # short kebab-case slug, stable across sessions for the same underlying preference
    description: str
    target: Literal["soul", "agents"]
    evidence: str  # verbatim user quote supporting the signal
    occurrences: int = 1
    timestamp: int  # Unix epoch milliseconds of the latest observation

    model_config = ConfigDict(extra="allow")


class AgentProfilePatch(BaseModel):
    """One section-level patch against SOUL.md (``file="soul"``) or AGENTS.md (``file="agents"``).

    ``action="add"`` appends ``new_text`` under ``section`` (created at end-of-file when missing);
    ``action="modify"`` replaces the verbatim ``old_text`` block with ``new_text``. Patches with
    ``is_conflict=True`` override an existing rule (the user changed their mind); they are applied
    like any other patch, and the flag lets the caller route them through a user-confirmation step
    or surface them in debugging.
    """

    file: Literal["soul", "agents"]
    action: Literal["add", "modify"]
    section: str  # markdown heading text the patch targets, without the leading hashes
    old_text: str = ""  # verbatim block being replaced; empty for action="add"
    new_text: str
    evidence: str = ""  # verbatim user quote that justifies the patch
    reason: str = ""
    is_conflict: bool = False

    model_config = ConfigDict(extra="allow")


class AgentProfileUpdate(BaseModel):
    """Result of one ``AgentProfileExtractor`` run; the default-constructed instance is a full noop.

    ``soul_diff`` / ``agents_diff`` are unified diffs of all applied patches; ``new_soul_md`` /
    ``new_agents_md`` carry the corresponding patched text and equal the inputs when nothing
    applied. ``signals`` are implicit signals observed (or recurrence-bumped) this run that remain
    below the gate — the caller persists them and passes them back as ``pending_signals`` on the
    next run.
    """

    patches: list[AgentProfilePatch] = []
    soul_diff: str = ""
    agents_diff: str = ""
    new_soul_md: str = ""
    new_agents_md: str = ""
    signals: list[AgentProfileSignal] = []

    model_config = ConfigDict(extra="allow")
