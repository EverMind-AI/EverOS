"""Each agent-memory call's reply model, the shape its prompt's output skeleton shows, and the LLM callsites held to it.

Each callsite makes one call and validates the reply against its model; there is no retry here, the host decides. Business checks on a valid reply stay with the operator that reads it.

Two groups of fields carry defaults, each deliberately: the keys of a
compressed message other than ``role``, because the input messages themselves carry ``content``, ``tool_calls`` and
``tool_call_id`` by role and the pre-compress prompt asks for every other field unchanged; and the fields of an
``update`` operation's ``data``, because both skill prompts show updates that restate only the fields they change.
A schema-constrained reply still writes every key, as ``null`` where it has nothing; a prompt-only reply may leave
them out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field

from everos.algo.llm import LLMMessage, StrictOutputModel, parse_reply

if TYPE_CHECKING:
    from everos.algo.llm.protocols import LLMClient


class _CompressedFunction(StrictOutputModel):
    """The ``function`` object of a compressed tool call, ``arguments`` being the compressed JSON text."""

    name: str
    arguments: str


class _CompressedToolCall(StrictOutputModel):
    """A tool call of a compressed assistant message, in the OpenAI shape the input used."""

    id: str
    type: Literal["function"]
    function: _CompressedFunction


class _CompressedMessage(StrictOutputModel):
    """One compressed message: an assistant tool-call request, a tool result, or a plain chat turn."""

    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[_CompressedToolCall] | None = None
    tool_call_id: str | None = None


class _ToolPreCompressReply(StrictOutputModel):
    """Tool pre-compress reply: the ``{"compressed_messages"}`` skeleton of its prompt."""

    compressed_messages: list[_CompressedMessage]


class _CaseFilterReply(StrictOutputModel):
    """Case filter reply: the ``{"has_exploration", "has_user_correction", "reason"}`` skeleton of its prompt."""

    has_exploration: bool
    has_user_correction: bool
    reason: str


class _CaseCompressReply(StrictOutputModel):
    """Case compress reply: the ``{"task_intent", "approach", "key_insight", "quality_score"}`` skeleton, in order."""

    task_intent: str
    approach: str
    key_insight: str
    quality_score: float


class _SkillAddData(StrictOutputModel):
    """The ``data`` of an ``add`` operation: the whole new skill."""

    name: str
    description: str
    content: str
    confidence: float


class _SkillUpdateData(StrictOutputModel):
    """The ``data`` of an ``update`` operation: only the fields it changes, the rest absent or ``null``."""

    name: str | None = None
    description: str | None = None
    content: str | None = None
    confidence: float | None = None


class _SkillAddOp(StrictOutputModel):
    """Skill ``add`` operation: a new skill for the cluster."""

    action: Literal["add"]
    data: _SkillAddData


class _SkillUpdateOp(StrictOutputModel):
    """Skill ``update`` operation: changes to the existing skill at ``index``."""

    action: Literal["update"]
    index: int
    data: _SkillUpdateData


class _SkillNoneOp(StrictOutputModel):
    """Skill ``none`` operation: the documented no-op."""

    action: Literal["none"]


_SkillOperation = Annotated[
    _SkillAddOp | _SkillUpdateOp | _SkillNoneOp, Field(discriminator="action")
]


class _SkillExtractReply(StrictOutputModel):
    """Skill extract reply, success and failure prompt alike: the ``{"operations", "update_note"}`` skeleton."""

    operations: list[_SkillOperation]
    update_note: str


class _MaturityReply(StrictOutputModel):
    """Maturity reply: the four 1-5 dimension scores and the ``reason`` of its prompt's skeleton, in order."""

    completeness: int
    executability: int
    evidence: int
    clarity: int
    reason: str


class _ProfilePatch(StrictOutputModel):
    """A candidate's proposed section-level edit."""

    action: Literal["add", "modify"]
    section: str
    old_text: str
    new_text: str


class _ProfileCandidate(StrictOutputModel):
    """One candidate signal with its routing, its gate verdicts and its proposed patch, in skeleton order."""

    target: Literal["soul", "agents", "none"]
    signal: str
    key: str
    evidence: str
    speech_act: Literal[
        "directive",
        "question",
        "venting",
        "withdrawn",
        "deferral",
        "denial",
        "material",
        "information",
    ]
    persistence: Literal["explicit", "implicit", "one_off"]
    directed_at: Literal["agent", "task", "user", "other"]
    novelty: Literal["new", "redundant", "conflict"]
    conflict_excerpt: str | None
    matched_pending_key: str | None
    reason: str
    patch: _ProfilePatch | None


class _ProfileUpdateReply(StrictOutputModel):
    """Profile update reply: the ``{"candidates"}`` skeleton of its prompt."""

    candidates: list[_ProfileCandidate]


async def _call_llm_for_tool_pre_compress(
    llm: LLMClient, messages: list[LLMMessage]
) -> _ToolPreCompressReply:
    """Request a tool pre-compress reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid tool pre-compress output.
    """
    text = await llm.achat(messages=messages, response_format=_ToolPreCompressReply)
    return parse_reply(_ToolPreCompressReply, text, operator="agent tool pre-compress")


async def _call_llm_for_case_filter(
    llm: LLMClient, messages: list[LLMMessage]
) -> _CaseFilterReply:
    """Request a case filter reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid case filter output.
    """
    text = await llm.achat(messages=messages, response_format=_CaseFilterReply)
    return parse_reply(_CaseFilterReply, text, operator="agent case filter")


async def _call_llm_for_case_compress(
    llm: LLMClient, messages: list[LLMMessage]
) -> _CaseCompressReply:
    """Request a case compress reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid case compress output.
    """
    text = await llm.achat(messages=messages, response_format=_CaseCompressReply)
    return parse_reply(_CaseCompressReply, text, operator="agent case compress")


async def _call_llm_for_skill_extract(
    llm: LLMClient, messages: list[LLMMessage]
) -> _SkillExtractReply:
    """Request a skill extract reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid skill extract output.
    """
    text = await llm.achat(messages=messages, response_format=_SkillExtractReply)
    return parse_reply(_SkillExtractReply, text, operator="agent skill extract")


async def _call_llm_for_maturity(
    llm: LLMClient, messages: list[LLMMessage]
) -> _MaturityReply:
    """Request a maturity reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid maturity output.
    """
    text = await llm.achat(messages=messages, response_format=_MaturityReply)
    return parse_reply(_MaturityReply, text, operator="agent skill maturity")


async def _call_llm_for_profile_update(
    llm: LLMClient, messages: list[LLMMessage]
) -> _ProfileUpdateReply:
    """Request a profile update reply and return it validated.

    Raises:
        LLMOutputError: If the reply is not a valid profile update output.
    """
    text = await llm.achat(messages=messages, response_format=_ProfileUpdateReply)
    return parse_reply(_ProfileUpdateReply, text, operator="agent profile update")
