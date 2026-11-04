"""Machine-readable skip / drop reasons for the agent-memory extractors.

:class:`~everos.algo.agent_memory.AgentCaseExtractor` and
:class:`~everos.algo.agent_memory.AgentSkillExtractor` reject their input at a dozen-plus distinct
gates. ``aextract`` reports every one of them the same way — an empty list plus a ``logger`` line —
so a caller staring at ``[]`` cannot tell a too-simple trajectory from an LLM-filtered one from a
malformed one. ``aextract_with_reason`` returns the same decisions as typed values instead, which is
what an upstream service needs to answer "why is this session's memory empty, and what do I do about
it?" without scraping logs.

The reasons here are **algorithmic**: each one names the gate that rejected the input and nothing
more. How a rejection should be attributed — caller mistake, not-yet-satisfied condition, platform
fault — depends on deployment context the library does not have, so that taxonomy belongs to the
caller. Likewise, ``detail`` carries the *numbers* behind a rejection (observed value, threshold) as
structured data rather than prose, so callers can compose their own user-facing message without
parsing strings.

Exceptions are not reasons. Whatever the LLM client raises, and malformed LLM output
(``LLMOutputError``), still propagate — they mean the pipeline could not run, not that the input
was judged unworthy.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from everos.algo.types import AgentCase, AgentSkill


__all__ = [
    "CaseExtractionResult",
    "CaseSkipReason",
    "OpOutcome",
    "SkillExtractionResult",
    "SkillSkipReason",
]


class CaseSkipReason(StrEnum):
    """Why :meth:`AgentCaseExtractor.aextract_with_reason` produced no :class:`AgentCase`.

    A MemCell yields at most one case, so at most one of these applies per call. Members are
    ordered by the pipeline step that emits them.

    - ``EMPTY_MEMCELL`` — the MemCell carried no items at all.
    - ``NO_MESSAGES_AFTER_STRIP`` — nothing survived dropping the system head (everything before the
      first user message), i.e. the cell contained no user message.
    - ``NO_USER_MESSAGE`` — defensive only. The strip step returns either an empty list or a list
      whose first element *is* a user message, so this is unreachable from the extraction pipeline;
      it can only surface if the underlying pre-filter is called directly.
    - ``NO_ASSISTANT_MESSAGE`` — user turns only; the agent never responded.
    - ``TRAJECTORY_NOT_CLOSED`` — the last item is not a final assistant text response (it is a
      pending tool call, a tool result, or a user turn). This is the "cell must be closed" gate:
      a trajectory that stops mid-tool-loop is incomplete, not merely uninteresting.
      ``detail={"last_item_kind": ...}``.
    - ``NO_TOOL_SINGLE_USER`` — no tool calls anywhere and only one user message.
      ``detail={"user_count": ...}``.
    - ``NO_TOOL_TOO_FEW_MESSAGES`` — no tool calls and the conversation is too short.
      ``detail={"messages": ..., "min": ...}``.
    - ``NO_TOOL_ASSISTANT_TOO_SHORT`` — no tool calls and the assistant's combined output is too
      brief to hold a reusable lesson. ``detail={"tokens": ..., "min": ...}``.
    - ``TOO_FEW_TOOL_ROUNDS`` — fewer tool-call rounds than ``min_tool_call_rounds``. A round is one
      ``ToolCallRequest``; parallel calls inside a single request count once.
      ``detail={"rounds": ..., "min_rounds": ...}``.
    - ``TRAJECTORY_TOO_LARGE`` — still over the hard ceiling after heuristic trimming; the trajectory
      cannot be compressed into a prompt. ``detail={"tokens": ..., "limit": ...}``.
    - ``FILTER_REJECTED`` — the LLM filter found neither exploration nor user correction.
      ``detail={"llm_reason": ...}`` carries the model's own one-line rationale, which is free text
      and may change between runs — display it, do not branch on it.
    - ``SUCCESS_GATE_REJECTED`` — a configured correctness gate returned probability below its threshold.
      ``detail={"case_gate": ...}`` records input size, method, probability, and thresholds.
    - ``COMPRESS_EMPTY_INTENT`` / ``COMPRESS_EMPTY_APPROACH`` — the compression call returned an
      empty required field, i.e. the LLM found nothing to say about the trajectory.
    """

    EMPTY_MEMCELL = "empty_memcell"
    NO_MESSAGES_AFTER_STRIP = "no_messages_after_strip"
    NO_USER_MESSAGE = "no_user_message"
    NO_ASSISTANT_MESSAGE = "no_assistant_message"
    TRAJECTORY_NOT_CLOSED = "trajectory_not_closed"
    NO_TOOL_SINGLE_USER = "no_tool_single_user"
    NO_TOOL_TOO_FEW_MESSAGES = "no_tool_too_few_messages"
    NO_TOOL_ASSISTANT_TOO_SHORT = "no_tool_assistant_too_short"
    SUCCESS_GATE_REJECTED = "success_gate_rejected"
    TOO_FEW_TOOL_ROUNDS = "too_few_tool_rounds"
    TRAJECTORY_TOO_LARGE = "trajectory_too_large"
    FILTER_REJECTED = "filter_rejected"
    COMPRESS_EMPTY_INTENT = "compress_empty_intent"
    COMPRESS_EMPTY_APPROACH = "compress_empty_approach"


class SkillSkipReason(StrEnum):
    """Why a skill operation produced no :class:`AgentSkill`.

    Unlike the case path, one skill extraction handles a *set* of operations — the LLM sees one new
    case against N pre-filtered existing skills and may add, update, or leave alone each one. So
    these reasons attach per operation, except ``CASE_QUALITY_BELOW_THRESHOLD`` which short-circuits
    before any operation exists and is reported through
    :attr:`SkillExtractionResult.pre_reason` / :attr:`SkillExtractionResult.pre_detail` rather than
    through an :class:`OpOutcome`.

    - ``CASE_QUALITY_BELOW_THRESHOLD`` — the source case scored below ``skip_quality_threshold``;
      no LLM call was made. ``pre_detail={"quality": ..., "threshold": ...}`` — note this one lands on
      ``pre_detail``, not on an ``OpOutcome.detail``, because it fires before any operation exists.
    - ``OP_ACTION_NONE`` — the LLM explicitly decided this skill needs no change. Not a failure:
      a result of all-``OP_ACTION_NONE`` means the case carried nothing new.
    - ``ADD_CONTENT_EMPTY`` — add operation with no content.
    - ``ADD_CONTENT_INSUFFICIENT`` — add operation whose content is too short or too few lines to be
      an actionable skill. ``detail={"lines": ..., "chars": ..., "min_lines": ..., "min_chars": ...}``.
    - ``ADD_NAME_AND_DESC_EMPTY`` — add operation with neither a name nor a description.
    - ``UPDATE_INDEX_OUT_OF_RANGE`` — ``index`` does not address any supplied existing skill.
      ``detail={"index": ..., "size": ...}``. A frequent cause is the caller passing an
      ``existing_relevant_skills`` list that does not match what the prompt was rendered against.
    - ``UPDATE_DUPLICATE_INDEX`` — a second operation targeting an already-updated skill.
      ``detail={"index": ...}``.
    - ``UPDATE_CONTENT_INSUFFICIENT`` — replacement content fails the same sufficiency check as add.
      ``detail`` carries the same four keys as ``ADD_CONTENT_INSUFFICIENT``.
    - ``UPDATE_NO_FIELD_CHANGED`` — the operation would change nothing. ``detail={"index": ...}``
      names the existing skill the no-op targeted.
    """

    CASE_QUALITY_BELOW_THRESHOLD = "case_quality_below_threshold"
    OP_ACTION_NONE = "op_action_none"
    ADD_CONTENT_EMPTY = "add_content_empty"
    ADD_CONTENT_INSUFFICIENT = "add_content_insufficient"
    ADD_NAME_AND_DESC_EMPTY = "add_name_and_desc_empty"
    UPDATE_INDEX_OUT_OF_RANGE = "update_index_out_of_range"
    UPDATE_DUPLICATE_INDEX = "update_duplicate_index"
    UPDATE_CONTENT_INSUFFICIENT = "update_content_insufficient"
    UPDATE_NO_FIELD_CHANGED = "update_no_field_changed"


class CaseExtractionResult(NamedTuple):
    """Return type of :meth:`AgentCaseExtractor.aextract_with_reason`.

    NamedTuple subclass of ``tuple`` — supports positional unpacking and named access::

        cases, reason, detail = await extractor.aextract_with_reason(memcell)
        result = await extractor.aextract_with_reason(memcell)
        result.cases  # list[AgentCase]

    Attributes:
        cases: Zero or one :class:`~everos.algo.types.AgentCase`. Non-empty iff ``reason is None``.
        reason: The gate that rejected the MemCell, or ``None`` on success.
        detail: Structured context for ``reason`` — observed values and the thresholds they missed.
            Admitted gated cases also return ``case_gate`` provenance on success; legacy success returns an empty dict.
    """

    cases: list[AgentCase]
    reason: CaseSkipReason | None
    detail: dict[str, Any]


class OpOutcome(NamedTuple):
    """What became of one LLM-proposed skill operation.

    Exactly one of ``skill`` / ``reason`` is set: the operation either produced a skill or was
    dropped.

    Attributes:
        op_index: Position in the LLM's ``operations`` list. Note this is *not* the ``index`` field
            inside an update operation — that one addresses ``existing_relevant_skills``.
        skill: The resulting :class:`~everos.algo.types.AgentSkill` (added, updated, or flagged for
            retirement), or ``None`` when the operation was dropped.
        reason: Why the operation was dropped, or ``None`` when it produced a skill.
        detail: Structured context for ``reason``; empty dict when there is none.
    """

    op_index: int
    skill: AgentSkill | None
    reason: SkillSkipReason | None
    detail: dict[str, Any]


class SkillExtractionResult(NamedTuple):
    """Return type of :meth:`AgentSkillExtractor.aextract_with_reason`.

    Three distinct empty states are distinguishable, which is the point of the split:

    ===========================  ==============  ==========  =====================================
    Situation                    ``pre_reason``  ``outcomes``  Meaning
    ===========================  ==============  ==========  =====================================
    Quality short-circuit        set             ``[]``      No LLM call was made.
    LLM proposed nothing         ``None``        ``[]``      LLM ran and found nothing to change.
    Every operation dropped      ``None``        non-empty   LLM proposed work; all of it failed
                                                             validation — see each ``reason``.
    ===========================  ==============  ==========  =====================================

    Attributes:
        pre_reason: A short-circuit that fired before any operation existed, or ``None``.
        pre_detail: Structured context for ``pre_reason`` — the observed value and the threshold it
            missed. Empty dict when ``pre_reason is None``. Per-operation rejections carry their own
            context on :attr:`OpOutcome.detail` instead; this field only ever describes
            ``pre_reason``.
        outcomes: One entry per LLM-proposed operation, in order —
            ``len(outcomes) == len(operations)``. Empty when ``pre_reason`` is set or when the LLM
            proposed nothing.
    """

    pre_reason: SkillSkipReason | None
    pre_detail: dict[str, Any]
    outcomes: list[OpOutcome]

    @property
    def skills(self) -> list[AgentSkill]:
        """The operations that succeeded, in operation order — same value ``aextract`` returns."""
        return [o.skill for o in self.outcomes if o.skill is not None]

    @property
    def dropped(self) -> list[OpOutcome]:
        """The operations that were dropped, in operation order."""
        return [o for o in self.outcomes if o.reason is not None]
