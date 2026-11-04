"""AgentCaseExtractor — distil one agent trajectory MemCell into at most one :class:`AgentCase`.

Typed :class:`ConversationItem` objects are used throughout; OpenAI-format dicts are produced only at
the LLM prompt boundary so algorithm-private fields never leak into prompts.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from everos.algo.agent_memory._replies import (
    _call_llm_for_case_compress,
    _call_llm_for_case_filter,
    _call_llm_for_tool_pre_compress,
    _CaseCompressReply,
    _CompressedMessage,
)
from everos.algo.agent_memory.prompts.case_compress import (
    AGENT_CASE_COMPRESS_SYSTEM_PROMPT,
    AGENT_CASE_COMPRESS_USER_PROMPT,
)
from everos.algo.agent_memory.prompts.case_filter import (
    AGENT_CASE_FILTER_SYSTEM_PROMPT,
    AGENT_CASE_FILTER_USER_PROMPT,
)
from everos.algo.agent_memory.prompts.tool_pre_compress import (
    AGENT_TOOL_PRE_COMPRESS_SYSTEM_PROMPT,
    AGENT_TOOL_PRE_COMPRESS_USER_PROMPT,
)
from everos.algo.agent_memory.reasons import CaseExtractionResult, CaseSkipReason
from everos.algo.llm import LLMMessage, LLMOutputError
from everos.algo.types import (
    AgentCase,
    ChatMessage,
    ConversationItem,
    MemCell,
    ToolCall,
    ToolCallRequest,
    ToolCallResult,
)
from everos.algo.utils import count_tokens, json_default, render_prompt, truncate_text

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.agent_memory.gate import AgentCaseGate
    from everos.algo.llm.protocols import LLMClient

logger = logging.getLogger(__name__)


__all__ = [
    # Re-exported system prompts — monkey-patch at startup to override the rules the LLM is given
    "AGENT_CASE_COMPRESS_SYSTEM_PROMPT",
    "AGENT_CASE_FILTER_SYSTEM_PROMPT",
    "AGENT_TOOL_PRE_COMPRESS_SYSTEM_PROMPT",
    "AgentCaseExtractor",
]


# ── Heuristic constants ─────────────────────────────────────────────────────────────────────────────
# Tunable algorithm-IP thresholds; override at startup via monkey-patch (DESIGN.md §4.6).

PRE_COMPRESS_CHUNK_SIZE = 100_000
HIGH_MESSAGE_COUNT_THRESHOLD = (
    100  # halve scale_trigger when message count exceeds this
)
MAX_TOOL_OUTPUT_TOKENS = 1000
MAX_TOOL_ARGS_TOKENS = 800
MAX_ASSISTANT_RESPONSE_TOKENS = 3000
MAX_TASK_INTENT_TOKENS = 300  # head-only truncation after LLM extraction
FILTER_NO_TOOL_MAX_MESSAGES = 4
FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS = 200

# Filter-only view caps. The worth-extracting signals (exploration / correction) are driven by user
# messages and the assistant's action trajectory — which tools, with what args — not by tool-result
# bodies. So the filter sees an aggressively slimmed view (tool outputs shrink to a head just large
# enough to reveal error-vs-success); the fuller `messages_json` still feeds the compress step.
FILTER_TOOL_OUTPUT_TOKENS = 80
FILTER_TOOL_ARGS_TOKENS = 120
FILTER_ASSISTANT_TOKENS = 600


class AgentCaseExtractor:
    """Distil one agent-trajectory MemCell into at most one AgentCase.

    Returns a list of length 0 (filtered out) or 1 (successful extraction).
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        min_tool_call_rounds: int = 3,
        complex_task_tool_call_round_threshold: int = 20,
        gate: AgentCaseGate | None = None,
    ) -> None:
        """Construct the extractor.

        Args:
            llm: LLM client used for the pre-compress / filter / compress steps.
            min_tool_call_rounds: When ``> 0``, trajectories whose tool-call round count is
                strictly below this value are rejected as too simple before any LLM call. Default
                ``3``; set to ``0`` to disable this gate.
            complex_task_tool_call_round_threshold: Tool-call rounds strictly greater than this
                value mark the trajectory as a complex task and fast-pass the LLM filter step.
                Default ``20``.
            gate: Optional correctness admission. Replaces complexity and value filters; structural and resource
                checks remain. The gate sees untrimmed input, before any extraction LLM call.
        """
        self._llm = llm
        self._min_tool_call_rounds = min_tool_call_rounds
        self._complex_task_round_threshold = complex_task_tool_call_round_threshold
        self._gate = gate

    async def aextract(self, memcell: MemCell) -> list[AgentCase]:
        """Run the 11-step pipeline on one MemCell; return ``[]`` (filtered) or ``[AgentCase]``.

        Thin wrapper over :meth:`aextract_with_reason` — use that one when an empty result needs to
        be explained rather than merely observed.

        Raises:
            LLMOutputError: If a pre-compress, filter or compress reply is not valid for its call (no
                internal retry).
        """
        result = await self.aextract_with_reason(memcell)
        return result.cases

    async def aextract_with_reason(
        self,
        memcell: MemCell,
    ) -> CaseExtractionResult:
        """Run the 11-step pipeline on one MemCell, reporting which gate rejected it when none passes.

        Same algorithm as :meth:`aextract`; the difference is that a rejection comes back as a typed
        :class:`CaseSkipReason` plus structured ``detail`` instead of only reaching the log.

        Raises:
            LLMOutputError: If a pre-compress, filter or compress reply is not valid for its call (no
                internal retry).
        """
        if not memcell.items:
            logger.info("no items on memcell (n_items=0), skipping")
            return CaseExtractionResult([], CaseSkipReason.EMPTY_MEMCELL, {})

        client = self._llm

        # Step 1+2: typed view from MemCell, strip system head (drop anything before first user)
        msgs = _strip_before_first_user(memcell.items)

        # Step 3: structural + heuristic pre-filter
        if skip := _should_skip(msgs, skip_complexity=self._gate is not None):
            reason, detail = skip
            logger.info(
                "skipping memcell (n_items=%d): %s %s",
                len(memcell.items),
                reason,
                detail,
            )
            return CaseExtractionResult([], reason, detail)

        gate_detail: dict[str, Any] = {}
        if self._gate is not None:
            decision = await self._gate.aevaluate(_dump_messages(msgs))
            gate_detail = {"case_gate": decision.model_dump(mode="json")}
            if not decision.accepted:
                return CaseExtractionResult(
                    [], CaseSkipReason.SUCCESS_GATE_REJECTED, gate_detail
                )

        # Step 3b: minimum-rounds gate — drop trajectories that are too simple to be worth learning
        if self._gate is None and self._min_tool_call_rounds > 0:
            rounds = _count_tool_call_rounds(msgs)
            if rounds < self._min_tool_call_rounds:
                logger.info(
                    "skipping memcell — only %d tool-call rounds < min %d",
                    rounds,
                    self._min_tool_call_rounds,
                )
                return CaseExtractionResult(
                    [],
                    CaseSkipReason.TOO_FEW_TOOL_ROUNDS,
                    {"rounds": rounds, "min_rounds": self._min_tool_call_rounds},
                )

        # Step 4: heuristic trim (with scale_trigger adaptation)
        msgs, total_tokens = _heuristic_trim(msgs)
        logger.info(
            "memcell pre-trim total_tokens=%d, message_count=%d",
            total_tokens,
            len(msgs),
        )

        # Step 5: over-size bail — re-count only when trim scaled down
        if total_tokens > PRE_COMPRESS_CHUNK_SIZE:
            trimmed_tokens = count_tokens(_dump_messages(msgs))
            if trimmed_tokens > PRE_COMPRESS_CHUNK_SIZE * 2:
                logger.info(
                    "memcell still %d tokens after trim (> %d), skipping",
                    trimmed_tokens,
                    PRE_COMPRESS_CHUNK_SIZE * 2,
                )
                return CaseExtractionResult(
                    [],
                    CaseSkipReason.TRAJECTORY_TOO_LARGE,
                    gate_detail
                    | {"tokens": trimmed_tokens, "limit": PRE_COMPRESS_CHUNK_SIZE * 2},
                )

        # Step 6: selective LLM pre-compression of largest tool-call groups
        msgs = await _pre_compress_to_list(msgs, client)
        messages_json = _dump_messages(msgs)

        # Step 7: LLM filter — many tool-call rounds auto-passes as complex task; else judge signals
        # on a slimmed view (tool-result bodies shrunk) to cut prompt tokens. `and` short-circuits, so
        # the slim view is built only when the trajectory actually reaches the LLM filter.
        if (
            self._gate is None
            and _count_tool_call_rounds(msgs) <= self._complex_task_round_threshold
        ):
            worth, llm_reason = await _is_worth_extracting(
                _dump_messages_for_filter(msgs), _count_user_messages(msgs), client
            )
            if not worth:
                return CaseExtractionResult(
                    [], CaseSkipReason.FILTER_REJECTED, {"llm_reason": llm_reason}
                )

        # Step 8: single LLM compress (no retry)
        exp, compress_reason = await _compress_experience(messages_json, client)
        if exp is None:
            # compress_reason is always set when exp is None; the assert-free narrowing keeps mypy happy.
            return CaseExtractionResult(
                [], compress_reason or CaseSkipReason.COMPRESS_EMPTY_INTENT, gate_detail
            )

        # Step 9: hard truncate task_intent (head-only)
        original_intent = exp.task_intent
        intent = truncate_text(original_intent, MAX_TASK_INTENT_TOKENS, head_ratio=1.0)
        if intent != original_intent:
            logger.info(
                "memcell truncated task_intent to %d tokens",
                MAX_TASK_INTENT_TOKENS,
            )

        case = AgentCase(
            id=uuid.uuid4().hex,
            timestamp=memcell.timestamp,
            task_intent=intent,
            approach=exp.approach,
            quality_score=_clamp_quality_score(exp.quality_score),
            key_insight=exp.key_insight,
        )
        if gate_detail:
            case = case.model_copy(update=gate_detail)
        return CaseExtractionResult([case], None, gate_detail)


# ── Serialization helpers (ConversationItem → OpenAI Chat Completions wire format) ─────────────────────────


def _to_openai_dict(msg: ConversationItem) -> dict[str, Any]:
    """Convert a ConversationItem to OpenAI Chat Completions wire-format, stripping algorithm-private fields."""
    if isinstance(msg, ChatMessage):
        d: dict[str, Any] = {"role": msg.role, "content": msg.content}
    elif isinstance(msg, ToolCallRequest):
        d = {"role": "assistant"}
        if msg.content is not None:
            d["content"] = msg.content
        d["tool_calls"] = [tc.model_dump(mode="json") for tc in msg.tool_calls]
    else:  # ToolCallResult
        d = {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
    return d


def _to_openai_dicts(messages: Sequence[ConversationItem]) -> list[dict[str, Any]]:
    """Vector form of :func:`_to_openai_dict`."""
    return [_to_openai_dict(m) for m in messages]


def _dump_messages(messages: Sequence[ConversationItem]) -> str:
    """JSON-dump messages in OpenAI wire format for LLM prompts and token counting."""
    return json.dumps(
        _to_openai_dicts(messages), ensure_ascii=False, default=json_default
    )


def _dump_messages_for_filter(messages: Sequence[ConversationItem]) -> str:
    """Slim JSON view for the worth-extracting filter (step 7).

    Keeps user messages whole and preserves the assistant's action trajectory (tool names + short
    args), but shrinks tool-result bodies to a short head — they barely inform the exploration /
    correction signals, so dropping them cuts filter-prompt tokens sharply on tool-heavy
    trajectories. The compress step still uses the fuller ``messages_json``.
    """
    slim = _apply_truncation(
        messages,
        FILTER_TOOL_OUTPUT_TOKENS,
        FILTER_TOOL_ARGS_TOKENS,
        FILTER_ASSISTANT_TOKENS,
        head_ratio=1.0,
    )
    return _dump_messages(slim)


# ── Module-level helpers ────────────────────────────────────────────────────────────────────────────


def _strip_before_first_user(
    messages: Sequence[ConversationItem],
) -> list[ConversationItem]:
    """Drop everything before the first user message (e.g. system prompts)."""
    for i, msg in enumerate(messages):
        if isinstance(msg, ChatMessage) and msg.role == "user":
            return list(messages[i:])
    return []


def _has_tool_calls(messages: Sequence[ConversationItem]) -> bool:
    """Return ``True`` iff any message is a ToolCallRequest or ToolCallResult."""
    return any(isinstance(msg, (ToolCallRequest, ToolCallResult)) for msg in messages)


def _count_tool_call_rounds(messages: Sequence[ConversationItem]) -> int:
    """Count ToolCallRequest messages (= tool-call rounds; parallel calls within one request count once)."""
    return sum(1 for msg in messages if isinstance(msg, ToolCallRequest))


def _count_user_messages(messages: Sequence[ConversationItem]) -> int:
    """Count user-role ChatMessage entries in the trajectory."""
    return sum(
        1 for msg in messages if isinstance(msg, ChatMessage) and msg.role == "user"
    )


def _item_kind(item: ConversationItem) -> str:
    """Short label for a trajectory item, used as ``detail`` on ``TRAJECTORY_NOT_CLOSED``."""
    if isinstance(item, ChatMessage):
        return f"text:{item.role}"
    return "tool_call" if isinstance(item, ToolCallRequest) else "tool_result"


def _should_skip(
    messages: Sequence[ConversationItem], *, skip_complexity: bool = False
) -> tuple[CaseSkipReason, dict[str, Any]] | None:
    """Pre-filter combining structural + heuristic checks. Returns ``(reason, detail)`` or ``None``."""
    if not messages:
        return CaseSkipReason.NO_MESSAGES_AFTER_STRIP, {}
    if not any(isinstance(msg, ChatMessage) and msg.role == "user" for msg in messages):
        return CaseSkipReason.NO_USER_MESSAGE, {}
    has_assistant = any(
        (isinstance(msg, ChatMessage) and msg.role == "assistant")
        or isinstance(msg, ToolCallRequest)
        for msg in messages
    )
    if not has_assistant:
        return CaseSkipReason.NO_ASSISTANT_MESSAGE, {}

    last_msg = messages[-1]
    if isinstance(last_msg, ToolCallRequest) or not (
        isinstance(last_msg, ChatMessage) and last_msg.role == "assistant"
    ):
        return CaseSkipReason.TRAJECTORY_NOT_CLOSED, {
            "last_item_kind": _item_kind(last_msg)
        }

    has_tools = _has_tool_calls(messages)
    if not has_tools and not skip_complexity:
        user_count = sum(
            1 for msg in messages if isinstance(msg, ChatMessage) and msg.role == "user"
        )
        if user_count < 2:
            return CaseSkipReason.NO_TOOL_SINGLE_USER, {"user_count": user_count}

        if len(messages) <= FILTER_NO_TOOL_MAX_MESSAGES:
            return (
                CaseSkipReason.NO_TOOL_TOO_FEW_MESSAGES,
                {"messages": len(messages), "min": FILTER_NO_TOOL_MAX_MESSAGES + 1},
            )

        assistant_content = " ".join(
            msg.content
            for msg in messages
            if isinstance(msg, ChatMessage) and msg.role == "assistant"
        )
        assistant_tokens = count_tokens(assistant_content)
        if assistant_tokens < FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS:
            return (
                CaseSkipReason.NO_TOOL_ASSISTANT_TOO_SHORT,
                {
                    "tokens": assistant_tokens,
                    "min": FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS,
                },
            )

    return None


def _calc_tool_content_size(msg: ConversationItem) -> int:
    """Tool-related token count of a single message."""
    if isinstance(msg, ToolCallResult):
        return count_tokens(msg.content or "")
    if isinstance(msg, ToolCallRequest):
        return sum(count_tokens(tc.function.arguments or "") for tc in msg.tool_calls)
    return 0


def _heuristic_trim(
    messages: Sequence[ConversationItem],
) -> tuple[list[ConversationItem], int]:
    """Truncate oversized tool outputs / args / assistant responses; auto-scales caps on overage.

    Returns ``(trimmed_messages, original_total_tokens)``.
    """
    total_tokens = count_tokens(_dump_messages(messages))

    # High message count signals lower per-message value: halve the trigger so trim kicks in earlier.
    scale_trigger = (
        PRE_COMPRESS_CHUNK_SIZE // 2
        if len(messages) > HIGH_MESSAGE_COUNT_THRESHOLD
        else PRE_COMPRESS_CHUNK_SIZE
    )
    needs_scale = total_tokens > scale_trigger
    if needs_scale:
        scale = scale_trigger / total_tokens
        trim_tool_output = max(200, int(MAX_TOOL_OUTPUT_TOKENS * scale))
        trim_tool_args = max(200, int(MAX_TOOL_ARGS_TOKENS * scale))
        trim_assistant = max(500, int(MAX_ASSISTANT_RESPONSE_TOKENS * scale))
        logger.info(
            "scale trim active: total=%d > trigger=%d, scale=%.2f -> output=%d args=%d assistant=%d",
            total_tokens,
            scale_trigger,
            scale,
            trim_tool_output,
            trim_tool_args,
            trim_assistant,
        )
    else:
        trim_tool_output = MAX_TOOL_OUTPUT_TOKENS
        trim_tool_args = MAX_TOOL_ARGS_TOKENS
        trim_assistant = MAX_ASSISTANT_RESPONSE_TOKENS

    trimmed = _apply_truncation(
        messages, trim_tool_output, trim_tool_args, trim_assistant
    )
    return trimmed, total_tokens


def _apply_truncation(
    messages: Sequence[ConversationItem],
    max_tool_output: int,
    max_tool_args: int,
    max_assistant: int,
    head_ratio: float = 0.7,
) -> list[ConversationItem]:
    """Apply per-message head+tail truncation on a deep copy."""
    result = [m.model_copy(deep=True) for m in messages]
    trimmed_count = 0
    for msg in result:
        if isinstance(msg, ToolCallResult) and msg.content:
            original = msg.content
            msg.content = truncate_text(
                original, max_tool_output, head_ratio=head_ratio
            )
            if msg.content != original:
                trimmed_count += 1
        elif isinstance(msg, ToolCallRequest):
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if args:
                    new_args = truncate_text(args, max_tool_args, head_ratio=head_ratio)
                    if new_args != args:
                        tc.function.arguments = new_args
                        trimmed_count += 1
            if msg.content:
                new_content = truncate_text(
                    msg.content, max_assistant, head_ratio=head_ratio
                )
                if new_content != msg.content:
                    msg.content = new_content
                    trimmed_count += 1
        elif (
            isinstance(msg, ChatMessage)
            and msg.role == "assistant"
            and isinstance(msg.content, str)
            and msg.content
        ):
            new_content = truncate_text(
                msg.content, max_assistant, head_ratio=head_ratio
            )
            if new_content != msg.content:
                msg.content = new_content
                trimmed_count += 1
    if trimmed_count > 0:
        logger.info("heuristic trim: truncated %d content fields", trimmed_count)
    return result


def _collect_tool_call_groups(items: Sequence[ConversationItem]) -> list[list[int]]:
    """Collect atomic ``ToolCallRequest + following ToolCallResult`` groups (must not be split across chunks)."""
    groups: list[list[int]] = []
    i = 0
    while i < len(items):
        msg = items[i]
        if isinstance(msg, ToolCallRequest):
            group = [i]
            j = i + 1
            while j < len(items) and isinstance(items[j], ToolCallResult):
                group.append(j)
                j += 1
            groups.append(group)
            i = j
        else:
            i += 1
    return groups


def _calc_group_size(items: Sequence[ConversationItem], group: list[int]) -> int:
    """Total tool-content tokens of a tool-call group."""
    return sum(_calc_tool_content_size(items[idx]) for idx in group)


async def _pre_compress_to_list(
    original_data: Sequence[ConversationItem],
    client: LLMClient,
) -> list[ConversationItem]:
    """Selectively compress the largest tool-call groups via parallel LLM calls.

    Targets the largest groups until estimated total drops below ``PRE_COMPRESS_CHUNK_SIZE``.

    Raises:
        LLMOutputError: If a chunk's reply is not a valid compression of that chunk.
    """
    items = [m.model_copy(deep=True) for m in original_data]

    groups = _collect_tool_call_groups(items)
    if not groups:
        return items

    groups_with_size = [
        (i, g, _calc_group_size(items, g)) for i, g in enumerate(groups)
    ]
    total_size = sum(s for _, _, s in groups_with_size)
    if total_size <= PRE_COMPRESS_CHUNK_SIZE:
        logger.debug(
            "tool content %d tokens <= %d, no compression",
            total_size,
            PRE_COMPRESS_CHUNK_SIZE,
        )
        return items

    # Pick the largest groups, assume ~90% reduction each, until estimated total drops below threshold.
    groups_by_size = sorted(groups_with_size, key=lambda x: x[2], reverse=True)
    compress_indices: set[int] = set()
    estimated_total: float = float(total_size)
    for idx, _g, size in groups_by_size:
        if estimated_total <= PRE_COMPRESS_CHUNK_SIZE:
            break
        compress_indices.add(idx)
        estimated_total -= size * 0.9

    groups_to_compress = [g for i, g in enumerate(groups) if i in compress_indices]
    logger.debug(
        "selective compression: %d/%d groups, %d total tokens",
        len(groups_to_compress),
        len(groups),
        total_size,
    )

    # Pack selected groups into chunks of PRE_COMPRESS_CHUNK_SIZE.
    chunks: list[list[list[int]]] = []
    current_chunk: list[list[int]] = []
    current_size = 0
    for group in groups_to_compress:
        group_size = _calc_group_size(items, group)
        if current_chunk and current_size + group_size > PRE_COMPRESS_CHUNK_SIZE:
            chunks.append(current_chunk)
            current_chunk = [group]
            current_size = group_size
        else:
            current_chunk.append(group)
            current_size += group_size
    if current_chunk:
        chunks.append(current_chunk)

    chunk_msg_lists: list[list[ConversationItem]] = []
    for chunk_groups in chunks:
        chunk_indices = [idx for group in chunk_groups for idx in group]
        chunk_msg_lists.append([items[idx] for idx in chunk_indices])

    # Compress all chunks in parallel.
    # Each chunk comes back the length it went in, so the results line up with the selected indices.
    results = await asyncio.gather(
        *(_compress_tool_chunk(chunk_msgs, client) for chunk_msgs in chunk_msg_lists)
    )
    all_compressed = [item for result in results for item in result]
    selected_indices = sorted(idx for group in groups_to_compress for idx in group)
    for idx, compressed in zip(selected_indices, all_compressed, strict=True):
        items[idx] = compressed
    return items


async def _compress_tool_chunk(
    messages: Sequence[ConversationItem],
    client: LLMClient,
) -> list[ConversationItem]:
    """LLM-compress one chunk of tool messages, message for message; preserves original timestamps.

    Raises:
        LLMOutputError: If the reply is not a valid tool pre-compress output, holds a different number of messages
            than the chunk, or holds a tool message with no ``tool_call_id``.
    """
    user = render_prompt(
        AGENT_TOOL_PRE_COMPRESS_USER_PROMPT,
        None,
        messages_json=_dump_messages(messages),
        new_count=len(messages),
    )
    reply = await _call_llm_for_tool_pre_compress(
        client,
        [
            LLMMessage(role="system", content=AGENT_TOOL_PRE_COMPRESS_SYSTEM_PROMPT),
            LLMMessage(role="user", content=user),
        ],
    )
    compressed = reply.compressed_messages
    if len(compressed) != len(messages):
        raise LLMOutputError(
            f"agent tool pre-compress reply holds {len(compressed)} messages for a chunk of {len(messages)}"
        )
    # Compressed messages carry no timestamp; each keeps its original's, so downstream ordering stays meaningful.
    return [
        _compressed_to_item(comp, orig.timestamp)
        for orig, comp in zip(messages, compressed, strict=True)
    ]


def _compressed_to_item(
    message: _CompressedMessage, timestamp: int
) -> ConversationItem:
    """Convert a compressed message back to a ConversationItem, inferring its kind from role / tool_calls.

    Raises:
        LLMOutputError: If a tool message carries no ``tool_call_id``, so it no longer answers any tool call.
    """
    if message.role == "tool":
        if message.tool_call_id is None:
            raise LLMOutputError(
                "agent tool pre-compress reply holds a tool message with no tool_call_id"
            )
        return ToolCallResult(
            kind="tool_result",
            tool_call_id=message.tool_call_id,
            content=message.content or "",
            timestamp=timestamp,
        )
    if message.role == "assistant" and message.tool_calls:
        return ToolCallRequest(
            kind="tool_call",
            tool_calls=[
                ToolCall.model_validate(call.model_dump())
                for call in message.tool_calls
            ],
            content=message.content,
            timestamp=timestamp,
            sender_id="assistant",
        )
    return ChatMessage(
        kind="text",
        id="",
        role=message.role,
        content=message.content or "",
        timestamp=timestamp,
        sender_id=message.role,
    )


async def _is_worth_extracting(
    messages_json: str,
    user_message_count: int,
    client: LLMClient,
) -> tuple[bool, str]:
    """LLM filter: ``True`` only if at least one signal (exploration / user correction) is True.

    Used only when the tool-call round count is at or below the extractor's configured threshold;
    higher-volume trajectories pass automatically as complex tasks (handled upstream, not here).

    ``has_user_correction`` is additionally validated against ``user_message_count``: a real
    correction requires at least two user messages (initial request + corrective follow-up). If the
    LLM claims a correction on a single-user-message trajectory, the signal is rejected as
    hallucination.

    Returns:
        ``(worth_extracting, llm_reason)`` — the model's one-line rationale is returned alongside the
        verdict so callers can surface *why* a trajectory was filtered out. It is free text and may
        vary between runs; display it, do not branch on it.

    Raises:
        LLMOutputError: If the reply is not a valid case filter output.
    """
    reply = await _call_llm_for_case_filter(
        client,
        [
            LLMMessage(role="system", content=AGENT_CASE_FILTER_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=render_prompt(
                    AGENT_CASE_FILTER_USER_PROMPT, None, messages=messages_json
                ),
            ),
        ],
    )
    llm_reason = reply.reason
    has_exploration = reply.has_exploration
    has_user_correction = reply.has_user_correction and user_message_count >= 2
    if reply.has_user_correction and user_message_count < 2:
        logger.info(
            "rejecting has_user_correction=True on single-user-message trajectory (likely hallucination): %s",
            llm_reason,
        )
    worth = has_exploration or has_user_correction
    if not worth:
        logger.info("filtered out by LLM: %s", llm_reason)
    return worth, llm_reason


async def _compress_experience(
    messages_json: str,
    client: LLMClient,
) -> tuple[_CaseCompressReply | None, CaseSkipReason | None]:
    """Extract task_intent / approach / quality_score / key_insight via one LLM call (no retry).

    Returns ``(None, reason)`` when the LLM emits an empty ``task_intent`` or ``approach`` — those are
    business validation failures (the trajectory isn't worth compressing), not parse errors. The two
    cases are reported separately because they mean different things: an empty intent says the LLM
    could not name the task at all, an empty approach that it found no method worth recording.

    Raises:
        LLMOutputError: If the reply is not a valid case compress output.
    """
    reply = await _call_llm_for_case_compress(
        client,
        [
            LLMMessage(role="system", content=AGENT_CASE_COMPRESS_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=render_prompt(
                    AGENT_CASE_COMPRESS_USER_PROMPT, None, messages=messages_json
                ),
            ),
        ],
    )
    if not reply.task_intent:
        logger.info("LLM returned empty 'task_intent', skipping")
        return None, CaseSkipReason.COMPRESS_EMPTY_INTENT
    if not reply.approach:
        logger.warning("LLM returned empty 'approach', skipping")
        return None, CaseSkipReason.COMPRESS_EMPTY_APPROACH
    return reply, None


def _clamp_quality_score(value: float) -> float:
    """Clamp a model-written score to [0.0, 1.0], the range the compress rubric asks for."""
    return max(0.0, min(1.0, value))
