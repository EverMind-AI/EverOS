"""Boundary detection: cut a conversation into MemCells, in one batch or one message at a time.

Batch (:func:`adetect_boundaries`): keep the chat messages → hard-limit pre-cut → render → one model call →
cut → carry the cuts back to the full item list. The hard limits count chat messages only, so the filter comes
first. Incremental (:meth:`BoundaryDetector.adetect_step`): one model
call over ``history`` and one new message, then the cut-and-bridge transition the caller feeds into its next step.

Every model call sends two messages: a system message with the rules, closed by a caller's
``custom_instructions=`` when given, and a user message with the rendered conversation, which the operator always
builds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple

from everos.algo.boundary._render import (
    _format_messages_step,
    _format_messages_with_indices,
)
from everos.algo.boundary._replies import _call_llm_for_batch, _call_llm_for_step
from everos.algo.boundary._segments import (
    _calculate_time_gap,
    _chat_view,
    _close_step,
    _cut_items,
    _force_split_cuts,
    _needs_force_split,
)
from everos.algo.boundary.prompts.en.chat import (
    BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT,
    BATCH_BOUNDARY_DETECT_USER_PROMPT,
    STEP_BOUNDARY_DETECT_SYSTEM_PROMPT,
    STEP_BOUNDARY_DETECT_USER_PROMPT,
    assemble_system_prompt,
)
from everos.algo.llm import LLMMessage
from everos.algo.utils import render_prompt

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.llm.protocols import LLMClient
    from everos.algo.types import ChatMessage, ConversationItem, MemCell


logger = logging.getLogger("everos.algo.boundary")

DEFAULT_HARD_TOKEN_LIMIT = 65_536
DEFAULT_HARD_MSG_LIMIT = 500


class DetectionResult(NamedTuple):
    """Return type of :func:`adetect_boundaries` and :class:`BoundaryDetector`.

    NamedTuple subclass of ``tuple`` — named access and index access::

        result = await adetect_boundaries(items, llm=client)
        result.cells  # list[MemCell]
        result.tail  # list[ConversationItem]
        result.tail_complete  # bool | None
        result[0], result[1], result[2]

    Attributes:
        cells: Zero or more :class:`~everos.algo.types.MemCell` instances the algorithm has confidently
            closed.  Caller persists immediately.
        tail: Trailing items the algorithm did NOT close — the LLM cannot judge whether the
            conversation continues beyond the last seen message, so the trailing segment is left as a
            tail for the caller's state machine.  When ``is_final=True``, tail is forced into the final
            MemCell and this field is guaranteed empty.
        tail_complete: Whether the trailing segment already forms a complete episode — the LLM's own
            verdict on it, not a restatement of ``tail`` being non-empty. ``True``: the segment after the
            last boundary (the whole window when there is none) is complete and the caller can close it
            into a MemCell now; a segment that ends with an acknowledgement or farewell closing its own
            topic counts as complete. ``False``: it only opens a topic, or is only media placeholders,
            intent-free short replies, system notifications or an ambiguous 30-minute-to-4-hour gap, so
            the caller should hold it for the next batch. A caller that closes every non-empty tail will
            close half-started episodes; one that holds every non-empty tail never closes one.

            ``None`` means no path judged it, and it is not interchangeable with ``False``:

            * :func:`adetect_boundaries` (batch) returns ``bool`` — the LLM's answer.
            * :meth:`BoundaryDetector.adetect_step` (single-step) returns ``None``.
              Its prompt answers a different question — has a new episode begun — and never evaluates the
              trailing segment's completeness. Deriving one from the other does not work: ``should_end``
              is "no" on nearly every step because the step prompt keeps related content together, so
              copying it would report "not complete" almost everywhere, and on the steps that do close an
              episode it would claim "complete" about a tail that path never looked at.
            * Inputs that reach no LLM at all (no ``ChatMessage`` among the items) return ``None``.
    """

    cells: list[MemCell]
    tail: list[ConversationItem]
    tail_complete: bool | None


class BoundaryDecision(NamedTuple):
    """Single-step boundary decision returned by :func:`adetect_boundary_step`.

    Attributes:
        should_end: Whether the current episode should end (i.e. the LLM is ready to commit a cut between
            ``conversation_history`` and ``new_messages``). ``True`` means the segment is ready to commit;
            ``False`` means keep buffering (the current episode is not yet closed).
        reasoning: One-sentence LLM rationale.
        confidence: LLM-reported confidence in [0.0, 1.0].
        topic_summary: Topic label when ``should_end=True``; empty string otherwise.
    """

    should_end: bool
    reasoning: str
    confidence: float
    topic_summary: str


class BoundaryDetector:
    """Boundary detection with the LLM client bound once. Native async.

    - :meth:`adetect` — batch path over a list of ``ConversationItem``; forwards to :func:`adetect_boundaries`.
    - :meth:`adetect_step` — incremental path over ``ChatMessage`` only; one LLM call plus the cut-and-bridge
      state transition for the caller.

    Args:
        llm: LLM client every call goes through.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def adetect(
        self,
        items: Sequence[ConversationItem],
        *,
        is_final: bool = False,
        custom_instructions: str | None = None,
        hard_token_limit: int = DEFAULT_HARD_TOKEN_LIMIT,
        hard_msg_limit: int = DEFAULT_HARD_MSG_LIMIT,
    ) -> DetectionResult:
        """Detect boundaries in ``items`` with the bound client; :func:`adetect_boundaries` documents the behaviour.

        Args:
            items: Ordered conversation items; tool items may be mixed in.
            is_final: When ``True``, forces the trailing segment into a cell so ``tail == []``.
            custom_instructions: The caller's instructions, appended verbatim after the bundled rules as the system
                message's last block, ``<custom_instructions>``. They add to the rules, never replace them, and are
                never rendered as a template, so ``{{`` stays literal. ``None`` or a blank string leaves the system
                message unchanged. The user message always carries the data.
            hard_token_limit: Token count at which leading cells are force-split off without the model.
            hard_msg_limit: Chat-message count at which leading cells are force-split off without the model.

        Returns:
            :class:`DetectionResult` ``(cells, tail, tail_complete)``.

        Raises:
            LLMOutputError: If the LLM reply is not a valid batch output.
            InvalidInputError: If a message the prompt shows has no timestamp (0).
        """
        return await adetect_boundaries(
            items,
            llm=self._llm,
            is_final=is_final,
            custom_instructions=custom_instructions,
            hard_token_limit=hard_token_limit,
            hard_msg_limit=hard_msg_limit,
        )

    async def adetect_step(
        self,
        history: list[ChatMessage],
        new: ChatMessage,
        *,
        smart_mask: bool = True,
        smart_mask_threshold: int = 5,
        hard_token_limit: int = DEFAULT_HARD_TOKEN_LIMIT,
        hard_message_limit: int = DEFAULT_HARD_MSG_LIMIT,
        custom_instructions: str | None = None,
    ) -> DetectionResult:
        """Single-step incremental boundary detection — one LLM call + state transition.

        Takes ``ChatMessage`` only; tool items are not filtered on this path. The caller owns only the outer
        loop (front-buffer + final-tail flush); this method encapsulates the smart-mask threshold check,
        masking, force-split short-circuit, LLM call, and the cut-and-bridge transition.

        Args:
            history: Prior accumulated messages.
            new: The single newly-arrived message whose boundary status is judged against ``history``.
            smart_mask: When ``True`` (default), enables the smart-mask path. Mask is only actually applied
                when ``len(history) > smart_mask_threshold``; otherwise the LLM sees the full history.
            smart_mask_threshold: Smart-mask only fires when ``len(history)`` exceeds this value.
            hard_token_limit: Force-split without an LLM call when ``history + [new]`` holds at least this
                many tokens and ``len(history) >= 2``.
            hard_message_limit: Force-split without an LLM call when ``len(history) + 1`` reaches this value
                and ``len(history) >= 2``.
            custom_instructions: The caller's instructions, appended verbatim after the bundled rules as the system
                message's last block, ``<custom_instructions>``. They add to the rules, never replace them, and are
                never rendered as a template, so ``{{`` stays literal. ``None`` or a blank string leaves the system
                message unchanged. The user message always carries the data.

        Returns:
            :class:`DetectionResult`:

            - ``cells`` — exactly one closed :class:`~everos.algo.types.MemCell` (built from ``history`` only)
              when the episode ends; empty otherwise.
            - ``tail`` — the new history the caller must feed into the next call:

              * episode ends with smart-mask active: ``[history[-1], new]`` (bridge).
              * episode ends without smart-mask: ``[new]`` (clean cut).
              * episode continues: ``[*history, new]`` (accumulate).

            - ``tail_complete`` — always ``None`` on this path. The step prompt answers whether a new
              episode has begun, and never evaluates whether the trailing segment is a complete episode;
              ``None`` says so rather than guessing. See :class:`DetectionResult` for why ``should_end`` is
              not a substitute.

        Raises:
            LLMOutputError: If the LLM reply is not a valid step output.
            InvalidInputError: If a message the prompt shows has no timestamp (0).
        """
        bridge = smart_mask and len(history) > smart_mask_threshold
        if _needs_force_split(history, new, hard_token_limit, hard_message_limit):
            return _closed_step(history, new, bridge=bridge)

        shown_history = history[:-1] if bridge else history
        decision = await adetect_boundary_step(
            shown_history,
            [new],
            llm=self._llm,
            custom_instructions=custom_instructions,
            time_gap_info=_calculate_time_gap(shown_history, [new]),
        )
        if not decision.should_end:
            return DetectionResult(cells=[], tail=[*history, new], tail_complete=None)
        if not decision.topic_summary:
            logger.warning(
                "BoundaryDetector: should_end=True but topic_summary is empty; using fallback"
            )
        return _closed_step(history, new, bridge=bridge)


async def adetect_boundaries(
    items: Sequence[ConversationItem],
    *,
    llm: LLMClient,
    is_final: bool = False,
    custom_instructions: str | None = None,
    hard_token_limit: int = DEFAULT_HARD_TOKEN_LIMIT,
    hard_msg_limit: int = DEFAULT_HARD_MSG_LIMIT,
) -> DetectionResult:
    """Detect MemCell boundaries in a conversation, in one LLM call.

    Only the ``ChatMessage`` items reach the model; ``ToolCallRequest`` / ``ToolCallResult`` items stay in the
    output cells at their place. Tool items between two chat messages go with the earlier one, tool items ahead
    of the first chat message with the first cell, and trailing tool items into the tail (or into the last cell
    when ``is_final``). For a chat-only input every item maps to itself. Every item lands exactly once:
    ``sum(len(c.items) for c in cells) + len(tail) == len(items)``.

    Args:
        items: Ordered conversation items, typically ``prior_tail + new_items``.
        llm: LLM client. Algorithm always requires an LLM; the caller must supply one.
        is_final: When ``True``, forces the trailing segment into a cell so ``tail == []``.
        custom_instructions: The caller's instructions, appended verbatim after the bundled rules as the system
            message's last block, ``<custom_instructions>``. They add to the rules, never replace them, and are never
            rendered as a template, so ``{{`` stays literal. ``None`` or a blank string leaves the system message
            unchanged. The user message always carries the data.
        hard_token_limit: Leading cells are force-split off, without the model, until the chat messages left
            hold fewer tokens than this.
        hard_msg_limit: Likewise, until fewer chat messages than this are left.

    Returns:
        :class:`DetectionResult`. Without any ``ChatMessage`` no LLM is called: ``cells`` is empty, ``tail``
        holds every item, and ``tail_complete`` is ``None``.

    Raises:
        LLMOutputError: If the LLM reply is not a valid batch output. The client's own exceptions propagate
            unchanged; retry policy is the caller's concern.
        InvalidInputError: If a chat message the prompt shows has text but no timestamp (0).
    """
    chat_positions, messages = _chat_view(items)
    if not messages:
        return DetectionResult(cells=[], tail=list(items), tail_complete=None)

    cuts = _force_split_cuts(messages, hard_token_limit, hard_msg_limit)
    window_start = cuts[-1] if cuts else 0
    boundaries, tail_complete = await _ask_for_boundaries(
        messages[window_start:], llm, custom_instructions
    )
    cuts += [window_start + boundary for boundary in boundaries]
    if is_final:
        cuts.append(len(messages))

    cells, tail = _cut_items(items, chat_positions, cuts)
    return DetectionResult(cells=cells, tail=tail, tail_complete=tail_complete)


async def adetect_boundary_step(
    conversation_history: list[ChatMessage],
    new_messages: list[ChatMessage],
    *,
    llm: LLMClient,
    custom_instructions: str | None = None,
    time_gap_info: str = "",
) -> BoundaryDecision:
    """Single-step LLM boundary decision over ``history + new_messages``.

    Caller is responsible for the outer accumulation loop (front-2-buffer / smart_mask / cut-and-bridge)
    and the force-split short-circuit; :meth:`BoundaryDetector.adetect_step` is that caller. This primitive
    only owns the LLM call.

    Args:
        conversation_history: Prior accumulated messages (may be empty).
        new_messages: Newly arrived messages whose boundary status is being judged.
        llm: LLM client. Algorithm always requires an LLM; the caller must supply one.
        custom_instructions: The caller's instructions, appended verbatim after the bundled rules as the system
            message's last block, ``<custom_instructions>``. They add to the rules, never replace them, and are never
            rendered as a template, so ``{{`` stays literal. ``None`` or a blank string leaves the system message
            unchanged. The user message always carries the data.
        time_gap_info: Pre-computed time-gap label rendered into the user message's ``<time_gap>``; an empty
            string renders as "No time gap information available".

    Returns:
        :class:`BoundaryDecision` with the LLM's verdict.

    Raises:
        LLMOutputError: If the LLM reply is not a valid step output. The client's own exceptions propagate
            unchanged; retry/fallback policy is the caller's concern.
        InvalidInputError: If a chat message the prompt shows has text but no timestamp (0).
    """
    if not conversation_history:
        return BoundaryDecision(
            should_end=False,
            reasoning="First messages in conversation",
            confidence=1.0,
            topic_summary="",
        )

    request = _step_request(
        conversation_history, new_messages, time_gap_info, custom_instructions
    )
    reply = await _call_llm_for_step(llm, request)
    logger.info(
        "boundary step reply: should_end=%s confidence=%.2f",
        reply.should_end,
        reply.confidence,
    )
    return BoundaryDecision(
        should_end=reply.should_end,
        reasoning=reply.reasoning,
        confidence=reply.confidence,
        topic_summary=reply.topic_summary,
    )


def _closed_step(
    history: list[ChatMessage], new: ChatMessage, *, bridge: bool
) -> DetectionResult:
    """The step result that closes ``history`` into a cell."""
    cell, next_history = _close_step(history, new, bridge=bridge)
    return DetectionResult(cells=[cell], tail=next_history, tail_complete=None)


async def _ask_for_boundaries(
    messages: Sequence[ChatMessage],
    llm: LLMClient,
    custom_instructions: str | None,
) -> tuple[list[int], bool]:
    """Call the LLM once; return the usable boundaries and its ``tail_complete`` verdict.

    The schema guarantees integers only; which of them can cut ``messages`` is checked here: indices outside
    ``1 <= b < len(messages)`` are dropped, the rest deduplicated and sorted.

    Raises:
        LLMOutputError: If the LLM reply is not a valid batch output.
        InvalidInputError: If a message the prompt shows has no timestamp (0).
    """
    reply = await _call_llm_for_batch(
        llm, _batch_request(messages, custom_instructions)
    )
    in_range = [b for b in reply.boundaries if 1 <= b < len(messages)]
    boundaries = sorted(set(in_range))
    logger.info(
        "boundary batch reply: messages=%d boundaries=%d tail_complete=%s dropped_out_of_range=%d",
        len(messages),
        len(boundaries),
        reply.tail_complete,
        len(reply.boundaries) - len(in_range),
    )
    return boundaries, reply.tail_complete


def _batch_request(
    messages: Sequence[ChatMessage], custom_instructions: str | None
) -> list[LLMMessage]:
    """The batch call's ``[system, user]`` pair; ``count`` is the window size, so it bounds the indices.

    Raises:
        InvalidInputError: If a message the prompt shows has no timestamp (0).
    """
    user = render_prompt(
        BATCH_BOUNDARY_DETECT_USER_PROMPT,
        None,
        count=len(messages),
        messages=_format_messages_with_indices(messages),
    )
    return [
        LLMMessage(
            role="system",
            content=assemble_system_prompt(
                BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT,
                custom_instructions=custom_instructions,
            ),
        ),
        LLMMessage(role="user", content=user),
    ]


def _step_request(
    conversation_history: list[ChatMessage],
    new_messages: list[ChatMessage],
    time_gap_info: str,
    custom_instructions: str | None,
) -> list[LLMMessage]:
    """The step call's ``[system, user]`` pair.

    Raises:
        InvalidInputError: If a message the prompt shows has no timestamp (0).
    """
    user = render_prompt(
        STEP_BOUNDARY_DETECT_USER_PROMPT,
        None,
        conversation_history=_format_messages_step(conversation_history),
        time_gap_info=time_gap_info or "No time gap information available",
        new_messages=_format_messages_step(new_messages),
    )
    return [
        LLMMessage(
            role="system",
            content=assemble_system_prompt(
                STEP_BOUNDARY_DETECT_SYSTEM_PROMPT,
                custom_instructions=custom_instructions,
            ),
        ),
        LLMMessage(role="user", content=user),
    ]
