"""
Agent MemCell Extractor for EverCore

Extends ConvMemCellExtractor for agent conversations in OpenAI chat completion format.

Strategy: maximally reuse the parent's three-phase pipeline
(force-split → LLM batch boundary detection → flush).

Only two customizations:
1. Guard: skip boundary detection when the agent turn is still in progress
   (last new message is a tool_call or tool response — the agent hasn't finished)
2. Override _detect_boundaries: filter out tool messages for LLM prompt,
   then remap boundary indices back to original message space so that
   resulting MemCells contain the full trajectory including tool calls.

Example of the index remapping:

    Original messages (0-indexed):
      [0] user: hello
      [1] assistant + tool_calls    ← intermediate, filtered out
      [2] tool: result              ← intermediate, filtered out
      [3] assistant: answer
      [4] user: new question
      [5] assistant + tool_calls    ← intermediate, filtered out
      [6] tool: API result          ← intermediate, filtered out
      [7] assistant: done

    Filtered for LLM (0-indexed):
      [0] user: hello        → orig 0
      [1] assistant: answer  → orig 3
      [2] user: new question → orig 4
      [3] assistant: done    → orig 7

    LLM returns: boundaries: [2]  (split after filtered[2], i.e. "assistant: answer")

    Remap: filtered_to_orig[2-1] + 1 = orig[3] + 1 = 4

    Result: MemCell_1 = original[0:4] = [user, tool_call, tool, assistant]  ← full turn
            MemCell_2 = original[4:]  = [user, tool_call, tool, assistant]  ← full turn
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from memory_layer.memcell_extractor.conv_memcell_extractor import (
    ConvMemCellExtractor,
    BatchBoundaryResult,
)
from memory_layer.memcell_extractor.base_memcell_extractor import (
    MemCellExtractRequest,
    StatusResult,
)
from memory_layer.llm.llm_provider import LLMProvider
from api_specs.memory_types import MemCell, RawDataType, is_intermediate_agent_step
from core.observation.logger import get_logger

logger = get_logger(__name__)

# Default hard limits for agent conversations
AGENT_DEFAULT_HARD_TOKEN_LIMIT = 32768
AGENT_DEFAULT_HARD_MESSAGE_LIMIT = 64


@dataclass
class AgentMemCellExtractRequest(MemCellExtractRequest):
    """Agent-specific MemCell extraction request."""

    pass


class AgentMemCellExtractor(ConvMemCellExtractor):
    """Agent MemCell Extractor — thin layer over ConvMemCellExtractor.

    Reuses the parent's full three-phase pipeline. Only customizes:
    - raw_data_type = AGENTCONVERSATION
    - Guard: skip when last new message is an intermediate agent step
    - _detect_boundaries: filter→detect→remap so LLM sees clean conversation
      but MemCells contain full trajectory
    """

    def __init__(
        self,
        llm_provider=LLMProvider,
        boundary_detection_prompt: Optional[str] = None,
        hard_token_limit: Optional[int] = None,
        hard_message_limit: Optional[int] = None,
    ):
        super().__init__(
            llm_provider=llm_provider,
            boundary_detection_prompt=boundary_detection_prompt,
            hard_token_limit=hard_token_limit or AGENT_DEFAULT_HARD_TOKEN_LIMIT,
            hard_message_limit=hard_message_limit or AGENT_DEFAULT_HARD_MESSAGE_LIMIT,
        )
        self.raw_data_type = RawDataType.AGENTCONVERSATION

    # ------------------------------------------------------------------
    # extract_memcell: single guard, then delegate to parent
    # ------------------------------------------------------------------

    async def extract_memcell(
        self, request: MemCellExtractRequest
    ) -> Tuple[List[MemCell], StatusResult]:
        """Guard + delegate.

        The only guard: if the agent turn is still in progress (last new
        message is a tool_call or tool response), skip — no point in
        running boundary detection mid-turn.

        Everything else (force-split, LLM detection, flush) is handled
        by the parent's pipeline. Our _detect_boundaries override ensures
        the LLM sees clean messages while MemCells get full trajectories.
        """
        if not request.flush:
            # Skip when only a single message total — not enough context
            total_msgs = len(request.history_raw_data_list) + len(
                request.new_raw_data_list
            )
            if total_msgs <= 1:
                logger.debug(
                    "[AgentMemCellExtractor] Skipping: only %d message(s), "
                    "waiting for more context",
                    total_msgs,
                )
                return ([], StatusResult(should_wait=True))

            if request.new_raw_data_list:
                # Skip when new messages are all user messages (no assistant response yet)
                all_user_only = all(
                    isinstance(rd.content, dict)
                    and rd.content.get("role") == "user"
                    for rd in request.new_raw_data_list
                )
                if all_user_only:
                    logger.debug(
                        "[AgentMemCellExtractor] Skipping: new messages contain "
                        "only user messages, waiting for assistant response",
                    )
                    return ([], StatusResult(should_wait=True))

                # Skip when the agent turn is still in progress
                last_content = request.new_raw_data_list[-1].content
                if isinstance(
                    last_content, dict
                ) and is_intermediate_agent_step(last_content):
                    logger.debug(
                        "[AgentMemCellExtractor] Skipping: last new message is "
                        "intermediate (role=%s)",
                        last_content.get("role"),
                    )
                    return ([], StatusResult(should_wait=True))

        return await super().extract_memcell(request)

    # ------------------------------------------------------------------
    # _find_force_split_point: respect tool-call boundaries
    # ------------------------------------------------------------------

    @staticmethod
    def _is_safe_split(messages: List[Dict[str, Any]], split_at: int) -> bool:
        """Check if split_at is a safe boundary.

        A safe boundary means:
        1. split_at is in valid range: 1 <= split_at <= len(messages) - 1
        2. messages[split_at - 1] is a final assistant response
           (not an intermediate step, not a user/tool message)
        """
        if split_at < 1 or split_at > len(messages) - 1:
            return False
        last_msg = messages[split_at - 1]
        return (
            last_msg.get("role") == "assistant"
            and not last_msg.get("tool_calls")
        )

    def _find_force_split_point(self, messages: List[Dict[str, Any]]) -> int:
        """Find force-split point that does not break a tool-call sequence.

        Gets the candidate from the parent, then adjusts so we never:
        1. Split in the middle of a tool-call sequence
        2. Cut out a chunk with only user messages (no assistant reply)

        Strategy: walk backwards from parent candidate to find a safe point.
        If that fails (e.g. all leading messages are intermediate), walk
        forward instead.
        """
        original = super()._find_force_split_point(messages)
        candidate = original

        # Walk backwards to find a safe boundary (min split_at=2 for a meaningful chunk)
        while candidate > 2 and not self._is_safe_split(messages, candidate):
            candidate -= 1

        # If walking back didn't find a safe point, walk forward
        if not self._is_safe_split(messages, candidate):
            candidate = original + 1
            while candidate < len(messages) and not self._is_safe_split(
                messages, candidate
            ):
                candidate += 1

        # If no safe split found in either direction, don't split —
        # keep all messages in one MemCell rather than breaking a tool sequence
        if not self._is_safe_split(messages, candidate):
            logger.warning(
                "[AgentMemCellExtractor] No safe split found among %d messages, "
                "keeping as one chunk",
                len(messages),
            )
            return len(messages)

        return candidate

    # ------------------------------------------------------------------
    # _detect_boundaries: filter → detect → remap
    # ------------------------------------------------------------------

    async def _detect_boundaries(
        self, messages: List[Dict[str, Any]]
    ) -> BatchBoundaryResult:
        """Detect boundaries on filtered messages, remap to original indices.

        1. Filter out tool messages → clean user/assistant conversation
        2. Run parent's LLM boundary detection on the filtered list
        3. Remap boundary indices back to original message space

        This ensures:
        - LLM sees a clean conversation without tool noise
        - MemCells contain the full agent trajectory including tool calls
        """
        # Build filtered list with original-index mapping
        filtered: List[Dict[str, Any]] = []
        filtered_to_orig: List[int] = []  # filtered_to_orig[i] = original index
        for orig_idx, msg in enumerate(messages):
            if not is_intermediate_agent_step(msg):
                filtered.append(msg)
                filtered_to_orig.append(orig_idx)

        if not filtered:
            return BatchBoundaryResult(boundaries=[], should_wait=True)

        # Detect boundaries on clean user-assistant conversation
        result = await super()._detect_boundaries(filtered)

        # Remap: filtered boundary b → original split point
        #   boundary b means "split after the b-th filtered message" (1-indexed)
        #   In original space: split right after orig[filtered_to_orig[b-1]]
        remapped: List[int] = []
        for b in result.boundaries:
            orig_split = filtered_to_orig[b - 1] + 1
            remapped.append(orig_split)

        result.boundaries = remapped
        return result
