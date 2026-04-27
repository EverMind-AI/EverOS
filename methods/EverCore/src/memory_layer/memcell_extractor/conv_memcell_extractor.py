"""
Simple Boundary Detection Base Class for EverCore

This module provides a simple and extensible base class for detecting
boundaries in various types of content (conversations, emails, notes, etc.).
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
import json, re, os
from core.di.utils import get_bean_by_type
from core.component.llm.tokenizer.tokenizer_factory import TokenizerFactory
from common_utils.datetime_utils import from_iso_format as dt_from_iso_format
from memory_layer.llm.llm_provider import LLMProvider
from api_specs.memory_types import RawDataType
from api_specs.memory_models import MessageSenderRole

from memory_layer.prompts import get_prompt_by
from memory_layer.memcell_extractor.base_memcell_extractor import (
    MemCellExtractor,
    MemCell,
    StatusResult,
    MemCellExtractRequest,
)
from api_specs.memory_types import get_text_from_content_items
from core.observation.logger import get_logger
from core.observation.stage_timer import timed
from agentic_layer.metrics.memorize_metrics import (
    record_boundary_detection,
    record_memcell_extracted,
    get_space_id_for_metrics,
)

logger = get_logger(__name__)


@dataclass
class BatchBoundaryResult:
    """Result from batch boundary detection."""

    # List of 1-indexed message numbers after which to split
    boundaries: List[int] = field(default_factory=list)
    should_wait: bool = False


@dataclass
class ConversationMemCellExtractRequest(MemCellExtractRequest):
    pass


class ConvMemCellExtractor(MemCellExtractor):
    """
    Conversation MemCell Extractor - Responsible only for boundary detection and creating basic MemCell

    Responsibilities:
    1. Boundary detection (determine where to split conversation into MemCells)
    2. Create basic MemCell (including basic fields such as original_data, summary, timestamp, etc.)

    Not included:
    - Episode extraction (handled by EpisodeMemoryExtractor)
    - Foresight extraction (handled by ForesightExtractor)
    - AtomicFact extraction (handled by AtomicFactExtractor)
    - Embedding computation (handled by MemoryManager)

    Language support:
    - Controlled by MEMORY_LANGUAGE env var: 'zh' (Chinese) or 'en' (English), default 'en'
    """

    # Default limits for force splitting (configurable via environment variables)
    DEFAULT_HARD_TOKEN_LIMIT = int(os.getenv("MEMCELL_HARD_TOKEN_LIMIT", "65536"))
    DEFAULT_HARD_MESSAGE_LIMIT = int(os.getenv("MEMCELL_HARD_MESSAGE_LIMIT", "500"))

    @classmethod
    def _get_tokenizer(cls):
        """Get the shared tokenizer from tokenizer factory (with caching)."""
        tokenizer_factory: TokenizerFactory = get_bean_by_type(TokenizerFactory)
        return tokenizer_factory.get_tokenizer_from_tiktoken("o200k_base")

    def __init__(
        self,
        llm_provider=LLMProvider,
        boundary_detection_prompt: Optional[str] = None,
        hard_token_limit: Optional[int] = None,
        hard_message_limit: Optional[int] = None,
    ):
        super().__init__(RawDataType.CONVERSATION, llm_provider)
        self.llm_provider = llm_provider

        # Force split limits
        self.hard_token_limit = hard_token_limit or self.DEFAULT_HARD_TOKEN_LIMIT
        self.hard_message_limit = hard_message_limit or self.DEFAULT_HARD_MESSAGE_LIMIT

        # Use custom prompt or get default via PromptManager
        self.conv_batch_boundary_detection_prompt = (
            boundary_detection_prompt
            or get_prompt_by("CONV_BATCH_BOUNDARY_DETECTION_PROMPT")
        )

    def shutdown(self) -> None:
        """Cleanup resources."""
        pass

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count total tokens in message list using tiktoken.

        Includes sender_name in token count since it's included when passed to LLM.

        Args:
            messages: List of message dictionaries

        Returns:
            Total token count
        """
        tokenizer = self._get_tokenizer()
        total = 0
        for msg in messages:
            if isinstance(msg, dict):
                speaker = msg.get('sender_name', '')
                content = get_text_from_content_items(msg.get('content', []))
                # Format matches what's sent to LLM: "sender_name: content"
                text = f"{speaker}: {content}" if speaker else content
            else:
                text = str(msg)
            total += len(tokenizer.encode(text))
        return total

    def _extract_participant_ids(
        self, chat_raw_data_list: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract user participant IDs from chat_raw_data_list

        Retrieves sender_id only from messages with role='user'.

        Args:
            chat_raw_data_list: List of raw chat data

        Returns:
            List[str]: List of deduplicated user participant IDs
        """
        participant_ids = set()

        for raw_data in chat_raw_data_list:
            if raw_data.get('role') == MessageSenderRole.USER.value and raw_data.get(
                'sender_id'
            ):
                participant_ids.add(raw_data['sender_id'])

        return list(participant_ids)

    def _find_force_split_point(self, messages: List[Dict[str, Any]]) -> int:
        """
        Find how many messages to include in a force-split chunk.

        Starts with hard_message_limit - 1, then reduces if token limit is exceeded.
        Guaranteed to return at least 1 and at most len(messages) - 1.

        Args:
            messages: All messages to consider for splitting

        Returns:
            Number of messages to include in the split chunk (exclusive end index)
        """
        if len(messages) <= 1:
            return len(messages)

        # Start with message limit (leave at least 1 for next iteration)
        candidate = min(self.hard_message_limit - 1, len(messages) - 1)

        # Reduce if token limit exceeded for the candidate chunk
        while (
            candidate > 1
            and self._count_tokens(messages[:candidate]) >= self.hard_token_limit
        ):
            candidate = max(1, candidate // 2)

        return candidate

    @staticmethod
    def _build_original_data_items(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Build original_data items in { message } format.

        Each message dict from RawData.content already carries the v1 message
        format (with content[] list). Wraps each into { message }.
        Parsed results (parsed_summary, parsed_content) are embedded
        directly in content items by the enrich provider.

        Args:
            messages: List of message dicts (v1 format with content[] list)

        Returns:
            List of { message } dicts for MemCell.original_data
        """
        items = []
        for msg in messages:
            msg.pop("_parse_info", None)  # Remove transient key if present
            items.append({"message": msg})
        return items

    def _create_memcell_directly(
        self,
        messages: List[Dict[str, Any]],
        request: ConversationMemCellExtractRequest,
        trigger_type: str,  # 'token_limit', 'message_limit', 'flush', 'llm'
    ) -> Optional[MemCell]:
        """
        Create MemCell directly without boundary detection.

        Used for force_split and flush modes where we skip LLM boundary detection.

        Args:
            messages: List of messages to include in the MemCell
            request: The extraction request
            trigger_type: Type of trigger ('token_limit', 'message_limit', 'flush', 'llm')

        Returns:
            MemCell or None if no messages
        """
        if not messages:
            logger.warning(
                "[ConvMemCellExtractor] _create_memcell_directly called with no messages"
            )
            return None

        # Parse timestamp from last message
        ts_value = messages[-1].get("timestamp")
        timestamp = dt_from_iso_format(ts_value)
        participants = self._extract_participant_ids(messages)

        # Build original_data in { message } format
        original_data_items = self._build_original_data_items(messages)

        memcell = MemCell(
            user_id_list=request.user_id_list,
            original_data=original_data_items,
            timestamp=timestamp,
            group_id=request.group_id,
            participants=participants,
            sender_ids=participants,
            type=self.raw_data_type,
        )

        # Record metrics
        result_type = (
            'flush'
            if trigger_type == 'flush'
            else ('should_end' if trigger_type == 'llm' else 'force_split')
        )
        record_boundary_detection(
            space_id=get_space_id_for_metrics(),
            raw_data_type=self.raw_data_type.value,
            result=result_type,
            trigger_type=trigger_type,
        )
        record_memcell_extracted(
            space_id=get_space_id_for_metrics(),
            raw_data_type=self.raw_data_type.value,
            trigger_type=trigger_type,
        )

        logger.info(
            f"[ConvMemCellExtractor] ✅ MemCell created: "
            f"messages={len(messages)}, trigger={trigger_type}"
        )

        return memcell

    def _format_messages_with_indices(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format messages with 1-based indices and timestamps for LLM input.

        Format: [N] [YYYY-MM-DD HH:MM:SS+TZ] sender_name: content

        Args:
            messages: List of message dictionaries

        Returns:
            Formatted string with numbered messages
        """
        lines = []
        for i, msg in enumerate(messages, start=1):
            content = get_text_from_content_items(msg.get("content", []))
            sender_name = msg.get("sender_name", "")
            timestamp = msg.get("timestamp", "")

            # Format timestamp with timezone offset
            time_str = ""
            if timestamp:
                try:
                    if isinstance(timestamp, datetime):
                        time_str = timestamp.isoformat(sep=" ", timespec="seconds")
                    elif isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        time_str = dt.isoformat(sep=" ", timespec="seconds")
                except (ValueError, AttributeError, TypeError):
                    pass

            if content:
                if time_str:
                    lines.append(f"[{i}] [{time_str}] {sender_name}: {content}")
                else:
                    lines.append(f"[{i}] {sender_name}: {content}")
            else:
                logger.debug(
                    f"[ConvMemCellExtractor] Warning: message {i} has no content"
                )

        return "\n".join(lines)

    def _parse_batch_boundary_response(
        self, resp: str
    ) -> Optional[BatchBoundaryResult]:
        """
        Parse LLM response for batch boundary detection.

        Handles nested JSON structures (unlike old single-level regex approach).

        Args:
            resp: Raw LLM response string

        Returns:
            BatchBoundaryResult or None if parsing fails
        """
        data = None

        # Try markdown code block first
        json_match = re.search(r'```json\s*(.*?)\s*```', resp, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try direct parse
        if data is None:
            try:
                data = json.loads(resp.strip())
            except json.JSONDecodeError:
                pass

        # Try extracting outermost {...} (handles nested braces)
        if data is None:
            start = resp.find('{')
            end = resp.rfind('}')
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(resp[start : end + 1])
                except json.JSONDecodeError:
                    pass

        if data is None:
            return None

        # Parse boundaries: new format is a flat list of integers
        raw_boundaries = data.get("boundaries", [])
        boundaries = []
        for item in raw_boundaries:
            try:
                boundaries.append(int(item))
            except (TypeError, ValueError):
                logger.warning(
                    f"[ConvMemCellExtractor] Skipping unparseable boundary value: {item!r}"
                )

        return BatchBoundaryResult(
            boundaries=boundaries, should_wait=bool(data.get("should_wait", False))
        )

    async def _detect_boundaries(
        self, messages: List[Dict[str, Any]]
    ) -> BatchBoundaryResult:
        """
        Use LLM to detect multiple boundary points in a message sequence.

        Args:
            messages: All messages to analyze (history + new, already within limits)

        Returns:
            BatchBoundaryResult with detected split points and should_wait flag
        """
        if not messages:
            return BatchBoundaryResult(boundaries=[], should_wait=False)

        messages_text = self._format_messages_with_indices(messages)

        logger.debug(
            f"[ConvMemCellExtractor] Detect boundaries – total messages: {len(messages)}, "
            f"formatted text length: {len(messages_text)}"
        )

        prompt = self.conv_batch_boundary_detection_prompt.format(
            messages=messages_text
        )

        logger.debug(
            f"[ConvMemCellExtractor] === BOUNDARY DETECTION PROMPT ===\n{prompt}\n"
            f"[ConvMemCellExtractor] === END PROMPT ==="
        )

        with timed("detect_boundaries"):
            # Retry only when LLM returns unparseable content.
            # Infrastructure errors (auth, rate-limit, network) are handled
            # by the lower layer and will propagate as exceptions.
            for i in range(5):
                resp = await self.llm_provider.generate(prompt)
                logger.debug(
                    f"[ConvMemCellExtractor] === BOUNDARY DETECTION RESPONSE (attempt {i+1}) ===\n"
                    f"{resp}\n"
                    f"[ConvMemCellExtractor] === END RESPONSE ==="
                )

                result = self._parse_batch_boundary_response(resp)
                if result is not None:
                    # Validate boundary indices
                    valid_boundaries = [
                        b for b in result.boundaries if 1 <= b < len(messages)
                    ]
                    if len(valid_boundaries) != len(result.boundaries):
                        logger.warning(
                            f"[ConvMemCellExtractor] Filtered {len(result.boundaries) - len(valid_boundaries)} "
                            f"out-of-range boundaries (total messages: {len(messages)})"
                        )
                    result.boundaries = sorted(valid_boundaries)

                    # Record metrics for the overall detection
                    detection_result = (
                        'should_end' if result.boundaries else 'should_wait'
                    )
                    record_boundary_detection(
                        space_id=get_space_id_for_metrics(),
                        raw_data_type=self.raw_data_type.value,
                        result=detection_result,
                        trigger_type='llm',
                    )
                    return result

                logger.warning(
                    f"[ConvMemCellExtractor] Failed to parse JSON from LLM response "
                    f"(attempt {i + 1}/5), response: {resp[:200]}..."
                )

            # All retries exhausted, raise error to interrupt the flow
            error_msg = (
                "[ConvMemCellExtractor] All 5 retries exhausted for boundary detection"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def extract_memcell(
        self, request: ConversationMemCellExtractRequest
    ) -> tuple[List[MemCell], StatusResult]:
        """
        Extract MemCells from the conversation using multi-split boundary detection.

        Algorithm:
          1. Combine history + new messages
          2. Force-split loop: while combined messages exceed hard limits,
             cut the front chunk into a MemCell
          3. LLM batch boundary detection on remaining messages
          4. If flush=True, force-cut any remaining tail into a final MemCell

        Returns:
            (list_of_memcells, StatusResult)
            - Empty list means no boundary detected; caller should accumulate messages
            - StatusResult.should_wait indicates the last segment has insufficient context
        """
        history_message_dict_list = [
            raw_data.content for raw_data in request.history_raw_data_list
        ]
        new_message_dict_list = [
            new_raw_data.content for new_raw_data in request.new_raw_data_list
        ]

        # Note: parsed results are embedded directly in content items by the enrich provider,
        # no need for separate parse_info passthrough.

        # flush=True with no new messages: treat history as the flush tail directly
        if not new_message_dict_list:
            if request.flush and history_message_dict_list:
                logger.info(
                    f"[ConvMemCellExtractor] Flush with no new messages: "
                    f"packing {len(history_message_dict_list)} history messages into final MemCell"
                )
                memcell = self._create_memcell_directly(
                    history_message_dict_list, request, 'flush'
                )
                result_memcells = [memcell] if memcell else []
                return result_memcells, StatusResult(should_wait=False)

            logger.warning(
                "[ConvMemCellExtractor] No valid new messages to process "
                "(possibly all filtered out)"
            )
            return [], StatusResult(should_wait=True)

        all_msgs = history_message_dict_list + new_message_dict_list
        result_memcells: List[MemCell] = []

        # === Phase 1: Force-split loop ===
        # While combined messages exceed hard limits, cut the front chunk
        while len(all_msgs) > 1:
            total_tokens = self._count_tokens(all_msgs)
            total_messages = len(all_msgs)

            exceeds_token = total_tokens >= self.hard_token_limit
            exceeds_count = total_messages >= self.hard_message_limit

            if not exceeds_token and not exceeds_count:
                break

            split_at = self._find_force_split_point(all_msgs)
            trigger_type = 'token_limit' if exceeds_token else 'message_limit'

            logger.debug(
                f"[ConvMemCellExtractor] Force split triggered: "
                f"tokens={total_tokens}/{self.hard_token_limit}, "
                f"messages={total_messages}/{self.hard_message_limit}, "
                f"split_at={split_at}"
            )

            memcell = self._create_memcell_directly(
                all_msgs[:split_at], request, trigger_type
            )
            if memcell:
                result_memcells.append(memcell)
            all_msgs = all_msgs[split_at:]

        # === Phase 2: LLM batch boundary detection ===
        should_wait = False
        if all_msgs:
            batch_result = await self._detect_boundaries(all_msgs)

            # Create MemCells for each detected boundary segment
            prev = 0
            for boundary in batch_result.boundaries:
                end = boundary  # 1-indexed integer, treat as exclusive end index
                segment = all_msgs[prev:end]
                if segment:
                    memcell = self._create_memcell_directly(segment, request, 'llm')
                    if memcell:
                        result_memcells.append(memcell)
                prev = end

            # Remaining messages after the last boundary
            all_msgs = all_msgs[prev:]
            should_wait = batch_result.should_wait

        # === Phase 3: Flush tail ===
        if request.flush and all_msgs:
            logger.info(
                f"[ConvMemCellExtractor] Flush mode: packing {len(all_msgs)} remaining "
                f"messages into final MemCell"
            )
            memcell = self._create_memcell_directly(all_msgs, request, 'flush')
            if memcell:
                result_memcells.append(memcell)
            all_msgs = []
            should_wait = False

        if result_memcells:
            logger.info(
                f"[ConvMemCellExtractor] ✅ Extracted {len(result_memcells)} MemCell(s), "
                f"remaining_msgs={len(all_msgs)}, should_wait={should_wait}"
            )
        else:
            logger.debug(
                f"[ConvMemCellExtractor] ⏳ No boundary detected, "
                f"remaining_msgs={len(all_msgs)}, should_wait={should_wait}"
            )

        return result_memcells, StatusResult(should_wait=should_wait)
