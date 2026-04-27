"""
Agent Experience Extractor for EverCore

Extracts AgentCase from agent MemCells (OpenAI chat completion format).

Pipeline:
1. Pre-compress: Build a structured list from raw messages. If total tool content
   exceeds a threshold, use LLM to compress tool call inputs/outputs in chunks.
2. Single LLM call: Extract one experience with task_intent, approach, quality_score.
3. Compute embedding on task_intent for retrieval.

OpenAI message format:
- role="user": User input (content only)
- role="assistant" with tool_calls: Agent decides to call tools
- role="tool" with tool_call_id: Tool execution result
- role="assistant" without tool_calls: Agent final response
"""

import asyncio
import copy
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from common_utils.json_utils import parse_json_response

from core.oxm.mongo.mongo_utils import generate_object_id_str
from memory_layer.llm.llm_provider import LLMProvider
from memory_layer.memory_extractor.base_memory_extractor import (
    MemoryExtractor,
    MemoryExtractRequest,
)
from memory_layer.prompts import get_prompt_by
from api_specs.memory_types import (
    MemCell,
    RawDataType,
    AgentCase,
    get_text_from_content_items,
)
from api_specs.memory_models import MemoryType
from agentic_layer.vectorize_service import get_vectorize_service
from core.di.utils import get_bean_by_type
from core.component.llm.tokenizer.tokenizer_factory import TokenizerFactory
from core.observation.logger import get_logger
from core.observation.stage_timer import timed

logger = get_logger(__name__)

# LLM pre-compression chunk size (tokens)
# Tool content below this threshold skips compression entirely
PRE_COMPRESS_CHUNK_SIZE = 100000

# When a conversation has many messages, each individual message carries less unique value,
# so we apply more aggressive trimming by halving the scale_trigger threshold.
# This constant defines the message count above which we switch to the tighter threshold.
HIGH_MESSAGE_COUNT_THRESHOLD = 100

# Heuristic trim: per-message token limits applied before LLM compression
MAX_TOOL_OUTPUT_TOKENS = 1000
MAX_TOOL_ARGS_TOKENS = 800
MAX_ASSISTANT_RESPONSE_TOKENS = 3000

# Hard cap for task_intent token length (truncated after LLM extraction)
MAX_TASK_INTENT_TOKENS = 300


@dataclass
class AgentCaseExtractRequest(MemoryExtractRequest):
    """Request for extracting AgentCase from a MemCell."""

    pass


class AgentCaseExtractor(MemoryExtractor):
    """
    Extracts AgentCase from an agent MemCell.

    Each MemCell produces at most one AgentCase.
    Multiple conversation turns solving the same problem are synthesized into one record.

    Pipeline:
    1. Pre-compress: Build structured list, LLM-compress tool content if over threshold
    2. Single LLM call: extract one experience record
    3. Compute embedding on task_intent for retrieval
    """

    # Heuristic: no-tool conversations meeting these thresholds are skipped
    # without an LLM filter call (saves cost for obvious non-extractable cases)
    FILTER_NO_TOOL_MAX_MESSAGES = 4
    FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS = 200

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        filter_prompt: Optional[str] = None,
        experience_compress_prompt: Optional[str] = None,
        tool_pre_compress_prompt: Optional[str] = None,
        pre_compress_chunk_size: int = PRE_COMPRESS_CHUNK_SIZE,
        max_tool_output_tokens: int = MAX_TOOL_OUTPUT_TOKENS,
        max_tool_args_tokens: int = MAX_TOOL_ARGS_TOKENS,
        max_assistant_response_tokens: int = MAX_ASSISTANT_RESPONSE_TOKENS,
    ):
        super().__init__(MemoryType.AGENT_CASE)
        self.llm_provider = llm_provider
        self.filter_prompt = filter_prompt or get_prompt_by("AGENT_CASE_FILTER_PROMPT")
        self.experience_compress_prompt = experience_compress_prompt or get_prompt_by(
            "AGENT_CASE_COMPRESS_PROMPT"
        )
        self.tool_pre_compress_prompt = tool_pre_compress_prompt or get_prompt_by(
            "AGENT_TOOL_PRE_COMPRESS_PROMPT"
        )
        self.pre_compress_chunk_size = pre_compress_chunk_size
        self.max_tool_output_tokens = max_tool_output_tokens
        self.max_tool_args_tokens = max_tool_args_tokens
        self.max_assistant_response_tokens = max_assistant_response_tokens

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """JSON encoder default for non-serializable types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)

    @classmethod
    def _get_tokenizer(cls):
        """Get the shared tokenizer from tokenizer factory."""
        tokenizer_factory: TokenizerFactory = get_bean_by_type(TokenizerFactory)
        return tokenizer_factory.get_tokenizer_from_tiktoken("o200k_base")

    @classmethod
    def _count_tokens(cls, text: str) -> int:
        """Count tokens in a string."""
        if not text:
            return 0
        tokenizer = cls._get_tokenizer()
        return len(tokenizer.encode(text))

    @classmethod
    def _calc_tool_content_size(cls, msg: Dict[str, Any]) -> int:
        """Calculate the tool-related content size of a message (in tokens)."""
        role = msg.get("role", "")
        if role == "tool":
            return cls._count_tokens(msg.get("content", ""))
        if role == "assistant" and msg.get("tool_calls"):
            return sum(
                cls._count_tokens(tc.get("function", {}).get("arguments", ""))
                for tc in msg["tool_calls"]
            )
        return 0

    @classmethod
    def _truncate_text(cls, text: str, max_tokens: int, head_ratio: float = 0.7) -> str:
        """Truncate text to max_tokens, keeping head and tail with a marker.

        When head_ratio=1.0, only the head is kept with "..." appended.
        """
        if not text or not isinstance(text, str):
            return text
        tokenizer = cls._get_tokenizer()
        tokens = tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        head_count = int(max_tokens * head_ratio)
        tail_count = max_tokens - head_count
        head_text = tokenizer.decode(tokens[:head_count])
        if tail_count <= 0:
            return head_text.rstrip() + "..."
        tail_text = tokenizer.decode(tokens[-tail_count:])
        trimmed = len(tokens) - max_tokens
        return f"{head_text}\n[... trimmed {trimmed} tokens ...]\n{tail_text}"

    @classmethod
    def _heuristic_trim_tool_outputs(
        cls,
        messages: List[Dict[str, Any]],
        max_tool_output_tokens: int,
        max_tool_args_tokens: int,
        max_assistant_response_tokens: int = MAX_ASSISTANT_RESPONSE_TOKENS,
        head_ratio: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Truncate oversized tool outputs, arguments, and assistant responses."""
        result = copy.deepcopy(messages)
        trimmed_count = 0
        for msg in result:
            role = msg.get("role", "")
            if role == "tool" and msg.get("content"):
                original = msg["content"]
                msg["content"] = cls._truncate_text(
                    original, max_tool_output_tokens, head_ratio
                )
                if msg["content"] != original:
                    trimmed_count += 1
            elif role == "assistant":
                # Trim tool call arguments
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        func = tc.get("function")
                        if not func:
                            continue
                        args = func.get("arguments", "")
                        if args:
                            new_args = cls._truncate_text(
                                args, max_tool_args_tokens, head_ratio
                            )
                            if new_args != args:
                                func["arguments"] = new_args
                                trimmed_count += 1
                # Trim assistant response content (non-tool-call messages)
                content = msg.get("content")
                if content and isinstance(content, str):
                    new_content = cls._truncate_text(
                        content, max_assistant_response_tokens, head_ratio
                    )
                    if new_content != content:
                        msg["content"] = new_content
                        trimmed_count += 1
        if trimmed_count > 0:
            logger.info(
                f"[AgentCaseExtractor] Heuristic trim: "
                f"truncated {trimmed_count} content fields"
            )
        return result

    def _collect_tool_call_groups(self, items: List[Dict[str, Any]]) -> List[List[int]]:
        """Collect atomic tool call groups from the message list.

        Each group is an assistant message with tool_calls + its corresponding
        tool response messages. These must not be split across chunks.
        """
        groups: List[List[int]] = []
        i = 0
        while i < len(items):
            msg = items[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                group = [i]
                j = i + 1
                while j < len(items) and items[j].get("role") == "tool":
                    group.append(j)
                    j += 1
                groups.append(group)
                i = j
            else:
                i += 1
        return groups

    def _calc_group_size(self, items: List[Dict[str, Any]], group: List[int]) -> int:
        """Calculate total tool content tokens of a group."""
        return sum(self._calc_tool_content_size(items[idx]) for idx in group)

    async def _pre_compress_to_list(
        self, original_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Pre-compress tool content using selective LLM compression.

        If total tool content <= pre_compress_chunk_size, return as-is.
        Otherwise, only compress the largest groups (by token size descending)
        until the estimated total drops below the threshold, leaving small
        groups uncompressed to save LLM calls.
        """
        items = copy.deepcopy(original_data)

        tool_call_groups = self._collect_tool_call_groups(items)
        if not tool_call_groups:
            return items

        # Calculate size per group
        groups_with_size = [
            (i, g, self._calc_group_size(items, g))
            for i, g in enumerate(tool_call_groups)
        ]
        total_size = sum(s for _, _, s in groups_with_size)
        if total_size <= self.pre_compress_chunk_size:
            logger.debug(
                f"[AgentCaseExtractor] Tool content {total_size} tokens "
                f"<= {self.pre_compress_chunk_size}, no compression needed"
            )
            return items

        # Select only the largest groups needed to bring total under threshold.
        # Assume ~90% reduction for compressed groups.
        groups_by_size = sorted(groups_with_size, key=lambda x: x[2], reverse=True)
        compress_indices: set = set()
        estimated_total = total_size
        for idx, group, size in groups_by_size:
            if estimated_total <= self.pre_compress_chunk_size:
                break
            compress_indices.add(idx)
            estimated_total -= size * 0.9  # estimated savings

        # Keep original order so chunk boundaries and replacement stay aligned
        groups_to_compress: List[List[int]] = [
            g for i, g in enumerate(tool_call_groups) if i in compress_indices
        ]

        logger.debug(
            f"[AgentCaseExtractor] Selective compression: "
            f"{len(groups_to_compress)}/{len(tool_call_groups)} groups, "
            f"{total_size} total tokens"
        )

        # Split selected groups into chunks of pre_compress_chunk_size
        chunks: List[List[List[int]]] = []
        current_chunk: List[List[int]] = []
        current_size = 0

        for group in groups_to_compress:
            group_size = self._calc_group_size(items, group)
            if (
                current_chunk
                and current_size + group_size > self.pre_compress_chunk_size
            ):
                chunks.append(current_chunk)
                current_chunk = [group]
                current_size = group_size
            else:
                current_chunk.append(group)
                current_size += group_size

        if current_chunk:
            chunks.append(current_chunk)

        # Build per-chunk message lists
        chunk_msg_lists: List[List[Dict[str, Any]]] = []
        for chunk_groups in chunks:
            chunk_indices = [idx for group in chunk_groups for idx in group]
            chunk_msg_lists.append([items[idx] for idx in chunk_indices])

        # Compress all chunks in parallel
        results = await asyncio.gather(
            *(self._compress_tool_chunk(chunk_msgs) for chunk_msgs in chunk_msg_lists),
            return_exceptions=True,
        )
        all_compressed: List[Dict[str, Any]] = []
        for round_idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    f"[AgentCaseExtractor] Chunk {round_idx + 1} compression error: "
                    f"{result}, keeping original messages"
                )
                all_compressed.extend(chunk_msg_lists[round_idx])
            elif result is not None:
                all_compressed.extend(result)
            else:
                logger.warning(
                    f"[AgentCaseExtractor] Chunk {round_idx + 1} compression failed, "
                    "keeping original messages"
                )
                all_compressed.extend(chunk_msg_lists[round_idx])

        # Replace only the selected groups' messages with compressed results
        selected_indices = sorted(idx for group in groups_to_compress for idx in group)

        if len(all_compressed) == len(selected_indices):
            for i, idx in enumerate(selected_indices):
                items[idx] = all_compressed[i]
        else:
            logger.warning(
                f"[AgentCaseExtractor] Compressed count {len(all_compressed)} "
                f"!= selected message count {len(selected_indices)}, keeping originals"
            )

        return items

    async def _compress_tool_chunk(
        self, messages: List[Dict[str, Any]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Compress a chunk of tool-related messages via LLM."""
        prompt = self.tool_pre_compress_prompt.format(
            messages_json=json.dumps(
                messages, ensure_ascii=False, indent=2, default=self._json_default
            ),
            new_count=len(messages),
        )

        for attempt in range(2):
            try:
                resp = await self.llm_provider.generate(prompt)
                data = parse_json_response(resp)
                if (
                    data
                    and "compressed_messages" in data
                    and isinstance(data["compressed_messages"], list)
                    and len(data["compressed_messages"]) == len(messages)
                ):
                    return data["compressed_messages"]
                logger.warning(
                    f"[AgentCaseExtractor] Tool pre-compress attempt {attempt+1}/2: "
                    f"invalid response format"
                )
            except Exception as e:
                logger.warning(
                    f"[AgentCaseExtractor] Tool pre-compress attempt {attempt+1}/2: {e}"
                )

        return None

    async def _filter_conversation(self, messages_json: str) -> bool:
        """LLM-based filter to determine if the conversation is worth extracting."""
        prompt = self.filter_prompt.format(messages=messages_json)
        try:
            resp = await self.llm_provider.generate(prompt)
            data = parse_json_response(resp)
            if data and "worth_extracting" in data:
                worth = data["worth_extracting"]
                if not worth:
                    reason = data.get("reason", "")
                    logger.info(f"[AgentCaseExtractor] Filtered out by LLM: {reason}")
                return bool(worth)
        except Exception as e:
            logger.warning(f"[AgentCaseExtractor] Filter failed: {e}")
        # Default to extracting if filter fails
        return True

    async def _compress_experience(
        self, messages_json: str
    ) -> Optional[Dict[str, Any]]:
        """Single LLM call to extract one experience with task_intent + approach + quality_score."""
        prompt = self.experience_compress_prompt.format(messages=messages_json)

        for attempt in range(2):
            try:
                resp = await self.llm_provider.generate(prompt)
                data = parse_json_response(resp)
                if data and "task_intent" in data:
                    if not data["task_intent"]:
                        logger.info(
                            "[AgentCaseExtractor] LLM returned empty 'task_intent', skipping"
                        )
                        return None
                    if not data.get("approach"):
                        logger.warning(
                            "[AgentCaseExtractor] LLM returned empty 'approach', skipping"
                        )
                        return None
                    return data
                logger.warning(
                    f"[AgentCaseExtractor] Compress attempt {attempt+1}/2: "
                    f"missing or invalid 'task_intent' field"
                )
            except Exception as e:
                logger.warning(
                    f"[AgentCaseExtractor] Compress attempt {attempt+1}/2: {e}"
                )

        logger.error(
            "[AgentCaseExtractor] Experience extraction failed after 2 attempts"
        )
        return None

    @staticmethod
    def _clamp_quality_score(value: Any) -> Optional[float]:
        """Clamp quality_score to [0.0, 1.0], return None if invalid."""
        if value is None:
            return None
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _unwrap_messages(original_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Unwrap messages from MemCell original_data items.

        MemCell.original_data items are in { "message": {...}, "parse_info": ... }
        format. This extracts the inner message dicts. Also normalizes the content
        field: v1 API uses content[] list, but downstream processing expects plain
        strings for tool/user messages.
        """
        messages = []
        for item in original_data:
            if not isinstance(item, dict):
                continue
            msg = item.get("message", item)
            # Normalize content: convert content items list to plain text string
            # for roles where downstream expects a string (user, tool).
            # Assistant messages may have tool_calls and null/empty content, leave as-is.
            content = msg.get("content")
            if isinstance(content, list):
                msg = copy.deepcopy(msg)
                msg["content"] = get_text_from_content_items(content)
            messages.append(msg)
        return messages

    @staticmethod
    def _has_tool_calls(messages: List[Dict[str, Any]]) -> bool:
        """Check if the conversation contains any tool calls or tool responses."""
        return any(
            msg.get("tool_calls") or msg.get("role") == "tool" for msg in messages
        )

    @staticmethod
    def _count_tool_call_rounds(messages: List[Dict[str, Any]]) -> int:
        """Count the number of assistant messages that contain tool_calls."""
        return sum(
            1
            for msg in messages
            if msg.get("role") == "assistant" and msg.get("tool_calls")
        )

    @staticmethod
    def _strip_before_first_user(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Drop messages before the first user message (e.g. system prompts)."""
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                return messages[i:]
        return []

    @classmethod
    def _should_skip(cls, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Pre-filter conversations that are not worth extracting.

        Combines structural checks with heuristic content checks to avoid
        a separate LLM filter call for low-tool-call conversations.
        """
        if not messages:
            return "No messages after stripping system prompts"

        if not any(msg.get("role") == "user" for msg in messages):
            return "No user messages found"

        if not any(msg.get("role") == "assistant" for msg in messages):
            return "No assistant messages found"

        last_msg = messages[-1]
        if last_msg.get("role") != "assistant" or last_msg.get("tool_calls"):
            return "Incomplete agent trajectory (last message is not a final assistant response)"

        has_tools = cls._has_tool_calls(messages)

        if not has_tools:
            user_count = sum(1 for msg in messages if msg.get("role") == "user")
            if user_count < 2:
                return "Single-turn conversation without tool calls"

            # Heuristic: no-tool conversations with very few messages are
            # unlikely to contain meaningful problem-solving — skip without
            # an LLM filter call to save cost.
            if len(messages) <= cls.FILTER_NO_TOOL_MAX_MESSAGES:
                return (
                    f"No-tool conversation with only {len(messages)} messages "
                    f"(max {cls.FILTER_NO_TOOL_MAX_MESSAGES}), skipping"
                )

            # Heuristic: no-tool conversations with very brief assistant
            # responses are likely simple Q&A or chitchat.
            assistant_content = " ".join(
                msg.get("content", "") or ""
                for msg in messages
                if msg.get("role") == "assistant" and not msg.get("tool_calls")
            )
            assistant_tokens = cls._count_tokens(assistant_content)
            if assistant_tokens < cls.FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS:
                return (
                    f"No-tool conversation with brief assistant response "
                    f"({assistant_tokens} tokens < {cls.FILTER_NO_TOOL_MIN_ASSISTANT_TOKENS}), skipping"
                )

        return None

    async def _compute_embedding(self, text: str) -> Optional[Dict[str, Any]]:
        """Compute embedding for the task intent."""
        try:
            if not text:
                return None
            vs = get_vectorize_service()
            vec = await vs.get_embedding(text)
            return {
                "embedding": vec.tolist() if hasattr(vec, "tolist") else list(vec),
                "vector_model": vs.get_model_name(),
            }
        except Exception as e:
            logger.error(f"[AgentCaseExtractor] Embedding failed: {e}")
            return None

    async def extract_memory(
        self, request: MemoryExtractRequest
    ) -> Optional[AgentCase]:
        """
        Extract AgentCase from a MemCell.

        Pipeline:
        1. Pre-compress: build structured list, LLM-compress tool content if over threshold
        2. Single LLM call: extract one experience record
        3. Compute embedding on task_intent
        """
        memcell = request.memcell
        if not memcell:
            return None

        if memcell.type != RawDataType.AGENTCONVERSATION:
            logger.warning(
                f"[AgentCaseExtractor] Expected AGENT_CONVERSATION, got {memcell.type}"
            )
            return None

        try:
            # Unwrap from { "message": ..., "parse_info": ... } format
            # and normalize content[] lists to plain strings
            raw_messages = self._unwrap_messages(memcell.original_data or [])
            original_data = self._strip_before_first_user(raw_messages)

            # Pre-filter: skip conversations not worth extracting
            skip_reason = self._should_skip(original_data)
            if skip_reason:
                logger.info(f"[AgentCaseExtractor] {skip_reason}, skipping")
                return None

            # Heuristic trim: truncate oversized tool outputs and assistant responses.
            # Start scaling when total tokens exceed half of PRE_COMPRESS_CHUNK_SIZE,
            # with limits inversely proportional to how far over the threshold we are.
            # After trim, skip entirely if still over 2x PRE_COMPRESS_CHUNK_SIZE.
            total_tokens = self._count_tokens(
                json.dumps(original_data, ensure_ascii=False, default=self._json_default)
            )
            logger.info(
                f"[AgentCaseExtractor] event_id={memcell.event_id}, "
                f"total_tokens={total_tokens}, message_count={len(original_data)}"
            )

            # High message count signals lower per-message value; use a tighter trigger
            # so trim kicks in earlier and compresses more aggressively.
            scale_trigger = (
                self.pre_compress_chunk_size // 2
                if len(original_data) > HIGH_MESSAGE_COUNT_THRESHOLD
                else self.pre_compress_chunk_size
            )
            needs_scale = total_tokens > scale_trigger
            if needs_scale:
                # Scale limits inversely proportional to how far over the trigger we are.
                # e.g. at 1x chunk_size -> scale=0.5, at 2x -> scale=0.25
                scale = scale_trigger / total_tokens
                trim_tool_output = max(200, int(self.max_tool_output_tokens * scale))
                trim_tool_args = max(200, int(self.max_tool_args_tokens * scale))
                trim_assistant = max(500, int(self.max_assistant_response_tokens * scale))
                logger.info(
                    f"[AgentCaseExtractor] Total tokens {total_tokens} > "
                    f"scale_trigger ({scale_trigger}), "
                    f"scale={scale:.2f} -> trim limits: "
                    f"tool_output={trim_tool_output}, tool_args={trim_tool_args}, "
                    f"assistant={trim_assistant}"
                )
            else:
                trim_tool_output = self.max_tool_output_tokens
                trim_tool_args = self.max_tool_args_tokens
                trim_assistant = self.max_assistant_response_tokens
            original_data = self._heuristic_trim_tool_outputs(
                original_data,
                trim_tool_output,
                trim_tool_args,
                trim_assistant,
            )

            # Only re-count after trim when scaling was applied — if total_tokens was
            # already <= scale_trigger, trimmed_tokens can't possibly exceed 2x chunk_size.
            if needs_scale:
                trimmed_tokens = self._count_tokens(
                    json.dumps(original_data, ensure_ascii=False, default=self._json_default)
                )
                if trimmed_tokens > self.pre_compress_chunk_size * 2:
                    logger.info(
                        f"[AgentCaseExtractor] Still {trimmed_tokens} tokens after trim "
                        f"(> 2x PRE_COMPRESS_CHUNK_SIZE {self.pre_compress_chunk_size * 2}), skipping extraction"
                    )
                    return None

            # Step 1: Pre-compress to JSON list (LLM-based if tool content is large)
            with timed("pre_compress"):
                pre_compressed_list = await self._pre_compress_to_list(original_data)
            messages_json = json.dumps(
                pre_compressed_list,
                ensure_ascii=False,
                indent=2,
                default=self._json_default,
            )

            logger.debug(
                f"[AgentCaseExtractor] Pre-compressed: "
                f"{len(pre_compressed_list)} items, {len(messages_json)} chars"
            )

            # Step 2: LLM filter — for conversations with only a single round
            # of tool calls (no-tool short conversations already skipped by
            # heuristic in _should_skip)
            tool_rounds = self._count_tool_call_rounds(original_data)
            if tool_rounds <= 1:
                with timed("filter_conversation"):
                    if not await self._filter_conversation(messages_json):
                        return None

            # Step 3: Single LLM call — returns experience dict or None
            with timed("compress_experience"):
                exp_dict = await self._compress_experience(messages_json)

            if not exp_dict:
                logger.info(
                    "[AgentCaseExtractor] No actionable experience extracted, skipping"
                )
                return None

            # Truncate task_intent to hard token cap (head only)
            original_intent = exp_dict.get("task_intent", "")
            raw_intent = self._truncate_text(
                original_intent, MAX_TASK_INTENT_TOKENS, head_ratio=1.0
            )
            if raw_intent != original_intent:
                logger.info(
                    f"[AgentCaseExtractor] Truncated task_intent to "
                    f"{MAX_TASK_INTENT_TOKENS} tokens, "
                    f"original: {original_intent}"
                )

            # Build AgentCase
            experience = AgentCase(
                id=generate_object_id_str(),
                memory_type=MemoryType.AGENT_CASE,
                user_id=request.user_id or "",
                timestamp=memcell.timestamp,
                group_id=request.group_id,
                participants=memcell.participants,
                sender_ids=memcell.sender_ids,
                task_intent=raw_intent,
                approach=exp_dict.get("approach", ""),
                quality_score=self._clamp_quality_score(
                    exp_dict.get("quality_score", 0.5)
                ),
                key_insight=exp_dict.get("key_insight", ""),
            )

            # Step 4: Compute embedding on task_intent for retrieval
            embedding_data = await self._compute_embedding(experience.task_intent)
            if embedding_data:
                experience.vector = embedding_data["embedding"]
                experience.vector_model = embedding_data["vector_model"]

            logger.debug(
                f"[AgentCaseExtractor] Extracted: "
                f"intent='{experience.task_intent[:80]}'"
            )

            return experience

        except Exception as e:
            logger.error(f"[AgentCaseExtractor] Extraction failed: {e}")
            return None
