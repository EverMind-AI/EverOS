"""
Simple Memory Extraction Base Class for EverMemOS

This module provides a simple base class for extracting memories
from boundary detection results (BoundaryResult).
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import re, json, asyncio, uuid


from memory_layer.prompts import get_prompt_by
from memory_layer.llm.llm_provider import LLMProvider

from memory_layer.memory_extractor.base_memory_extractor import MemoryExtractor, MemoryExtractRequest
from api_specs.memory_types import MemoryType, EpisodeMemory, RawDataType, MemCell

from common_utils.datetime_utils import get_now_with_timezone
from agentic_layer.vectorize_service import get_vectorize_service

from core.observation.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EpisodeMemoryExtractRequest(MemoryExtractRequest):
    """Episode extraction request (inherited from base class)"""

    pass


class EpisodeMemoryExtractor(MemoryExtractor):
    """
    Episode memory extractor - responsible only for extracting Episodes from MemCell

    Responsibilities:
    1. Extract group Episodes from MemCell's original_data
    2. Extract personal Episodes from MemCell's original_data

    Not included:
    - Foresight extraction (handled by ForesightExtractor)
    - EventLog extraction (handled by EventLogExtractor)
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        episode_prompt: Optional[str] = None,
        group_episode_prompt: Optional[str] = None,
        custom_instructions: Optional[str] = None,
    ):
        """
        Initialize Episode Extractor
        
        Args:
            llm_provider: LLM provider
            episode_prompt: Optional custom personal Episode prompt (uses default if not provided)
            group_episode_prompt: Optional custom group Episode prompt (uses default if not provided)
            custom_instructions: Optional custom instructions (uses default if not provided)
        """
        super().__init__(MemoryType.EPISODIC_MEMORY)
        self.llm_provider = llm_provider
        
        # Use custom prompts or get default via PromptManager
        self.episode_generation_prompt = episode_prompt or get_prompt_by("EPISODE_GENERATION_PROMPT")
        self.group_episode_generation_prompt = group_episode_prompt or get_prompt_by("GROUP_EPISODE_GENERATION_PROMPT")
        self.default_custom_instructions = custom_instructions or get_prompt_by("DEFAULT_CUSTOM_INSTRUCTIONS")

    def _parse_timestamp(self, timestamp) -> datetime:
        """
        Parse timestamp into datetime object
        Supports multiple formats: numeric timestamp, ISO format string, numeric string, etc.
        """
        if isinstance(timestamp, datetime):
            return timestamp
        elif isinstance(timestamp, (int, float)):
            return datetime.fromtimestamp(timestamp)
        elif isinstance(timestamp, str):
            # Handle string timestamps (could be ISO format or timestamp string)
            try:
                if timestamp.isdigit():
                    return datetime.fromtimestamp(int(timestamp))
                else:
                    # Try parsing as ISO format
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                # Fallback to current time if parsing fails
                logger.error(f"Failed to parse timestamp: {timestamp}")
                return get_now_with_timezone()
        else:
            # Unknown format, fallback to current time
            logger.error(f"Failed to parse timestamp: {timestamp}")
            return get_now_with_timezone()

    def _format_timestamp(self, dt: datetime) -> str:
        """
        Format datetime into a human-readable string
        """
        weekday = dt.strftime("%A")  # Monday, Tuesday, etc.
        month_day = dt.strftime("%B %d, %Y")  # March 14, 2024
        time_of_day = dt.strftime("%I:%M %p")  # 3:00 PM
        return f"{month_day} ({weekday}) at {time_of_day} UTC"

    def get_conversation_text(self, data_list):
        lines = []
        for data in data_list:
            # Handle both RawData objects and dict objects
            if hasattr(data, 'content'):
                # RawData object
                speaker = data.content.get('speaker_name') or data.content.get(
                    'sender', 'Unknown'
                )
                content = data.content['content']
                timestamp = data.content['timestamp']
            else:
                # Dict object
                speaker = data.get('speaker_name') or data.get('sender', 'Unknown')
                content = data['content']
                timestamp = data['timestamp']

            if timestamp:
                lines.append(f"[{timestamp}] {speaker}: {content}")
            else:
                lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def get_conversation_json_text(self, data_list):
        lines = []
        for data in data_list:
            # Handle both RawData objects and dict objects
            if hasattr(data, 'content'):
                # RawData object
                speaker = data.content.get('speaker_name') or data.content.get(
                    'sender', 'Unknown'
                )
                content = data.content['content']
                timestamp = data.content['timestamp']
            else:
                # Dict object
                speaker = data.get('speaker_name') or data.get('sender', 'Unknown')
                content = data['content']
                timestamp = data['timestamp']

            if timestamp:
                lines.append(
                    f"""
                {{
                    "timestamp": {timestamp},
                    "speaker": {speaker},
                    "content": {content}
                }}"""
                )
            else:
                lines.append(
                    f"""
                {{
                    "speaker": {speaker},
                    "content": {content}
                }}"""
                )
        return "\n".join(lines)

    def get_speaker_name_map(self, data_list: List[Dict[str, Any]]) -> Dict[str, str]:
        speaker_name_map = {}
        for data in data_list:
            if hasattr(data, 'content'):
                speaker_name_map[data.content.get('speaker_id')] = data.content.get(
                    'speaker_name'
                )
            else:
                speaker_name_map[data.get('speaker_id')] = data.get('speaker_name')
        return speaker_name_map

    def _extract_participant_name_map(
        self, chat_raw_data_list: List[Dict[str, Any]]
    ) -> List[str]:
        participant_name_map = {}
        for raw_data in chat_raw_data_list:
            if 'speaker_name' in raw_data and raw_data['speaker_name']:
                participant_name_map[raw_data['speaker_id']] = raw_data['speaker_name']
            if 'referList' in raw_data and raw_data['referList']:
                for refer_item in raw_data['referList']:
                    if isinstance(refer_item, dict):
                        if 'name' in refer_item and refer_item['_id']:
                            participant_name_map[refer_item['_id']] = refer_item['name']
        return participant_name_map

    async def _extract_episode(
        self, request: EpisodeMemoryExtractRequest, use_group_prompt: bool = False
    ) -> Optional[EpisodeMemory]:
        """
        Extract Episode memory (internal method, single extraction)

        Args:
            request: Episode extraction request (contains single memcell and optional user_id)
            use_group_prompt: Whether to use group prompt
                - True: Extract group Episode (user_id=None)
                - False: Extract personal Episode (user_id from request.user_id)

        Returns:
            EpisodeMemory (contains episode field)
        """
        logger.debug(
            f"📚 Starting Episode extraction, use_group_prompt={use_group_prompt}"
        )

        memcell = request.memcell
        if not memcell:
            return None

        # Prepare conversation text
        if memcell.type == RawDataType.CONVERSATION:
            conversation_text = self.get_conversation_json_text(memcell.original_data)

            # Select prompt and parameters
            if use_group_prompt:
                prompt_template = self.group_episode_generation_prompt
                content_key = "conversation"
                time_key = "conversation_start_time"
            else:
                prompt_template = self.episode_generation_prompt
                content_key = "conversation"
                time_key = "conversation_start_time"
            default_title = "Conversation Episode"
        else:
            return None

        # Format timestamp
        start_time = self._parse_timestamp(memcell.timestamp)
        start_time_str = self._format_timestamp(start_time)

        # Build prompt parameters
        format_params = {
            time_key: start_time_str,
            content_key: conversation_text,
            "custom_instructions": self.default_custom_instructions,
        }

        # Get participant information
        participants_name_map = self.get_speaker_name_map(memcell.original_data)
        participants_name_map.update(
            self._extract_participant_name_map(memcell.original_data)
        )

        # Determine user_id and user_name
        user_id = None
        user_name = None
        if use_group_prompt:
            # Group mode: user_id is None, user_name is None
            user_id = None
            user_name = None
        else:
            # Personal mode: get from request.user_id
            if request.user_id:
                user_id = request.user_id
                user_name = participants_name_map.get(user_id, user_id)
                format_params["user_name"] = user_name

        # Call LLM (with retry)
        data = None
        for i in range(5):
            try:
                prompt = prompt_template.format(**format_params)
                response = await self.llm_provider.generate(prompt)

                # Parse JSON
                if '```json' in response:
                    start = response.find('```json') + 7
                    end = response.find('```', start)
                    if end > start:
                        json_str = response[start:end].strip()
                        data = json.loads(json_str)
                    else:
                        data = json.loads(response)
                else:
                    json_match = re.search(
                        r'\{[^{}]*"title"[^{}]*"content"[^{}]*\}', response, re.DOTALL
                    )
                    if json_match:
                        data = json.loads(json_match.group())
                    else:
                        data = json.loads(response)

                # Validate required fields: title and content must exist
                if "title" not in data or not data["title"]:
                    raise ValueError("LLM response missing title field")
                if "content" not in data or not data["content"]:
                    raise ValueError("LLM response missing content field")

                # Validation passed, exit retry loop
                break
            except Exception as e:
                logger.warning(f"Episode extraction retry {i+1}/5: {e}")
                if i == 4:
                    raise Exception("Episode memory extraction failed after 5 retries")
                continue

        # Use first 200 characters of content as default summary if summary is missing
        if "summary" not in data or not data["summary"]:
            data["summary"] = data["content"][:200]

        title = data["title"]
        content = data["content"]
        summary = data["summary"]

        # Collect participants
        participants = memcell.participants if memcell.participants else []

        # Compute Embedding
        embedding_data = await self._compute_embedding(content)

        # Create EpisodeMemory object
        episode_memory = EpisodeMemory(
            memory_type=MemoryType.EPISODIC_MEMORY,
            user_id=user_id,
            user_name=user_name,
            ori_event_id_list=[memcell.event_id],
            timestamp=start_time,
            subject=title,
            summary=summary,
            episode=content,
            group_id=request.group_id,
            participants=participants,
            type=memcell.type,
            memcell_event_id_list=[memcell.event_id],
            extend=embedding_data,  # Add embedding to extend field
        )

        logger.debug(f"✅ Episode extraction completed: subject='{title}'")
        return episode_memory

    async def extract_memory(self, request: MemoryExtractRequest) -> Optional[EpisodeMemory]:
        """
        Extract Episode memory from MemCell (implement abstract method from base class)

        Automatically determine whether to extract group or personal Episode based on request.user_id:
        - user_id=None: extract group Episode (using group prompt)
        - user_id!=None: extract personal Episode (using personal prompt, focusing on user's perspective)

        Args:
            request: Memory extraction request, containing:
                - memcell: MemCell to extract
                - user_id: User ID (None means group)
                - group_id: Group ID
                - Other optional fields

        Returns:
            EpisodeMemory: Episode memory object
                - Group Episode: user_id=None, episode contains global view of entire conversation
                - Personal Episode: user_id=<user_id>, episode contains personal view of the user
        """
        # Determine if it's a group or personal Episode
        is_group_episode = request.user_id is None

        logger.debug(
            f"[extract_memory] Extracting {'group' if is_group_episode else 'personal'} Episode, "
            f"user_id={request.user_id}, group_id={request.group_id}"
        )

        # Build EpisodeMemoryExtractRequest
        episode_request = EpisodeMemoryExtractRequest(
            memcell=request.memcell,
            user_id=request.user_id,
            group_id=request.group_id,
            group_name=request.group_name,
            participants=request.participants,
            old_memory_list=request.old_memory_list,
            user_organization=request.user_organization,
        )

        # Call internal extraction method
        return await self._extract_episode(
            request=episode_request,
            use_group_prompt=is_group_episode,  # Group uses group prompt, personal uses personal prompt
        )

    async def _compute_embedding(self, text: str) -> Optional[dict]:
        """Compute embedding for Episode text"""
        try:
            if not text:
                return None

            vs = get_vectorize_service()
            vec = await vs.get_embedding(text)

            return {
                "embedding": vec.tolist() if hasattr(vec, "tolist") else list(vec),
                "vector_model": vs.get_model_name(),  # Use unified get_model_name() method
            }
        except Exception as e:
            logger.error(f"Episode Embedding computation failed: {e}")
            return None
