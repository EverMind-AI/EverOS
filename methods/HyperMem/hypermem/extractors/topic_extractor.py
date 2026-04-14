"""
Topic Extractor (Retrieval-based) - Four-stage extraction pipeline

Core concept: A topic represents "the same scenario", not a simple similarity aggregation.

Stage 1: Episode similarity detection (retrieval-based)
- Uses retrieval algorithms (BM25/Embedding) to find historical episodes similar to the new episode
- Output: List of similar episodes (with scores)

Stage 2: Topic similarity detection (retrieval-based)
- Uses retrieval algorithms to find existing topics related to the new episode
- Output: List of similar topics (with scores)

Stage 3: Topic extraction/update (using LLM)
- Case 1: No similar episodes -> Create a new topic
- Case 2.1: Similar episodes found, no similar topics -> Create a new topic
- Case 2.2: Similar episodes found, similar topics found -> Update the topic

Stage 4: Episode role and weight assignment (using LLM)
- Assign roles and importance weights to each episode in the topic
- Build episode hyperedge (EpisodeHyperedge)
"""

import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

from hypermem.utils.logger import get_logger
from hypermem.llm.llm_provider import LLMProvider
from hypermem.types import Episode, Topic
from hypermem.structure import EpisodeHyperedge, EpisodeRole
from hypermem.prompts.topic_prompts import (
    TOPIC_EXTRACTION_PROMPT,
    TOPIC_UPDATE_PROMPT,
    EPISODE_ROLE_WEIGHT_ASSIGNMENT_PROMPT,
    TOPIC_MATCH_PROMPT
)

logger = get_logger(__name__)


@dataclass
class SimilarEpisode:
    """Similar episode information"""
    episode_id: str
    similarity_score: float  # Similarity score (0.0-1.0)
    reasoning: str  # Reason for similarity (retrieval method name)


@dataclass
class SimilarTopic:
    """Similar topic information"""
    topic_id: str
    similarity_score: float  # Similarity score (0.0-1.0)
    reasoning: str  # Reason for similarity (retrieval method name)


@dataclass
class SimilarEpisodeResult:
    """Stage 1 output: Episode similarity detection result"""
    has_similar: bool  # Whether there are similar episodes
    similar_episodes: List[SimilarEpisode]  # List of similar episodes
    reasoning: str  # Overall reasoning explanation

    def get_episode_ids(self) -> List[str]:
        """Get list of similar episode IDs"""
        return [mc.episode_id for mc in self.similar_episodes]


@dataclass
class SimilarTopicResult:
    """Stage 2 output: Topic similarity detection result"""
    has_similar: bool  # Whether there are similar topics
    similar_topics: List[SimilarTopic]  # List of similar topics
    reasoning: str  # Overall reasoning explanation

    def get_topic_ids(self) -> List[str]:
        """Get list of similar topic IDs"""
        return [s.topic_id for s in self.similar_topics]


@dataclass
class TopicExtractRequest:
    """Topic extraction request"""
    history_episode_list: List[Episode]
    new_episode: Episode
    existing_topics: Optional[List[Topic]] = None


@dataclass
class TopicExtractResult:
    """Topic extraction result (Stage 3 output)"""
    topics: List[Topic]  # List of topics
    action: str  # "create_new", "update_existing"

    # Stage 1 result: Episode similarity detection (retrieval)
    similar_episode_result: Optional[SimilarEpisodeResult] = None
    
    # Stage 2 result: Topic similarity detection
    similar_topic_result: Optional[SimilarTopicResult] = None


@dataclass
class EpisodeRoleWeightAssignmentResult:
    """Stage 4 output: Episode role and weight assignment result"""
    episode_roles: Dict[str, EpisodeRole]  # episode_id -> role
    episode_weights: Dict[str, float]  # episode_id -> weight
    coherence_score: float  # Topic coherence score
    reasoning: str


@dataclass
class EpisodeHyperedgeExtractResult:
    """Episode hyperedge extraction final result"""
    topic_id: str
    episode_hyperedge: Optional[EpisodeHyperedge]  # Episode hyperedge, None when extraction fails

    # Stage result
    role_weight_result: Optional[EpisodeRoleWeightAssignmentResult] = None

    # Statistics
    episode_count: int = 0

    def __post_init__(self):
        """Compute statistics"""
        # Episode count is calculated from the number of connections in the hyperedge
        if self.episode_hyperedge:
            self.episode_count = len(self.episode_hyperedge.relation)
        else:
            self.episode_count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "topic_id": self.topic_id,
            "episode_count": self.episode_count,
            "episode_hyperedge": self.episode_hyperedge.model_dump() if self.episode_hyperedge else None
        }


class TopicExtractor:
    """
    Topic Extractor — Four-stage extraction pipeline using retrieval-augmented topic detection.
    """

    def __init__(
        self,
        llm_provider=LLMProvider,
        topic_match_batch_size: int = 10,  # Max topics per LLM matching call
    ):
        """
        Initialize the topic extractor (LLM-based matching)

        Args:
            llm_provider: LLM provider
            topic_match_batch_size: Max number of topics per LLM matching call (batched if more)
        """
        self.llm_provider = llm_provider
        self.topic_match_batch_size = topic_match_batch_size
    
    def _format_episode_display(self, episode: Episode, simple_id: str = None) -> str:
        """Format a single episode (for LLM display). If simple_id provided, use it instead of UUID."""
        lines = []
        lines.append(f"Episode ID: {simple_id if simple_id else episode.event_id}")

        if episode.participants:
            lines.append(f"Participants: {', '.join(episode.participants)}")

        if episode.subject:
            lines.append(f"Content: {episode.subject}")
        elif episode.summary:
            lines.append(f"Content: {episode.summary}")

        if episode.timestamp:
            lines.append(f"Timestamp: {episode.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        if episode.keywords:
            keywords_display = ', '.join(episode.keywords[:8])
            lines.append(f"Keywords: {keywords_display}")

        if episode.episode_description:
            lines.append(f"Episode: {episode.episode_description}")
        
        return "\n".join(lines)
    
    def _format_episode_list(self, episode_list: List[Episode], use_simple_ids: bool = False) -> str:
        """Format a list of episodes. If use_simple_ids, show episode_1/episode_2 instead of UUIDs."""
        return "\n\n".join([
            f"--- Episode {i+1} ---\n{self._format_episode_display(mc, simple_id=f'episode_{i+1}' if use_simple_ids else None)}"
            for i, mc in enumerate(episode_list)
        ])
    
    def _format_topic_display(self, topic: Topic) -> str:
        """Format a single topic (for LLM display)"""
        lines = []
        lines.append(f"Topic ID: {topic.topic_id}")
        lines.append(f"Title: {topic.title}")

        if topic.timestamp:
            lines.append(f"Last Updated: {topic.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        if topic.participants:
            lines.append(f"Participants: {', '.join(topic.participants)}")

        lines.append(f"Summary: {topic.summary}")

        episode_count = len(topic.episode_ids)
        lines.append(f"Episode Count: {episode_count}")
        if episode_count > 0:
            if episode_count <= 5:
                lines.append(f"Episode IDs: {', '.join(topic.episode_ids)}")
            else:
                first_five = ', '.join(topic.episode_ids[:5])
                lines.append(f"Episode IDs (first 5): {first_five}, ... (+{episode_count - 5} more)")

        if topic.keywords:
            lines.append(f"Keywords: {', '.join(topic.keywords)}")

        return "\n".join(lines)
    
    def _format_topic_list(self, topic_list: List[Topic]) -> str:
        """Format a list of topics"""
        return "\n\n".join([
            f"--- Topic {i+1} ---\n{self._format_topic_display(s)}"
            for i, s in enumerate(topic_list)
        ])
    
    def _validate_new_topic_extraction(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate the correctness of new topic extraction results"""
        errors = []
        
        if "title" not in data:
            errors.append("Missing required field 'title'")
        elif not data.get("title"):
            errors.append("Field 'title' cannot be empty")
        
        if "summary" not in data:
            errors.append("Missing required field 'summary'")
        elif not data.get("summary"):
            errors.append("Field 'summary' cannot be empty")
        
        if "keywords" in data and not isinstance(data.get("keywords"), list):
            errors.append("Field 'keywords' must be a list")
        
        return len(errors) == 0, errors
    
    async def _extract_new_topic(
        self,
        episode_list: List[Episode]
    ) -> Optional[Topic]:
        """
        Stage 3: Create a new topic (with retry mechanism)
        """
        episodes_text = self._format_episode_list(episode_list)

        logger.info(f"[Stage3] Creating new topic - episode count: {len(episode_list)}")

        prompt = TOPIC_EXTRACTION_PROMPT.format(
            episodes=episodes_text,
            episode_count=len(episode_list)
        )
        
        print("\n" + "="*80)
        print("[Stage 3] LLM Input - Create New Topic")
        print("="*80)
        max_display_length = 1000
        if len(prompt) > max_display_length:
            print(prompt[:max_display_length])
            print(f"\n... (truncated, total length: {len(prompt)} characters)")
        else:
            print(prompt)
        print("="*80)
        
        attempt = 0
        last_feedback = None

        while True:
            try:
                attempt += 1

                # Build current prompt: original prompt + optional validation feedback
                if last_feedback:
                    current_prompt = prompt + f"\n\n[IMPORTANT] Previous attempt failed with the following errors, please fix:\n{last_feedback}"
                else:
                    current_prompt = prompt

                resp = await self.llm_provider.generate(
                    current_prompt,
                    response_format={"type": "json_object"}
                )

                print("\n" + "="*80)
                print(f"[Stage 3] LLM Output (attempt {attempt})")
                print("="*80)
                print(resp)
                print("="*80)

                # JSON mode should guarantee valid JSON; use json_repair as fallback
                try:
                    data = json.loads(resp)
                except json.JSONDecodeError:
                    if HAS_JSON_REPAIR:
                        data = json_repair.loads(resp)
                    else:
                        raise

                is_valid, validation_errors = self._validate_new_topic_extraction(data)

                if not is_valid:
                    error_msg = "New topic extraction validation errors:\n" + "\n".join(validation_errors)
                    raise ValueError(error_msg)

                import uuid
                topic_id_val = f"topic_{str(uuid.uuid4())}"

                episode_ids = [mc.event_id for mc in episode_list]

                user_id_set = set()
                participant_set = set()
                for mc in episode_list:
                    user_id_set.update(mc.user_id_list)
                    if mc.participants:
                        participant_set.update(mc.participants)

                last_episode = episode_list[-1]
                timestamp = last_episode.timestamp
                if isinstance(timestamp, int):
                    timestamp = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                topic = Topic(
                    topic_id=topic_id_val,
                    title=data.get("title", ""),
                    summary=data.get("summary", ""),
                    episode_ids=episode_ids,
                    timestamp=timestamp,
                    user_id_list=list(user_id_set),
                    participants=list(participant_set) if participant_set else None,
                    keywords=data.get("keywords", [])
                )

                logger.info(f"[Stage3] Created new topic: {topic.title} (attempt {attempt})")
                print(f"  ✓ New topic extraction validation passed (attempt {attempt})")

                return topic
                
            except (json.JSONDecodeError, ValueError, Exception) as e:
                print(f"  Attempt {attempt} failed: {type(e).__name__}")
                print(f"  Error details: {str(e)}")

                if isinstance(e, json.JSONDecodeError):
                    last_feedback = f"JSON parsing failed. Please provide valid JSON format with the following structure:\n{{\n  \"title\": \"...\",\n  \"summary\": \"...\",\n  \"keywords\": [...]\n}}"
                elif isinstance(e, ValueError) and "validation errors" in str(e):
                    last_feedback = str(e) + "\n\nPlease ensure:\n1. 'title' is present and not empty\n2. 'summary' is present and not empty\n3. 'keywords' is optional but must be a list if provided"
                else:
                    last_feedback = f"Processing error: {str(e)}\n\nPlease check the output format and retry."

    def _validate_topic_update(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate the correctness of topic update results"""
        errors = []
        
        if "title" not in data:
            errors.append("Missing required field 'title'")
        elif not data.get("title"):
            errors.append("Field 'title' cannot be empty")
        
        if "summary" not in data:
            errors.append("Missing required field 'summary'")
        elif not data.get("summary"):
            errors.append("Field 'summary' cannot be empty")
        
        if "keywords" in data and not isinstance(data.get("keywords"), list):
            errors.append("Field 'keywords' must be a list")
        
        return len(errors) == 0, errors
    
    async def _update_existing_topic(
        self,
        topic: Topic,
        new_episode: Episode
    ) -> Optional[Topic]:
        """Stage 3: Update an existing topic (with retry mechanism)"""
        topic_text = self._format_topic_display(topic)
        episode_text = self._format_episode_display(new_episode)

        logger.info(f"[Stage3] Updating topic: {topic.topic_id}")

        prompt = TOPIC_UPDATE_PROMPT.format(
            existing_topic=topic_text,
            new_episode=episode_text
        )

        print("\n" + "="*80)
        print(f"[Stage 3] LLM Input - Update Topic {topic.topic_id}")
        print("="*80)
        max_display_length = 1000
        if len(prompt) > max_display_length:
            print(prompt[:max_display_length])
            print(f"\n... (truncated, total length: {len(prompt)} characters)")
        else:
            print(prompt)
        print("="*80)
        
        attempt = 0
        last_feedback = None

        while True:
            try:
                attempt += 1

                # Build current prompt: original prompt + optional validation feedback
                if last_feedback:
                    current_prompt = prompt + f"\n\n[IMPORTANT] Previous attempt failed with the following errors, please fix:\n{last_feedback}"
                else:
                    current_prompt = prompt

                resp = await self.llm_provider.generate(
                    current_prompt,
                    response_format={"type": "json_object"}
                )

                print("\n" + "="*80)
                print(f"[Stage 3] LLM Output - Topic {topic.topic_id} (attempt {attempt})")
                print("="*80)
                print(resp)
                print("="*80)

                # JSON mode should guarantee valid JSON; use json_repair as fallback
                try:
                    data = json.loads(resp)
                except json.JSONDecodeError:
                    if HAS_JSON_REPAIR:
                        data = json_repair.loads(resp)
                    else:
                        raise
                
                is_valid, validation_errors = self._validate_topic_update(data)
                
                if not is_valid:
                    error_msg = "Topic update validation errors:\n" + "\n".join(validation_errors)
                    raise ValueError(error_msg)
                
                updated_episode_ids = topic.episode_ids.copy()
                if new_episode.event_id not in updated_episode_ids:
                    updated_episode_ids.append(new_episode.event_id)

                user_id_set = set(topic.user_id_list)
                user_id_set.update(new_episode.user_id_list)

                participant_set = set(topic.participants) if topic.participants else set()
                if new_episode.participants:
                    participant_set.update(new_episode.participants)

                timestamp = new_episode.timestamp
                if isinstance(timestamp, int):
                    timestamp = datetime.fromtimestamp(timestamp)
                elif isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

                updated_topic = Topic(
                    topic_id=topic.topic_id,
                    title=data.get("title", topic.title),
                    summary=data.get("summary", topic.summary),
                    episode_ids=updated_episode_ids,
                    timestamp=timestamp,
                    user_id_list=list(user_id_set),
                    participants=list(participant_set) if participant_set else None,
                    keywords=data.get("keywords", topic.keywords)
                )

                logger.info(f"[Stage3] Updated topic: {updated_topic.title} (attempt {attempt})")
                print(f"  ✓ Topic update validation passed (attempt {attempt})")

                return updated_topic
                
            except (json.JSONDecodeError, ValueError, Exception) as e:
                print(f"  Attempt {attempt} failed: {type(e).__name__}")
                print(f"  Error details: {str(e)}")

                if isinstance(e, json.JSONDecodeError):
                    last_feedback = f"JSON parsing failed. Please provide valid JSON format with the following structure:\n{{\n  \"title\": \"...\",\n  \"summary\": \"...\",\n  \"keywords\": [...]\n}}"
                elif isinstance(e, ValueError) and "validation errors" in str(e):
                    last_feedback = str(e) + "\n\nPlease ensure:\n1. 'title' is present and not empty\n2. 'summary' is present and not empty\n3. 'keywords' is optional but must be a list if provided"
                else:
                    last_feedback = f"Processing error: {str(e)}\n\nPlease check the output format and retry."
    
    async def _build_episode_hyperedge(
        self,
        topic: Topic,
        episodes: List[Episode]
    ) -> EpisodeHyperedgeExtractResult:
        """Build episode hyperedge (includes Stage 4)"""
        if not episodes:
            logger.warning("[Stage4] No episodes, returning empty hyperedge")
            return EpisodeHyperedgeExtractResult(
                topic_id=topic.topic_id,
                episode_hyperedge=None
            )

        role_weight_result = await self._assign_episode_roles_and_weights(topic, episodes)

        topic_uuid = topic.topic_id.replace("topic_", "") if topic.topic_id.startswith("topic_") else topic.topic_id
        hyperedge_id = f"episode_hyperedge_{topic_uuid}"
        
        relation_map = {}
        weights_map = {}
        
        for ep in episodes:
            episode_id = ep.event_id
            role = role_weight_result.episode_roles.get(episode_id, EpisodeRole.DEVELOPING)
            weight = role_weight_result.episode_weights.get(episode_id, 0.5)

            relation_map[episode_id] = role.value
            weights_map[episode_id] = weight

        episode_hyperedge = EpisodeHyperedge(
            id=hyperedge_id,
            relation=relation_map,
            weights=weights_map,
            topic_node_id=topic.topic_id,
            created_at=datetime.now(),
            coherence_score=role_weight_result.coherence_score
        )
        
        hyperedge_result = EpisodeHyperedgeExtractResult(
            topic_id=topic.topic_id,
            episode_hyperedge=episode_hyperedge,
            role_weight_result=role_weight_result
        )
        
        logger.info(f"[Stage4] Completed hyperedge construction - episode count: {len(episodes)}, hyperedge ID: {hyperedge_id}")
        
        return hyperedge_result
    
    def _validate_episode_role_weight_assignment(
        self,
        data: Dict[str, Any],
        episodes: List[Episode]
    ) -> tuple[bool, List[str]]:
        """Validate the correctness of episode role and weight assignment results"""
        errors = []
        
        if "episode_roles" not in data:
            errors.append("Missing required field 'episode_roles'")
            return False, errors
        
        if "coherence_score" not in data:
            errors.append("Missing required field 'coherence_score'")
        
        if "reasoning" not in data:
            errors.append("Missing required field 'reasoning'")
        
        episode_roles_list = data.get("episode_roles", [])
        
        if not isinstance(episode_roles_list, list):
            errors.append("'episode_roles' must be a list")
            return False, errors
        
        episode_ids = {mc.event_id for mc in episodes}
        valid_roles = {role.value for role in EpisodeRole}

        assigned_episode_ids = set()
        for i, item in enumerate(episode_roles_list):
            if not isinstance(item, dict):
                errors.append(f"Episode role assignment #{i+1} must be a dict")
                continue

            episode_id = item.get("episode_id", "")
            if not episode_id:
                errors.append(f"Episode role assignment #{i+1} missing 'episode_id'")
                continue

            assigned_episode_ids.add(episode_id)

            if episode_id not in episode_ids:
                errors.append(f"episode_id '{episode_id}' not in topic's episodes (bidirectional link validation failed)")

            role = item.get("role", "")
            if not role:
                errors.append(f"Episode '{episode_id}' missing 'role' field")
            elif role not in valid_roles:
                errors.append(f"Episode '{episode_id}' has invalid role '{role}'. Valid roles: {', '.join(valid_roles)}")

            weight = item.get("weight")
            if weight is None:
                errors.append(f"Episode '{episode_id}' missing 'weight' field")
            else:
                try:
                    weight_float = float(weight)
                    if not (0.0 <= weight_float <= 1.0):
                        errors.append(f"Episode '{episode_id}' weight {weight_float} must be in [0.0, 1.0]")
                except (ValueError, TypeError):
                    errors.append(f"Episode '{episode_id}' weight '{weight}' must be numeric")

        missing_episode_ids = episode_ids - assigned_episode_ids
        if missing_episode_ids:
            errors.append(f"Following episodes missing role/weight assignment (bidirectional link incomplete): {', '.join(missing_episode_ids)}")
        
        coherence_score = data.get("coherence_score")
        if coherence_score is not None:
            try:
                score_float = float(coherence_score)
                if not (0.0 <= score_float <= 1.0):
                    errors.append(f"coherence_score {score_float} must be in [0.0, 1.0]")
            except (ValueError, TypeError):
                errors.append(f"coherence_score '{coherence_score}' must be numeric")
        
        return len(errors) == 0, errors
    
    async def _assign_episode_roles_and_weights(
        self,
        topic: Topic,
        episodes: List[Episode]
    ) -> EpisodeRoleWeightAssignmentResult:
        """Stage 4: Episode role and weight assignment (with retry mechanism)"""
        logger.info(f"[Stage4] Starting episode role and weight assignment - episode count: {len(episodes)}")

        # Build simple_id <-> real_id mapping to avoid LLM copying long UUIDs
        simple_to_real = {f"episode_{i+1}": ep.event_id for i, ep in enumerate(episodes)}

        topic_text = self._format_topic_display(topic)
        episodes_text = self._format_episode_list(episodes, use_simple_ids=True)

        prompt = EPISODE_ROLE_WEIGHT_ASSIGNMENT_PROMPT.format(
            topic_content=topic_text,
            episodes=episodes_text
        )

        print("\n" + "="*80)
        print("[Stage 4] LLM Input - Episode Role and Weight Assignment")
        print("="*80)
        max_display_length = 1000
        if len(prompt) > max_display_length:
            print(prompt[:max_display_length])
            print(f"\n... (truncated, total length: {len(prompt)} characters)")
        else:
            print(prompt)
        print("="*80)
        
        attempt = 0
        last_feedback = None

        while True:
            try:
                attempt += 1

                # Build current prompt: original prompt + optional validation feedback
                if last_feedback:
                    current_prompt = prompt + f"\n\n[IMPORTANT] Previous attempt failed with the following errors, please fix:\n{last_feedback}"
                else:
                    current_prompt = prompt

                resp = await self.llm_provider.generate(
                    current_prompt,
                    response_format={"type": "json_object"}
                )

                print("\n" + "="*80)
                print(f"[Stage 4] LLM Output (attempt {attempt})")
                print("="*80)
                print(resp)
                print("="*80)

                # JSON mode should guarantee valid JSON; use json_repair as fallback
                try:
                    data = json.loads(resp)
                except json.JSONDecodeError:
                    if HAS_JSON_REPAIR:
                        data = json_repair.loads(resp)
                    else:
                        raise
                
                episode_roles = {}
                episode_weights = {}

                invalid_ids = []
                for item in data.get("episode_roles", []):
                    simple_id = item.get("episode_id", "")
                    # Map simple_id (episode_1, episode_2, ...) back to real UUID
                    real_id = simple_to_real.get(simple_id)
                    if not real_id:
                        invalid_ids.append(simple_id)
                        continue
                    role_str = item.get("role", "developing")
                    weight = item.get("weight", 0.5)

                    try:
                        role = EpisodeRole(role_str)
                    except ValueError:
                        logger.warning(f"[Stage4] Unknown role type: {role_str}, using default value developing")
                        role = EpisodeRole.DEVELOPING

                    episode_roles[real_id] = role
                    episode_weights[real_id] = float(weight)

                # Strict validation: all episodes must be assigned, no invalid IDs
                errors = []
                if invalid_ids:
                    errors.append(f"Invalid episode IDs: {', '.join(invalid_ids)}. Valid IDs are: {', '.join(simple_to_real.keys())}")
                missing = [f"episode_{i+1}" for i, ep in enumerate(episodes) if ep.event_id not in episode_roles]
                if missing:
                    errors.append(f"Missing role/weight assignment for: {', '.join(missing)}")
                if errors:
                    raise ValueError("Episode role/weight assignment validation errors:\n" + "\n".join(errors))

                coherence_score = float(data.get("coherence_score", 0.8))
                reasoning = data.get("reasoning", "")

                logger.info(f"[Stage4] Completed role and weight assignment - {len(episode_roles)} episodes (attempt {attempt})")
                print(f"  ✓ Episode role/weight assignment validation passed (including bidirectional link verification) (attempt {attempt})")

                return EpisodeRoleWeightAssignmentResult(
                    episode_roles=episode_roles,
                    episode_weights=episode_weights,
                    coherence_score=coherence_score,
                    reasoning=reasoning
                )
                
            except (json.JSONDecodeError, ValueError, Exception) as e:
                print(f"  Attempt {attempt} failed: {type(e).__name__}")
                print(f"  Error details: {str(e)}")

                valid_roles_str = ', '.join([role.value for role in EpisodeRole])
                if isinstance(e, json.JSONDecodeError):
                    last_feedback = f"JSON parsing failed. Please provide valid JSON format with the following structure:\n{{\n  \"episode_roles\": [...],\n  \"coherence_score\": 0.0-1.0,\n  \"reasoning\": \"...\"\n}}"
                elif isinstance(e, ValueError) and "validation errors" in str(e):
                    last_feedback = str(e) + f"\n\nPlease ensure:\n1. 'episode_roles' is a list\n2. Must assign roles/weights to all episodes (bidirectional link integrity)\n3. Each item contains: episode_id, role, weight\n4. role must be one of: {valid_roles_str}\n5. weight must be in [0.0, 1.0]\n6. coherence_score must be in [0.0, 1.0]\n\nEpisode IDs that need assignment: {', '.join(simple_to_real.keys())}"
                else:
                    last_feedback = f"Processing error: {str(e)}\n\nPlease check the output format and retry."
    
    async def _llm_match_topics_batch(
        self,
        episode: Episode,
        topics_batch: List[Topic],
        batch_id_offset: int = 0
    ) -> List[str]:
        """Match an episode against a batch of topics using LLM.

        Returns list of matched topic_ids.
        """
        ep_subject = episode.subject or ""
        ep_summary = episode.summary or ""

        # Build topics text with simple IDs
        topic_lines = []
        simple_to_real = {}
        for i, t in enumerate(topics_batch):
            simple_id = f"topic_{batch_id_offset + i + 1}"
            simple_to_real[simple_id] = t.topic_id
            topic_lines.append(f"- {simple_id}: {t.title}\n  Summary: {t.summary}")

        topics_text = "\n".join(topic_lines)

        prompt = TOPIC_MATCH_PROMPT.format(
            episode_subject=ep_subject,
            episode_summary=ep_summary,
            num_topics=len(topics_batch),
            topics_text=topics_text
        )

        resp = await self.llm_provider.generate(prompt, response_format={"type": "json_object"})

        try:
            data = json.loads(resp)
        except json.JSONDecodeError:
            if HAS_JSON_REPAIR:
                data = json_repair.loads(resp)
            else:
                raise

        matched_ids = []
        for item in data.get("results", []):
            if item.get("match") is True:
                simple_id = item.get("topic_id", "")
                real_id = simple_to_real.get(simple_id)
                if real_id:
                    matched_ids.append(real_id)
        return matched_ids

    async def _llm_match_topics(
        self,
        episode: Episode,
        existing_topics: List[Topic]
    ) -> List[str]:
        """Match an episode against all existing topics using LLM, with batching.

        Returns list of matched topic_ids.
        """
        batch_size = self.topic_match_batch_size
        all_matched_ids = []

        for i in range(0, len(existing_topics), batch_size):
            batch = existing_topics[i:i + batch_size]
            matched = await self._llm_match_topics_batch(episode, batch, batch_id_offset=i)
            all_matched_ids.extend(matched)

        return all_matched_ids

    async def extract_topic(
        self,
        request: TopicExtractRequest
    ) -> tuple[Optional[TopicExtractResult], List[EpisodeHyperedgeExtractResult]]:
        """
        Topic extraction pipeline: LLM-based topic matching → create or update.

        Args:
            request: Topic extraction request

        Returns:
            (TopicExtractResult, List[EpisodeHyperedgeExtractResult]) tuple
        """
        logger.info("[TopicExtractor] Starting topic extraction")

        # No existing topics → create new
        if not request.existing_topics:
            logger.info("No existing topics, creating a new topic")
            topic = await self._extract_new_topic([request.new_episode])
            if topic:
                topic_result = TopicExtractResult(
                    topics=[topic],
                    action="create_new",
                    similar_episode_result=None,
                    similar_topic_result=None
                )
                hyperedge = await self._build_episode_hyperedge(
                    topic=topic, episodes=[request.new_episode]
                )
                return topic_result, [hyperedge]
            return None, []

        # LLM-based matching against existing topics
        matched_topic_ids = await self._llm_match_topics(
            episode=request.new_episode,
            existing_topics=request.existing_topics
        )
        logger.info(f"LLM matched {len(matched_topic_ids)} topics: {matched_topic_ids}")
        print(f"  [Topic Match] Episode matched {len(matched_topic_ids)}/{len(request.existing_topics)} topics")

        # No match → create new topic
        if not matched_topic_ids:
            logger.info("No matching topics, creating a new topic")
            topic = await self._extract_new_topic([request.new_episode])
            if topic:
                topic_result = TopicExtractResult(
                    topics=[topic],
                    action="create_new",
                    similar_episode_result=None,
                    similar_topic_result=None
                )
                hyperedge = await self._build_episode_hyperedge(
                    topic=topic, episodes=[request.new_episode]
                )
                return topic_result, [hyperedge]
            return None, []

        # Matched → update all matched topics
        matched_topics = [t for t in request.existing_topics if t.topic_id in matched_topic_ids]
        logger.info(f"Updating {len(matched_topics)} matched topics")

        updated_topics = []
        for topic_to_update in matched_topics:
            updated_topic = await self._update_existing_topic(
                topic=topic_to_update,
                new_episode=request.new_episode
            )
            if updated_topic:
                updated_topics.append(updated_topic)
                logger.info(f"  Updated topic: {updated_topic.title}")

        if updated_topics:
            topic_result = TopicExtractResult(
                topics=updated_topics,
                action="update_existing",
                similar_episode_result=None,
                similar_topic_result=None
            )

            hyperedge_results = []
            for topic in updated_topics:
                topic_episodes = [
                    mc for mc in request.history_episode_list
                    if mc.event_id in topic.episode_ids
                ]
                if request.new_episode.event_id in topic.episode_ids:
                    if request.new_episode.event_id not in [mc.event_id for mc in topic_episodes]:
                        topic_episodes.append(request.new_episode)

                hyperedge = await self._build_episode_hyperedge(
                    topic=topic, episodes=topic_episodes
                )
                hyperedge_results.append(hyperedge)

            return topic_result, hyperedge_results

        return None, []


