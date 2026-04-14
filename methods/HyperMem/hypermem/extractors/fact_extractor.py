"""
Topic-based Fact Extractor - Two-stage Extraction Pipeline

Stage 1: Fact Extraction
- Extract key facts from a Topic and its associated Episodes
- Each fact is a complete statement that can directly answer user queries
- Supports cross-episode information merging
- Output: Fact list

Stage 2: Role and Weight Assignment
- Assign roles and importance weights to each fact within the topic
- Build fact hyperedges (FactHyperedge), connecting facts to episodes
- Output: Fact hyperedges

Methods:
- extract_facts(): Execute the complete two-stage pipeline
  Returns: (FactExtractResult, FactHyperedgeExtractResult)
"""

import re
import json
import uuid
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

# Approach 3.1: Use json_repair library for JSON repair
try:
    import json_repair
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False
    print("Warning: json_repair library is not installed. Standard json parsing will be used. Install with: pip install json-repair")

from hypermem.utils.logger import get_logger
from hypermem.llm.llm_provider import LLMProvider
from hypermem.types import Fact, Episode, Topic
from hypermem.structure import FactHyperedge, FactRole
from hypermem.prompts.fact_prompts import (
    FACT_EXTRACTION_PROMPT,
    FACT_ROLE_ASSIGNMENT_PROMPT
)

logger = get_logger(__name__)


@dataclass
class FactExtractResult:
    """Fact extraction result"""
    topic_id: str
    facts: List[Fact]
    reasoning: str = ""
    
    def __post_init__(self):
        """Compute statistics"""
        self.fact_count = len(self.facts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "topic_id": self.topic_id,
            "fact_count": self.fact_count,
            "facts": [fact.to_dict() for fact in self.facts],
            "reasoning": self.reasoning
        }
    
    def get_facts_as_text(self) -> List[str]:
        """Get text representations of all facts"""
        return [fact.to_text() for fact in self.facts]


@dataclass
class RoleAssignmentResult:
    """Role assignment result"""
    fact_roles: Dict[str, FactRole]  # fact_id -> role
    fact_weights: Dict[str, float]     # fact_id -> weight
    extraction_confidence: float
    reasoning: str


@dataclass
class FactHyperedgeExtractResult:
    """Fact hyperedge extraction result"""
    topic_id: str
    episode_id: str
    fact_hyperedge: Optional[FactHyperedge]
    
    role_assignment_result: Optional[RoleAssignmentResult] = None
    fact_count: int = 0
    
    def __post_init__(self):
        """Compute statistics"""
        if self.fact_hyperedge:
            self.fact_count = len(self.fact_hyperedge.relation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "topic_id": self.topic_id,
            "episode_id": self.episode_id,
            "fact_count": self.fact_count,
            "fact_hyperedge": self.fact_hyperedge.model_dump() if self.fact_hyperedge else None
        }


class FactExtractor:
    """
    Topic-based Fact Extractor - Two-stage Extraction Pipeline

    Design Goals:
    1. Extract facts from topics and associated episodes that can directly answer user queries
    2. Each fact is a complete statement containing sufficient context information
    3. Support cross-episode information merging (multiple episodes under the same topic may describe the same fact)
    4. Assign roles and weights to facts, and build hyperedge connections

    Usage:
    - extract_facts(): Two-stage pipeline, returns fact list and fact hyperedges
      Returns: (FactExtractResult, List[FactHyperedgeExtractResult])
    """
    
    def __init__(self, llm_provider=LLMProvider, **llm_kwargs):
        """Initialize the fact extractor"""
        self.llm_provider = llm_provider
        self.llm_kwargs = llm_kwargs
    
    def _format_topic_context(self, topic: Topic) -> Tuple[str, str, str]:
        """
        Format topic context

        Args:
            topic: Topic object

        Returns:
            (topic_id, topic_title, topic_summary)
        """
        return topic.topic_id, topic.title, topic.summary
    
    def _format_episodes_content(self, episodes: List[Episode]) -> str:
        """
        Format episode list content

        Args:
            episodes: List of Episodes

        Returns:
            Formatted text
        """
        lines = []

        for i, episode in enumerate(episodes):
            lines.append(f"--- Episode {i+1} (ID: episode_{i+1}) ---")

            # Subject
            if episode.subject:
                lines.append(f"Subject: {episode.subject}")

            # Summary
            if episode.summary:
                lines.append(f"Summary: {episode.summary}")

            # Episodic memory
            if episode.episode_description:
                lines.append(f"Episode: {episode.episode_description}")

            # Keywords
            if episode.keywords:
                lines.append(f"Keywords: {', '.join(episode.keywords)}")

            # Original data
            if episode.original_data:
                lines.append("Original Content:")
                for j, data in enumerate(episode.original_data):
                    if isinstance(data, dict):
                        if 'content' in data and 'speaker_name' in data:
                            lines.append(f"  [{j+1}] {data.get('speaker_name', 'Unknown')}: {data.get('content', '')}")
                        elif 'text' in data:
                            lines.append(f"  [{j+1}] {data.get('text', '')}")
                        else:
                            lines.append(f"  [{j+1}] {json.dumps(data, ensure_ascii=False)}")
                    else:
                        lines.append(f"  [{j+1}] {data}")

            lines.append("")  # Blank line separator

        return "\n".join(lines)
    
    def _format_facts(self, facts: List[Fact]) -> str:
        """
        Format fact list as text

        Args:
            facts: List of facts

        Returns:
            Formatted text
        """
        lines = []
        for i, fact in enumerate(facts):
            lines.append(f"--- Fact {i+1} ---")
            lines.append(f"ID: {fact.fact_id}")
            lines.append(f"Content: {fact.content}")
            lines.append(f"Episode IDs: {', '.join(fact.episode_ids)}")
            lines.append(f"Confidence: {fact.confidence}")
            if fact.temporal:
                lines.append(f"Temporal: {fact.temporal}")
            if fact.spatial:
                lines.append(f"Spatial: {fact.spatial}")
            if fact.keywords:
                lines.append(f"Keywords: {', '.join(fact.keywords)}")
            if fact.query_patterns:
                lines.append(f"Query Patterns: {fact.query_patterns}")
            lines.append("")
        return "\n".join(lines)
    
    def _get_reference_time(self, episodes: List[Episode]) -> str:
        """
        Get reference time (use the latest episode timestamp)

        Args:
            episodes: List of Episodes

        Returns:
            Reference time string
        """
        if not episodes:
            return "Not specified"

        # Find the latest timestamp
        latest_timestamp = None
        for episode in episodes:
            if episode.timestamp:
                if latest_timestamp is None or episode.timestamp > latest_timestamp:
                    latest_timestamp = episode.timestamp
        
        if latest_timestamp:
            if isinstance(latest_timestamp, datetime):
                return latest_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            return str(latest_timestamp)
        
        return "Not specified"
    
    def _validate_fact_extraction(
        self,
        data: Dict[str, Any],
        valid_episode_ids: set
    ) -> Tuple[bool, List[str]]:
        """
        Validate the correctness of fact extraction results

        Args:
            data: JSON data returned by the LLM
            valid_episode_ids: Set of valid episode IDs

        Returns:
            (is_valid, error_list)
        """
        errors = []
        
        # Check required fields
        if "facts" not in data:
            errors.append("Missing required field 'facts'")
            return False, errors

        facts = data.get("facts", [])

        # Check if facts is a list
        if not isinstance(facts, list):
            errors.append("'facts' must be a list type")
            return False, errors

        # Check that at least one fact was extracted
        if len(facts) == 0:
            errors.append("At least one fact must be extracted")
            return False, errors

        # Check required fields for each fact
        fact_ids = set()
        for i, fact in enumerate(facts):
            if not isinstance(fact, dict):
                errors.append(f"Fact #{i+1} must be a dict type")
                continue

            # Check required fields
            required_fields = ["fact_id", "content", "episode_ids"]
            for field in required_fields:
                if field not in fact:
                    errors.append(f"Fact #{i+1} is missing required field '{field}'")
                elif not fact[field]:
                    errors.append(f"Fact #{i+1} field '{field}' cannot be empty")

            # Check fact_id uniqueness
            fact_id = fact.get("fact_id", "")
            if fact_id in fact_ids:
                errors.append(f"Fact ID '{fact_id}' is duplicated")
            fact_ids.add(fact_id)

            # Check if episode_ids are valid
            episode_ids_list = fact.get("episode_ids", [])
            if not isinstance(episode_ids_list, list):
                errors.append(f"Fact '{fact_id}' 'episode_ids' must be a list type")
            else:
                for ep_id in episode_ids_list:
                    if ep_id not in valid_episode_ids:
                        errors.append(f"Fact '{fact_id}' references non-existent episode '{ep_id}'")

            # Check optional field types
            if "keywords" in fact and fact["keywords"] is not None:
                if not isinstance(fact["keywords"], list):
                    errors.append(f"Fact '{fact_id}' 'keywords' must be a list type")

            if "query_patterns" in fact and fact["query_patterns"] is not None:
                if not isinstance(fact["query_patterns"], list):
                    errors.append(f"Fact '{fact_id}' 'query_patterns' must be a list type")
        
        return len(errors) == 0, errors
    
    async def _extract_facts_stage(
        self,
        topic: Topic,
        episodes: List[Episode]
    ) -> FactExtractResult:
        """
        Stage 1: Fact extraction (with retry mechanism)

        Args:
            topic: Topic object
            episodes: Associated Episode list

        Returns:
            Fact extraction result
        """
        logger.info(f"[Stage1] Starting fact extraction - Topic: {topic.topic_id}, Episodes: {len(episodes)}")

        topic_id, topic_title, topic_summary = self._format_topic_context(topic)
        episodes_content = self._format_episodes_content(episodes)
        reference_time = self._get_reference_time(episodes)
        # Build simple_id <-> real_id mapping
        simple_to_real = {f"episode_{i+1}": ep.event_id for i, ep in enumerate(episodes)}
        valid_episode_ids = set(simple_to_real.keys())  # Validate against simple IDs

        prompt = FACT_EXTRACTION_PROMPT.format(
            topic_id=topic_id,
            topic_title=topic_title,
            topic_summary=topic_summary,
            episodes_content=episodes_content,
            reference_time=reference_time
        )
        
        # Print initial input (with length limit)
        print("\n" + "="*80)
        print("[Stage 1] LLM Input - Fact Extraction")
        print("="*80)
        max_display_length = 1500
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

                # Enable JSON Mode, forcing LLM to return valid JSON
                resp = await self.llm_provider.generate(
                    current_prompt,
                    response_format={"type": "json_object"}
                )

                # Print output
                print("\n" + "="*80)
                print(f"[Stage 1] LLM Output (attempt {attempt})")
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
                
                # Validate output correctness
                is_valid, validation_errors = self._validate_fact_extraction(data, valid_episode_ids)

                if not is_valid:
                    error_msg = "Fact extraction validation errors:\n" + "\n".join(validation_errors)
                    raise ValueError(error_msg)
                
                # Validation passed, parse fact list
                facts = []
                for item in data.get("facts", []):
                    # Generate UUID-format fact_id
                    fact_id_val = f"fact_{str(uuid.uuid4())}"

                    # Handle spatial field: LLM may return a list, needs conversion to string
                    spatial = item.get("spatial")
                    if isinstance(spatial, list):
                        spatial = ', '.join(str(s) for s in spatial) if spatial else None

                    # Handle temporal field: LLM may return a list, needs conversion to string
                    temporal = item.get("temporal")
                    if isinstance(temporal, list):
                        temporal = ', '.join(str(t) for t in temporal) if temporal else None

                    # Map simple episode IDs back to real UUIDs
                    raw_episode_ids = item.get("episode_ids", [])
                    real_episode_ids = [simple_to_real.get(eid, eid) for eid in raw_episode_ids]

                    fact = Fact(
                        fact_id=fact_id_val,
                        content=item.get("content", ""),
                        episode_ids=real_episode_ids,
                        topic_id=topic_id,
                        confidence=item.get("confidence", 0.8),
                        temporal=temporal,
                        spatial=spatial,
                        keywords=item.get("keywords", []),
                        query_patterns=item.get("query_patterns", []),
                        timestamp=datetime.now()
                    )
                    facts.append(fact)
                
                reasoning = data.get("reasoning", "")
                
                logger.info(f"[Stage1] Extracted {len(facts)} facts (attempt {attempt})")
                print(f"  OK: Fact extraction validation passed (attempt {attempt})")
                
                return FactExtractResult(
                    topic_id=topic_id,
                    facts=facts,
                    reasoning=reasoning
                )
                
            except (json.JSONDecodeError, ValueError, Exception) as e:
                print(f"  Attempt {attempt} failed: {type(e).__name__}")
                print(f"  Error details: {str(e)}")

                # Record feedback for next attempt (injected into original prompt, not accumulated)
                if isinstance(e, json.JSONDecodeError):
                    last_feedback = "JSON parsing failed. Please provide valid JSON format."
                elif isinstance(e, ValueError) and "validation errors" in str(e):
                    last_feedback = str(e) + f"\n\nAvailable episode IDs: {', '.join(valid_episode_ids)}"
                else:
                    last_feedback = f"Processing error: {str(e)}\n\nPlease check the output format and retry."

    def _validate_fact_role_assignment(
        self,
        data: Dict[str, Any],
        facts: List[Fact]
    ) -> Tuple[bool, List[str]]:
        """
        Validate the correctness of role assignment results

        Args:
            data: JSON data returned by the LLM
            facts: List of facts

        Returns:
            (is_valid, error_list)
        """
        errors = []
        
        # Check required fields
        if "fact_roles" not in data:
            errors.append("Missing required field 'fact_roles'")
            return False, errors

        # extraction_confidence is optional, defaults to 0.8 in parsing

        fact_roles_list = data.get("fact_roles", [])

        if not isinstance(fact_roles_list, list):
            errors.append("'fact_roles' must be a list type")
            return False, errors

        # Build fact ID set
        fact_ids = {fact.fact_id for fact in facts}

        # Valid role enum values
        valid_roles = {role.value for role in FactRole}

        # Check each role assignment
        assigned_fact_ids = set()
        for i, item in enumerate(fact_roles_list):
            if not isinstance(item, dict):
                errors.append(f"Role assignment item #{i+1} must be a dict type")
                continue

            fact_id = item.get("fact_id", "")
            if not fact_id:
                errors.append(f"Role assignment item #{i+1} is missing 'fact_id'")
                continue

            assigned_fact_ids.add(fact_id)

            # Check role field
            role = item.get("role", "")
            if not role:
                errors.append(f"Fact '{fact_id}' is missing 'role' field")
            elif role not in valid_roles:
                errors.append(f"Fact '{fact_id}' has invalid role '{role}'. Valid roles: {', '.join(valid_roles)}")

            # Check weight field (optional, defaults to 0.5 in parsing)
            weight = item.get("weight")
            if weight is not None:
                try:
                    weight_float = float(weight)
                    if not (0.0 <= weight_float <= 1.0):
                        errors.append(f"Fact '{fact_id}' weight must be in the range [0.0, 1.0]")
                except (ValueError, TypeError):
                    errors.append(f"Fact '{fact_id}' weight must be a numeric type")

        # Check extraction_confidence
        extraction_confidence = data.get("extraction_confidence")
        if extraction_confidence is not None:
            try:
                confidence_float = float(extraction_confidence)
                if not (0.0 <= confidence_float <= 1.0):
                    errors.append("extraction_confidence must be in the range [0.0, 1.0]")
            except (ValueError, TypeError):
                errors.append("extraction_confidence must be a numeric type")
        
        return len(errors) == 0, errors
    
    async def _assign_fact_roles_stage(
        self,
        topic: Topic,
        facts: List[Fact]
    ) -> RoleAssignmentResult:
        """
        Stage 2: Role assignment (with retry mechanism)

        Args:
            topic: Topic object
            facts: List of facts

        Returns:
            Role assignment result
        """
        logger.info(f"[Stage2] Starting role assignment - Fact count: {len(facts)}")

        topic_id, topic_title, topic_summary = self._format_topic_context(topic)
        facts_text = self._format_facts(facts)
        
        prompt = FACT_ROLE_ASSIGNMENT_PROMPT.format(
            topic_id=topic_id,
            topic_title=topic_title,
            topic_summary=topic_summary,
            facts=facts_text
        )
        
        # Print initial input
        print("\n" + "="*80)
        print("[Stage 2] LLM Input - Role Assignment")
        print("="*80)
        max_display_length = 1500
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

                # Enable JSON Mode, forcing LLM to return valid JSON
                resp = await self.llm_provider.generate(
                    current_prompt,
                    response_format={"type": "json_object"}
                )

                print("\n" + "="*80)
                print(f"[Stage 2] LLM Output (attempt {attempt})")
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
                
                is_valid, validation_errors = self._validate_fact_role_assignment(data, facts)
                
                if not is_valid:
                    error_msg = "Role assignment validation errors:\n" + "\n".join(validation_errors)
                    raise ValueError(error_msg)
                
                # Parse roles and weights
                # Build mapping from LLM-returned fact_id to actual fact_id
                llm_id_to_actual = {}
                for i, fact in enumerate(facts):
                    llm_id = f"fact_{i+1}"  # LLM typically returns fact_1, fact_2, ...
                    llm_id_to_actual[llm_id] = fact.fact_id

                fact_roles = {}
                fact_weights = {}

                for item in data.get("fact_roles", []):
                    llm_fact_id = item.get("fact_id", "")
                    # Convert to actual fact_id
                    actual_fact_id = llm_id_to_actual.get(llm_fact_id, llm_fact_id)

                    role_str = item.get("role", "detail")
                    weight = item.get("weight", 0.5)

                    try:
                        role = FactRole(role_str)
                    except ValueError:
                        logger.warning(f"[Stage2] Unknown role type: {role_str}, using default value 'detail'")
                        role = FactRole.DETAIL

                    fact_roles[actual_fact_id] = role
                    fact_weights[actual_fact_id] = float(weight)
                
                extraction_confidence = float(data.get("extraction_confidence", 0.8))
                reasoning = data.get("reasoning", "")
                
                logger.info(f"[Stage2] Completed role assignment - {len(fact_roles)} facts (attempt {attempt})")
                print(f"  OK: Role assignment validation passed (attempt {attempt})")
                
                return RoleAssignmentResult(
                    fact_roles=fact_roles,
                    fact_weights=fact_weights,
                    extraction_confidence=extraction_confidence,
                    reasoning=reasoning
                )
                
            except (json.JSONDecodeError, ValueError, Exception) as e:
                print(f"  Attempt {attempt} failed: {type(e).__name__}")
                print(f"  Error details: {str(e)}")

                valid_roles_str = ', '.join([role.value for role in FactRole])
                if isinstance(e, json.JSONDecodeError):
                    last_feedback = "JSON parsing failed. Please provide valid JSON format."
                elif isinstance(e, ValueError) and "validation errors" in str(e):
                    last_feedback = str(e) + f"\n\nValid roles: {valid_roles_str}"
                else:
                    last_feedback = f"Processing error: {str(e)}\n\nPlease check the output format and retry."
    
    def _build_fact_hyperedges(
        self,
        topic: Topic,
        facts: List[Fact],
        role_result: RoleAssignmentResult
    ) -> List[FactHyperedgeExtractResult]:
        """
        Build fact hyperedges (one hyperedge per episode)

        Args:
            topic: Topic object
            facts: List of facts
            role_result: Role assignment result

        Returns:
            List of fact hyperedge results
        """
        hyperedge_results = []

        # Build one hyperedge for each episode
        for episode_id in topic.episode_ids:
            # Find facts associated with this episode
            related_facts = [
                fact for fact in facts
                if episode_id in fact.episode_ids
            ]

            if not related_facts:
                continue

            # Build role mapping and weight mapping
            relation_map = {}
            weights_map = {}

            for fact in related_facts:
                fact_id = fact.fact_id
                role = role_result.fact_roles.get(fact_id, FactRole.DETAIL)
                weight = role_result.fact_weights.get(fact_id, 0.5)

                relation_map[fact_id] = role.value
                weights_map[fact_id] = weight

            # Generate hyperedge ID
            hyperedge_id = f"fact_hyperedge_{episode_id}"

            # Create fact hyperedge
            fact_hyperedge = FactHyperedge(
                id=hyperedge_id,
                relation=relation_map,
                weights=weights_map,
                episode_node_id=episode_id,
                created_at=datetime.now(),
                extraction_confidence=role_result.extraction_confidence
            )
            
            hyperedge_result = FactHyperedgeExtractResult(
                topic_id=topic.topic_id,
                episode_id=episode_id,
                fact_hyperedge=fact_hyperedge,
                role_assignment_result=role_result
            )
            
            hyperedge_results.append(hyperedge_result)
        
        return hyperedge_results
    
    async def extract_facts(
        self,
        topic: Topic,
        episodes: List[Episode]
    ) -> Tuple[Optional[FactExtractResult], List[FactHyperedgeExtractResult]]:
        """
        Two-stage fact extraction pipeline

        Args:
            topic: Topic object
            episodes: Associated Episode list

        Returns:
            (FactExtractResult, List[FactHyperedgeExtractResult]) tuple
        """
        logger.info(f"[FactExtractor] Starting two-stage fact extraction - Topic: {topic.topic_id}")

        # ========== Stage 1: Fact Extraction ==========
        fact_result = await self._extract_facts_stage(topic, episodes)

        if not fact_result.facts:
            logger.warning("[FactExtractor] No facts were extracted")
            return fact_result, []

        # ========== Stage 2: Role Assignment ==========
        role_result = await self._assign_fact_roles_stage(topic, fact_result.facts)

        # ========== Build Fact Hyperedges ==========
        hyperedge_results = self._build_fact_hyperedges(
            topic, fact_result.facts, role_result
        )

        logger.info(f"[FactExtractor] Extraction complete - Facts: {fact_result.fact_count}, Hyperedges: {len(hyperedge_results)}")

        return fact_result, hyperedge_results
