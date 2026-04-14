"""
Hypergraph Extraction (Topic-based Fact Extraction)

Features:
1. Read episode files
2. Use topic_extractor to extract topics and episode hyperedges (first)
3. Use fact_extractor to extract facts and fact hyperedges based on topics (second)
4. Use hypergraph_extractor to build the complete hypergraph
5. Save hypergraph results

Data flow:
episodes → topic extraction → fact extraction (per topic) → hypergraph building → save
"""

import json
import sys
import asyncio
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hypermem.utils.datetime_utils import from_iso_format, get_now_with_timezone, to_iso_format
from hypermem.llm.llm_provider import LLMProvider
from hypermem.types import Episode, RawDataType, Fact, Topic
from hypermem.extractors.fact_extractor import (
    FactExtractor,
    FactExtractResult,
    FactHyperedgeExtractResult
)
from hypermem.extractors.topic_extractor import (
    TopicExtractor,
    TopicExtractRequest,
    TopicExtractResult,
    EpisodeHyperedgeExtractResult,
)
from hypermem.extractors.hypergraph_extractor import HypergraphExtractor
from hypermem.structure import Hypergraph, FactHyperedge, EpisodeHyperedge

from hypermem.config import ExperimentConfig
import dataclasses

console = Console()

# Extraction failure retry configuration
MAX_EXTRACTION_RETRIES = 100


class TopicExtractorType:
    """Topic extractor type"""
    LLM = "llm"
    RETRIEVAL = "retrieval"


def serialize_to_json(obj):
    """Recursively serialize an object to a JSON-compatible dictionary"""
    if obj is None:
        return None
    elif isinstance(obj, datetime):
        return to_iso_format(obj)
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        result = {}
        for field in dataclasses.fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = serialize_to_json(value)
        return result
    elif hasattr(obj, 'model_dump'):
        try:
            return obj.model_dump(mode='json')
        except:
            dumped = obj.model_dump()
            return serialize_to_json(dumped)
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: serialize_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_to_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        return str(obj)


# ==================== Fact Result Save/Load ====================

def save_fact_results(
    conv_id: str,
    fact_results: List[FactExtractResult],
    fact_hyperedge_results: List[FactHyperedgeExtractResult],
    save_dir: Path
) -> None:
    """
    Save fact extraction results

    Args:
        conv_id: Conversation ID
        fact_results: List of fact extraction results (one per topic)
        fact_hyperedge_results: List of fact hyperedge extraction results
        save_dir: Save directory
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"facts_conv_{conv_id}.json"

    data = {
        "fact_results": [serialize_to_json(r) for r in fact_results],
        "fact_hyperedge_results": [serialize_to_json(r) for r in fact_hyperedge_results]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reconstruct_fact(data: dict) -> Fact:
    """Reconstruct a Fact object from a dictionary"""
    timestamp = None
    if data.get('timestamp'):
        try:
            timestamp = from_iso_format(data['timestamp'])
        except:
            pass
    
    # Handle spatial field: LLM may return a list, need to convert to string
    spatial = data.get('spatial')
    if isinstance(spatial, list):
        spatial = ', '.join(str(s) for s in spatial) if spatial else None
    
    # Handle temporal field: LLM may return a list, need to convert to string
    temporal = data.get('temporal')
    if isinstance(temporal, list):
        temporal = ', '.join(str(t) for t in temporal) if temporal else None
    
    return Fact(
        fact_id=data['fact_id'],
        content=data['content'],
        episode_ids=data.get('episode_ids', []),
        topic_id=data.get('topic_id', ''),
        confidence=data.get('confidence', 0.8),
        temporal=temporal,
        spatial=spatial,
        keywords=data.get('keywords', []),
        query_patterns=data.get('query_patterns', []),
        timestamp=timestamp
    )


def reconstruct_fact_hyperedge(data: dict) -> FactHyperedge:
    """Reconstruct a FactHyperedge object from a dictionary"""
    created_at = None
    if data.get('created_at'):
        try:
            created_at = from_iso_format(data['created_at'])
        except:
            pass
    
    return FactHyperedge(
        id=data['id'],
        relation=data.get('relation', {}),
        weights=data.get('weights'),
        episode_node_id=data.get('episode_node_id', ''),
        created_at=created_at,
        extraction_confidence=data.get('extraction_confidence', 0.8)
    )


def load_fact_results(
    conv_id: str,
    save_dir: Path
) -> Optional[tuple[List[FactExtractResult], List[FactHyperedgeExtractResult]]]:
    """
    Load fact extraction results

    Args:
        conv_id: Conversation ID
        save_dir: Save directory

    Returns:
        (list of fact extraction results, list of fact hyperedge extraction results) or None
    """
    input_file = save_dir / f"facts_conv_{conv_id}.json"

    if not input_file.exists():
        return None

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Rebuild FactExtractResult list
        fact_results = []
        for item in data['fact_results']:
            facts = [reconstruct_fact(e) for e in item.get('facts', [])]
            result = FactExtractResult(
                topic_id=item['topic_id'],
                facts=facts,
                reasoning=item.get('reasoning', '')
            )
            fact_results.append(result)

        # Rebuild FactHyperedgeExtractResult list
        fact_hyperedge_results = []
        for item in data['fact_hyperedge_results']:
            fact_hyperedge = None
            if item.get('fact_hyperedge'):
                fact_hyperedge = reconstruct_fact_hyperedge(item['fact_hyperedge'])
            
            result = FactHyperedgeExtractResult(
                topic_id=item['topic_id'],
                episode_id=item['episode_id'],
                fact_hyperedge=fact_hyperedge,
                role_assignment_result=None
            )
            fact_hyperedge_results.append(result)
        
        return (fact_results, fact_hyperedge_results)

    except Exception as e:
        console.print(f"[yellow][!] Failed to load fact results: {e}[/yellow]")
        import traceback
        traceback.print_exc()
        return None


# ==================== Token Statistics Save/Load ====================

def save_token_stats(
    conv_id: str,
    topic_token_stats: Optional[Dict],
    fact_token_stats: Optional[Dict],
    save_dir: Path
) -> None:
    """
    Save token usage statistics

    Note: The total field will be calculated collectively after stage4 ends; it is not calculated here.

    Args:
        conv_id: Conversation ID
        topic_token_stats: Token statistics for topic extraction
        fact_token_stats: Token statistics for fact extraction
        save_dir: Save directory
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"token_stats_conv_{conv_id}.json"

    # Note: total is not calculated here; it will be calculated collectively after stage4 ends
    data = {
        "conv_id": conv_id,
        "topic_extraction": topic_token_stats,
        "fact_extraction": fact_token_stats,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_token_stats(conv_id: str, save_dir: Path) -> Optional[Dict]:
    """Load token statistics"""
    input_file = save_dir / f"token_stats_conv_{conv_id}.json"
    
    if not input_file.exists():
        return None
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ==================== Topic Result Save/Load ====================

def save_topic_results(
    conv_id: str,
    topic_result: Optional[TopicExtractResult],
    episode_hyperedge_results: List[EpisodeHyperedgeExtractResult],
    save_dir: Path
) -> None:
    """Save topic extraction results"""
    save_dir.mkdir(parents=True, exist_ok=True)
    output_file = save_dir / f"topics_conv_{conv_id}.json"

    data = {
        "topic_result": serialize_to_json(topic_result),
        "episode_hyperedge_results": [serialize_to_json(r) for r in episode_hyperedge_results]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def reconstruct_topic(data: dict) -> Topic:
    """Reconstruct a Topic object from a dictionary"""
    timestamp = None
    if data.get('timestamp'):
        try:
            timestamp = from_iso_format(data['timestamp'])
        except:
            pass
    
    return Topic(
        topic_id=data['topic_id'],
        title=data['title'],
        summary=data['summary'],
        episode_ids=data.get('episode_ids', []),
        timestamp=timestamp,
        user_id_list=data.get('user_id_list', []),
        participants=data.get('participants', []),
        keywords=data.get('keywords', []),
    )


def reconstruct_episode_hyperedge(data: dict) -> EpisodeHyperedge:
    """Reconstruct an EpisodeHyperedge object from a dictionary"""
    created_at = None
    if data.get('created_at'):
        try:
            created_at = from_iso_format(data['created_at'])
        except:
            pass
    
    return EpisodeHyperedge(
        id=data['id'],
        relation=data.get('relation', {}),
        weights=data.get('weights'),
        topic_node_id=data.get('topic_node_id', ''),
        created_at=created_at,
        coherence_score=data.get('coherence_score', 0.8)
    )


def load_topic_results(
    conv_id: str,
    save_dir: Path
) -> Optional[tuple[Optional[TopicExtractResult], List[EpisodeHyperedgeExtractResult]]]:
    """Load topic extraction results"""
    input_file = save_dir / f"topics_conv_{conv_id}.json"

    if not input_file.exists():
        return None

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        topic_result = None
        if data['topic_result']:
            topics = [reconstruct_topic(s) for s in data['topic_result']['topics']]
            topic_result = TopicExtractResult(
                topics=topics,
                action=data['topic_result'].get('action', 'create_new'),
                similar_topic_result=None
            )
        
        episode_hyperedge_results = []
        for item in data['episode_hyperedge_results']:
            episode_hyperedge = None
            if item['episode_hyperedge']:
                episode_hyperedge = reconstruct_episode_hyperedge(item['episode_hyperedge'])
            
            result = EpisodeHyperedgeExtractResult(
                topic_id=item['topic_id'],
                episode_hyperedge=episode_hyperedge,
                role_weight_result=None
            )
            episode_hyperedge_results.append(result)
        
        return (topic_result, episode_hyperedge_results)

    except Exception as e:
        console.print(f"[yellow][!] Failed to load topic results: {e}[/yellow]")
        import traceback
        traceback.print_exc()
        return None


def load_episodes_from_json(file_path: str) -> List[Episode]:
    """Load a list of Episodes from a JSON file"""
    with open(file_path, "r", encoding="utf-8") as f:
        episode_dicts = json.load(f)

    episodes = []
    for episode_dict in episode_dicts:
        if "timestamp" in episode_dict and episode_dict["timestamp"]:
            ts = episode_dict["timestamp"]
            if isinstance(ts, str):
                episode_dict["timestamp"] = from_iso_format(ts)
            elif isinstance(ts, (int, float)):
                episode_dict["timestamp"] = datetime.fromtimestamp(ts)

        if "type" in episode_dict and episode_dict["type"]:
            try:
                episode_dict["type"] = RawDataType(episode_dict["type"])
            except ValueError:
                episode_dict["type"] = RawDataType.CONVERSATION


        episode = Episode(**episode_dict)
        episodes.append(episode)

    return episodes


# ==================== Topic Extraction ====================

async def extract_topics_for_episodes(
    episodes: List[Episode],
    llm_provider: LLMProvider,
    extractor_type: str = TopicExtractorType.LLM,
    embedding_provider=None,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None
) -> tuple[Optional[TopicExtractResult], List[EpisodeHyperedgeExtractResult]]:
    """
    Extract topics and episode hyperedges for a list of Episodes

    Args:
        episodes: List of Episodes
        llm_provider: LLM provider
        extractor_type: Topic extractor type
        embedding_provider: Embedding provider
        progress: Progress bar object
        task_id: Progress task ID

    Returns:
        (topic extraction result, list of episode hyperedge extraction results)
    """
    if not episodes:
        return None, []
    
    console.print(f"  [*] Using LLM-based topic matching (TopicExtractor)")
    topic_extractor = TopicExtractor(
        llm_provider=llm_provider,
        topic_match_batch_size=10,
    )

    # Maintain deduplicated topic dictionary and hyperedge dictionary in real-time
    topic_map = {}
    hyperedge_map = {}
    
    # Create the initial topic for the first episode
    if len(episodes) >= 1:
        try:
            first_episode = episodes[0]
            console.print(f"  [*] Creating initial topic for the first episode...")

            topic = await topic_extractor._extract_new_topic([first_episode])

            if topic:
                topic_map[topic.topic_id] = topic
                hyperedge_result = await topic_extractor._build_episode_hyperedge(
                    topic=topic,
                    episodes=[first_episode]
                )
                hyperedge_map[topic.topic_id] = hyperedge_result
                console.print(f"  [+] Created initial topic: {topic.title}")
            else:
                raise RuntimeError("Topic creation for the first episode returned None")

        except Exception as e:
            console.print(f"[red][!] Topic creation for the first episode failed: {e}[/red]")
            raise RuntimeError(f"Topic creation for the first episode failed: {e}") from e
    
    if len(episodes) == 1:
        if topic_map:
            return TopicExtractResult(
                topics=list(topic_map.values()),
                action="merged"
            ), list(hyperedge_map.values())
        return None, []

    # Incrementally add subsequent episodes
    for idx in range(1, len(episodes)):
        if progress and task_id is not None:
            progress.update(task_id, completed=idx - 1)

        retry_count = 0
        success = False

        while retry_count < MAX_EXTRACTION_RETRIES and not success:
            try:
                new_episode = episodes[idx]
                history_episodes = episodes[:idx]
                existing_topics = list(topic_map.values())

                topic_request = TopicExtractRequest(
                    history_episode_list=history_episodes,
                    new_episode=new_episode,
                    existing_topics=existing_topics
                )

                topic_result, episode_hyperedge_results = await topic_extractor.extract_topic(topic_request)

                if topic_result:
                    for topic in topic_result.topics:
                        topic_map[topic.topic_id] = topic

                if episode_hyperedge_results:
                    for result in episode_hyperedge_results:
                        if result.episode_hyperedge:
                            hyperedge_map[result.topic_id] = result

                success = True

            except Exception as e:
                retry_count += 1
                if retry_count < MAX_EXTRACTION_RETRIES:
                    console.print(f"[yellow][!] Topic extraction failed (idx={idx}, retry {retry_count}/{MAX_EXTRACTION_RETRIES}): {e}[/yellow]")
                else:
                    console.print(f"[red][X] Topic extraction failed (idx={idx}), max retries reached: {e}[/red]")
                    break
    
    if progress and task_id is not None:
        progress.update(task_id, completed=len(episodes) - 1)
    
    if topic_map:
        return TopicExtractResult(
            topics=list(topic_map.values()),
            action="merged"
        ), list(hyperedge_map.values())
    
    return None, []


# ==================== Topic-based Fact Extraction ====================

async def extract_facts_for_topics(
    topics: List[Topic],
    episodes: List[Episode],
    llm_provider: LLMProvider,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None
) -> tuple[List[FactExtractResult], List[FactHyperedgeExtractResult]]:
    """
    Extract facts based on topics

    Args:
        topics: List of topics
        episodes: List of all Episodes
        llm_provider: LLM provider
        progress: Progress bar object
        task_id: Progress task ID

    Returns:
        (list of fact extraction results, list of fact hyperedge extraction results)
    """
    fact_extractor = FactExtractor(llm_provider=llm_provider)
    
    # Build episode_id -> Episode mapping
    episode_map: Dict[str, Episode] = {mc.event_id: mc for mc in episodes}
    
    fact_results = []
    fact_hyperedge_results = []

    for idx, topic in enumerate(topics):
        if progress and task_id is not None:
            progress.update(task_id, completed=idx)

        # Get episodes associated with this topic
        topic_episodes = [
            episode_map[mc_id]
            for mc_id in topic.episode_ids
            if mc_id in episode_map
        ]

        if not topic_episodes:
            console.print(f"  [yellow][!] Topic {topic.topic_id} has no associated episodes, skipping[/yellow]")
            continue

        # Fact extraction retry logic
        retry_count = 0
        success = False

        while retry_count < MAX_EXTRACTION_RETRIES and not success:
            try:
                # Extract facts based on topics and associated episodes
                fact_result, hyperedge_result_list = await fact_extractor.extract_facts(
                    topic=topic,
                    episodes=topic_episodes
                )

                if fact_result:
                    fact_results.append(fact_result)
                if hyperedge_result_list:
                    fact_hyperedge_results.extend(hyperedge_result_list)

                success = True

            except Exception as e:
                retry_count += 1
                if retry_count < MAX_EXTRACTION_RETRIES:
                    console.print(f"[yellow][!] Topic {topic.topic_id} fact extraction failed (retry {retry_count}/{MAX_EXTRACTION_RETRIES}): {e}[/yellow]")
                else:
                    console.print(f"[red][X] Topic {topic.topic_id} fact extraction failed, max retries reached: {e}[/red]")
                    break
    
    if progress and task_id is not None:
        progress.update(task_id, completed=len(topics))
    
    return fact_results, fact_hyperedge_results


# ==================== Process Single Conversation ====================

async def process_single_conversation(
    conv_id: str,
    episodes_file: Path,
    save_dir: Path,
    llm_provider: LLMProvider,
    extractor_type: str = TopicExtractorType.LLM,
    embedding_provider=None,
    progress: Optional[Progress] = None,
    conv_task_id: Optional[int] = None,
    skip_existing: bool = True,
    facts_dir: Optional[Path] = None,
    topics_dir: Optional[Path] = None
) -> Optional[Hypergraph]:
    """
    Process hypergraph extraction for a single conversation (new pipeline: topics first, then facts)

    Args:
        conv_id: Conversation ID
        episodes_file: Episode JSON file path
        save_dir: Save directory (hypergraphs/)
        llm_provider: LLM provider
        extractor_type: Topic extractor type
        embedding_provider: Embedding provider
        progress: Progress bar object
        conv_task_id: Conversation task ID
        skip_existing: Whether to skip existing hypergraph files
        facts_dir: Fact results save directory
        topics_dir: Topic results save directory

    Returns:
        The constructed hypergraph object
    """
    try:
        output_file = save_dir / f"hypergraph_conv_{conv_id}.json"
        console.print(f"  [dim]Conversation {conv_id}: checking path {output_file}[/dim]")
        
        if skip_existing and output_file.exists():
            console.print(f"  [cyan][√] Conversation {conv_id}: hypergraph already exists, skipping[/cyan]")
            if progress and conv_task_id is not None:
                progress.update(conv_task_id, description=f"[cyan]Conversation {conv_id}[/cyan]", status="[cyan]exists[/cyan]", completed=1)
            
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    hypergraph_dict = json.load(f)
                return Hypergraph.from_dict(hypergraph_dict)
            except Exception as e:
                console.print(f"  [yellow][!] Conversation {conv_id}: failed to load existing hypergraph: {e}, will reprocess[/yellow]")
        
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, description=f"[cyan]Conversation {conv_id}[/cyan]", status="loading data")
        
        # Step 1: Load Episodes
        episodes = load_episodes_from_json(str(episodes_file))
        console.print(f"  [+] Conversation {conv_id}: loaded {len(episodes)} Episodes")
        
        if not episodes:
            console.print(f"  [yellow][!] Conversation {conv_id}: no Episodes, skipping[/yellow]")
            return None
        
        # Step 2: Extract topics (first)
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="checking topic cache", total=1, completed=0)

        # Topic extraction token statistics
        topic_token_stats = None

        cached_topics = None
        if topics_dir and topics_dir.exists():
            topic_cache_file = topics_dir / f"topics_conv_{conv_id}.json"
            if topic_cache_file.exists():
                console.print(f"  [cyan]✓ Found topic cache file: {topic_cache_file.name}[/cyan]")
                try:
                    cached_topics = load_topic_results(conv_id, topics_dir)
                    if cached_topics is not None:
                        console.print(f"  [green]✓ Topic cache loaded successfully[/green]")
                except Exception as e:
                    console.print(f"  [yellow]⚠ Topic cache loading error: {e}, will re-extract[/yellow]")
                    cached_topics = None

        if cached_topics is not None:
            console.print(f"  [cyan][√] Conversation {conv_id}: using cached topic results[/cyan]")
            topic_result, episode_hyperedge_results = cached_topics
            num_topics = len(topic_result.topics) if topic_result else 0
            console.print(f"  [cyan]   └─ Number of topics: {num_topics}[/cyan]")
        else:
            if progress and conv_task_id is not None:
                progress.update(conv_task_id, status="extract topic", total=len(episodes) - 1, completed=0)

            console.print(f"  [yellow]→ Conversation {conv_id}: starting topic extraction...[/yellow]")
            try:
                # Reset statistics, record accumulated values before topic extraction
                llm_provider.reset_accumulated_stats()

                topic_result, episode_hyperedge_results = await extract_topics_for_episodes(
                    episodes=episodes,
                    llm_provider=llm_provider,
                    extractor_type=extractor_type,
                    embedding_provider=embedding_provider,
                    progress=progress,
                    task_id=conv_task_id
                )
                
                # Get token statistics for topic extraction
                topic_token_stats = llm_provider.get_accumulated_stats()
                if topic_token_stats:
                    console.print(f"  [dim]   └─ Topic extraction Token: prompt={topic_token_stats['prompt_tokens']:,}, "
                                  f"completion={topic_token_stats['completion_tokens']:,}, "
                                  f"total={topic_token_stats['total_tokens']:,}[/dim]")

                num_topics = len(topic_result.topics) if topic_result else 0
                console.print(f"  [+] Conversation {conv_id}: extracted {num_topics} topics")
            except Exception as topic_extract_error:
                console.print(f"[red][-] Conversation {conv_id}: topic extraction failed: {topic_extract_error}[/red]")
                import traceback
                traceback.print_exc()
                raise

            if topics_dir:
                save_topic_results(conv_id, topic_result, episode_hyperedge_results, topics_dir)
                console.print(f"  [+] Conversation {conv_id}: saved topic results to topics/")
        
        # Step 3: Extract facts based on topics (second)
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="checking fact cache", total=1, completed=0)

        # Fact extraction token statistics
        fact_token_stats = None

        cached_facts = None
        if facts_dir and facts_dir.exists():
            fact_cache_file = facts_dir / f"facts_conv_{conv_id}.json"
            if fact_cache_file.exists():
                console.print(f"  [cyan]✓ Found fact cache file: {fact_cache_file.name}[/cyan]")
                try:
                    cached_facts = load_fact_results(conv_id, facts_dir)
                    if cached_facts is not None:
                        console.print(f"  [green]✓ Fact cache loaded successfully[/green]")
                except Exception as e:
                    console.print(f"  [yellow]⚠ Fact cache loading error: {e}, will re-extract[/yellow]")
                    cached_facts = None

        if cached_facts is not None:
            console.print(f"  [cyan][√] Conversation {conv_id}: using cached fact results[/cyan]")
            fact_results, fact_hyperedge_results = cached_facts
        else:
            # Get topic list
            topics = topic_result.topics if topic_result else []

            if not topics:
                console.print(f"  [yellow][!] Conversation {conv_id}: no topics, skipping fact extraction[/yellow]")
                fact_results = []
                fact_hyperedge_results = []
            else:
                if progress and conv_task_id is not None:
                    progress.update(conv_task_id, status="extract fact", total=len(topics), completed=0)

                console.print(f"  [yellow]→ Conversation {conv_id}: extracting facts based on {len(topics)} topics...[/yellow]")

                # Reset statistics, record accumulated values before fact extraction
                llm_provider.reset_accumulated_stats()
                
                fact_results, fact_hyperedge_results = await extract_facts_for_topics(
                    topics=topics,
                    episodes=episodes,
                    llm_provider=llm_provider,
                    progress=progress,
                    task_id=conv_task_id
                )
                
                # Get token statistics for fact extraction
                fact_token_stats = llm_provider.get_accumulated_stats()
                if fact_token_stats:
                    console.print(f"  [dim]   └─ Fact extraction Token: prompt={fact_token_stats['prompt_tokens']:,}, "
                                  f"completion={fact_token_stats['completion_tokens']:,}, "
                                  f"total={fact_token_stats['total_tokens']:,}[/dim]")

                # Count facts
                total_facts = sum(len(r.facts) for r in fact_results)
                if total_facts > 0:
                    console.print(f"  [+] Conversation {conv_id}: extracted {total_facts} facts")
                else:
                    console.print(f"  [red][-] Conversation {conv_id}: fact extraction returned 0 facts[/red]")
                    if progress and conv_task_id is not None:
                        progress.update(conv_task_id, status="[yellow]facts=0[/yellow]")

            if facts_dir:
                save_fact_results(conv_id, fact_results, fact_hyperedge_results, facts_dir)
                console.print(f"  [+] Conversation {conv_id}: saved fact results to facts/")

        # Step 4: Build hypergraph
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="building hypergraph", total=1, completed=0)
        
        hypergraph_extractor = HypergraphExtractor()
        hypergraph = hypergraph_extractor.build_hypergraph(
            episodes=episodes,
            fact_results=fact_results,
            fact_hyperedge_results=fact_hyperedge_results,
            topic_extract_result=topic_result,
            episode_hyperedge_results=episode_hyperedge_results
        )
        
        stats = hypergraph.get_stats()
        console.print(f"  [+] Conversation {conv_id}: built hypergraph - {stats}")
        
        # Step 5: Save hypergraph
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="saving hypergraph", completed=1)
        
        hypergraph_dict = hypergraph.to_dict()
        output_file = save_dir / f"hypergraph_conv_{conv_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(hypergraph_dict, f, ensure_ascii=False, indent=2)
        
        console.print(f"  [+] Conversation {conv_id}: saved hypergraph to {output_file.name}")
        
        # Step 6: Save token statistics (if available)
        if topic_token_stats or fact_token_stats:
            token_stats_dir = save_dir.parent / "token_stats"
            save_token_stats(conv_id, topic_token_stats, fact_token_stats, token_stats_dir)

            # Calculate total token count
            total_prompt = (topic_token_stats or {}).get('prompt_tokens', 0) + (fact_token_stats or {}).get('prompt_tokens', 0)
            total_completion = (topic_token_stats or {}).get('completion_tokens', 0) + (fact_token_stats or {}).get('completion_tokens', 0)
            total_tokens = total_prompt + total_completion
            console.print(f"  [dim]   └─ Total Token: prompt={total_prompt:,}, completion={total_completion:,}, total={total_tokens:,}[/dim]")
        
        if progress and conv_task_id is not None:
            total_facts = sum(len(r.facts) for r in fact_results) if fact_results else 0
            if total_facts > 0:
                progress.update(conv_task_id, status="[green]done[/green]", completed=1)
            else:
                progress.update(conv_task_id, status="[yellow]done (facts=0)[/yellow]", completed=1)

        return hypergraph
        
    except Exception as e:
        console.print(f"[red][-] Conversation {conv_id} processing failed: {e}[/red]")
        if progress and conv_task_id is not None:
            progress.update(conv_task_id, status="[red]failed[/red]")
        import traceback
        traceback.print_exc()
        return None


async def process_conversation_with_semaphore(
    semaphore: asyncio.Semaphore,
    conv_id: str,
    episodes_file: Path,
    save_dir: Path,
    llm_provider: LLMProvider,
    extractor_type: str = TopicExtractorType.LLM,
    embedding_provider=None,
    progress: Optional[Progress] = None,
    conv_task_id: Optional[int] = None,
    skip_existing: bool = True,
    facts_dir: Optional[Path] = None,
    topics_dir: Optional[Path] = None
) -> tuple[str, Optional[Hypergraph]]:
    """Conversation processing with semaphore-controlled concurrency"""
    async with semaphore:
        if progress and conv_task_id is not None:
            progress.start_task(conv_task_id)

        hypergraph = await process_single_conversation(
            conv_id=conv_id,
            episodes_file=episodes_file,
            save_dir=save_dir,
            llm_provider=llm_provider,
            extractor_type=extractor_type,
            embedding_provider=embedding_provider,
            progress=progress,
            conv_task_id=conv_task_id,
            skip_existing=skip_existing,
            facts_dir=facts_dir,
            topics_dir=topics_dir
        )
        
        return (conv_id, hypergraph)


async def main():
    """Main function: batch process hypergraph extraction for all conversations"""
    config = ExperimentConfig()
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Stage 2: Hypergraph Extraction[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Configure topic extractor type
    extractor_type = TopicExtractorType.RETRIEVAL

    # Configure paths
    episodes_dir = config.episodes_dir()
    hypergraph_dir = config.hypergraph_dir()
    facts_dir = config.facts_dir()
    topics_dir = config.topics_dir()
    token_stats_dir = config.token_stats_dir()
    
    # Create directories
    hypergraph_dir.mkdir(parents=True, exist_ok=True)
    facts_dir.mkdir(parents=True, exist_ok=True)
    topics_dir.mkdir(parents=True, exist_ok=True)
    token_stats_dir.mkdir(parents=True, exist_ok=True)
    
    # Concurrency configuration
    max_concurrent_tasks = 10
    skip_existing = True
    
    console.print(f"[bold]Experiment name:[/bold] {config.experiment_name}")
    console.print(f"[bold]Topic extractor type:[/bold] {extractor_type}")
    console.print(f"[bold]Number of conversations:[/bold] {config.num_conv}")
    console.print(f"[bold]Concurrency:[/bold] {max_concurrent_tasks}")
    console.print(f"[bold]Skip existing:[/bold] {'Yes' if skip_existing else 'No'}")
    console.print(f"[bold]Pipeline:[/bold] episode → topic → fact → hypergraph\n")
    
    # Initialize LLM Provider (with statistics enabled)
    llm_config = config.llm_config[config.llm_service].copy()
    provider_type = llm_config.pop('llm_provider', 'openai')
    llm_config['enable_stats'] = True  # Enable token statistics
    llm_provider = LLMProvider(provider_type=provider_type, **llm_config)
    
    # Initialize Embedding Provider
    embedding_provider = None
    if extractor_type == TopicExtractorType.RETRIEVAL:
        try:
            from hypermem.llm.embedding_provider import EmbeddingProvider
            embedding_config = config.embedding_config
            embedding_provider = EmbeddingProvider(
                base_url=embedding_config["base_url"],
                model_name=embedding_config["model_name"]
            )
            console.print(f"[green][OK][/green] Embedding Provider initialized successfully\n")
        except Exception as e:
            console.print(f"[red][X] Failed to initialize Embedding Provider: {e}[/red]")
            console.print(f"[yellow][!] Falling back to LLM version[/yellow]\n")
            extractor_type = TopicExtractorType.LLM
    
    # Display final configuration
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Final Configuration[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print(f"[bold]Topic extractor type:[/bold] {extractor_type}")
    console.print(f"[bold]Episode directory:[/bold] {episodes_dir}")
    console.print(f"[bold]Hypergraph save directory:[/bold] {hypergraph_dir}")
    console.print(f"[bold]Topic cache directory:[/bold] {topics_dir}")
    console.print(f"[bold]Fact cache directory:[/bold] {facts_dir}")
    console.print(f"[bold]Token statistics directory:[/bold] {token_stats_dir}")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Collect all conversation files
    conv_files = []
    for i in range(config.num_conv):
        episode_file = episodes_dir / f"episode_list_conv_{i}.json"
        if episode_file.exists():
            conv_files.append((str(i), episode_file))
        else:
            console.print(f"[yellow][!] File not found: {episode_file}[/yellow]")
    
    if not conv_files:
        console.print("[red][-] No Episode files found[/red]")
        return
    
    console.print(f"[green][OK][/green] Found {len(conv_files)} conversation files\n")
    
    # Create semaphore
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed:>3}/{task.total:<3}"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[bold]{task.fields[status]}"),
        console=console,
        transient=False
    ) as progress:
        
        tasks = []
        for conv_id, episode_file in conv_files:
            task_id = progress.add_task(
                f"[cyan]Conversation {conv_id}[/cyan]",
                total=1,
                status="waiting",
                start=False
            )
            tasks.append((conv_id, episode_file, task_id))
        
        coroutines = [
            process_conversation_with_semaphore(
                semaphore=semaphore,
                conv_id=conv_id,
                episodes_file=episode_file,
                save_dir=hypergraph_dir,
                llm_provider=llm_provider,
                extractor_type=extractor_type,
                embedding_provider=embedding_provider,
                progress=progress,
                conv_task_id=task_id,
                skip_existing=skip_existing,
                facts_dir=facts_dir,
                topics_dir=topics_dir
            )
            for conv_id, episode_file, task_id in tasks
        ]
        
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                conv_id = tasks[i][0]
                console.print(f"[red][-] Conversation {conv_id} processing error: {result}[/red]")
                processed_results.append((conv_id, None))
            else:
                processed_results.append(result)
    
    # Summarize results
    successful = sum(1 for _, hg in processed_results if hg is not None)
    failed = len(processed_results) - successful

    # Aggregate Token statistics
    total_token_summary = {
        'topic_extraction': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'call_count': 0, 'total_duration': 0.0},
        'fact_extraction': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'call_count': 0, 'total_duration': 0.0},
        'total': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'call_count': 0, 'total_duration': 0.0}
    }

    # Read all token statistics files
    for conv_id, _ in processed_results:
        token_stats = load_token_stats(conv_id, token_stats_dir)
        if token_stats:
            for stage in ['topic_extraction', 'fact_extraction']:
                if token_stats.get(stage):
                    for key in ['prompt_tokens', 'completion_tokens', 'total_tokens', 'call_count']:
                        total_token_summary[stage][key] += token_stats[stage].get(key, 0)
                        total_token_summary['total'][key] += token_stats[stage].get(key, 0)
                    # Accumulate total_duration
                    total_token_summary[stage]['total_duration'] += token_stats[stage].get('total_duration', 0.0)
                    total_token_summary['total']['total_duration'] += token_stats[stage].get('total_duration', 0.0)
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Processing Complete[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    console.print(f"[bold green][OK] Succeeded:[/bold green] {successful}/{len(processed_results)} conversations")
    if failed > 0:
        console.print(f"[bold red][X] Failed:[/bold red] {failed}/{len(processed_results)} conversations")
    
    # Print Token statistics summary
    if total_token_summary['total']['total_tokens'] > 0:
        console.print("\n[bold yellow]Token Usage Summary:[/bold yellow]")
        console.print(f"  Topic extraction: prompt={total_token_summary['topic_extraction']['prompt_tokens']:,}, "
                      f"completion={total_token_summary['topic_extraction']['completion_tokens']:,}, "
                      f"total={total_token_summary['topic_extraction']['total_tokens']:,}, "
                      f"calls={total_token_summary['topic_extraction']['call_count']}")
        console.print(f"  Fact extraction: prompt={total_token_summary['fact_extraction']['prompt_tokens']:,}, "
                      f"completion={total_token_summary['fact_extraction']['completion_tokens']:,}, "
                      f"total={total_token_summary['fact_extraction']['total_tokens']:,}, "
                      f"calls={total_token_summary['fact_extraction']['call_count']}")
        console.print(f"  [bold]Total: prompt={total_token_summary['total']['prompt_tokens']:,}, "
                      f"completion={total_token_summary['total']['completion_tokens']:,}, "
                      f"total={total_token_summary['total']['total_tokens']:,}, "
                      f"calls={total_token_summary['total']['call_count']}[/bold]")
        
        # Save summary statistics
        summary_file = token_stats_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "experiment_name": config.experiment_name,
                "num_conversations": len(processed_results),
                "successful": successful,
                "failed": failed,
                "token_usage": total_token_summary
            }, f, ensure_ascii=False, indent=2)
        console.print(f"\n[dim]Token statistics summary saved to: {summary_file}[/dim]")
    
    console.print(f"\n[bold]Hypergraphs saved at:[/bold] {hypergraph_dir}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
