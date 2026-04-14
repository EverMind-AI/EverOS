"""
Hypergraph Index Building

Features:
1. Read hypergraph files
2. Build indexes for facts
3. Build indexes for episodes
4. Build indexes for topics
5. Save index results

Data flow:
hypergraphs → fact indexing → episode indexing → topic indexing → save
"""

import sys
import json
import pickle
import asyncio
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

import numpy as np
import torch
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hypermem.config import ExperimentConfig
from hypermem.llm.embedding_provider import EmbeddingProvider

console = Console()


def ensure_nltk_data():
    """Ensure NLTK data is downloaded"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def tokenize(text: str, stemmer: PorterStemmer, stop_words: set) -> List[str]:
    """
    NLTK-based tokenization with stemming and stop word filtering

    Args:
        text: Text to tokenize
        stemmer: Stemmer
        stop_words: Set of stop words

    Returns:
        List of processed tokens
    """
    if not text:
        return []

    tokens = word_tokenize(text.lower())
    
    processed_tokens = [
        stemmer.stem(token) 
        for token in tokens 
        if token.isalpha() and len(token) >= 2 and token not in stop_words
    ]
    
    return processed_tokens


def build_fact_searchable_text(fact: Dict[str, Any]) -> str:
    """
    Build searchable text for a fact (with weighting)

    Supports two fact formats:
    - Triplet format: subject_name + relation + object_name
    - Full sentence format: content field

    Weighting strategy:
    - Core content (content or triplet): repeated 3 times (core information)
    - query_patterns: repeated 2 times (query patterns)
    - keywords/tags, background, impact: repeated 1 time (auxiliary information)

    Args:
        fact: Fact dictionary

    Returns:
        Searchable text
    """
    parts = []

    # Core content (highest weight)
    if fact.get("content"):
        # Full sentence format: use content field
        parts.extend([fact["content"]] * 3)
    else:
        # Triplet format: use subject/relation/object
        core_parts = []
        if fact.get("subject_name"):
            core_parts.append(fact["subject_name"])
        if fact.get("relation"):
            core_parts.append(fact["relation"])
        if fact.get("object_name"):
            core_parts.append(fact["object_name"])

        if core_parts:
            core_text = " ".join(core_parts)
            parts.extend([core_text] * 3)

    # Query patterns (medium weight)
    if fact.get("query_patterns"):
        query_text = " ".join(fact["query_patterns"])
        parts.extend([query_text] * 2)

    # Keywords (compatible with keywords and tags)
    keywords = fact.get("keywords") or fact.get("tags") or []
    if keywords:
        if isinstance(keywords, list):
            parts.append(" ".join(keywords))
        else:
            parts.append(str(keywords))

    # Auxiliary information (lower weight)
    if fact.get("background"):
        parts.append(fact["background"])
    if fact.get("impact"):
        parts.append(fact["impact"])
    if fact.get("temporal"):
        parts.append(str(fact["temporal"]))
    if fact.get("spatial"):
        parts.append(str(fact["spatial"]))

    return " ".join(str(part) for part in parts if part)


def build_episode_searchable_text(episode: Dict[str, Any]) -> str:
    """
    Build searchable text for an episode (with weighting)

    Weighting strategy:
    - subject: repeated 3 times (title has highest weight)
    - summary: repeated 2 times (summary has medium weight)
    - episode: repeated 1 time (content has base weight)

    Args:
        episode: Episode dictionary

    Returns:
        Searchable text
    """
    parts = []

    # Title (highest weight)
    if episode.get("subject"):
        parts.extend([episode["subject"]] * 3)

    # Summary (medium weight)
    if episode.get("summary"):
        parts.extend([episode["summary"]] * 2)

    # Full content
    if episode.get("episode_description"):
        parts.append(episode["episode_description"])

    # Keywords
    if episode.get("keywords"):
        if isinstance(episode["keywords"], list):
            parts.append(" ".join(episode["keywords"]))
        else:
            parts.append(str(episode["keywords"]))

    return " ".join(str(part) for part in parts if part)


def build_topic_searchable_text(topic: Dict[str, Any]) -> str:
    """
    Build searchable text for a topic (with weighting)

    Weighting strategy:
    - title: repeated 3 times (title has highest weight)
    - keywords: repeated 2 times (keywords have medium weight)
    - summary: repeated 1 time (summary has base weight)

    Args:
        topic: Topic dictionary

    Returns:
        Searchable text
    """
    parts = []

    # Title (highest weight)
    if topic.get("title"):
        parts.extend([topic["title"]] * 3)

    # Keywords (medium weight)
    if topic.get("keywords"):
        keywords_text = " ".join(topic["keywords"])
        parts.extend([keywords_text] * 2)

    # Summary
    if topic.get("summary"):
        parts.append(topic["summary"])

    return " ".join(str(part) for part in parts if part)


def build_bm25_index_for_single_conv(
    conv_id: int,
    data_dir: Path,
    bm25_save_dir: Path,
    stemmer: PorterStemmer,
    stop_words: set,
    skip_existing: bool = True
) -> tuple[bool, str]:
    """
    Build BM25 index for a single conversation's hypergraph data

    Args:
        conv_id: Conversation ID
        data_dir: Hypergraph data directory
        bm25_save_dir: BM25 index save directory
        stemmer: Stemmer
        stop_words: Set of stop words
        skip_existing: Whether to skip existing index files

    Returns:
        (success, status: "completed"/"skipped"/"failed")
    """
    try:
        # Check if output file already exists (checkpoint resume)
        output_path = bm25_save_dir / f"hypergraph_bm25_index_conv_{conv_id}.pkl"
        if skip_existing and output_path.exists():
            return (True, "skipped")
        
        hypergraph_file = data_dir / f"hypergraph_conv_{conv_id}.json"
        if not hypergraph_file.exists():
            return (False, "failed")
        
        # Read hypergraph data
        with open(hypergraph_file, "r", encoding="utf-8") as f:
            hypergraph = json.load(f)

        # ===== 1. Build index for facts =====
        facts = hypergraph.get("facts", {})
        fact_corpus = []
        fact_docs = []

        for fact_id, fact_data in facts.items():
            fact_docs.append({
                "id": fact_id,
                "type": "fact",
                "data": fact_data
            })
            searchable_text = build_fact_searchable_text(fact_data)
            tokenized_text = tokenize(searchable_text, stemmer, stop_words)
            fact_corpus.append(tokenized_text)
        
        # ===== 2. Build index for episodes =====
        episodes = hypergraph.get("episodes", {})
        episode_corpus = []
        episode_docs = []

        for episode_id, episode_data in episodes.items():
            episode_docs.append({
                "id": episode_id,
                "type": "episode",
                "data": episode_data
            })
            searchable_text = build_episode_searchable_text(episode_data)
            tokenized_text = tokenize(searchable_text, stemmer, stop_words)
            episode_corpus.append(tokenized_text)
        
        # ===== 3. Build index for topics =====
        topics = hypergraph.get("topics", {})
        topic_corpus = []
        topic_docs = []

        for topic_id, topic_data in topics.items():
            topic_docs.append({
                "id": topic_id,
                "type": "topic",
                "data": topic_data
            })
            searchable_text = build_topic_searchable_text(topic_data)
            tokenized_text = tokenize(searchable_text, stemmer, stop_words)
            topic_corpus.append(tokenized_text)

        # ===== 4. Build unified BM25 index =====
        all_corpus = fact_corpus + episode_corpus + topic_corpus
        all_docs = fact_docs + episode_docs + topic_docs

        if not all_corpus:
            console.print(f"  [yellow][!] Conversation {conv_id}: no documents, skipping index creation[/yellow]")
            return False

        bm25 = BM25Okapi(all_corpus)

        # ===== 5. Save index =====
        index_data = {
            "bm25": bm25,
            "docs": all_docs,
            "fact_count": len(fact_docs),
            "episode_count": len(episode_docs),
            "topic_count": len(topic_docs)
        }
        
        with open(output_path, "wb") as f:
            pickle.dump(index_data, f)
        
        return (True, "completed")
        
    except Exception as e:
        console.print(f"  [red][X] Conversation {conv_id}: BM25 index building failed - {e}[/red]")
        return (False, "failed")


async def build_bm25_index_for_hypergraph(
    config: ExperimentConfig,
    data_dir: Path,
    bm25_save_dir: Path,
    progress: Progress = None,
    max_workers: int = 10,
    skip_existing: bool = True
):
    """
    Build BM25 indexes for hypergraph data in parallel

    Args:
        config: Experiment configuration
        data_dir: Hypergraph data directory
        bm25_save_dir: BM25 index save directory
        progress: Rich progress bar object
        max_workers: Maximum concurrency
        skip_existing: Whether to skip existing index files (checkpoint resume)
    """
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Starting Hypergraph BM25 Index Building (Parallel Mode)[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    console.print(f"[bold]Skip existing:[/bold] {'Yes' if skip_existing else 'No'}\n")

    # Initialize NLTK
    console.print("Ensuring NLTK data is available...")
    ensure_nltk_data()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_workers)

    async def process_with_semaphore(conv_id: int, task_id: int):
        """Processing function with semaphore-controlled concurrency"""
        async with semaphore:
            if progress:
                progress.start_task(task_id)
                progress.update(task_id, status="processing")

            # Execute CPU-intensive operations in thread pool
            loop = asyncio.get_event_loop()
            success, status = await loop.run_in_executor(
                None,
                build_bm25_index_for_single_conv,
                conv_id,
                data_dir,
                bm25_save_dir,
                stemmer,
                stop_words,
                skip_existing
            )
            
            if progress:
                if status == "skipped":
                    progress.update(task_id, status="[cyan]exists[/cyan]", completed=1)
                elif status == "completed":
                    progress.update(task_id, status="[green]done[/green]", completed=1)
                else:
                    progress.update(task_id, status="[yellow]skipped[/yellow]", completed=1)
            
            return (conv_id, success, status)
    
    # If progress bar is available, use progress bar processing
    if progress:
        # Create tasks for each conversation
        tasks = []
        for i in range(config.num_conv):
            task_id = progress.add_task(
                f"[cyan]BM25 Index - Conversation {i}[/cyan]",
                total=1,
                status="waiting",
                start=False
            )
            tasks.append((i, task_id))
        
        # Parallel processing
        coroutines = [process_with_semaphore(conv_id, task_id) for conv_id, task_id in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    else:
        # No progress bar mode (simple print)
        coroutines = []
        for i in range(config.num_conv):
            async def simple_process(conv_id):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    build_bm25_index_for_single_conv,
                    conv_id,
                    data_dir,
                    bm25_save_dir,
                    stemmer,
                    stop_words,
                    skip_existing
                )
            coroutines.append(simple_process(i))
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    # Summarize results
    completed_count = 0
    skipped_count = 0
    failed_count = 0
    for r in results:
        if isinstance(r, tuple) and len(r) >= 3:
            success, status = r[1], r[2]
            if status == "skipped":
                skipped_count += 1
            elif status == "completed":
                completed_count += 1
            else:
                failed_count += 1
        elif isinstance(r, Exception):
            failed_count += 1

    console.print(f"\n[bold green][OK] BM25 index building complete[/bold green]")
    console.print(f"    New: {completed_count}, Skipped: {skipped_count}, Failed: {failed_count}")


def build_embedding_index_for_single_conv(
    conv_id: int,
    data_dir: Path,
    emb_save_dir: Path,
    embedding_provider: EmbeddingProvider,
    alpha: float = 1.0,
    aggregate_type: str = "mean",
    batch_size: int = 256,
    skip_existing: bool = True
) -> tuple[bool, str]:
    """
    Build vector embedding index for a single conversation's hypergraph data

    Implementation steps:
    1. Build node embeddings based on nodes' raw text
    2. Build hyperedge embeddings based on node embeddings and hyperedge weights (softmax-normalized weighted sum)
    3. Update node embeddings based on hyperedge embeddings (aggregate hyperedge embeddings, multiply by alpha, and add to original node embeddings)
    4. Save all node and hyperedge embeddings

    Args:
        conv_id: Conversation ID
        data_dir: Hypergraph data directory
        emb_save_dir: Vector index save directory
        embedding_provider: Embedding provider
        alpha: Update weight of hyperedge embeddings on node embeddings, range 0-1
        aggregate_type: Hyperedge aggregation method, "mean" or "sum"
            - mean: all nodes receive the same magnitude of adjustment (more stable)
            - sum: nodes connected to more hyperedges receive larger adjustments (highlights hub nodes)
        batch_size: Batch size
        skip_existing: Whether to skip existing index files

    Returns:
        (success, status: "completed"/"skipped"/"failed")
    """
    try:
        # Check if output file already exists (checkpoint resume)
        output_path = emb_save_dir / f"hypergraph_embedding_index_conv_{conv_id}.pkl"
        output_path_enhanced = emb_save_dir / f"hypergraph_embedding_index_enhanced_conv_{conv_id}.pkl"
        if skip_existing and output_path.exists() and output_path_enhanced.exists():
            return (True, "skipped")
        
        hypergraph_file = data_dir / f"hypergraph_conv_{conv_id}.json"
        if not hypergraph_file.exists():
            return (False, "failed")
        
        # Read hypergraph data
        with open(hypergraph_file, "r", encoding="utf-8") as f:
            hypergraph = json.load(f)

        # ===== Step 1: Collect node texts and generate initial node embeddings =====
        print("  [Step 1] Generating initial node embeddings...")
        
        texts_to_embed = []
        node_text_map = []  # Record node info for each text
        
        # 1.1 Fact nodes (supports triplet and full sentence formats)
        facts = hypergraph.get("facts", {})
        for fact_id, fact_data in facts.items():
            # Build complete text representation for each fact (consistent with BM25 index)
            parts = []

            # Core content
            if fact_data.get("content"):
                # Full sentence format: use content field
                parts.append(fact_data["content"])
            else:
                # Triplet format: use subject/relation/object
                core_text = f"{fact_data.get('subject_name', '')} {fact_data.get('relation', '')} {fact_data.get('object_name', '')}"
                if core_text.strip():
                    parts.append(core_text)

            # Query patterns (important! helps match user questions)
            if fact_data.get("query_patterns"):
                parts.append(" ".join(fact_data["query_patterns"]))

            # Keywords (compatible with keywords and tags)
            keywords = fact_data.get("keywords") or fact_data.get("tags") or []
            if keywords:
                if isinstance(keywords, list):
                    parts.append(" ".join(keywords))
                else:
                    parts.append(str(keywords))

            # Background information
            if fact_data.get("background"):
                parts.append(fact_data["background"])

            # Temporal information
            if fact_data.get("temporal"):
                parts.append(str(fact_data["temporal"]))

            # Spatial information
            if fact_data.get("spatial"):
                parts.append(str(fact_data["spatial"]))

            # Impact/result
            if fact_data.get("impact"):
                parts.append(fact_data["impact"])

            fact_text = " ".join(parts)
            if fact_text.strip():
                texts_to_embed.append(fact_text)
                node_text_map.append({
                    "node_type": "fact",
                    "node_id": fact_id,
                    "data": fact_data
                })
        
        # 1.2 Episode nodes
        episodes = hypergraph.get("episodes", {})
        for episode_id, episode_data in episodes.items():
            # Merge subject, summary, and episode as the complete text representation for the episode
            parts = []
            if episode_data.get("subject"):
                parts.append(episode_data["subject"])
            if episode_data.get("summary"):
                parts.append(episode_data["summary"])
            if episode_data.get("episode_description"):
                parts.append(episode_data["episode_description"])

            if parts:
                episode_text = " ".join(parts)
                texts_to_embed.append(episode_text)
                node_text_map.append({
                    "node_type": "episode",
                    "node_id": episode_id,
                    "data": episode_data
                })
        
        # 1.3 Topic nodes
        topics = hypergraph.get("topics", {})
        for topic_id, topic_data in topics.items():
            # Merge title and summary as the complete text representation for the topic
            parts = []
            if topic_data.get("title"):
                parts.append(topic_data["title"])
            if topic_data.get("summary"):
                parts.append(topic_data["summary"])

            if parts:
                topic_text = " ".join(parts)
                texts_to_embed.append(topic_text)
                node_text_map.append({
                    "node_type": "topic",
                    "node_id": topic_id,
                    "data": topic_data
                })
        
        if not texts_to_embed:
            return (False, "failed")
        
        # Batch generate node embeddings
        all_node_embeddings = []
        for j in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[j:j+batch_size]
            batch_embeddings = embedding_provider.embed(batch_texts)
            all_node_embeddings.extend(batch_embeddings)
        
        # Build node ID to embedding mapping
        node_embeddings = {}  # {(node_type, node_id): embedding}
        embedding_dim = None
        for node_info, embedding in zip(node_text_map, all_node_embeddings):
            key = (node_info["node_type"], node_info["node_id"])
            emb_array = np.array(embedding)
            node_embeddings[key] = emb_array
            if embedding_dim is None:
                embedding_dim = len(emb_array)
        
        # ===== Step 2: Build hyperedge embeddings based on node embeddings =====
        
        hyperedge_embeddings = {}  # {(edge_type, edge_id): embedding}
        
        # 2.1 Fact hyperedges (fact_hyperedges)
        fact_hyperedges = hypergraph.get("fact_hyperedges", {})
        for edge_id, edge_data in fact_hyperedges.items():
            weights_dict = edge_data.get("weights", {})
            relation_dict = edge_data.get("relation", {})

            if not weights_dict or not relation_dict:
                continue

            # Get connected fact nodes and their weights
            fact_ids = list(weights_dict.keys())
            weights = [weights_dict[fid] for fid in fact_ids]

            # Softmax normalize weights (using torch.softmax)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            normalized_weights = torch.softmax(weights_tensor, dim=0).numpy()

            # Weighted sum to compute hyperedge embedding
            edge_embedding = None
            valid_nodes = 0
            for fact_id, weight in zip(fact_ids, normalized_weights):
                node_key = ("fact", fact_id)
                if node_key in node_embeddings:
                    if edge_embedding is None:
                        edge_embedding = np.zeros_like(node_embeddings[node_key])
                    edge_embedding += weight * node_embeddings[node_key]
                    valid_nodes += 1
            
            if valid_nodes > 0 and edge_embedding is not None:
                hyperedge_embeddings[("fact_hyperedge", edge_id)] = edge_embedding
        
        # 2.2 Episode hyperedges (episode_hyperedges)
        episode_hyperedges = hypergraph.get("episode_hyperedges", {})
        for edge_id, edge_data in episode_hyperedges.items():
            weights_dict = edge_data.get("weights", {})
            relation_dict = edge_data.get("relation", {})

            if not weights_dict or not relation_dict:
                continue

            # Get connected episode nodes and their weights
            episode_ids = list(weights_dict.keys())
            weights = [weights_dict[eid] for eid in episode_ids]

            # Softmax normalize weights (using torch.softmax)
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            normalized_weights = torch.softmax(weights_tensor, dim=0).numpy()

            # Weighted sum to compute hyperedge embedding
            edge_embedding = None
            valid_nodes = 0
            for episode_id, weight in zip(episode_ids, normalized_weights):
                node_key = ("episode", episode_id)
                if node_key in node_embeddings:
                    if edge_embedding is None:
                        edge_embedding = np.zeros_like(node_embeddings[node_key])
                    edge_embedding += weight * node_embeddings[node_key]
                    valid_nodes += 1

            if valid_nodes > 0 and edge_embedding is not None:
                hyperedge_embeddings[("episode_hyperedge", edge_id)] = edge_embedding
        
        # ===== Step 3: Update node embeddings based on hyperedge embeddings =====

        # Record hyperedges connected to each node
        node_to_hyperedges = defaultdict(list)  # {(node_type, node_id): [(edge_type, edge_id), ...]}
        
        # 3.1 Build connections from fact hyperedges
        for edge_id, edge_data in fact_hyperedges.items():
            relation_dict = edge_data.get("relation", {})
            for fact_id in relation_dict.keys():
                node_key = ("fact", fact_id)
                edge_key = ("fact_hyperedge", edge_id)
                if edge_key in hyperedge_embeddings:
                    node_to_hyperedges[node_key].append(edge_key)
        
        # 3.2 Build connections from episode hyperedges
        for edge_id, edge_data in episode_hyperedges.items():
            relation_dict = edge_data.get("relation", {})
            for episode_id in relation_dict.keys():
                node_key = ("episode", episode_id)
                edge_key = ("episode_hyperedge", edge_id)
                if edge_key in hyperedge_embeddings:
                    node_to_hyperedges[node_key].append(edge_key)
        
        # 3.3 Update node embeddings
        updated_node_embeddings = {}
        for node_key, original_embedding in node_embeddings.items():
            if node_key in node_to_hyperedges:
                # Get all connected hyperedge embeddings
                connected_edges = node_to_hyperedges[node_key]
                edge_embeddings_list = [hyperedge_embeddings[edge_key] for edge_key in connected_edges]
                
                # Choose aggregation method based on aggregation type
                if aggregate_type == "sum":
                    # sum: nodes connected to more hyperedges receive larger adjustments, highlights hub nodes
                    final_edge_embedding = np.sum(edge_embeddings_list, axis=0)
                else:
                    # mean (default): all nodes receive the same magnitude of adjustment, more stable
                    final_edge_embedding = np.mean(edge_embeddings_list, axis=0)

                # Update node embedding: new = original + alpha * final_edge_embedding
                updated_embedding = original_embedding + alpha * final_edge_embedding
                updated_node_embeddings[node_key] = updated_embedding
            else:
                # Node has no connected hyperedges, keep original embedding
                updated_node_embeddings[node_key] = original_embedding
        
        updated_count = sum(1 for k in updated_node_embeddings if k in node_to_hyperedges)
        
        # ===== Step 4: Organize and save all embeddings =====

        # Complete structure containing nodes, hyperedges, and metadata
        embedding_index_new = {
            "nodes": {},
            "hyperedges": {},
            "metadata": {
                "alpha": alpha,
                "aggregate_type": aggregate_type,
                "num_nodes": len(updated_node_embeddings),
                "num_hyperedges": len(hyperedge_embeddings),
                "num_updated_nodes": updated_count
            }
        }
        
        # List format for existing retrieval code
        embedding_index = []
        
        # Save node embeddings
        for (node_type, node_id), embedding in updated_node_embeddings.items():
            if node_type not in embedding_index_new["nodes"]:
                embedding_index_new["nodes"][node_type] = {}
            
            # Get original data
            if node_type == "fact":
                original_data = facts.get(node_id, {})
            elif node_type == "episode":
                original_data = episodes.get(node_id, {})
            elif node_type == "topic":
                original_data = topics.get(node_id, {})
            else:
                original_data = {}
            
            embedding_index_new["nodes"][node_type][node_id] = {
                "embedding": embedding.tolist(),
                "data": original_data,
                "is_updated": (node_type, node_id) in node_to_hyperedges
            }
            
            embedding_index.append({
                "type": node_type,
                "id": node_id,
                "field": "combined",  # Indicates this is a combined node embedding
                "embedding": embedding.tolist(),
                "data": original_data
            })
        
        # Save hyperedge embeddings
        for (edge_type, edge_id), embedding in hyperedge_embeddings.items():
            if edge_type not in embedding_index_new["hyperedges"]:
                embedding_index_new["hyperedges"][edge_type] = {}
            
            # Get original hyperedge data
            if edge_type == "fact_hyperedge":
                original_data = fact_hyperedges.get(edge_id, {})
            elif edge_type == "episode_hyperedge":
                original_data = episode_hyperedges.get(edge_id, {})
            else:
                original_data = {}
            
            embedding_index_new["hyperedges"][edge_type][edge_id] = {
                "embedding": embedding.tolist(),
                "data": original_data
            }
        
        # Save standard format
        output_path = emb_save_dir / f"hypergraph_embedding_index_conv_{conv_id}.pkl"
        emb_save_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(embedding_index, f)
        
        # Save enhanced format with additional metadata
        with open(output_path_enhanced, "wb") as f:
            pickle.dump(embedding_index_new, f)
        
        return (True, "completed")
        
    except Exception as e:
        console.print(f"  [red][X] Conversation {conv_id}: Embedding index building failed - {e}[/red]")
        return (False, "failed")


async def build_embedding_index_for_hypergraph(
    config: ExperimentConfig,
    data_dir: Path,
    emb_save_dir: Path,
    alpha: float = 1.0,
    aggregate_type: str = "mean",
    progress: Progress = None,
    max_workers: int = 10,
    skip_existing: bool = True
):
    """
    Build vector embedding indexes for hypergraph data in parallel

    Args:
        config: Experiment configuration
        data_dir: Hypergraph data directory
        emb_save_dir: Vector index save directory
        alpha: Update weight of hyperedge embeddings on node embeddings
        aggregate_type: Hyperedge aggregation method, "mean" or "sum"
        progress: Rich progress bar object
        max_workers: Maximum concurrency
        skip_existing: Whether to skip existing index files (checkpoint resume)
    """
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Starting Hypergraph Vector Embedding Index Building (Parallel Mode, with Hyperedge Propagation)[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    console.print(f"[bold]Alpha parameter:[/bold] {alpha} (influence weight of hyperedge info on nodes)")
    console.print(f"[bold]Aggregation method:[/bold] {aggregate_type} (mean=stable/sum=highlights hubs)")
    console.print(f"[bold]Skip existing:[/bold] {'Yes' if skip_existing else 'No'}\n")
    
    # Initialize embedding provider
    embedding_provider = EmbeddingProvider(
        base_url=config.embedding_config["base_url"],
        model_name=config.embedding_config["model_name"]
    )
    BATCH_SIZE = 256
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_workers)

    async def process_with_semaphore(conv_id: int, task_id: int):
        """Processing function with semaphore-controlled concurrency"""
        async with semaphore:
            if progress:
                progress.start_task(task_id)
                progress.update(task_id, status="processing")

            # Execute IO-intensive operations in thread pool
            loop = asyncio.get_event_loop()
            success, status = await loop.run_in_executor(
                None,
                build_embedding_index_for_single_conv,
                conv_id,
                data_dir,
                emb_save_dir,
                embedding_provider,
                alpha,
                aggregate_type,
                BATCH_SIZE,
                skip_existing
            )
            
            if progress:
                if status == "skipped":
                    progress.update(task_id, status="[cyan]exists[/cyan]", completed=1)
                elif status == "completed":
                    progress.update(task_id, status="[green]done[/green]", completed=1)
                else:
                    progress.update(task_id, status="[red]failed[/red]", completed=1)
            
            return (conv_id, success, status)
    
    # If progress bar is available, use progress bar processing
    if progress:
        # Create tasks for each conversation
        tasks = []
        for i in range(config.num_conv):
            task_id = progress.add_task(
                f"[cyan]Embedding Index - Conversation {i}[/cyan]",
                total=1,
                status="waiting",
                start=False
            )
            tasks.append((i, task_id))
        
        # Parallel processing
        coroutines = [process_with_semaphore(conv_id, task_id) for conv_id, task_id in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    else:
        # No progress bar mode
        coroutines = []
        for i in range(config.num_conv):
            async def simple_process(conv_id):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    build_embedding_index_for_single_conv,
                    conv_id,
                    data_dir,
                    emb_save_dir,
                    embedding_provider,
                    alpha,
                    aggregate_type,
                    BATCH_SIZE,
                    skip_existing
                )
            coroutines.append(simple_process(i))
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    # Summarize results
    completed_count = 0
    skipped_count = 0
    failed_count = 0
    for r in results:
        if isinstance(r, tuple) and len(r) >= 3:
            success, status = r[1], r[2]
            if status == "skipped":
                skipped_count += 1
            elif status == "completed":
                completed_count += 1
            else:
                failed_count += 1
        elif isinstance(r, Exception):
            failed_count += 1

    console.print(f"\n[bold green][OK] Embedding index building complete[/bold green]")
    console.print(f"    New: {completed_count}, Skipped: {skipped_count}, Failed: {failed_count}")


async def main():
    """Main function: build hypergraph indexes in parallel"""
    config = ExperimentConfig()
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Stage 3: Hypergraph Index Building[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Configure paths
    hypergraph_dir = config.hypergraph_dir()
    bm25_save_dir = config.bm25_index_dir()
    emb_save_dir = config.vectors_dir()

    # Create save directories
    bm25_save_dir.mkdir(parents=True, exist_ok=True)
    emb_save_dir.mkdir(parents=True, exist_ok=True)
    
    # Concurrency setting
    max_concurrent_tasks = 1
    
    # Checkpoint resume setting: whether to skip existing index files
    skip_existing = True  # True: skip existing files; False: force regeneration of all indexes
    
    console.print(f"[bold]Experiment name:[/bold] {config.experiment_name}")
    console.print(f"[bold]Hypergraph directory:[/bold] {hypergraph_dir}")
    console.print(f"[bold]BM25 index save directory:[/bold] {bm25_save_dir}")
    console.print(f"[bold]Vector index save directory:[/bold] {emb_save_dir}")
    console.print(f"[bold]Number of conversations:[/bold] {config.num_conv}")
    console.print(f"[bold]Concurrency:[/bold] {max_concurrent_tasks}")
    console.print(f"[bold]Checkpoint resume:[/bold] {'Enabled' if skip_existing else 'Disabled'}\n")
    
    # Create progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed:>3}/{task.total:<3}"),  # Right-align completed count, left-align total count
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[bold]{task.fields[status]}"),
        console=console,
        transient=False
    ) as progress:
        # Build BM25 index
        await build_bm25_index_for_hypergraph(
            config, 
            hypergraph_dir, 
            bm25_save_dir,
            progress=progress,
            max_workers=max_concurrent_tasks,
            skip_existing=skip_existing
        )
        
        # Build vector embedding index (if retrieval type requires vectors)
        retrieval_type = getattr(config, 'retrieval_type', 'rrf').lower()
        need_emb = retrieval_type in ('vector', 'rrf')
        if need_emb:
            # Read parameters from configuration
            aggregate_type = getattr(config, 'hyperedge_emb_aggregate_type', 'mean')
            alpha = getattr(config, 'node_emb_update_weight', 1.0)
            await build_embedding_index_for_hypergraph(
                config, 
                hypergraph_dir, 
                emb_save_dir,
                alpha=alpha,
                aggregate_type=aggregate_type,
                progress=progress,
                max_workers=max_concurrent_tasks,
                skip_existing=skip_existing
            )
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]All Hypergraph Index Building Complete![/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")


if __name__ == "__main__":
    asyncio.run(main())

