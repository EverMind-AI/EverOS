"""
Hypergraph Hierarchical Retrieval

Features:
1. Top-down three-layer retrieval mode
   - Layer 1: Retrieve relevant topics
   - Layer 2: Retrieve relevant memory cells (episodes) from relevant topics
   - Layer 3: Retrieve relevant fact triplets from relevant memory cells
2. Supports BM25 and vector retrieval
3. Supports reranking optimization
4. Leverages hyperedge information and weights for retrieval optimization

Data flow:
hypergraph indexes → topic retrieval → episode retrieval → fact retrieval → reranking → formatting
"""

import sys
import pickle
import json
import asyncio
from pathlib import Path
import nltk
import numpy as np
from typing import List, Tuple, Dict, Any, Set, Optional
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn,
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hypermem.llm.embedding_provider import EmbeddingProvider
from hypermem.llm.reranker_provider import RerankerProvider
from hypermem.config import ExperimentConfig

console = Console()

# RRF (Reciprocal Rank Fusion) constant
RRF_K = 60  # k value in the RRF formula, typically set to 60


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[Dict, float]]],
    top_n: int,
    k: int = RRF_K
) -> List[Tuple[Dict, float]]:
    """
    Fuse multiple retrieval result lists using Reciprocal Rank Fusion (RRF)

    RRF formula: RRF_score(d) = Σ 1 / (k + rank(d))

    Args:
        results_list: Multiple retrieval result lists, each containing (doc, score) tuples
        top_n: Return top N results
        k: RRF parameter, typically set to 60

    Returns:
        Fused result list [(doc, rrf_score), ...]
    """
    # Store RRF scores and document data for each document
    doc_scores = {}  # doc_id -> {"doc": doc, "score": rrf_score}
    
    for results in results_list:
        for rank, (doc, _) in enumerate(results):
            # Get document ID
            doc_id = doc.get("id")
            if doc_id is None:
                # If no id exists, use the hash of the document content as identifier
                doc_id = hash(str(doc))
            
            # Calculate RRF score
            rrf_score = 1.0 / (k + rank + 1)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += rrf_score
            else:
                doc_scores[doc_id] = {"doc": doc, "score": rrf_score}
    
    # Sort by RRF score
    sorted_results = sorted(
        [(item["doc"], item["score"]) for item in doc_scores.values()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results[:top_n]


def build_retrieval_template(input_type: str = "011") -> str:
    """
    Build retrieval result template based on input type

    Args:
        input_type: Three-digit binary code, corresponding to topic, episode, fact from left to right
            - "111": topic + episode + fact
            - "110": topic + episode
            - "101": topic + fact
            - "100": topic only
            - "011": episode + fact
            - "010": episode only
            - "001": fact only

    Returns:
        Formatted template string
    """
    # Parse input type
    if len(input_type) != 3 or not all(c in '01' for c in input_type):
        console.print(f"[yellow]Warning: Invalid input_type '{input_type}', using default '011'[/yellow]")
        input_type = "011"
    
    use_topic = input_type[0] == '1'
    use_episode = input_type[1] == '1'
    use_fact = input_type[2] == '1'
    
    # Build template
    template_parts = ["Episodes memories for conversation between {speaker_1} and {speaker_2}:"]
    
    if use_topic:
        template_parts.append("""
## Relevant Topics:
{topics}""")

    if use_episode:
        template_parts.append("""
## Relevant Memory Cells:
{episodes}""")

    if use_fact:
        template_parts.append("""
## Relevant Facts:
{facts}""")
    
    return "".join(template_parts)

# Temporal keywords (used to detect temporal questions)
TEMPORAL_KEYWORDS = [
    'when', 'what time', 'what date', 'which year', 'which month', 'which day',
    'how long', 'how many years', 'how many months', 'how many days',
    'since when', 'until when', 'before', 'after', 'during',
    'recently', 'last', 'next', 'ago', 'in the past', 'in the future',
    'date', 'time', 'year', 'month', 'week', 'day'
]

# Factual question keywords
FACTUAL_KEYWORDS = [
    'what is', 'who is', 'who are', 'where is', 'where are',
    'what did', 'who did', "what's", "who's",
    'which', 'name', 'identity', 'called'
]

# Reasoning question keywords
REASONING_KEYWORDS = [
    'why', 'how', 'would', 'could', 'should',
    'likely', 'probably', 'consider', 'think',
    'reason', 'because', 'explain', 'infer'
]

# Commonsense question keywords
COMMONSENSE_KEYWORDS = [
    'usually', 'normally', 'typically', 'generally',
    'common', 'often', 'always', 'never',
    'most', 'least', 'best', 'worst'
]


def detect_question_type(query: str) -> str:
    """
    Detect question type

    Args:
        query: Query question

    Returns:
        Question type: 'temporal', 'factual', 'reasoning', 'commonsense', 'default'
    """
    query_lower = query.lower()
    
    # Priority 1: Detect temporal questions
    if any(keyword in query_lower for keyword in TEMPORAL_KEYWORDS):
        return 'temporal'
    
    # Priority 2: Detect reasoning questions
    if any(keyword in query_lower for keyword in REASONING_KEYWORDS):
        return 'reasoning'
    
    # Priority 3: Detect factual questions
    if any(keyword in query_lower for keyword in FACTUAL_KEYWORDS):
        return 'factual'
    
    # Priority 4: Detect commonsense questions
    if any(keyword in query_lower for keyword in COMMONSENSE_KEYWORDS):
        return 'commonsense'
    
    # Default
    return 'default'


def is_temporal_question(query: str) -> bool:
    """
    Determine whether the question is a temporal question

    Args:
        query: Query question

    Returns:
        True if temporal question, False otherwise
    """
    return detect_question_type(query) == 'temporal'


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


def tokenize(text: str, stemmer, stop_words: set) -> list[str]:
    """
    NLTK tokenization, consistent with index building

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


def search_with_bm25(
    query: str, 
    bm25, 
    docs: List[Dict], 
    doc_type: str,
    top_n: int = 5
) -> List[Tuple[Dict, float]]:
    """
    Retrieve using BM25 index, filtered by type

    Args:
        query: Query text
        bm25: BM25 index object
        docs: Document list
        doc_type: Document type filter ("topic", "episode", "fact")
        top_n: Return top N results

    Returns:
        List of (document data, score) tuples (returns content of the data field)
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokenized_query = tokenize(query, stemmer, stop_words)
    
    if not tokenized_query:
        print(f"Warning: Query is empty after tokenization for {doc_type}")
        return []

    # Get scores for all documents
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Filter documents of the specified type and extract the data field
    filtered_results = [
        (doc.get("data", doc), score) for doc, score in zip(docs, doc_scores)
        if doc.get("type") == doc_type
    ]
    
    # Sort by score
    sorted_results = sorted(filtered_results, key=lambda x: x[1], reverse=True)

    return sorted_results[:top_n]


def search_with_emb(
    query: str,
    emb_index: List[Dict],
    embedding_provider: EmbeddingProvider,
    doc_type: str,
    top_n: int = 5
) -> List[Tuple[Dict, float]]:
    """
    Retrieve using vector embedding index, filtered by type

    Args:
        query: Query text
        emb_index: Vector index list
        embedding_provider: Embedding provider
        doc_type: Document type filter
        top_n: Return top N results

    Returns:
        List of (document data, score) tuples
    """
    query_vec = np.array(embedding_provider.embed([query])[0])
    
    # Filter documents of the specified type
    filtered_items = [
        item for item in emb_index
        if item.get("type") == doc_type
    ]
    
    if not filtered_items:
        return []
    
    # Extract vectors and corresponding data
    embeddings = [item["embedding"] for item in filtered_items]
    embeddings_np = np.array(embeddings)
    
    # L2 norm normalization
    query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)

    # Calculate cosine similarity
    scores = embedding_provider.cosine_similarity(query_vec, embeddings_np)
    
    # Build result list (return complete data dict)
    results_with_scores = [
        (filtered_items[i]["data"], scores[i])
        for i in range(len(filtered_items))
    ]
    
    # Sort by score
    sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    
    return sorted_results[:top_n]


def rerank_results(
    query: str,
    results: List[Tuple[Dict, float]],
    reranker_provider: RerankerProvider,
    text_field: str,
    top_n: int = 5
) -> List[Tuple[Dict, float]]:
    """
    Rerank retrieval results using a Reranker

    Args:
        query: Query text
        results: Initial retrieval results
        reranker_provider: Reranker provider
        text_field: Text field used for reranking
        top_n: Return top N results

    Returns:
        Reranked results
    """
    if not results:
        return []
    
    # Extract documents and text
    docs = []
    doc_texts = []
    for doc, score in results:
        text = doc.get(text_field, "")
        if text:
            docs.append(doc)
            doc_texts.append(text)
    
    if not doc_texts:
        return []
    
    # Prepare query list
    queries = [query] * len(doc_texts)
    
    # Get reranking scores
    rerank_scores = reranker_provider.rerank(queries, doc_texts)
    
    # Build new result list
    reranked_results = list(zip(docs, rerank_scores))
    
    # Sort by reranking score
    sorted_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
    
    return sorted_results[:top_n]


def get_connected_episodes(
    topic_ids: Set[str],
    hypergraph: Dict[str, Any]
) -> Dict[str, float]:
    """
    Get connected episodes and their weights based on topic IDs

    Args:
        topic_ids: Set of topic IDs
        hypergraph: Complete hypergraph data

    Returns:
        {episode_id: weight} dictionary
    """
    episode_weights = defaultdict(float)

    # Traverse episode_hyperedges to find episodes connected to the topics
    episode_hyperedges = hypergraph.get("episode_hyperedges", {})

    for hyperedge_id, hyperedge_data in episode_hyperedges.items():
        topic_node_id = hyperedge_data.get("topic_node_id")

        # If the topic connected by this hyperedge is among our retrieved topics
        if topic_node_id in topic_ids:
            # Get all connected episodes and their weights
            episode_relations = hyperedge_data.get("relation", {})
            weights = hyperedge_data.get("weights", {})

            for episode_id, relation_type in episode_relations.items():
                weight = weights.get(episode_id, 0.5)
                # Accumulate weights (if an episode belongs to multiple relevant topics)
                episode_weights[episode_id] = max(episode_weights[episode_id], weight)

    return dict(episode_weights)


def get_connected_facts(
    episode_ids: Set[str],
    hypergraph: Dict[str, Any]
) -> Dict[str, Tuple[float, str]]:
    """
    Get connected fact triplets, their weights, and relation types based on episode IDs

    Args:
        episode_ids: Set of episode IDs
        hypergraph: Complete hypergraph data

    Returns:
        {fact_id: (weight, relation_type)} dictionary
    """
    fact_info = {}

    # Traverse fact_hyperedges to find facts connected to the episodes
    fact_hyperedges = hypergraph.get("fact_hyperedges", {})

    for hyperedge_id, hyperedge_data in fact_hyperedges.items():
        episode_node_id = hyperedge_data.get("episode_node_id")

        # If the episode connected by this hyperedge is among our retrieved episodes
        if episode_node_id in episode_ids:
            # Get all connected facts and their weights and relation types
            fact_relations = hyperedge_data.get("relation", {})
            weights = hyperedge_data.get("weights", {})

            for fact_id, relation_type in fact_relations.items():
                weight = weights.get(fact_id, 0.5)
                # Keep the information with the highest weight
                if fact_id not in fact_info or fact_info[fact_id][0] < weight:
                    fact_info[fact_id] = (weight, relation_type)

    return fact_info


def hierarchical_retrieval(
    query: str,
    hypergraph: Dict[str, Any],
    bm25=None,
    docs=None,
    emb_index=None,
    embedding_provider: EmbeddingProvider = None,
    reranker_provider: RerankerProvider = None,
    config: ExperimentConfig = None
) -> Dict[str, List[Tuple[Dict, float]]]:
    """
    Top-down hierarchical retrieval

    Args:
        query: Query question
        hypergraph: Complete hypergraph data structure
        bm25: BM25 index (if using BM25)
        docs: Document list (if using BM25)
        emb_index: Vector index (if using vector retrieval)
        embedding_provider: Embedding provider
        reranker_provider: Reranker provider
        config: Experiment configuration

    Returns:
        Dictionary containing three-layer retrieval results
    """
    results = {
        "topics": [],
        "episodes": [],
        "facts": []
    }

    # Retrieval log: captures the full retrieval flow for debugging
    retrieval_log = {
        "query": query,
        "config": {},
        "layer1_topic": {},
        "layer2_episode": {},
        "layer3_fact": {},
    }

    # Parse retrieval output type to determine which layers need to be retrieved
    # Three-digit binary code: topic/episode/fact
    output_type = getattr(config, 'hypergraph_retrieval_output_type', '011')
    if len(output_type) != 3:
        output_type = '011'
    need_topic_output = output_type[0] == '1'
    need_episode_output = output_type[1] == '1'
    need_fact_output = output_type[2] == '1'
    
    # Detect question type
    question_type = detect_question_type(query)
    
    # Differentiated retrieval: select retrieval parameters based on question type
    rerank_type = getattr(config, 'rerank_type', 'ada').lower()
    if rerank_type == 'ada' and question_type in config.adaptive_retrieval_config:
        adaptive_config = config.adaptive_retrieval_config[question_type]
        topic_top_k = adaptive_config["topic_top_k"]
        episode_top_k = adaptive_config["episode_top_k"]
        fact_top_k = adaptive_config["fact_top_k"]
        initial_candidates = adaptive_config["initial_candidates"]
        print(f"  [ADAPTIVE] Detected {question_type} type question, using adaptive config: topic={topic_top_k}, episode={episode_top_k}, fact={fact_top_k}")
    else:
        # Use default configuration
        topic_top_k = config.retrieval_config["topic_top_k"]
        episode_top_k = config.retrieval_config["episode_top_k"]
        fact_top_k = config.retrieval_config["fact_top_k"]
        initial_candidates = config.retrieval_config["initial_candidates"]
    
    # Temporal enhancement detection
    is_temporal = config.temporal_enhancement and (question_type == 'temporal')
    if is_temporal:
        print(f"  [TEMPORAL] Temporal information enhancement enabled")
    
    # === Layer 1: Retrieve relevant topics ===

    print(f"  [Layer 1] Retrieving relevant topics...")
    retrieve_top_n = initial_candidates if config.use_reranker else topic_top_k
    
    # Select retrieval method based on retrieval type: keyword (BM25 only), vector (vector only), rrf (fusion)
    retrieval_type = getattr(config, 'retrieval_type', 'rrf').lower()
    use_emb = retrieval_type in ('vector', 'rrf')
    use_bm25 = retrieval_type in ('keyword', 'rrf')
    use_rrf = retrieval_type == 'rrf'

    retrieval_log["config"] = {
        "question_type": question_type,
        "topic_top_k": topic_top_k,
        "episode_top_k": episode_top_k,
        "fact_top_k": fact_top_k,
        "initial_candidates": initial_candidates,
        "output_type": output_type,
        "use_reranker": config.use_reranker,
        "retrieval_type": retrieval_type,
    }

    if use_rrf and emb_index and bm25 and docs:
        # RRF hybrid retrieval: use both BM25 and vector retrieval, then fuse
        bm25_topic_results = search_with_bm25(
            query=query,
            bm25=bm25,
            docs=docs,
            doc_type="topic",
            top_n=retrieve_top_n
        )
        emb_topic_results = search_with_emb(
            query=query,
            emb_index=emb_index,
            embedding_provider=embedding_provider,
            doc_type="topic",
            top_n=retrieve_top_n
        )
        topic_results = reciprocal_rank_fusion(
            [bm25_topic_results, emb_topic_results],
            top_n=retrieve_top_n
        )
        print(f"    [RRF] Fused BM25({len(bm25_topic_results)}) + Vector({len(emb_topic_results)}) → {len(topic_results)}")
        retrieval_log["layer1_topic"]["bm25"] = [(d.get("id",""), float(round(s,3))) for d,s in bm25_topic_results]
        retrieval_log["layer1_topic"]["emb"] = [(d.get("id",""), float(round(s,3))) for d,s in emb_topic_results]
        retrieval_log["layer1_topic"]["rrf"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
    elif use_emb and emb_index:
        topic_results = search_with_emb(
            query=query,
            emb_index=emb_index,
            embedding_provider=embedding_provider,
            doc_type="topic",
            top_n=retrieve_top_n
        )
        print(f"    [Vector] Retrieved {len(topic_results)} topics")
        retrieval_log["layer1_topic"]["emb"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
    else:
        topic_results = search_with_bm25(
            query=query,
            bm25=bm25,
            docs=docs,
            doc_type="topic",
            top_n=retrieve_top_n
        )
        print(f"    [BM25] Retrieved {len(topic_results)} topics")
        retrieval_log["layer1_topic"]["bm25"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]

    # Rerank topics
    pre_rerank_topics = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
    if config.use_reranker and topic_results and reranker_provider:
        try:
            topic_results = rerank_results(
                query=query,
                results=topic_results,
                reranker_provider=reranker_provider,
                text_field="summary",
                top_n=topic_top_k
            )
            retrieval_log["layer1_topic"]["reranked"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
        except Exception as e:
            print(f"    [WARNING] Reranker failed, using original results: {e}")
            topic_results = topic_results[:topic_top_k]
    else:
        topic_results = topic_results[:topic_top_k]
    retrieval_log["layer1_topic"]["pre_rerank"] = pre_rerank_topics
    retrieval_log["layer1_topic"]["final"] = [(d.get("id",""), float(round(s,3))) for d,s in topic_results]
    retrieval_log["layer1_topic"]["final_titles"] = [d.get("title","")[:60] for d,s in topic_results]

    results["topics"] = topic_results
    relevant_topic_ids = {topic_data["id"] for topic_data, _ in topic_results}
    print(f"    Found {len(relevant_topic_ids)} relevant topics")

    if not relevant_topic_ids:
        print("    Warning: No relevant topics found, skipping subsequent retrieval")
        results["retrieval_log"] = retrieval_log
        return results
    
    # === Layer 2: Get connected episodes from relevant topics ===
    # Optimization: skip episode retrieval if neither episode nor fact output is needed
    if not need_episode_output and not need_fact_output:
        print(f"  [Layer 2] Skipping episode retrieval (output_type={output_type})")
        print(f"  [Layer 3] Skipping fact retrieval (output_type={output_type})")
        return results

    print(f"  [Layer 2] Retrieving episodes from relevant topics...")
    connected_episodes = get_connected_episodes(relevant_topic_ids, hypergraph)
    print(f"    Found {len(connected_episodes)} episodes via hyperedge connections")
    retrieval_log["layer2_episode"]["connected_count"] = len(connected_episodes)

    if not connected_episodes:
        print("    Warning: No connected episodes found, skipping fact retrieval")
        results["retrieval_log"] = retrieval_log
        return results

    # Retrieve within connected episodes
    # episode_top_k was set at the beginning of the function based on question type
    episode_retrieve_top_n = initial_candidates if config.use_reranker else episode_top_k

    
    # Define helper function for BM25 episode retrieval
    def search_episodes_bm25():
        bm25_episode_results = search_with_bm25(
            query=query,
            bm25=bm25,
            docs=docs,
            doc_type="episode",
            top_n=episode_retrieve_top_n * 2  # Retrieve more, will be filtered later
        )
        # Filter connected episodes (using BM25 scores only, without weighting)
        filtered_results = []
        for doc, score in bm25_episode_results:
            doc_data = doc.get("data", doc) if isinstance(doc, dict) and "data" in doc else doc
            episode_id = doc_data.get("id")
            if episode_id in connected_episodes:
                # Use BM25 score directly, without fusing with hyperedge weights
                filtered_results.append((doc_data, score))
        return sorted(filtered_results, key=lambda x: x[1], reverse=True)[:episode_retrieve_top_n]

    # Define helper function for vector episode retrieval
    def search_episodes_emb():
        filtered_emb_index = [
            item for item in emb_index
            if item.get("type") == "episode" and item.get("id") in connected_episodes
        ]
        if not filtered_emb_index:
            return []
        
        query_vec = np.array(embedding_provider.embed([query])[0])
        embeddings = [item["embedding"] for item in filtered_emb_index]
        embeddings_np = np.array(embeddings)
        
        # L2 norm normalization
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)

        scores = embedding_provider.cosine_similarity(query_vec, embeddings_np)

        # Use vector similarity scores directly, without fusing with hyperedge weights
        emb_results = []
        for i, item in enumerate(filtered_emb_index):
            emb_results.append((item["data"], scores[i]))

        return sorted(emb_results, key=lambda x: x[1], reverse=True)[:episode_retrieve_top_n]

    if use_rrf and emb_index and bm25 and docs:
        # RRF hybrid retrieval
        bm25_episode_results = search_episodes_bm25()
        emb_episode_results = search_episodes_emb()
        episode_results = reciprocal_rank_fusion(
            [bm25_episode_results, emb_episode_results],
            top_n=episode_retrieve_top_n
        )
        print(f"    [RRF] Fused BM25({len(bm25_episode_results)}) + Vector({len(emb_episode_results)}) → {len(episode_results)}")
        retrieval_log["layer2_episode"]["bm25"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in bm25_episode_results[:20]]
        retrieval_log["layer2_episode"]["emb"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in emb_episode_results[:20]]
        retrieval_log["layer2_episode"]["rrf"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in episode_results[:20]]
    elif use_emb and emb_index:
        episode_results = search_episodes_emb()
        print(f"    [Vector] Retrieved {len(episode_results)} episodes")
        retrieval_log["layer2_episode"]["emb"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in episode_results[:20]]
    else:
        episode_results = search_episodes_bm25()
        print(f"    [BM25] Retrieved {len(episode_results)} episodes")
        retrieval_log["layer2_episode"]["bm25"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in episode_results[:20]]

    # Rerank episodes
    pre_rerank_episodes = [(d.get("id","")[:25], float(round(s,3))) for d,s in episode_results]
    if config.use_reranker and episode_results and reranker_provider:
        try:
            episode_results = rerank_results(
                query=query,
                results=episode_results,
                reranker_provider=reranker_provider,
                text_field="episode_description",
                top_n=episode_top_k
            )
            retrieval_log["layer2_episode"]["reranked"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in episode_results]
        except Exception as e:
            print(f"    [WARNING] Episode reranker failed, using original results: {e}")
            episode_results = episode_results[:episode_top_k]
    else:
        episode_results = episode_results[:episode_top_k]
    retrieval_log["layer2_episode"]["pre_rerank"] = pre_rerank_episodes
    retrieval_log["layer2_episode"]["final"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in episode_results]
    retrieval_log["layer2_episode"]["final_subjects"] = [d.get("subject","")[:50] for d,s in episode_results]

    results["episodes"] = episode_results
    relevant_episode_ids = {episode_data["id"] for episode_data, _ in episode_results}
    print(f"    Retrieved {len(relevant_episode_ids)} relevant episodes")

    if not relevant_episode_ids:
        print("    Warning: No relevant episodes found, skipping fact retrieval")
        results["retrieval_log"] = retrieval_log
        return results

    # === Layer 3: Get connected fact triplets from relevant episodes ===
    # Optimization: skip fact retrieval if fact output is not needed
    if not need_fact_output:
        print(f"  [Layer 3] Skipping fact retrieval (output_type={output_type})")
        return results

    print(f"  [Layer 3] Retrieving fact triplets from relevant episodes...")
    connected_facts = get_connected_facts(relevant_episode_ids, hypergraph)
    print(f"    Found {len(connected_facts)} facts via hyperedge connections")
    retrieval_log["layer3_fact"]["connected_count"] = len(connected_facts)

    if not connected_facts:
        print("    Warning: No connected fact triplets found")
        results["retrieval_log"] = retrieval_log
        return results

    # fact_top_k was set at the beginning of the function based on question type
    fact_retrieve_top_n = initial_candidates if config.use_reranker else fact_top_k
    
    # Define helper function for BM25 fact retrieval
    def search_facts_bm25():
        bm25_fact_results = search_with_bm25(
            query=query,
            bm25=bm25,
            docs=docs,
            doc_type="fact",
            top_n=fact_retrieve_top_n * 2
        )
        # Filter connected facts (using BM25 scores only, without weighting)
        filtered_results = []
        for doc, score in bm25_fact_results:
            doc_data = doc.get("data", doc) if isinstance(doc, dict) and "data" in doc else doc
            fact_id = doc_data.get("id")
            if fact_id in connected_facts:
                # Use BM25 score directly, without fusing with hyperedge weights and relation types
                filtered_results.append((doc_data, score))
        return sorted(filtered_results, key=lambda x: x[1], reverse=True)[:fact_retrieve_top_n]

    # Define helper function for vector fact retrieval
    def search_facts_emb():
        filtered_emb_index = [
            item for item in emb_index
            if item.get("type") == "fact" and item.get("id") in connected_facts
        ]
        if not filtered_emb_index:
            return []

        query_vec = np.array(embedding_provider.embed([query])[0])
        embeddings = [item["embedding"] for item in filtered_emb_index]
        embeddings_np = np.array(embeddings)

        # L2 norm normalization
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        embeddings_np = embeddings_np / (np.linalg.norm(embeddings_np, axis=1, keepdims=True) + 1e-8)

        scores = embedding_provider.cosine_similarity(query_vec, embeddings_np)

        # Use vector similarity scores directly, without fusing with hyperedge weights and relation types
        emb_results = []
        for i, item in enumerate(filtered_emb_index):
            emb_results.append((item["data"], scores[i]))

        return sorted(emb_results, key=lambda x: x[1], reverse=True)[:fact_retrieve_top_n]

    # Retrieve within connected facts
    if use_rrf and emb_index and bm25 and docs:
        # RRF hybrid retrieval
        bm25_fact_results = search_facts_bm25()
        emb_fact_results = search_facts_emb()
        fact_results = reciprocal_rank_fusion(
            [bm25_fact_results, emb_fact_results],
            top_n=fact_retrieve_top_n
        )
        print(f"    [RRF] Fused BM25({len(bm25_fact_results)}) + Vector({len(emb_fact_results)}) → {len(fact_results)}")
        retrieval_log["layer3_fact"]["bm25"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in bm25_fact_results[:20]]
        retrieval_log["layer3_fact"]["emb"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in emb_fact_results[:20]]
        retrieval_log["layer3_fact"]["rrf"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in fact_results[:20]]
    elif use_emb and emb_index:
        fact_results = search_facts_emb()
        print(f"    [Vector] Retrieved {len(fact_results)} facts")
        retrieval_log["layer3_fact"]["emb"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in fact_results[:20]]
    else:
        fact_results = search_facts_bm25()
        print(f"    [BM25] Retrieved {len(fact_results)} facts")
        retrieval_log["layer3_fact"]["bm25"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in fact_results[:20]]

    # Rerank fact triplets
    pre_rerank_facts = [(d.get("id","")[:25], float(round(s,3))) for d,s in fact_results]
    if config.use_reranker and fact_results and reranker_provider:
        try:
            fact_texts = []
            fact_docs = []
            for doc, score in fact_results:
                text_parts = []
                if doc.get("content"):
                    text_parts.append(doc["content"])
                else:
                    if doc.get("subject_name"):
                        text_parts.append(doc["subject_name"])
                    if doc.get("relation"):
                        text_parts.append(doc["relation"])
                    if doc.get("object_name"):
                        text_parts.append(doc["object_name"])
                    if doc.get("background"):
                        text_parts.append(doc["background"])

                fact_text = " ".join(text_parts)
                if fact_text:
                    fact_docs.append(doc)
                    fact_texts.append(fact_text)

            if fact_texts:
                queries = [query] * len(fact_texts)
                rerank_scores = reranker_provider.rerank(queries, fact_texts)
                fact_results = sorted(
                    zip(fact_docs, rerank_scores),
                    key=lambda x: x[1],
                    reverse=True
                )[:fact_top_k]
                retrieval_log["layer3_fact"]["reranked"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in fact_results]
        except Exception as e:
            print(f"    [WARNING] Fact reranker failed, using original results: {e}")
            fact_results = fact_results[:fact_top_k]
    else:
        fact_results = fact_results[:fact_top_k]
    retrieval_log["layer3_fact"]["pre_rerank"] = pre_rerank_facts
    retrieval_log["layer3_fact"]["final"] = [(d.get("id","")[:25], float(round(s,3))) for d,s in fact_results]
    retrieval_log["layer3_fact"]["final_contents"] = [d.get("content","")[:60] for d,s in fact_results]

    results["facts"] = fact_results
    results["retrieval_log"] = retrieval_log
    print(f"    Retrieved {len(fact_results)} relevant fact triplets")

    return results


def format_hierarchical_results(
    results: Dict[str, List[Tuple[Dict, float]]],
    speaker_a: str,
    speaker_b: str,
    input_type: str = "011"
) -> str:
    """
    Format hierarchical retrieval results into readable text

    Args:
        results: Hierarchical retrieval results
        speaker_a: Speaker A
        speaker_b: Speaker B
        input_type: Three-digit binary code controlling which content to output (topic/episode/fact)
            - "111": topic + episode + fact
            - "011": episode + fact (default)
            - "001": fact only

    Returns:
        Formatted context string
    """
    # Format topics
    topic_texts = []
    for idx, (topic_data, score) in enumerate(results.get("topics", []), 1):
        title = topic_data.get('title', '')
        summary = topic_data.get('summary', '')
        timestamp = topic_data.get('timestamp', '')
        keywords = topic_data.get('keywords', [])
        
        topic_text = f"[Topic {idx}] {title}"
        if timestamp:
            topic_text += f"\n  Time: {timestamp}"
        if summary:
            topic_text += f"\n  Summary: {summary}"
        if keywords:
            keywords_str = ", ".join(keywords[:5])  # Only display the first 5 keywords
            topic_text += f"\n  Keywords: {keywords_str}"
        
        topic_texts.append(topic_text)
    
    topics_str = "\n\n".join(topic_texts) if topic_texts else "No relevant topics found."
    
    # Format episodes
    episode_texts = []
    for idx, (episode_data, score) in enumerate(results.get("episodes", []), 1):
        episode = episode_data.get('episode_description', '')
        timestamp = episode_data.get('timestamp', '')

        episode_text = f"[Memory {idx}] {episode}"
        if timestamp:
            episode_text += f"\n  Time: {timestamp}"

        episode_texts.append(episode_text)

    episodes_str = "\n\n".join(episode_texts) if episode_texts else "No relevant memory cells found."
    
    # Format facts
    fact_texts = []
    for idx, (fact_data, score) in enumerate(results.get("facts", []), 1):
        # Check fact format: full sentence (content field) or triplet
        content = fact_data.get('content', '')

        if content:
            # Full sentence format: use content field
            fact_text = f"[Fact {idx}] {content}"
            temporal = fact_data.get('temporal', '')
            spatial = fact_data.get('spatial', '')

            if temporal:
                fact_text += f"\n  Time: {temporal}"
            if spatial:
                fact_text += f"\n  Location: {spatial}"
        else:
            # Triplet format
            triple = f"{fact_data.get('subject_name', '?')} - {fact_data.get('relation', '?')} - {fact_data.get('object_name', '?')}"
            fact_text = f"[Fact {idx}] {triple}"
            background = fact_data.get('background', '')
            impact = fact_data.get('impact', '')
            temporal = fact_data.get('temporal', '')

            if temporal:
                fact_text += f"\n  Time: {temporal}"
            if background:
                fact_text += f"\n  Background: {background}"
            if impact:
                fact_text += f"\n  Impact: {impact}"

        fact_texts.append(fact_text)

    facts_str = "\n\n".join(fact_texts) if fact_texts else "No relevant facts found."
    
    # Build dynamic template based on input_type and format
    template = build_retrieval_template(input_type)
    
    # Prepare format arguments (only include fields needed by the template)
    format_args = {
        "speaker_1": speaker_a,
        "speaker_2": speaker_b,
    }
    if "{topics}" in template:
        format_args["topics"] = topics_str
    if "{episodes}" in template:
        format_args["episodes"] = episodes_str
    if "{facts}" in template:
        format_args["facts"] = facts_str
    
    context = template.format(**format_args)
    
    return context


def get_query_count(conversation_data: Dict[str, Any]) -> int:
    """
    Count the number of queries to process in a conversation (excluding category 5)

    Args:
        conversation_data: Conversation data

    Returns:
        Number of queries
    """
    if "qa" not in conversation_data:
        return 0
    
    count = 0
    for qa_pair in conversation_data["qa"]:
        if qa_pair.get("question") and qa_pair.get("category") != 5:
            count += 1
    return count


def process_single_conversation_retrieval(
    conv_id: int,
    conversation_data: Dict[str, Any],
    config: ExperimentConfig,
    hypergraph_dir: Path,
    index_dir: Path,
    embedding_provider: Optional[EmbeddingProvider],
    reranker_provider: Optional[RerankerProvider],
    progress_callback: Optional[callable] = None
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Process retrieval task for a single conversation

    Args:
        conv_id: Conversation ID
        conversation_data: Conversation data
        config: Experiment configuration
        hypergraph_dir: Hypergraph directory
        index_dir: Index directory
        embedding_provider: Embedding provider
        reranker_provider: Reranker provider
        progress_callback: Progress callback function, called after each query is processed

    Returns:
        (conversation ID, list of retrieval results)
    """
    try:
        conv_id_str = f"locomo_exp_user_{conv_id}"
        speaker_a = conversation_data["conversation"].get("speaker_a", "Speaker A")
        speaker_b = conversation_data["conversation"].get("speaker_b", "Speaker B")
        
        if "qa" not in conversation_data:
            console.print(f"  [yellow][!] Conversation {conv_id}: 'qa' field not found[/yellow]")
            return (conv_id_str, [])
        
        # === Load hypergraph data ===
        hypergraph_file = hypergraph_dir / f"hypergraph_conv_{conv_id}.json"
        if not hypergraph_file.exists():
            console.print(f"  [yellow][!] Conversation {conv_id}: Hypergraph file not found[/yellow]")
            return (conv_id_str, [])
        
        with open(hypergraph_file, "r", encoding="utf-8") as f:
            hypergraph = json.load(f)
        
        # === Load indexes (hybrid retrieval requires loading both BM25 and vector indexes) ===
        bm25 = None
        docs = None
        emb_index = None
        
        # Always try to load BM25 index (for hybrid retrieval)
        bm25_index_dir = hypergraph_dir.parent / "bm25_index"
        bm25_index_file = bm25_index_dir / f"hypergraph_bm25_index_conv_{conv_id}.pkl"
        if bm25_index_file.exists():
            try:
                with open(bm25_index_file, "rb") as f:
                    index_data = pickle.load(f)
                bm25 = index_data["bm25"]
                docs = index_data["docs"]
            except (EOFError, pickle.UnpicklingError) as e:
                console.print(f"  [red][!] Conversation {conv_id}: BM25 index file corrupted ({e}), please re-run stage 3[/red]")
        
        # If vector retrieval is enabled, load vector index
        retrieval_type = getattr(config, 'retrieval_type', 'rrf').lower()
        need_emb = retrieval_type in ('vector', 'rrf')
        if need_emb:
            emb_index_dir = hypergraph_dir.parent / "vectors"
            emb_index_file = emb_index_dir / f"hypergraph_embedding_index_conv_{conv_id}.pkl"
            if emb_index_file.exists():
                try:
                    with open(emb_index_file, "rb") as f:
                        emb_index = pickle.load(f)
                except (EOFError, pickle.UnpicklingError) as e:
                    console.print(f"  [red][!] Conversation {conv_id}: Vector index file corrupted ({e}), please re-run stage 3[/red]")
            else:
                console.print(f"  [yellow][!] Conversation {conv_id}: Vector index file not found, using BM25 only[/yellow]")
        
        # Check if at least one index is available
        if bm25 is None and emb_index is None:
            console.print(f"  [yellow][!] Conversation {conv_id}: No available index found[/yellow]")
            return (conv_id_str, [])
        
        # === Perform hierarchical retrieval for each question ===
        results_for_conv = []
        for qa_pair in conversation_data["qa"]:
            question = qa_pair.get("question")
            if not question:
                continue
            
            # Skip category 5 questions
            if qa_pair.get("category") == 5:
                continue
            
            # Execute hierarchical retrieval
            hierarchical_results = hierarchical_retrieval(
                query=question,
                hypergraph=hypergraph,
                bm25=bm25,
                docs=docs,
                emb_index=emb_index,
                embedding_provider=embedding_provider,
                reranker_provider=reranker_provider,
                config=config
            )
            
            # Format results (based on configured input type)
            input_type = getattr(config, 'hypergraph_retrieval_output_type', '011')
            context_str = format_hierarchical_results(
                results=hierarchical_results,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                input_type=input_type
            )
            
            # Save results
            results_for_conv.append({
                "query": question,
                "context": context_str,
                "hierarchical_results": {
                    "topics_count": len(hierarchical_results.get("topics", [])),
                    "episodes_count": len(hierarchical_results.get("episodes", [])),
                    "facts_count": len(hierarchical_results.get("facts", []))
                },
                "retrieval_log": hierarchical_results.get("retrieval_log", {}),
                "original_qa": qa_pair
            })
        
            # Call progress callback
            if progress_callback:
                progress_callback()
        
        return (conv_id_str, results_for_conv)
        
    except Exception as e:
        console.print(f"  [red][X] Conversation {conv_id}: Retrieval failed - {e}[/red]")
        import traceback
        traceback.print_exc()
        return (f"locomo_exp_user_{conv_id}", [])


async def main():
    """Main function: execute batch hypergraph hierarchical retrieval in parallel"""
    # === Configuration ===
    config = ExperimentConfig()
    
    console.print("\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print("[bold cyan]Stage 4: Hypergraph Retrieval[/bold cyan]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")
    
    # Retrieval mode: keyword (BM25 only), vector (vector only), rrf (fusion)
    retrieval_type = getattr(config, 'retrieval_type', 'rrf').lower()
    retrieval_mode_display = {
        'rrf': '[green]RRF Hybrid Retrieval (BM25 + Vector)[/green]',
        'vector': '[cyan]Vector Retrieval (Vector)[/cyan]',
        'keyword': '[yellow]Keyword Retrieval (BM25)[/yellow]',
    }.get(retrieval_type, f'[red]Unknown mode: {retrieval_type}[/red]')
    console.print(f"[bold]Retrieval mode:[/bold] {retrieval_mode_display}")
    
    # Index directory
    index_dir = config.vectors_dir()

    # Hypergraph data directory
    hypergraph_dir = config.hypergraph_dir()

    # Output directory
    save_dir = config.experiment_dir()
    results_output_path = save_dir / "search_results.json"
    
    # Concurrency settings
    max_concurrent_tasks = 10
    
    # Dataset path
    dataset_path = Path(config.dataset_path)
    
    # Initialize services
    embedding_provider = None
    need_emb = retrieval_type in ('vector', 'rrf')
    if need_emb:
        embedding_provider = EmbeddingProvider(
            base_url=config.embedding_config["base_url"],
            model_name=config.embedding_config["model_name"],
            max_retries=config.embedding_max_retries
        )
    
    reranker_provider = None
    if config.use_reranker:
        reranker_provider = RerankerProvider(
            base_url=config.reranker_config["base_url"],
            model_name=config.reranker_config["model_name"],
            max_retries=config.reranker_max_retries
        )
    
    console.print(f"[bold]Concurrency:[/bold] {max_concurrent_tasks}\n")
    
    # Ensure NLTK data is available
    ensure_nltk_data()
    
    # Load dataset
    console.print(f"[bold]Loading dataset:[/bold] {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    console.print(f"[bold]Number of conversations:[/bold] {len(dataset)}\n")
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_with_semaphore(conv_id: int, conversation_data: Dict, task_id: int, progress: Progress, query_count: int):
        """Processing function with semaphore-based concurrency control"""
        async with semaphore:
            progress.start_task(task_id)
            progress.update(task_id, status="Processing")

            # Get the event loop in the main coroutine (before entering the thread pool)
            main_loop = asyncio.get_running_loop()
            
            # Create thread-safe progress callback (capturing main_loop via closure)
            def progress_callback():
                # Use call_soon_threadsafe to ensure thread safety
                main_loop.call_soon_threadsafe(
                    progress.advance, task_id, 1
                )
            
            # Execute retrieval in thread pool (since retrieval involves CPU-intensive operations)
            result = await main_loop.run_in_executor(
                None,
                process_single_conversation_retrieval,
                conv_id,
                conversation_data,
                config,
                hypergraph_dir,
                index_dir,
                embedding_provider,
                reranker_provider,
                progress_callback
            )
            
            conv_id_str, results_for_conv = result
            
            # Ensure progress bar is completed
            progress.update(task_id, completed=query_count, status=f"[green]Done[/green]")
            
            return result
    
    # Create progress bar for parallel processing
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.completed:>3}/{task.total:<3}"),  # Right-align completed count, left-align total
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        TextColumn("•"),
        TextColumn("[bold]{task.fields[status]}"),
        console=console,
        transient=False
    ) as progress:
        # Create tasks for each conversation and count queries
        tasks = []
        for i, conversation_data in enumerate(dataset):
            query_count = get_query_count(conversation_data)
            task_id = progress.add_task(
                f"[cyan]Conv {i}[/cyan]",
                total=query_count if query_count > 0 else 1,
                status="Waiting",
                start=False
            )
            tasks.append((i, conversation_data, task_id, query_count))
        
        # Parallel processing
        coroutines = [
            process_with_semaphore(conv_id, conv_data, task_id, progress, query_count)
            for conv_id, conv_data, task_id, query_count in tasks
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
    
    # Organize results
    all_search_results = {}
    for result in results:
        if isinstance(result, tuple):
            conv_id_str, results_for_conv = result
            all_search_results[conv_id_str] = results_for_conv
        elif isinstance(result, Exception):
            console.print(f"[red][X] Processing exception: {result}[/red]")

    # === Save all results ===
    console.print(f"\n[bold cyan]" + "="*80 + "[/bold cyan]")
    console.print(f"[bold]Saving retrieval results to:[/bold] {results_output_path}")
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(all_search_results, f, indent=2, ensure_ascii=False)

    # === Save retrieval logs separately for analysis ===
    retrieval_logs_path = save_dir / "retrieval_logs.json"
    all_logs = {}
    for conv_id, items in all_search_results.items():
        all_logs[conv_id] = [
            {"query": item.get("query", ""), "retrieval_log": item.get("retrieval_log", {})}
            for item in items
        ]
    with open(retrieval_logs_path, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, indent=2, ensure_ascii=False)
    console.print(f"[bold]Saving retrieval logs to:[/bold] {retrieval_logs_path}")

    console.print(f"[bold green][SUCCESS] Hypergraph hierarchical retrieval completed![/bold green]")
    console.print("[bold cyan]" + "="*80 + "[/bold cyan]\n")


if __name__ == "__main__":
    asyncio.run(main())

