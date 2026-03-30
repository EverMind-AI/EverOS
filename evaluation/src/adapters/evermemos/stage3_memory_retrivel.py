import json
import os
import sys
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import time

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import asyncio
from tqdm import tqdm


from evaluation.src.adapters.evermemos.config import ExperimentConfig

# from evaluation.src.adapters.evermemos.tools.embedding_provider import EmbeddingProvider
# from evaluation.src.adapters.evermemos.tools.reranker_provider import RerankerProvider
from agentic_layer.vectorize_service import get_vectorize_service
from agentic_layer import rerank_service

from evaluation.src.adapters.evermemos.tools import agentic_utils

from memory_layer.llm.llm_provider import LLMProvider


# This file depends on the rank_bm25 library.
# If you haven't installed it yet, run: pip install rank_bm25


def ensure_nltk_data():
    """Ensure required NLTK data is downloaded."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading punkt...")
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("Downloading punkt_tab...")
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        print("Downloading stopwords...")
        nltk.download("stopwords", quiet=True)

    # Verify stopwords availability
    try:
        from nltk.corpus import stopwords

        test_stopwords = stopwords.words("english")
        if not test_stopwords:
            raise ValueError("Stopwords is empty")
    except Exception as e:
        print(f"Warning: NLTK stopwords error: {e}")
        print("Re-downloading stopwords...")
        nltk.download("stopwords", quiet=False, force=True)


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Calculates cosine similarity between a query vector and multiple document vectors.

    Args:
        query_vec: A 1D numpy array for the query.
        doc_vecs: A 2D numpy array where each row is a document vector.

    Returns:
        A 1D numpy array of cosine similarity scores.
    """
    # Calculate dot product
    dot_product = np.dot(doc_vecs, query_vec)

    # Calculate norms
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)

    # Calculate cosine similarity, handling potential division by zero
    denominator = query_norm * doc_norms
    # Replace 0s in denominator with a small number to avoid division by zero
    denominator[denominator == 0] = 1e-9

    similarity_scores = dot_product / denominator

    return similarity_scores


def compute_maxsim_score(
    query_emb: np.ndarray, atomic_fact_embs: List[np.ndarray]
) -> float:
    """
    Compute maximum similarity between query and multiple atomic_fact embeddings (MaxSim strategy).

    Core idea of MaxSim strategy:
    - Find the single most relevant atomic_fact to the query
    - If any atomic_fact is strongly related to query, consider the entire event_log relevant
    - Avoid score dilution by irrelevant facts
    - Suitable for memory retrieval where users typically focus on one aspect

    Optimization: use vectorized matrix operations for 2-3x speed boost.

    Args:
        query_emb: Query embedding vector (1D numpy array)
        atomic_fact_embs: List of atomic_fact embedding vectors

    Returns:
        Maximum similarity score (float, range [-1, 1], typically [0, 1])
    """
    if not atomic_fact_embs:
        return 0.0

    query_norm = np.linalg.norm(query_emb)
    if query_norm == 0:
        return 0.0

    # Optimization: use matrix operations instead of loops (2-3x speedup)
    try:
        # Convert list to matrix: shape = (n_facts, embedding_dim)
        fact_matrix = np.array(atomic_fact_embs)

        # Batch compute norms for all facts
        fact_norms = np.linalg.norm(fact_matrix, axis=1)

        # Filter zero vectors
        valid_mask = fact_norms > 0
        if not np.any(valid_mask):
            return 0.0

        # Vectorized computation of all similarities
        # sims = (fact_matrix @ query_emb) / (query_norm * fact_norms)
        dot_products = np.dot(fact_matrix[valid_mask], query_emb)
        sims = dot_products / (query_norm * fact_norms[valid_mask])

        # Return maximum similarity
        return float(np.max(sims))

    except Exception as e:
        # Fall back to loop method (compatibility guarantee)
        similarities = []
        for fact_emb in atomic_fact_embs:
            fact_norm = np.linalg.norm(fact_emb)
            if fact_norm == 0:
                continue
            sim = np.dot(query_emb, fact_emb) / (query_norm * fact_norm)
            similarities.append(sim)
        return max(similarities) if similarities else 0.0


def tokenize(text: str, stemmer, stop_words: set) -> list[str]:
    """
    NLTK-based tokenization with stemming and stopword removal.
    This must be identical to the tokenization used during indexing.
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


def search_with_bm25_index(query: str, bm25, docs, top_n: int = 5):
    """
    Performs BM25 search using a pre-loaded index.
    """
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    tokenized_query = tokenize(query, stemmer, stop_words)

    if not tokenized_query:
        print("Warning: Query is empty after tokenization.")
        return []

    doc_scores = bm25.get_scores(tokenized_query)
    results_with_scores = list(zip(docs, doc_scores))
    sorted_results = sorted(results_with_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


def reciprocal_rank_fusion(
    emb_results: List[Tuple[dict, float]],
    bm25_results: List[Tuple[dict, float]],
    k: int = 60,
) -> List[Tuple[dict, float]]:
    """
    Fuse Embedding and BM25 retrieval results using RRF (Reciprocal Rank Fusion).

    RRF is a normalization-free fusion strategy sensitive to ranking position.
    Formula: RRF_score(doc) = Σ(1 / (k + rank_i))

    Advantages:
    1. No need to normalize scores (Embedding and BM25 have different score ranges)
    2. Simple and effective, widely validated in industry (Elasticsearch, etc.)
    3. More sensitive to top results (high ranks contribute more)
    4. No parameter tuning needed (k=60 is empirically optimal)

    Args:
        emb_results: Embedding retrieval results [(doc, score), ...]
        bm25_results: BM25 retrieval results [(doc, score), ...]
        k: RRF constant, typically 60 (empirical value)

    Returns:
        Fused results [(doc, rrf_score), ...], sorted by RRF score descending

    Example:
        emb_results = [(doc1, 0.92), (doc2, 0.87), (doc3, 0.81)]
        bm25_results = [(doc2, 15.3), (doc1, 12.7), (doc4, 10.2)]

        Doc1: 1/(60+1) + 1/(60+2) = 0.0323
        Doc2: 1/(60+2) + 1/(60+1) = 0.0323
        Doc3: 1/(60+3) + 0        = 0.0159
        Doc4: 0        + 1/(60+3) = 0.0159

        Fused result: [(doc1, 0.0323), (doc2, 0.0323), (doc3, 0.0159), (doc4, 0.0159)]
    """
    doc_rrf_scores = {}  # {event_id: rrf_score}
    doc_map = {}  # {event_id: doc}

    # Process Embedding retrieval results
    for rank, (doc, score) in enumerate(emb_results, start=1):
        doc_id = doc.get("event_id", id(doc))  # Prefer event_id, fall back to id()
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Process BM25 retrieval results
    for rank, (doc, score) in enumerate(bm25_results, start=1):
        doc_id = doc.get("event_id", id(doc))  # Prefer event_id, fall back to id()
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Convert back to (doc, score) format
    fused_results = [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]

    return fused_results


def multi_rrf_fusion(
    results_list: List[List[Tuple[dict, float]]], k: int = 60
) -> List[Tuple[dict, float]]:
    """
    Fuse multiple query retrieval results using RRF (multi-query fusion).

    Similar to dual-path RRF, but supports fusing arbitrary number of retrieval results.
    Each result set contributes score: 1 / (k + rank)

    Principle:
    - Documents ranking high across multiple queries accumulate high scores and rank high finally
    - This is a "voting mechanism": documents deemed relevant by multiple queries are more likely truly relevant

    Args:
        results_list: Multiple retrieval result lists [
            [(doc1, score), (doc2, score), ...],  # Query 1 results
            [(doc3, score), (doc1, score), ...],  # Query 2 results
            [(doc4, score), (doc2, score), ...],  # Query 3 results
        ]
        k: RRF constant (default 60)

    Returns:
        Fused results [(doc, rrf_score), ...], sorted by RRF score descending

    Example:
        Query 1 results: [(doc_A, 0.9), (doc_B, 0.8), (doc_C, 0.7)]
        Query 2 results: [(doc_B, 0.88), (doc_D, 0.82), (doc_A, 0.75)]
        Query 3 results: [(doc_A, 0.92), (doc_E, 0.85), (doc_B, 0.80)]

        RRF score calculation:
        doc_A: 1/(60+1) + 1/(60+3) + 1/(60+1) = 0.0323  <- appears in Q1,Q2,Q3
        doc_B: 1/(60+2) + 1/(60+1) + 1/(60+3) = 0.0323  <- appears in Q1,Q2,Q3
        doc_C: 1/(60+3) + 0        + 0        = 0.0159  <- only in Q1
        doc_D: 0        + 1/(60+2) + 0        = 0.0161  <- only in Q2
        doc_E: 0        + 0        + 1/(60+2) = 0.0161  <- only in Q3

        Fused result: doc_A and doc_B rank highest (endorsed by multiple queries)
    """
    if not results_list:
        return []

    # If only one result set, return directly
    if len(results_list) == 1:
        return results_list[0]

    doc_rrf_scores = {}  # {event_id: rrf_score}
    doc_map = {}  # {event_id: doc}

    # Iterate through each query's retrieval results
    for query_results in results_list:
        for rank, (doc, score) in enumerate(query_results, start=1):
            doc_id = doc.get("event_id", id(doc))  # Prefer event_id, fall back to id()
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            # Accumulate RRF score
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Convert back to (doc, score) format
    fused_results = [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]

    return fused_results


async def lightweight_retrieval(
    query: str, emb_index, bm25, docs, config: ExperimentConfig
) -> Tuple[List[Tuple[dict, float]], dict]:
    """
    Lightweight fast retrieval (no LLM calls, pure algorithmic retrieval).

    Supports three search modes (controlled by config.lightweight_search_mode):
    - "bm25_only": Only use BM25 search (fast, lexical matching)
    - "hybrid": BM25 + Embedding + RRF fusion (balanced)
    - "emb_only": Only use Embedding search (semantic matching)

    Advantages:
    - Fast: no LLM calls, pure vector/lexical retrieval
    - Low cost: no LLM API consumption
    - Stable: no network dependency, pure local computation

    Suitable scenarios:
    - Latency-sensitive scenarios
    - Budget-constrained scenarios
    - Simple and clear query scenarios

    Args:
        query: User query
        emb_index: Embedding index
        bm25: BM25 index
        docs: Document list
        config: Experiment configuration

    Returns:
        (final_results, metadata)
    """
    start_time = time.time()

    # Get search mode from config (default to "bm25_only")
    search_mode = getattr(config, 'lightweight_search_mode', 'bm25_only')

    metadata = {
        "retrieval_mode": "lightweight",
        "lightweight_search_mode": search_mode,
        "emb_count": 0,
        "bm25_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    # Execute retrieval based on search mode
    if search_mode == "bm25_only":
        # BM25 only mode: fast lexical matching
        bm25_results = await asyncio.to_thread(
            search_with_bm25_index, query, bm25, docs, config.lightweight_bm25_top_n
        )
        metadata["bm25_count"] = len(bm25_results)
        final_results = bm25_results[: config.lightweight_final_top_n]

    elif search_mode == "emb_only":
        # Embedding only mode: semantic matching
        emb_results = await search_with_emb_index(
            query, emb_index, top_n=config.lightweight_emb_top_n
        )
        metadata["emb_count"] = len(emb_results)
        final_results = emb_results[: config.lightweight_final_top_n]

    else:
        # Hybrid mode (default fallback): BM25 + Embedding + RRF fusion
        # Execute Embedding and BM25 retrieval in parallel
        emb_task = search_with_emb_index(
            query, emb_index, top_n=config.lightweight_emb_top_n
        )
        bm25_task = asyncio.to_thread(
            search_with_bm25_index, query, bm25, docs, config.lightweight_bm25_top_n
        )

        emb_results, bm25_results = await asyncio.gather(emb_task, bm25_task)

        metadata["emb_count"] = len(emb_results)
        metadata["bm25_count"] = len(bm25_results)

        # RRF fusion
        if not emb_results and not bm25_results:
            final_results = []
        elif not emb_results:
            final_results = bm25_results[: config.lightweight_final_top_n]
        elif not bm25_results:
            final_results = emb_results[: config.lightweight_final_top_n]
        else:
            # Use RRF fusion
            fused_results = reciprocal_rank_fusion(
                emb_results, bm25_results, k=60  # Standard RRF parameter
            )
            final_results = fused_results[: config.lightweight_final_top_n]

    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000

    return final_results, metadata


async def search_with_emb_index(
    query: str,
    emb_index,
    top_n: int = 5,
    query_embedding: Optional[np.ndarray] = None,  # Support pre-computed embedding
):
    """
    Execute embedding retrieval using MaxSim strategy.

    For documents containing atomic_facts:
    - Calculate similarity between query and each atomic_fact
    - Take maximum similarity as document score (MaxSim strategy)

    For traditional documents:
    - Fall back to using subject/summary/episode fields
    - Take maximum similarity among these fields

    Optimization: support pre-computed query embedding to avoid repeated API calls.

    Args:
        query: Query text
        emb_index: Pre-built embedding index
        top_n: Number of results to return
        query_embedding: Optional pre-computed query embedding (avoid redundant computation)

    Returns:
        Sorted (document, score) list
    """
    # Get query embedding (call API if not provided)
    if query_embedding is not None:
        query_vec = query_embedding
    else:
        query_vec = np.array(await get_vectorize_service().get_embedding(query))

    query_norm = np.linalg.norm(query_vec)

    # If query vector is zero, return empty result
    if query_norm == 0:
        return []

    # Store MaxSim score for each document
    doc_scores = []

    for item in emb_index:
        doc = item.get("doc")
        embeddings = item.get("embeddings", {})

        if not embeddings:
            continue

        # Prefer atomic_facts (MaxSim strategy)
        if "atomic_facts" in embeddings:
            atomic_fact_embs = embeddings["atomic_facts"]
            if atomic_fact_embs:
                # Use MaxSim to compute score
                score = compute_maxsim_score(query_vec, atomic_fact_embs)
                doc_scores.append((doc, score))
                continue

        # Fall back to traditional fields (maintain backward compatibility)
        # For traditional fields, also use MaxSim strategy (take maximum)
        field_scores = []
        for field in ["subject", "summary", "episode"]:
            if field in embeddings:
                field_emb = embeddings[field]
                field_norm = np.linalg.norm(field_emb)

                if field_norm > 0:
                    sim = np.dot(query_vec, field_emb) / (query_norm * field_norm)
                    field_scores.append(sim)

        if field_scores:
            score = max(field_scores)
            doc_scores.append((doc, score))

    if not doc_scores:
        return []

    # Sort by score descending and return Top-N
    sorted_results = sorted(doc_scores, key=lambda x: x[1], reverse=True)
    return sorted_results[:top_n]


async def hybrid_search_with_rrf(
    query: str,
    emb_index,
    bm25,
    docs,
    top_n: int = 40,
    emb_candidates: int = 50,
    bm25_candidates: int = 50,
    rrf_k: int = 60,
    query_embedding: Optional[np.ndarray] = None,  # Support pre-computed embedding
) -> List[Tuple[dict, float]]:
    """
    Fuse Embedding and BM25 retrieval results using RRF (hybrid retrieval).

    Execution flow:
    1. Execute Embedding (MaxSim) and BM25 retrieval in parallel
    2. Each method recalls top-N candidate documents
    3. Fuse two result sets using RRF
    4. Return fused Top-N documents

    Why use hybrid retrieval:
    - Embedding: good at semantic matching, but weak on rare words and exact matching
    - BM25: good at exact matching and rare words, but weak semantic understanding
    - RRF fusion: combines advantages of both, improves recall rate by 15-20%

    Args:
        query: User query
        emb_index: Embedding index
        bm25: BM25 index
        docs: Document list (for BM25)
        top_n: Final number of results to return (default 40)
        emb_candidates: Number of Embedding retrieval candidates (default 50)
        bm25_candidates: Number of BM25 retrieval candidates (default 50)
        rrf_k: RRF parameter k (default 60, empirically optimal)

    Returns:
        Fused Top-N results [(doc, rrf_score), ...]

    Example:
        Query: "What does he like to eat?"

        Embedding Top-3:
        - (doc_A: "User loves Sichuan cuisine", 0.92)  # Semantic match "like"="love"
        - (doc_B: "User prefers light flavors", 0.78)
        - (doc_C: "Chengdu is a food paradise", 0.65)

        BM25 Top-3:
        - (doc_A: "User loves Sichuan cuisine", 15.3)  # Exact match "likes"
        - (doc_D: "Likes eating hotpot", 12.7)  # Exact match "likes eating"
        - (doc_E: "Favorite dish is Mapo Tofu", 10.2)

        RRF fusion:
        - doc_A: Ranks high in both results -> highest score
        - doc_D: Only ranks high in BM25
        - doc_B: Only ranks high in Embedding

        Final: [(doc_A, 0.0323), (doc_D, 0.0161), (doc_B, 0.0161), ...]
    """
    # Execute Embedding and BM25 retrieval in parallel (improve efficiency)
    emb_task = search_with_emb_index(
        query, emb_index, top_n=emb_candidates, query_embedding=query_embedding
    )
    bm25_task = asyncio.to_thread(
        search_with_bm25_index, query, bm25, docs, bm25_candidates
    )

    # Wait for both retrieval tasks to complete
    emb_results, bm25_results = await asyncio.gather(emb_task, bm25_task)

    # If one retrieval result is empty, return the other
    if not emb_results and not bm25_results:
        return []
    elif not emb_results:
        print(f"Warning: Embedding search returned no results for query: {query}")
        return bm25_results[:top_n]
    elif not bm25_results:
        print(f"Warning: BM25 search returned no results for query: {query}")
        return emb_results[:top_n]

    # Use RRF to fuse two retrieval results
    fused_results = reciprocal_rank_fusion(emb_results, bm25_results, k=rrf_k)

    # Print fusion statistics (for debugging)
    print(
        f"Hybrid search: Emb={len(emb_results)}, BM25={len(bm25_results)}, Fused={len(fused_results)}, Returning top-{top_n}"
    )

    return fused_results[:top_n]


async def agentic_retrieval(
    query: str,
    config: ExperimentConfig,
    llm_provider: LLMProvider,  # Use LLMProvider
    llm_config: dict,
    emb_index,
    bm25,
    docs,
) -> Tuple[List[Tuple[dict, float]], dict]:
    """
    Agentic multi-round retrieval (LLM-guided) - new process.

    Process:
    1. Round 1: Hybrid search -> Top 20 -> Rerank -> Top 5 -> LLM judges sufficiency
    2. If sufficient: return original Top 20 (before rerank)
    3. If insufficient:
       - LLM generates improved query
       - Round 2: retrieve and merge to 40
       - Rerank 40 -> return final results

    Args:
        query: User query
        config: Experiment configuration
        llm_provider: LLM Provider (Memory Layer)
        llm_config: LLM configuration dict
        emb_index: Embedding index
        bm25: BM25 index
        docs: Document list

    Returns:
        (final_results, metadata)
    """
    import time

    start_time = time.time()

    metadata = {
        "is_multi_round": False,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "round2_count": 0,
        "is_sufficient": None,
        "reasoning": None,
        "refined_query": None,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    print(f"\n{'='*60}")
    print(f"Agentic Retrieval: {query[:60]}...")
    print(f"{'='*60}")
    print(f"  [Start] Time: {time.strftime('%H:%M:%S')}")

    # Round 1: Hybrid search Top 20
    print(f"  [Round 1] Hybrid search for Top 20...")

    round1_top20 = await hybrid_search_with_rrf(
        query=query,
        emb_index=emb_index,
        bm25=bm25,
        docs=docs,
        top_n=20,
        emb_candidates=config.hybrid_emb_candidates,
        bm25_candidates=config.hybrid_bm25_candidates,
        rrf_k=config.hybrid_rrf_k,
    )

    metadata["round1_count"] = len(round1_top20)
    print(f"  [Round 1] Retrieved {len(round1_top20)} documents")

    if not round1_top20:
        print(f"  [Warning] No results from Round 1")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # Rerank Top 20 to Top 10 for Sufficiency Check
    print(f"  [Rerank] Reranking Top 20 to get Top 10 for sufficiency check...")

    if config.use_reranker:
        reranked_top10 = await reranker_search(
            query=query,
            results=round1_top20,
            top_n=10,
            reranker_instruction=config.reranker_instruction,
            batch_size=config.reranker_batch_size,
            max_retries=config.reranker_max_retries,
            retry_delay=config.reranker_retry_delay,
            timeout=config.reranker_timeout,
            fallback_threshold=config.reranker_fallback_threshold,
            config=config,
        )
        metadata["round1_reranked_count"] = len(reranked_top10)
        print(f"  [Rerank] Got Top 10 for sufficiency check")
    else:
        # If not using reranker, take first 10 directly
        reranked_top10 = round1_top20[:10]
        metadata["round1_reranked_count"] = 10
        print(f"  [No Rerank] Using original Top 10 for sufficiency check")

    if not reranked_top10:
        print(f"  [Warning] Reranking failed, falling back to original Top 20")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_top20, metadata

    # LLM Sufficiency Check
    print(f"  [LLM] Checking sufficiency on Top 10...")

    is_sufficient, reasoning, missing_info, key_info = (
        await agentic_utils.check_sufficiency(
            query=query,
            results=reranked_top10,  # Use reranked Top 10
            llm_provider=llm_provider,  # Use LLMProvider
            llm_config=llm_config,
            max_docs=10,  # Explicitly check only 10 documents
        )
    )

    metadata["is_sufficient"] = is_sufficient
    metadata["reasoning"] = reasoning
    metadata["key_information_found"] = key_info  # New: record found key information

    print(f"  [LLM] Result: {'✅ Sufficient' if is_sufficient else '❌ Insufficient'}")
    print(f"  [LLM] Reasoning: {reasoning}")
    if key_info:  # New: print found key information
        print(f"  [LLM] Key Info Found: {', '.join(key_info)}")

    if is_sufficient:
        print(f"  [Decision] Sufficient! Using reranked Top 10 results")

        final_results = reranked_top10
        metadata["final_count"] = len(final_results)
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000

        print(f"  [Complete] Latency: {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata

    # If insufficient: enter Round 2
    metadata["is_multi_round"] = True
    metadata["missing_info"] = missing_info
    print(f"  [Decision] Insufficient, entering Round 2")
    print(f"  [Missing] {', '.join(missing_info) if missing_info else 'N/A'}")

    # LLM generates multiple improved queries (multi-query strategy)
    use_multi_query = getattr(
        config, 'use_multi_query', True
    )  # Default enable multi-query

    if use_multi_query:
        print(f"  [LLM] Generating multiple refined queries...")

        # Generate 2-3 complementary queries
        refined_queries, query_strategy = await agentic_utils.generate_multi_queries(
            original_query=query,
            results=reranked_top10,  # Based on Top 10 generate improved queries
            missing_info=missing_info,
            llm_provider=llm_provider,  # Use LLMProvider
            llm_config=llm_config,
            key_info=key_info,  # New: pass found key information
            max_docs=10,
            num_queries=3,  # Expect to generate 3 queries
        )

        metadata["refined_queries"] = refined_queries
        metadata["query_strategy"] = query_strategy
        metadata["num_queries"] = len(refined_queries)

        # Round 2: Execute multiple query retrieval in parallel
        print(f"  [Round 2] Executing {len(refined_queries)} queries in parallel...")

        # Execute hybrid search for all queries in parallel
        multi_query_tasks = [
            hybrid_search_with_rrf(
                query=q,
                emb_index=emb_index,
                bm25=bm25,
                docs=docs,
                top_n=50,  # Each query recalls 50 candidates
                emb_candidates=config.hybrid_emb_candidates,
                bm25_candidates=config.hybrid_bm25_candidates,
                rrf_k=config.hybrid_rrf_k,
            )
            for q in refined_queries
        ]

        # Wait for all queries to complete
        multi_query_results = await asyncio.gather(*multi_query_tasks)

        # Print recall count for each query
        for i, results in enumerate(multi_query_results, 1):
            print(f"    Query {i}: Retrieved {len(results)} documents")

        # Use RRF to fuse results from multiple queries
        print(f"  [Multi-RRF] Fusing results from {len(refined_queries)} queries...")

        # Use multi-query RRF fusion
        round2_results = multi_rrf_fusion(
            results_list=multi_query_results,
            k=config.hybrid_rrf_k,  # Use same k parameter
        )

        # Take Top 40 for subsequent merging
        round2_results = round2_results[:40]

        metadata["round2_count"] = len(round2_results)
        metadata["multi_query_total_docs"] = sum(len(r) for r in multi_query_results)

        print(
            f"  [Multi-RRF] Fused {metadata['multi_query_total_docs']} → {len(round2_results)} unique documents"
        )

    else:
        # Fall back to single-query mode (maintain backward compatibility)
        print(f"  [LLM] Generating single refined query (legacy mode)...")

        refined_query = await agentic_utils.generate_refined_query(
            original_query=query,
            results=reranked_top10,
            missing_info=missing_info,
            llm_provider=llm_provider,
            llm_config=llm_config,
            key_info=key_info,  # New: pass found key information
            max_docs=10,
        )

        metadata["refined_query"] = refined_query
        print(f"  [LLM] Refined query: {refined_query}")

        # Round 2: Retrieve using single refined query
        print(f"  [Round 2] Hybrid search with refined query...")

        round2_results = await hybrid_search_with_rrf(
            query=refined_query,
            emb_index=emb_index,
            bm25=bm25,
            docs=docs,
            top_n=40,
            emb_candidates=config.hybrid_emb_candidates,
            bm25_candidates=config.hybrid_bm25_candidates,
            rrf_k=config.hybrid_rrf_k,
        )

        metadata["round2_count"] = len(round2_results)
        print(f"  [Round 2] Retrieved {len(round2_results)} documents")

    # Merge: ensure total of 40 documents
    print(f"  [Merge] Combining Round 1 and Round 2 to ensure 40 documents...")

    # Use event_id for deduplication instead of Python memory address
    # Reason: BM25 and Embedding indices load JSON separately, creating different Python objects
    round1_ids = {doc.get("event_id", id(doc)) for doc, _ in round1_top20}
    round2_unique = [
        (doc, score)
        for doc, score in round2_results
        if doc.get("event_id", id(doc)) not in round1_ids
    ]

    # Merge: Round1 Top20 + Round2 deduplicated documents (ensure total=40)
    combined_results = round1_top20.copy()  # First add 20 from Round1
    needed_from_round2 = 40 - len(combined_results)  # Need 20 more
    combined_results.extend(round2_unique[:needed_from_round2])

    actual_count = len(combined_results)
    duplicates_removed = len(round2_results) - len(round2_unique)
    round2_added = len(round2_unique[:needed_from_round2])

    print(f"  [Merge] Round1 Top20: 20 documents")
    print(f"  [Merge] Round2 duplicates removed: {duplicates_removed} documents")
    print(f"  [Merge] Round2 unique added: {round2_added} documents")
    print(f"  [Merge] Combined total: {actual_count} documents (target: 40)")

    # Rerank the merged 40 documents
    if config.use_reranker and len(combined_results) > 0:
        print(f"  [Rerank] Reranking {len(combined_results)} documents...")

        final_results = await reranker_search(
            query=query,  # Use original query for rerank
            results=combined_results,
            top_n=20,  # Return Top 20 as final result
            reranker_instruction=config.reranker_instruction,
            batch_size=config.reranker_batch_size,
            max_retries=config.reranker_max_retries,
            retry_delay=config.reranker_retry_delay,
            timeout=config.reranker_timeout,
            fallback_threshold=config.reranker_fallback_threshold,
            config=config,
        )

        print(f"  [Rerank] Final Top 20 selected")
    else:
        # Not using Reranker, return Top 20 directly
        final_results = combined_results[:20]
        print(f"  [No Rerank] Returning Top 20 from combined results")

    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000

    print(
        f"  [Complete] Final: {len(final_results)} docs | Latency: {metadata['total_latency_ms']:.0f}ms"
    )
    print(f"{'='*60}\n")

    return final_results, metadata


async def reranker_search(
    query: str,
    results: List[Tuple[dict, float]],
    top_n: int = 20,
    reranker_instruction: str = None,
    batch_size: int = 10,  # Batch size (Reranker API usually limited)
    max_retries: int = 3,  # Maximum retry attempts
    retry_delay: float = 2.0,  # Base retry delay
    timeout: float = 30.0,  # Single batch timeout
    fallback_threshold: float = 0.3,  # Fallback threshold
    config: ExperimentConfig = None,  # Experiment configuration (for getting concurrency)
):
    """
    Rerank retrieval results using reranker model (supports batch concurrent processing + enhanced stability).

    For documents containing event_log:
    - Format as multi-line text: time + each atomic_fact on separate line
    - Example:
      2024-10-31 14:30:00
      User likes Sichuan cuisine
      User's favorite Sichuan dish is Mapo Tofu
      User dislikes overly spicy dishes

    For traditional documents:
    - Fall back to using episode field

    Optimization strategy (stability first):
    - Process documents in batches (batch_size per batch)
    - Serial batch processing (avoid API rate limiting)
    - Each batch supports retry and exponential backoff
    - Auto-downgrade to original ranking when success rate too low
    - Single batch timeout protection

    Args:
        query: User query
        results: Initial retrieval results (from embedding or BM25)
        top_n: Number of results to return (default 20)
        reranker_instruction: Reranker instruction
        batch_size: Number of documents per batch (default 10)
        max_retries: Maximum retry attempts per batch (default 3)
        retry_delay: Base retry delay in seconds (default 2.0, exponential backoff)
        timeout: Single batch timeout in seconds (default 30)
        fallback_threshold: Fallback when success rate below this value (default 0.3)

    Returns:
        Reranked Top-N result list
    """
    if not results:
        return []

    # Step 1: Format documents
    docs_with_episode = []
    doc_texts = []
    original_indices = []  # Record original indices for restoration

    for idx, (doc, score) in enumerate(results):
        # Prefer using event_log to format text (if exists)
        if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
            event_log = doc["event_log"]
            time_str = event_log.get("time", "")
            atomic_facts = event_log.get("atomic_fact", [])

            if isinstance(atomic_facts, list) and atomic_facts:
                # Format as multi-line text (time + each atomic_fact on separate line)
                formatted_lines = []
                if time_str:
                    formatted_lines.append(time_str)

                # Fix: compatible with both formats (string / dict)
                for fact in atomic_facts:
                    if isinstance(fact, dict) and "fact" in fact:
                        # New format: {"fact": "...", "embedding": [...]}
                        formatted_lines.append(fact["fact"])
                    elif isinstance(fact, str):
                        # Old format: pure string
                        formatted_lines.append(fact)

                formatted_text = "\n".join(formatted_lines)

                docs_with_episode.append(doc)
                doc_texts.append(formatted_text)
                original_indices.append(idx)
                continue

        # Fall back to original episode field (maintain backward compatibility)
        if episode_text := doc.get("episode"):
            docs_with_episode.append(doc)
            doc_texts.append(episode_text)
            original_indices.append(idx)

    if not doc_texts:
        return []

    reranker = rerank_service.get_rerank_service()
    print(f"Reranking query: {query}")
    print(f"Reranking {len(doc_texts)} documents in batches of {batch_size}...")
    print(f"Reranking instruction: {reranker_instruction}")

    # Step 2: Batch processing (serial + retry + fallback)
    # Split documents into batches, batch_size per batch
    batches = []
    for i in range(0, len(doc_texts), batch_size):
        batch = doc_texts[i : i + batch_size]
        batches.append((i, batch))  # Save start index and batch data

    print(f"Split into {len(batches)} batches for serial reranking")

    # Process single batch (with retry + timeout + exponential backoff)
    async def process_batch_with_retry(start_idx: int, batch_texts: List[str]):
        """Process single batch (with retry and timeout)."""
        for attempt in range(max_retries):
            try:
                # Add timeout protection
                batch_results = await asyncio.wait_for(
                    reranker.rerank_documents(
                        query, batch_texts, instruction=reranker_instruction
                    ),
                    timeout=timeout,
                )

                # Both vllm and DeepInfra now return same format:
                # {"results": [{"index": ..., "score": ..., "rank": ...}, ...]}
                for item in batch_results["results"]:
                    item["global_index"] = start_idx + item["index"]

                if attempt > 0:
                    print(
                        f"  ✓ Batch at {start_idx} succeeded on attempt {attempt + 1}"
                    )
                return batch_results["results"]

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (
                        2**attempt
                    )  # Exponential backoff: 2s, 4s, 8s
                    print(
                        f"  ⏱️  Batch at {start_idx} timeout (attempt {attempt + 1}), retrying in {wait_time:.1f}s"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    print(
                        f"  ❌ Batch at {start_idx} timeout after {max_retries} attempts"
                    )
                    return []

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    print(
                        f"  ⚠️  Batch at {start_idx} failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}"
                    )
                    await asyncio.sleep(wait_time)
                else:
                    print(
                        f"  ❌ Batch at {start_idx} failed after {max_retries} attempts: {e}"
                    )
                    return []

    # Controlled concurrent processing (conservative concurrency strategy)
    # Get concurrency count from config (default 2, conservative value)
    max_concurrent = getattr(config, 'reranker_concurrent_batches', 2)

    batch_results_list = []
    successful_batches = 0

    # Process in groups, max max_concurrent batches concurrently per group
    for group_start in range(0, len(batches), max_concurrent):
        group_batches = batches[group_start : group_start + max_concurrent]

        print(
            f"  Processing batch group {group_start//max_concurrent + 1} ({len(group_batches)} batches in parallel)..."
        )

        # Process all batches in current group concurrently
        tasks = [
            process_batch_with_retry(start_idx, batch)
            for start_idx, batch in group_batches
        ]
        group_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful batches
        for result in group_results:
            if isinstance(result, list) and result:
                batch_results_list.append(result)
                successful_batches += 1
            else:
                batch_results_list.append([])

        # Inter-group delay (further reduced, aggressive acceleration)
        if group_start + max_concurrent < len(batches):
            await asyncio.sleep(0.3)  # Inter-group 0.3s interval (reduced from 0.8s)

    # Step 3: Merge all batch results + fallback strategy
    all_rerank_results = []
    for batch_results in batch_results_list:
        all_rerank_results.extend(batch_results)

    # Calculate success rate
    success_rate = successful_batches / len(batches) if batches else 0.0
    print(
        f"Reranker success rate: {success_rate:.1%} ({successful_batches}/{len(batches)} batches)"
    )

    # Fallback strategy 1: Complete failure
    if not all_rerank_results:
        print(
            "⚠️ Warning: All reranker batches failed, using original ranking as fallback"
        )
        return results[:top_n]

    # Fallback strategy 2: Success rate too low
    if success_rate < fallback_threshold:
        print(
            f"⚠️ Warning: Reranker success rate too low ({success_rate:.1%} < {fallback_threshold:.1%}), using original ranking"
        )
        return results[:top_n]

    print(f"Reranking complete: {len(all_rerank_results)} documents scored")

    # Step 4: Sort by reranker score and return Top-N
    sorted_results = sorted(all_rerank_results, key=lambda x: x["score"], reverse=True)[
        :top_n
    ]

    # Map back to original documents
    final_results = [
        (results[original_indices[item["global_index"]]][0], item["score"])
        for item in sorted_results
    ]

    return final_results


async def main():
    """Main function to perform batch search and save results in nemori format."""
    # --- Configuration ---
    config = ExperimentConfig()
    bm25_index_dir = Path(__file__).parent / config.experiment_name / "bm25_index"
    emb_index_dir = Path(__file__).parent / config.experiment_name / "vectors"
    save_dir = Path(__file__).parent / config.experiment_name

    dataset_path = config.datase_path
    results_output_path = save_dir / "search_results.json"

    # Checkpoint resume: checkpoint file path
    checkpoint_path = save_dir / "search_results_checkpoint.json"

    # Ensure NLTK data is ready
    ensure_nltk_data()

    # Initialize LLM Provider (for Agentic retrieval)
    llm_provider = None
    llm_config = None
    if config.use_agentic_retrieval:
        if agentic_utils is None:
            print("Error: agentic_utils not found, cannot use agentic retrieval")
            print("Please check that tools/agentic_utils.py exists")
            return

        llm_config = config.llm_config.get(
            config.llm_service, config.llm_config["openai"]
        )

        # Use Memory Layer's LLMProvider instead of AsyncOpenAI
        llm_provider = LLMProvider(
            provider_type="openai",
            model=llm_config["model"],
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
            temperature=llm_config.get("temperature", 0.3),
            max_tokens=llm_config.get("max_tokens", 32768),
        )
        print(f"✅ LLM Provider initialized for agentic retrieval")
        print(f"   Model: {llm_config['model']}")

    # Load the dataset
    print(f"Loading dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Checkpoint resume: load existing checkpoint
    all_search_results = {}
    processed_conversations = set()

    if checkpoint_path.exists():
        print(f"\n🔄 Found checkpoint file: {checkpoint_path}")
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                all_search_results = json.load(f)
            processed_conversations = set(all_search_results.keys())
            print(
                f"✅ Loaded {len(processed_conversations)} conversations from checkpoint"
            )
            print(f"   Already processed: {sorted(processed_conversations)}")
        except Exception as e:
            print(f"⚠️  Failed to load checkpoint: {e}")
            print(f"   Starting from scratch...")
            all_search_results = {}
            processed_conversations = set()
    else:
        print(f"\n🆕 No checkpoint found, starting from scratch")

    # Iterate through the dataset, assuming the index of the dataset list
    # corresponds to the conversation index number.
    for i, conversation_data in enumerate(dataset):
        conv_id = f"locomo_exp_user_{i}"

        # Checkpoint resume: skip processed conversations
        if conv_id in processed_conversations:
            print(f"\n⏭️  Skipping Conversation ID: {conv_id} (already processed)")
            continue

        speaker_a = conversation_data["conversation"].get("speaker_a")
        speaker_b = conversation_data["conversation"].get("speaker_b")
        print(f"\n--- Processing Conversation ID: {conv_id} ({i+1}/{len(dataset)}) ---")

        if "qa" not in conversation_data:
            print(f"Warning: No 'qa' key found in conversation #{i}. Skipping.")
            continue

        # --- Load index once per conversation ---
        # If using hybrid search, need to load both Embedding and BM25 indices
        if config.use_hybrid_search:
            # Load Embedding index
            emb_index_path = emb_index_dir / f"embedding_index_conv_{i}.pkl"
            if not emb_index_path.exists():
                print(
                    f"Error: Embedding index not found at {emb_index_path}. Skipping conversation."
                )
                continue
            with open(emb_index_path, "rb") as f:
                emb_index = pickle.load(f)

            # Load BM25 index
            bm25_index_path = bm25_index_dir / f"bm25_index_conv_{i}.pkl"
            if not bm25_index_path.exists():
                print(
                    f"Error: BM25 index not found at {bm25_index_path}. Skipping conversation."
                )
                continue
            with open(bm25_index_path, "rb") as f:
                index_data = pickle.load(f)
            bm25 = index_data["bm25"]
            docs = index_data["docs"]

            print(
                f"Loaded both Embedding and BM25 indexes for conversation {i} (Hybrid Search)"
            )

        elif config.use_emb:
            # Load Embedding index only
            emb_index_path = emb_index_dir / f"embedding_index_conv_{i}.pkl"
            if not emb_index_path.exists():
                print(
                    f"Error: Index file not found at {emb_index_path}. Skipping conversation."
                )
                continue
            with open(emb_index_path, "rb") as f:
                emb_index = pickle.load(f)
        else:
            # Load BM25 index only
            bm25_index_path = bm25_index_dir / f"bm25_index_conv_{i}.pkl"
            if not bm25_index_path.exists():
                print(
                    f"Error: Index file not found at {bm25_index_path}. Skipping conversation."
                )
                continue
            with open(bm25_index_path, "rb") as f:
                index_data = pickle.load(f)
            bm25 = index_data["bm25"]
            docs = index_data["docs"]

        # Parallelize per-question retrieval with bounded concurrency
        # Increase concurrency: also use higher concurrency for Agentic retrieval (10 -> 20)
        max_concurrent = 20 if config.use_agentic_retrieval else 128
        sem = asyncio.Semaphore(max_concurrent)

        if config.use_agentic_retrieval:
            print(
                f"  🚀 Agentic retrieval enabled with HIGH CONCURRENCY: {max_concurrent} concurrent requests"
            )

        async def process_single_qa(qa_pair):
            """Process single QA pair (supports multiple retrieval modes)."""
            question = qa_pair.get("question")
            if not question:
                return None
            if qa_pair.get("category") == 5:
                print(f"Skipping question {question} because it is category 5")
                return None

            # Start timing
            qa_start_time = time.time()

            try:
                async with sem:
                    retrieval_metadata = {}

                    # Retrieval mode selection
                    if config.retrieval_mode == "agentic":
                        # Agentic multi-round retrieval (complex but high quality)
                        top_results, retrieval_metadata = await agentic_retrieval(
                            query=question,
                            config=config,
                            llm_provider=llm_provider,  # Use LLMProvider
                            llm_config=llm_config,
                            emb_index=emb_index,
                            bm25=bm25,
                            docs=docs,
                        )

                    elif config.retrieval_mode == "lightweight":
                        # Lightweight fast retrieval (fast but slightly lower quality)
                        top_results, retrieval_metadata = await lightweight_retrieval(
                            query=question,
                            emb_index=emb_index,
                            bm25=bm25,
                            docs=docs,
                            config=config,
                        )

                    else:
                        # Traditional retrieval branch (maintain backward compatibility)
                        if config.use_reranker:
                            # Stage 1: Initial retrieval, recall Top-N candidates
                            if config.use_hybrid_search:
                                # Hybrid search: Embedding (MaxSim) + BM25 + RRF fusion
                                results = await hybrid_search_with_rrf(
                                    query=question,
                                    emb_index=emb_index,
                                    bm25=bm25,
                                    docs=docs,
                                    top_n=config.emb_recall_top_n,
                                    emb_candidates=config.hybrid_emb_candidates,
                                    bm25_candidates=config.hybrid_bm25_candidates,
                                    rrf_k=config.hybrid_rrf_k,
                                )
                            elif config.use_emb:
                                # Use Embedding + MaxSim retrieval only
                                results = await search_with_emb_index(
                                    query=question,
                                    emb_index=emb_index,
                                    top_n=config.emb_recall_top_n,
                                )
                            else:
                                # Use BM25 retrieval only
                                results = await asyncio.to_thread(
                                    search_with_bm25_index,
                                    question,
                                    bm25,
                                    docs,
                                    config.emb_recall_top_n,
                                )

                            # Stage 2: Reranker reordering
                            top_results = await reranker_search(
                                query=question,
                                results=results,
                                top_n=config.reranker_top_n,
                                reranker_instruction=config.reranker_instruction,
                                batch_size=config.reranker_batch_size,
                                max_retries=config.reranker_max_retries,
                                retry_delay=config.reranker_retry_delay,
                                timeout=config.reranker_timeout,
                                fallback_threshold=config.reranker_fallback_threshold,
                                config=config,
                            )
                        else:
                            # Single-stage retrieval (not using Reranker)
                            if config.use_hybrid_search:
                                top_results = await hybrid_search_with_rrf(
                                    query=question,
                                    emb_index=emb_index,
                                    bm25=bm25,
                                    docs=docs,
                                    top_n=20,
                                    emb_candidates=config.hybrid_emb_candidates,
                                    bm25_candidates=config.hybrid_bm25_candidates,
                                    rrf_k=config.hybrid_rrf_k,
                                )
                            elif config.use_emb:
                                top_results = await search_with_emb_index(
                                    query=question, emb_index=emb_index, top_n=20
                                )
                            else:
                                top_results = await asyncio.to_thread(
                                    search_with_bm25_index, question, bm25, docs, 20
                                )

                        # Add retrieval time statistics
                        retrieval_metadata = {
                            "retrieval_mode": "traditional",
                            "use_reranker": config.use_reranker,
                            "use_hybrid_search": config.use_hybrid_search,
                        }

                    # Extract event_ids
                    event_ids = []
                    if top_results:
                        for doc, score in top_results:
                            event_id = doc.get('event_id')
                            if event_id:
                                event_ids.append(event_id)

                    # Calculate processing time
                    qa_latency_ms = (time.time() - qa_start_time) * 1000

                    result = {
                        "query": question,
                        "event_ids": event_ids,  # Return event_ids instead of context
                        "original_qa": qa_pair,
                        "retrieval_metadata": {
                            **retrieval_metadata,
                            "qa_latency_ms": qa_latency_ms,
                            "target_event_ids_count": len(
                                top_results
                            ),  # Record target count
                            "actual_event_ids_count": len(
                                event_ids
                            ),  # Record actual extracted count
                        },
                    }

                    return result

            except Exception as e:
                print(f"Error processing question '{question}': {e}")
                import traceback

                traceback.print_exc()
                return None

        tasks = [
            asyncio.create_task(process_single_qa(qa_pair))
            for qa_pair in conversation_data["qa"]
        ]
        results_for_conv = [
            res for res in await asyncio.gather(*tasks) if res is not None
        ]

        all_search_results[conv_id] = results_for_conv

        # Checkpoint resume: save checkpoint after each conversation
        try:
            print(f"💾 Saving checkpoint after conversation {conv_id}...")
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(all_search_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Checkpoint saved: {len(all_search_results)} conversations")
        except Exception as e:
            print(f"⚠️  Failed to save checkpoint: {e}")

    # Save all results to a single JSON file in the specified format
    print(f"\n{'='*60}")
    print(f"🎉 All conversations processed!")
    print(f"{'='*60}")
    print(f"\nSaving final results to: {results_output_path}")
    with open(results_output_path, "w", encoding="utf-8") as f:
        json.dump(all_search_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Batch search and retrieval complete!")
    print(f"   Total conversations: {len(all_search_results)}")

    # Checkpoint resume: delete checkpoint file after completion
    if checkpoint_path.exists():
        try:
            checkpoint_path.unlink()
            print(f"🗑️  Checkpoint file removed (task completed)")
        except Exception as e:
            print(f"⚠️  Failed to remove checkpoint: {e}")

    # Clean up resources
    reranker = rerank_service.get_rerank_service()
    # Assuming the service is DeepInfraRerankService, which has a close method.
    if hasattr(reranker, 'close') and callable(getattr(reranker, 'close')):
        await reranker.close()


if __name__ == "__main__":
    asyncio.run(main())
