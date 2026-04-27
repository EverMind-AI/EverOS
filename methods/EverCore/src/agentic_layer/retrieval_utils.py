"""Retrieval utility functions

Provides implementations of various retrieval strategies:
- Embedding vector retrieval
- BM25 keyword retrieval
- RRF fusion retrieval
- Agentic retrieval (LLM-guided multi-round retrieval)
"""

import re
import time
import jieba
import numpy as np
import logging
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from core.nlp.stopwords_utils import filter_stopwords as filter_chinese_stopwords
from .vectorize_service import get_vectorize_service

logger = logging.getLogger(__name__)


def build_bm25_index(candidates):
    """Build BM25 index (supports Chinese and English)"""
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        return None, None, None, None

    # Ensure NLTK data is downloaded
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))

    # Extract text and tokenize (supports Chinese and English)
    tokenized_docs = []
    for mem in candidates:
        text = getattr(mem, "episode", None) or getattr(mem, "summary", "") or ""
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))

        if has_chinese:
            tokens = list(jieba.cut(text))
            processed_tokens = filter_chinese_stopwords(tokens)
        else:
            tokens = word_tokenize(text.lower())
            processed_tokens = [
                stemmer.stem(token)
                for token in tokens
                if token.isalpha() and len(token) >= 2 and token not in stop_words
            ]

        tokenized_docs.append(processed_tokens)

    bm25 = BM25Okapi(tokenized_docs)
    return bm25, tokenized_docs, stemmer, stop_words


async def search_with_bm25(
    query: str, bm25, candidates, stemmer, stop_words, top_k: int = 50
) -> List[Tuple]:
    """BM25 retrieval (supports Chinese and English)"""
    if bm25 is None:
        return []

    try:
        from nltk.tokenize import word_tokenize
    except ImportError:
        return []

    # Tokenize query (supports Chinese and English)
    has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))

    if has_chinese:
        tokens = list(jieba.cut(query))
        tokenized_query = filter_chinese_stopwords(tokens)
    else:
        tokens = word_tokenize(query.lower())
        tokenized_query = [
            stemmer.stem(token)
            for token in tokens
            if token.isalpha() and len(token) >= 2 and token not in stop_words
        ]

    if not tokenized_query:
        return []

    # Calculate BM25 scores
    scores = bm25.get_scores(tokenized_query)

    # Sort and return Top-K
    results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:top_k]

    return results


def reciprocal_rank_fusion(
    results1: List[Tuple], results2: List[Tuple], k: int = 60
) -> List[Tuple]:
    """RRF fusion of two retrieval results"""
    doc_rrf_scores = {}
    doc_map = {}

    # Process first result set
    for rank, (doc, score) in enumerate(results1, start=1):
        doc_id = doc.get('id')
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Process second result set
    for rank, (doc, score) in enumerate(results2, start=1):
        doc_id = doc.get('id')
        if doc_id not in doc_map:
            doc_map[doc_id] = doc
        doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Convert to list and sort
    fused_results = [
        (doc_map[doc_id], rrf_score) for doc_id, rrf_score in doc_rrf_scores.items()
    ]
    fused_results.sort(key=lambda x: x[1], reverse=True)

    return fused_results


def vector_anchored_fusion(
    vector_results: List[Tuple[str, float]],
    keyword_results: List[Tuple[str, float]],
    saturation_k: float = 5.0,
    alpha: float = 0.7,
) -> List[Tuple[str, float]]:
    """Fuse vector and keyword search results with saturated BM25 weighting.

    BM25 raw scores are compressed via a saturation function
    ``sat = raw / (raw + saturation_k)`` which maps them to [0, 1) while
    preserving discrimination (e.g. BM25=0.4 -> 0.07, BM25=10 -> 0.67).

    The final score is a weighted sum ``alpha * vec + (1-alpha) * sat_bm25``.

    For docs appearing in only one result set, the missing side defaults
    to the minimum score of that set ("not recalled" != "not relevant").

    Args:
        vector_results: Vector search results as [(doc_id, score), ...].
        keyword_results: Keyword search results as [(doc_id, bm25_score), ...].
        saturation_k: Saturation constant for BM25 normalization.
            Controls the inflection point -- scores around this value map
            to ~0.5.  Default 5.0 works well for typical ES BM25 ranges.
        alpha: Vector weight in the fusion.  Higher values trust vector
            more; lower values trust BM25 more.

    Returns:
        Fused results [(doc_id, score), ...] sorted by score descending.
    """
    vec_score_map: Dict[str, float] = {doc_id: s for doc_id, s in vector_results}

    # Saturate BM25 scores into [0, 1)
    kw_sat_map: Dict[str, float] = {}
    for doc_id, raw in keyword_results:
        kw_sat_map[doc_id] = raw / (raw + saturation_k) if raw > 0 else 0.0

    # Floor defaults for docs missing from one path.
    # "Not recalled" != "not relevant" -- assume at least as good as the
    # weakest doc that *was* recalled by that path.
    vec_floor = min(vec_score_map.values()) if vec_score_map else 0.0
    kw_floor = min(kw_sat_map.values()) if kw_sat_map else 0.0

    # Weighted sum fusion
    all_ids = set(vec_score_map.keys()) | set(kw_sat_map.keys())
    scored: List[Tuple[str, float]] = []
    for doc_id in all_ids:
        vs = vec_score_map.get(doc_id, vec_floor)
        ks = kw_sat_map.get(doc_id, kw_floor)
        scored.append((doc_id, alpha * vs + (1.0 - alpha) * ks))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


async def lightweight_retrieval(
    query: str,
    candidates,
    emb_top_n: int = 50,
    bm25_top_n: int = 50,
    final_top_n: int = 20,
) -> Tuple:
    """Lightweight retrieval (Embedding + BM25 + RRF fusion)"""
    start_time = time.time()

    metadata = {
        "retrieval_mode": "lightweight",
        "emb_count": 0,
        "bm25_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    if not candidates:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # Build BM25 index
    bm25, tokenized_docs, stemmer, stop_words = build_bm25_index(candidates)

    # Embedding retrieval
    emb_results = []
    try:
        vectorize_service = get_vectorize_service()
        query_vec = await vectorize_service.get_embedding(query)
        query_norm = np.linalg.norm(query_vec)

        if query_norm > 0:
            scores = []
            for mem in candidates:
                try:
                    doc_vec = np.array(mem.extend.get("embedding", []))
                    if len(doc_vec) > 0:
                        doc_norm = np.linalg.norm(doc_vec)
                        if doc_norm > 0:
                            sim = np.dot(query_vec, doc_vec) / (query_norm * doc_norm)
                            scores.append((mem, float(sim)))
                except:
                    continue

            emb_results = sorted(scores, key=lambda x: x[1], reverse=True)[:emb_top_n]
    except Exception as e:
        pass

    metadata["emb_count"] = len(emb_results)

    # BM25 retrieval
    bm25_results = []
    if bm25 is not None:
        bm25_results = await search_with_bm25(
            query, bm25, candidates, stemmer, stop_words, top_k=bm25_top_n
        )

    metadata["bm25_count"] = len(bm25_results)

    # RRF fusion
    if not emb_results and not bm25_results:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata
    elif not emb_results:
        final_results = bm25_results[:final_top_n]
    elif not bm25_results:
        final_results = emb_results[:final_top_n]
    else:
        fused_results = reciprocal_rank_fusion(emb_results, bm25_results, k=60)
        final_results = fused_results[:final_top_n]

    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000

    return final_results, metadata


def multi_rrf_fusion(results_list: List[List[Tuple]], k: int = 60) -> List[Tuple]:
    """
    Fuse multiple queries' retrieval results using RRF (multi-query fusion)

    Similar to dual-path RRF, but supports fusing any number of retrieval results.
    Score contributed by each result set: 1 / (k + rank)

    Principle:
    - Documents ranked high across multiple queries → high accumulated score → higher final ranking
    - This is a "voting mechanism": documents considered relevant by multiple queries are more likely truly relevant

    Args:
        results_list: List of multiple retrieval results [
            [(doc1, score), (doc2, score), ...],  # Query 1 results
            [(doc3, score), (doc1, score), ...],  # Query 2 results
            [(doc4, score), (doc2, score), ...],  # Query 3 results
        ]
        k: RRF constant (default 60)

    Returns:
        Fused results [(doc, rrf_score), ...], sorted by RRF score in descending order

    Example:
        Query 1 results: [(doc_A, 0.9), (doc_B, 0.8), (doc_C, 0.7)]
        Query 2 results: [(doc_B, 0.88), (doc_D, 0.82), (doc_A, 0.75)]
        Query 3 results: [(doc_A, 0.92), (doc_E, 0.85), (doc_B, 0.80)]

        RRF score calculation:
        doc_A: 1/(60+1) + 1/(60+3) + 1/(60+1) = 0.0323  ← appears in Q1,Q2,Q3
        doc_B: 1/(60+2) + 1/(60+1) + 1/(60+3) = 0.0323  ← appears in Q1,Q2,Q3
        doc_C: 1/(60+3) + 0        + 0        = 0.0159  ← only in Q1
        doc_D: 0        + 1/(60+2) + 0        = 0.0161  ← only in Q2
        doc_E: 0        + 0        + 1/(60+2) = 0.0161  ← only in Q3

        Fused results: doc_A and doc_B rank highest (recognized by multiple queries)
    """
    if not results_list:
        return []

    # If only one result set, return directly
    if len(results_list) == 1:
        return results_list[0]

    # Use document's memory address as unique identifier
    doc_rrf_scores = {}  # {doc_id: rrf_score}
    doc_map = {}  # {doc_id: doc}

    # Iterate through each query's retrieval results
    for query_results in results_list:
        for rank, (doc, score) in enumerate(query_results, start=1):
            doc_id = id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            # Accumulate RRF score
            doc_rrf_scores[doc_id] = doc_rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by RRF score
    sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Convert back to (doc, score) format
    fused_results = [(doc_map[doc_id], rrf_score) for doc_id, rrf_score in sorted_docs]

    return fused_results


async def multi_query_retrieval(
    queries: List[str],
    candidates,
    emb_top_n: int = 50,
    bm25_top_n: int = 50,
    final_top_n: int = 40,
    rrf_k: int = 60,
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    Multi-query parallel retrieval + RRF fusion

    Perform hybrid retrieval (Embedding + BM25) for each query, then fuse all results using RRF.
    This strategy captures relevant information from different angles, improving recall.

    Process:
    1. Execute hybrid retrieval for all queries in parallel
    2. Use multi-query RRF to fuse results
    3. Return Top-N documents

    Args:
        queries: List of queries (2-3)
        candidates: Candidate memory list
        emb_top_n: Number of Embedding candidates per query
        bm25_top_n: Number of BM25 candidates per query
        final_top_n: Number of documents to return after fusion
        rrf_k: RRF parameter

    Returns:
        (results, metadata)
        - results: Fused Top-N results
        - metadata: Contains performance metrics and statistics

    Example:
        >>> queries = [
        ...     "What is the user's favorite cuisine?",
        ...     "What flavors does the user like?",
        ...     "What are the user's eating habits?"
        ... ]
        >>> results, metadata = await multi_query_retrieval(queries, candidates)
        >>> print(len(results))  # 40
        >>> print(metadata["num_queries"])  # 3
    """
    start_time = time.time()

    metadata = {
        "retrieval_mode": "multi_query",
        "num_queries": len(queries),
        "per_query_results": [],
        "total_docs_before_fusion": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    if not queries or not candidates:
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    logger.info(f"Executing {len(queries)} queries in parallel...")

    # Execute hybrid retrieval for all queries in parallel
    tasks = [
        lightweight_retrieval(q, candidates, emb_top_n, bm25_top_n, final_top_n)
        for q in queries
    ]

    multi_query_results = await asyncio.gather(*tasks, return_exceptions=True)
    from common_utils.async_utils import reraise_critical_errors

    reraise_critical_errors(multi_query_results)

    # Collect valid results
    valid_results = []
    for i, result in enumerate(multi_query_results, 1):
        if isinstance(result, Exception):
            logger.error(f"Query {i} failed: {result}")
            continue

        results, query_metadata = result
        if results:
            valid_results.append(results)
            metadata["per_query_results"].append(
                {
                    "query_index": i,
                    "count": len(results),
                    "latency_ms": query_metadata.get("total_latency_ms", 0),
                }
            )
            logger.debug(f"Query {i}: Retrieved {len(results)} documents")

    if not valid_results:
        logger.warning("All queries failed")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # Count total documents before fusion
    metadata["total_docs_before_fusion"] = sum(len(r) for r in valid_results)

    # Use multi-query RRF fusion
    logger.info(f"Fusing {len(valid_results)} query results...")
    fused_results = multi_rrf_fusion(valid_results, k=rrf_k)

    # Take Top-N
    final_results = fused_results[:final_top_n]

    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000

    logger.info(
        f"Multi-query retrieval: {metadata['total_docs_before_fusion']} → {len(final_results)} docs"
    )

    return final_results, metadata


async def rerank_candidates(
    query: str, candidates: List[Tuple], top_n: int, rerank_service
) -> List[Tuple]:
    """
    Rerank candidate results

    Use Rerank service to reorder retrieval results, improving precision.

    Args:
        query: User query
        candidates: Candidate results [(doc, score), ...]
        top_n: Number of Top-N to return
        rerank_service: Rerank service instance

    Returns:
        Reranked Top-N results [(doc, new_score), ...]

    Note:
        - If Rerank fails, fallback to original ranking
        - Use batch processing to avoid API rate limiting
    """
    if not candidates:
        return []

    try:
        logger.debug(
            f"Reranking {len(candidates)} candidates for query: {query[:50]}..."
        )

        # 🔥 Convert format: transform [(doc, score)] to format expected by rerank service
        # rerank_service.rerank_memories expects List[Dict[str, Any]]
        candidates_for_rerank = []
        for idx, (doc, score) in enumerate(candidates):
            # Build hit dictionary with sufficient information for rerank
            hit = {"index": idx, "score": score}

            # If doc is dict, merge directly
            if isinstance(doc, dict):
                hit.update(doc)
            else:
                # If doc is object, extract key fields
                hit["episode"] = getattr(doc, "episode", "")
                hit["summary"] = getattr(doc, "summary", "")
                hit["subject"] = getattr(doc, "subject", "")

                # Try to extract atomic_fact (if exists)
                if hasattr(doc, "atomic_fact"):
                    atomic_fact_val = doc.atomic_fact
                    if isinstance(atomic_fact_val, dict):
                        hit["atomic_fact"] = atomic_fact_val
                    elif atomic_fact_val:
                        # If it's an object, convert to dictionary
                        hit["atomic_fact"] = {
                            "atomic_fact": getattr(atomic_fact_val, "atomic_fact", []),
                            "time": getattr(atomic_fact_val, "time", ""),
                        }

            candidates_for_rerank.append(hit)

        # Call rerank service
        reranked_hits = await rerank_service.rerank_memories(
            query, candidates_for_rerank, top_k=top_n
        )

        # Convert format: from rerank returned format to (doc, score) format
        if reranked_hits:
            # reranked_hits format: [{"index": ..., "score": ...}, ...]
            # candidates format: [(doc, score), ...]

            reranked_results = []
            for hit in reranked_hits[:top_n]:
                # Extract index
                if isinstance(hit, dict):
                    idx = hit.get("index", hit.get("global_index", 0))
                    new_score = hit.get("score", 0.0)
                else:
                    # If returned is tuple, format is wrong, skip
                    logger.warning(f"Unexpected rerank result type: {type(hit)}")
                    continue

                if 0 <= idx < len(candidates):
                    doc = candidates[idx][0]
                    reranked_results.append((doc, new_score))

            logger.debug(f"Rerank complete: {len(reranked_results)} results")
            return reranked_results if reranked_results else candidates[:top_n]
        else:
            logger.warning("Rerank returned empty results, using original")
            return candidates[:top_n]

    except Exception as e:
        logger.error(f"Rerank failed: {e}, using original ranking", exc_info=True)
        return candidates[:top_n]


async def agentic_retrieval(
    query: str, candidates, llm_provider, config: Optional[Any] = None
) -> Tuple[List[Tuple], Dict[str, Any]]:
    """
    Agentic multi-round retrieval (LLM-guided)

    Use LLM to judge retrieval sufficiency and perform multi-round retrieval when necessary.

    Process:
    1. Round 1: Hybrid retrieval → Top 20
    2. Rerank → Top 10 → LLM judge sufficiency
    3. If sufficient: return original Top 20
    4. If insufficient:
       - LLM generates multiple improved queries (2-3)
       - Round 2: Parallel retrieval for all queries
       - Use RRF fusion → deduplicate and merge to 40
       - Rerank → return final Top 20

    Args:
        query: User query
        candidates: Candidate memory list
        llm_provider: LLM Provider (Memory Layer)
        config: Agentic configuration (optional)

    Returns:
        (final_results, metadata)
        - final_results: Final retrieval results [(doc, score), ...]
        - metadata: Contains detailed retrieval process information

    Example:
        >>> from agentic_layer.agentic_utils import AgenticConfig
        >>> config = AgenticConfig(use_reranker=True)
        >>> results, metadata = await agentic_retrieval(
        ...     query="What does the user like to eat?",
        ...     candidates=memcells,
        ...     llm_provider=llm,
        ...     config=config
        ... )
        >>> print(metadata["is_sufficient"])  # False
        >>> print(metadata["refined_queries"])  # ["User's favorite cuisine?", ...]
    """
    # Import configuration and tools
    from .agentic_utils import AgenticConfig, check_sufficiency, generate_multi_queries
    from .rerank_service import get_rerank_service

    # Use default config or provided config
    if config is None:
        config = AgenticConfig()

    start_time = time.time()

    metadata = {
        "retrieval_mode": "agentic",
        "is_multi_round": False,
        "round1_count": 0,
        "round1_reranked_count": 0,
        "is_sufficient": None,
        "reasoning": None,
        "missing_info": None,
        "refined_queries": None,
        "round2_count": 0,
        "final_count": 0,
        "total_latency_ms": 0.0,
    }

    logger.info(f"{'='*60}")
    logger.info(f"Agentic Retrieval: {query[:60]}...")
    logger.info(f"{'='*60}")

    # ========== Round 1: Hybrid search Top 20 ==========
    logger.info("Round 1: Hybrid search for Top 20...")

    try:
        round1_results, round1_metadata = await lightweight_retrieval(
            query=query,
            candidates=candidates,
            emb_top_n=config.round1_emb_top_n,
            bm25_top_n=config.round1_bm25_top_n,
            final_top_n=config.round1_top_n,
        )

        metadata["round1_count"] = len(round1_results)
        metadata["round1_latency_ms"] = round1_metadata.get("total_latency_ms", 0)

        logger.info(f"Round 1: Retrieved {len(round1_results)} documents")

        if not round1_results:
            logger.warning("Round 1 returned no results")
            metadata["total_latency_ms"] = (time.time() - start_time) * 1000
            return [], metadata

    except Exception as e:
        logger.error(f"Round 1 failed: {e}")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return [], metadata

    # ========== Rerank Top 20 → Top 10 for Sufficiency Check ==========
    if config.use_reranker:
        logger.info("Reranking Top 20 to get Top 10 for sufficiency check...")

        try:
            rerank_service = get_rerank_service()
            reranked_top5 = await rerank_candidates(
                query=query,
                candidates=round1_results,
                top_n=config.round1_rerank_top_n,
                rerank_service=rerank_service,
            )

            metadata["round1_reranked_count"] = len(reranked_top5)
            logger.info(f"Rerank: Got Top {len(reranked_top5)} for sufficiency check")

        except Exception as e:
            logger.error(f"Rerank failed: {e}, using original Top 10")
            reranked_top5 = round1_results[: config.round1_rerank_top_n]
            metadata["round1_reranked_count"] = len(reranked_top5)
    else:
        # No reranker, directly take first 10
        reranked_top5 = round1_results[: config.round1_rerank_top_n]
        metadata["round1_reranked_count"] = len(reranked_top5)
        logger.info("No Rerank: Using original Top 10 for sufficiency check")

    if not reranked_top5:
        logger.warning("No results for sufficiency check, returning Round 1 results")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_results, metadata

    # ========== LLM Sufficiency Check ==========
    logger.info("LLM: Checking sufficiency on Top 10...")

    try:
        is_sufficient, reasoning, missing_info = await check_sufficiency(
            query=query,
            results=reranked_top5,
            llm_provider=llm_provider,
            max_docs=config.round1_rerank_top_n,
        )

        metadata["is_sufficient"] = is_sufficient
        metadata["reasoning"] = reasoning
        metadata["missing_info"] = missing_info

        logger.info(
            f"LLM Result: {'✅ Sufficient' if is_sufficient else '❌ Insufficient'}"
        )
        logger.info(f"LLM Reasoning: {reasoning}")

    except Exception as e:
        logger.error(f"Sufficiency check failed: {e}, assuming sufficient")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_results, metadata

    # ========== If sufficient: return original Round 1 Top 20 ==========
    if is_sufficient:
        logger.info("Decision: Sufficient! Using Round 1 Top 20 results")

        final_results = round1_results
        metadata["final_count"] = len(final_results)
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000

        logger.info(f"Complete: Latency {metadata['total_latency_ms']:.0f}ms")
        return final_results, metadata

    # ========== If insufficient: enter Round 2 ==========
    metadata["is_multi_round"] = True
    logger.info("Decision: Insufficient, entering Round 2")
    if missing_info:
        logger.info(f"Missing: {', '.join(missing_info)}")

    # ========== LLM generate multiple refined queries ==========
    if config.enable_multi_query:
        logger.info("LLM: Generating multiple refined queries...")

        try:
            refined_queries, query_strategy = await generate_multi_queries(
                original_query=query,
                results=reranked_top5,
                missing_info=missing_info,
                llm_provider=llm_provider,
                max_docs=config.round1_rerank_top_n,
                num_queries=config.num_queries,
            )

            metadata["refined_queries"] = refined_queries
            metadata["query_strategy"] = query_strategy
            metadata["num_queries"] = len(refined_queries)

            logger.info(f"Generated {len(refined_queries)} queries")
            for i, q in enumerate(refined_queries, 1):
                logger.debug(f"  Query {i}: {q[:80]}...")

        except Exception as e:
            logger.error(f"Query generation failed: {e}, using original query")
            refined_queries = [query]
            metadata["refined_queries"] = refined_queries
            metadata["num_queries"] = 1
    else:
        # Single query mode (backward compatibility)
        refined_queries = [query]
        metadata["refined_queries"] = refined_queries
        metadata["num_queries"] = 1

    # ========== Round 2: Execute multiple queries retrieval in parallel ==========
    logger.info(f"Round 2: Executing {len(refined_queries)} queries in parallel...")

    try:
        round2_results, round2_metadata = await multi_query_retrieval(
            queries=refined_queries,
            candidates=candidates,
            emb_top_n=config.round1_emb_top_n,
            bm25_top_n=config.round1_bm25_top_n,
            final_top_n=config.round2_per_query_top_n,
            rrf_k=60,
        )

        metadata["round2_count"] = len(round2_results)
        metadata["round2_latency_ms"] = round2_metadata.get("total_latency_ms", 0)
        metadata["multi_query_total_docs"] = round2_metadata.get(
            "total_docs_before_fusion", 0
        )

        logger.info(f"Round 2: Retrieved {len(round2_results)} unique documents")

    except Exception as e:
        logger.error(f"Round 2 failed: {e}, using Round 1 results")
        metadata["total_latency_ms"] = (time.time() - start_time) * 1000
        return round1_results, metadata

    # ========== Merge: ensure total 40 documents ==========
    logger.info("Merge: Combining Round 1 and Round 2...")

    # Deduplicate: use document ID
    round1_ids = {id(doc) for doc, _ in round1_results}
    round2_unique = [
        (doc, score) for doc, score in round2_results if id(doc) not in round1_ids
    ]

    # Merge: Round1 Top20 + Round2 deduplicated documents (ensure total <= 40)
    combined_results = round1_results.copy()
    needed_from_round2 = config.combined_total - len(combined_results)
    combined_results.extend(round2_unique[:needed_from_round2])

    logger.info(
        f"Merge: Round1={len(round1_results)}, Round2_unique={len(round2_unique[:needed_from_round2])}, Total={len(combined_results)}"
    )

    # ========== Rerank merged documents ==========
    if config.use_reranker and len(combined_results) > 0:
        logger.info(f"Rerank: Reranking {len(combined_results)} documents...")

        try:
            rerank_service = get_rerank_service()
            final_results = await rerank_candidates(
                query=query,  # Use original query for rerank
                candidates=combined_results,
                top_n=config.final_top_n,
                rerank_service=rerank_service,
            )

            logger.info(f"Rerank: Final Top {len(final_results)} selected")

        except Exception as e:
            logger.error(f"Final rerank failed: {e}, using top {config.final_top_n}")
            final_results = combined_results[: config.final_top_n]
    else:
        # No Reranker, directly return Top N
        final_results = combined_results[: config.final_top_n]
        logger.info(f"No Rerank: Returning Top {len(final_results)}")

    metadata["final_count"] = len(final_results)
    metadata["total_latency_ms"] = (time.time() - start_time) * 1000

    logger.info(
        f"Complete: Final {len(final_results)} docs | Latency {metadata['total_latency_ms']:.0f}ms"
    )
    logger.info(f"{'='*60}\n")

    return final_results, metadata
