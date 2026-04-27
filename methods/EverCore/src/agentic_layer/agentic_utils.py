"""
Agentic Retrieval utility functions

Provides tools required for LLM-guided multi-round retrieval:
1. Sufficiency Check: Determine if retrieval results are sufficient
2. Multi-Query Generation: Generate multiple complementary improved queries
3. Document Formatting: Format documents for LLM usage
"""

import os
import json
import asyncio
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ==================== Prompt Templates ====================

SUFFICIENCY_CHECK_PROMPT = """You are a memory retrieval evaluation expert. Please assess whether the currently retrieved memories are sufficient to answer the user's query.

User query:
{query}

Retrieved memories:
{retrieved_docs}

Please determine whether these memories are sufficient to answer the user's query.

Output format (JSON):
{{
    "is_sufficient": true/false,
    "reasoning": "Your reasoning for the judgment",
    "missing_information": ["Missing information 1", "Missing information 2"]
}}

Requirements:
1. If the memories contain key information needed to answer the query, judge as sufficient (true)
2. If key information is missing, judge as insufficient (false), and list the missing information
3. reasoning should be concise and clear
4. missing_information should only be filled when insufficient, otherwise empty array
"""


MULTI_QUERY_GENERATION_PROMPT = """You are a query optimization expert. The user's original query failed to retrieve sufficient information; please generate multiple complementary improved queries.

Original query:
{original_query}

Currently retrieved memories:
{retrieved_docs}

Missing information:
{missing_info}

Please generate 2-3 complementary queries to help find the missing information. These queries should:
1. Focus on different missing information points
2. Use different expressions
3. Avoid being identical to the original query
4. Remain concise and clear

Output format (JSON):
{{
    "queries": [
        "Improved query 1",
        "Improved query 2",
        "Improved query 3"
    ],
    "reasoning": "Explanation of query generation strategy"
}}

Requirements:
1. queries array contains 2-3 queries
2. Each query length between 5-200 characters
3. reasoning explains the generation strategy
"""


# ==================== Configuration Class ====================


@dataclass
class AgenticConfig:
    """Agentic retrieval configuration"""

    # Round 1 configuration
    round1_emb_top_n: int = 50  # Number of embedding candidates
    round1_bm25_top_n: int = 50  # Number of BM25 candidates
    round1_top_n: int = 20  # Number returned after RRF fusion
    round1_rerank_top_n: int = int(os.getenv("AGENTIC_ROUND1_RERANK_TOP_N", "10"))  # Number after reranking used for LLM judgment

    # LLM configuration
    llm_temperature: float = 0.0  # Low temperature for judgment
    llm_max_tokens: int = 500

    # Round 2 configuration
    enable_multi_query: bool = True  # Whether to enable multi-query
    num_queries: int = 3  # Desired number of generated queries
    round2_per_query_top_n: int = 50  # Number recalled per query

    # Fusion configuration
    combined_total: int = 40  # Total number after merging
    final_top_n: int = 20  # Final number returned

    # Rerank configuration
    use_reranker: bool = True
    reranker_instruction: str = (
        "Determine if the passage contains specific facts, entities (names, dates, locations), "
        "or details that directly answer the question."
    )
    reranker_batch_size: int = 10
    reranker_timeout: float = 30.0

    # Fallback strategy
    fallback_on_error: bool = True  # Fallback when LLM fails
    timeout: float = 60.0  # Overall timeout (seconds)


# ==================== Utility Functions ====================


def format_documents_for_llm(
    results: List[Tuple[Any, float]], max_docs: int = 10
) -> str:
    """
    Format retrieval results for LLM usage

    Args:
        results: List of retrieval results [(candidate, score), ...]
        max_docs: Maximum number of documents to include

    Returns:
        Formatted document string
    """
    formatted_docs = []

    for i, (candidate, score) in enumerate(results[:max_docs], 1):
        # Extract memory content
        timestamp = getattr(candidate, 'timestamp', 'N/A')
        if hasattr(timestamp, 'strftime'):
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            timestamp_str = str(timestamp)

        # Prioritize episode (core content of MemCell)
        content = getattr(candidate, 'episode', None)
        if not content:
            content = getattr(candidate, 'summary', None)
        if not content:
            content = getattr(candidate, 'subject', 'N/A')

        # Build document entry
        doc_entry = f"[Memory {i}]\n"
        doc_entry += f"Time: {timestamp_str}\n"
        doc_entry += f"Content: {content}\n"
        doc_entry += f"Relevance score: {score:.4f}\n"

        formatted_docs.append(doc_entry)

    return "\n".join(formatted_docs) if formatted_docs else "No retrieval results"


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON response returned by LLM

    Args:
        response: Raw response string from LLM

    Returns:
        Parsed dictionary

    Raises:
        ValueError: JSON parsing failed
    """
    try:
        # Extract JSON part (may contain additional text)
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1

        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")

        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.debug(f"Raw response: {response[:500]}")
        raise ValueError(f"JSON parse error: {e}")


def parse_sufficiency_response(response: str) -> Tuple[bool, str, List[str]]:
    """
    Parse sufficiency judgment response

    Args:
        response: Raw response from LLM

    Returns:
        (is_sufficient, reasoning, missing_information)
    """
    try:
        result = parse_json_response(response)

        # Validate required fields
        if "is_sufficient" not in result:
            raise ValueError("Missing 'is_sufficient' field")

        is_sufficient = bool(result["is_sufficient"])
        reasoning = result.get("reasoning", "No reasoning provided")
        missing_info = result.get("missing_information", [])

        # Validate types
        if not isinstance(missing_info, list):
            missing_info = []

        return is_sufficient, reasoning, missing_info

    except Exception as e:
        logger.error(f"Failed to parse sufficiency response: {e}")
        # Conservative fallback: assume sufficient
        return True, f"Parse error: {str(e)}", []


def parse_multi_query_response(
    response: str, original_query: str
) -> Tuple[List[str], str]:
    """
    Parse multi-query generation response

    Args:
        response: Raw response from LLM
        original_query: Original query (for fallback)

    Returns:
        (queries_list, reasoning)
    """
    try:
        result = parse_json_response(response)

        # Validate required fields
        if "queries" not in result or not isinstance(result["queries"], list):
            raise ValueError("Missing or invalid 'queries' field")

        queries = result["queries"]
        reasoning = result.get("reasoning", "No reasoning provided")

        # Filter and validate queries
        valid_queries = []
        for q in queries:
            if isinstance(q, str) and 5 <= len(q) <= 300:
                # Avoid being identical to original query
                if q.lower().strip() != original_query.lower().strip():
                    valid_queries.append(q.strip())

        # At least return 1 query
        if not valid_queries:
            logger.warning("No valid queries generated, using original")
            return [original_query], "Fallback: used original query"

        # Limit to maximum 3 queries
        valid_queries = valid_queries[:3]

        logger.info(f"Generated {len(valid_queries)} valid queries")
        return valid_queries, reasoning

    except Exception as e:
        logger.error(f"Failed to parse multi-query response: {e}")
        # Fallback: return original query
        return [original_query], f"Parse error: {str(e)}"


# ==================== Core LLM Utility Functions ====================


async def check_sufficiency(
    query: str, results: List[Tuple[Any, float]], llm_provider, max_docs: int = 5
) -> Tuple[bool, str, List[str]]:
    """
    Check if retrieval results are sufficient

    Use LLM to judge whether currently retrieved memories are sufficient to answer the user's query.
    If insufficient, return a list of missing information.

    Args:
        query: User query
        results: Retrieval results (Top N)
        llm_provider: LLM Provider (Memory Layer)
        max_docs: Maximum number of documents to evaluate

    Returns:
        (is_sufficient, reasoning, missing_information)
        - is_sufficient: True means sufficient, False means insufficient
        - reasoning: LLM's judgment reasoning
        - missing_information: List of missing information (only populated when insufficient)

    Example:
        >>> is_sufficient, reasoning, missing = await check_sufficiency(
        ...     query="What does the user like to eat?",
        ...     results=[(mem1, 0.92), (mem2, 0.85)],
        ...     llm_provider=llm
        ... )
        >>> print(is_sufficient)  # False
        >>> print(missing)  # ["User's specific cuisine preferences", "Taste preferences"]
    """
    try:
        # 1. Format documents
        retrieved_docs = format_documents_for_llm(results, max_docs=max_docs)

        # 2. Build Prompt
        prompt = SUFFICIENCY_CHECK_PROMPT.format(
            query=query, retrieved_docs=retrieved_docs
        )

        # 3. Call LLM
        logger.debug(f"Calling LLM for sufficiency check on query: {query[:50]}...")
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.0,  # Low temperature for more stable judgment
            max_tokens=500,
        )

        # 4. Parse response
        is_sufficient, reasoning, missing_info = parse_sufficiency_response(result_text)

        logger.info(f"Sufficiency check result: {is_sufficient}")
        logger.debug(f"Reasoning: {reasoning}")

        return is_sufficient, reasoning, missing_info

    except asyncio.TimeoutError:
        logger.error("Sufficiency check timeout")
        # Timeout fallback: assume sufficient (to avoid infinite retries)
        return True, "Timeout: LLM took too long", []

    except Exception as e:
        logger.error(f"Sufficiency check failed: {e}", exc_info=True)
        # Conservative fallback: assume sufficient
        return True, f"Error: {str(e)}", []


async def generate_multi_queries(
    original_query: str,
    results: List[Tuple[Any, float]],
    missing_info: List[str],
    llm_provider,
    max_docs: int = 5,
    num_queries: int = 3,
) -> Tuple[List[str], str]:
    """
    Generate multiple complementary improved queries

    Based on the original query, current retrieval results, and missing information, generate multiple complementary queries.
    These queries are used for multi-query retrieval to help find missing information.

    Args:
        original_query: Original query
        results: Round 1 retrieval results
        missing_info: List of missing information
        llm_provider: LLM Provider
        max_docs: Maximum number of documents to use
        num_queries: Desired number of queries to generate (actual may be 1-3)

    Returns:
        (queries_list, reasoning)
        - queries_list: List of generated queries (1-3)
        - reasoning: LLM's explanation of generation strategy

    Example:
        >>> queries, reasoning = await generate_multi_queries(
        ...     original_query="What does the user like to eat?",
        ...     results=[(mem1, 0.9)],
        ...     missing_info=["cuisine preference", "taste"],
        ...     llm_provider=llm
        ... )
        >>> print(queries)
        ['What is the user's favorite cuisine?', 'What taste does the user prefer?', 'What are the user's eating habits?']
    """
    try:
        # 1. Format documents and missing information
        retrieved_docs = format_documents_for_llm(results, max_docs=max_docs)
        missing_info_str = ", ".join(missing_info) if missing_info else "N/A"

        # 2. Build Prompt
        prompt = MULTI_QUERY_GENERATION_PROMPT.format(
            original_query=original_query,
            retrieved_docs=retrieved_docs,
            missing_info=missing_info_str,
        )

        # 3. Call LLM
        logger.debug(f"Generating multi-queries for: {original_query[:50]}...")
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.4,  # Slightly higher temperature to increase query diversity
            max_tokens=300,
        )

        # 4. Parse response
        queries, reasoning = parse_multi_query_response(result_text, original_query)

        logger.info(f"Generated {len(queries)} queries")
        for i, q in enumerate(queries, 1):
            logger.debug(f"  Query {i}: {q[:80]}{'...' if len(q) > 80 else ''}")

        return queries, reasoning

    except asyncio.TimeoutError:
        logger.error("Multi-query generation timeout")
        # Timeout fallback: use original query
        return [original_query], "Timeout: used original query"

    except Exception as e:
        logger.error(f"Multi-query generation failed: {e}", exc_info=True)
        # Fallback to original query
        return [original_query], f"Error: {str(e)}"
