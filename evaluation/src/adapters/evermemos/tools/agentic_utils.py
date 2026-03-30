"""
Agentic Retrieval utility functions.

Provides tools for LLM-guided multi-round retrieval:
1. Sufficiency Check: Determine if retrieval results are sufficient
2. Query Refinement: Generate improved queries
3. Document Formatting: Format documents for LLM consumption
"""

import json
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional

# Import prompts from Python files
from evaluation.src.adapters.evermemos.prompts.sufficiency_check_prompts import SUFFICIENCY_CHECK_PROMPT
from evaluation.src.adapters.evermemos.prompts.refined_query_prompts import REFINED_QUERY_PROMPT
from evaluation.src.adapters.evermemos.prompts.multi_query_prompts import MULTI_QUERY_GENERATION_PROMPT


def format_documents_for_llm(
    results: List[Tuple[dict, float]],
    max_docs: int = 10,
    use_episode: bool = True
) -> str:
    """
    Format retrieval results for LLM consumption.
    
    Args:
        results: Retrieval results [(doc, score), ...]
        max_docs: Maximum number of documents to include
        use_episode: True=use Episode Memory, False=use Event Log
    
    Returns:
        Formatted document string
    """
    formatted_docs = []
    
    for i, (doc, score) in enumerate(results[:max_docs], start=1):
        subject = doc.get("subject", "N/A")
        
        # Choose format based on use_episode parameter
        if use_episode:
            # Use Episode Memory format (full narrative)
            episode = doc.get("episode", "N/A")
            
            # Limit episode length to avoid overly long prompts
            if len(episode) > 500:
                episode = episode[:500] + "..."
            
            doc_text = (
                f"Document {i}:\n"
                f"  Title: {subject}\n"
                f"  Content: {episode}\n"
            )
            formatted_docs.append(doc_text)
        else:
            # Use Event Log format (atomic facts)
            if doc.get("event_log") and doc["event_log"].get("atomic_fact"):
                event_log = doc["event_log"]
                time_str = event_log.get("time", "N/A")
                atomic_facts = event_log.get("atomic_fact", [])
                
                if isinstance(atomic_facts, list) and atomic_facts:
                    # Format as: Document N: title + time + fact list
                    facts_text = "\n     ".join(atomic_facts[:5])  # Show max 5 facts
                    if len(atomic_facts) > 5:
                        facts_text += f"\n     ... and {len(atomic_facts) - 5} more facts"
                    
                    doc_text = (
                        f"Document {i}:\n"
                        f"  Title: {subject}\n"
                        f"  Time: {time_str}\n"
                        f"  Facts:\n"
                        f"     {facts_text}\n"
                    )
                    formatted_docs.append(doc_text)
                    continue
            
            # Fall back to episode if no event_log
            episode = doc.get("episode", "N/A")
            if len(episode) > 500:
                episode = episode[:500] + "..."
            
            doc_text = (
                f"Document {i}:\n"
                f"  Title: {subject}\n"
                f"  Content: {episode}\n"
            )
            formatted_docs.append(doc_text)
    
    return "\n".join(formatted_docs)


def parse_json_response(response: str) -> dict:
    """
    Parse LLM JSON response with robust error handling.
    
    Args:
        response: Raw LLM response string
    
    Returns:
        Parsed JSON dictionary
    """
    try:
        # Extract JSON (LLM may add extra text before/after)
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)
        
        # Validate required fields
        if "is_sufficient" not in result:
            raise ValueError("Missing 'is_sufficient' field")
        
        # Set default values
        result.setdefault("reasoning", "No reasoning provided")
        result.setdefault("missing_information", [])
        result.setdefault("key_information_found", [])
        
        return result
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  Failed to parse LLM response: {e}")
        print(f"  Raw response: {response[:200]}...")
        
        # Conservative fallback: assume sufficient to avoid unnecessary second round
        return {
            "is_sufficient": True,
            "reasoning": f"Failed to parse: {str(e)}",
            "missing_information": [],
            "key_information_found": []
        }


def parse_refined_query(response: str, original_query: str) -> str:
    """
    Parse refined query from LLM response.
    
    Args:
        response: LLM response
        original_query: Original query (for fallback)
    
    Returns:
        Refined query string
    """
    refined = response.strip()
    
    # Remove common prefixes
    prefixes = ["Refined Query:", "Output:", "Answer:", "Query:"]
    for prefix in prefixes:
        if refined.startswith(prefix):
            refined = refined[len(prefix):].strip()
    
    # Validate length
    if len(refined) < 5 or len(refined) > 300:
        print(f"  ⚠️  Invalid refined query length ({len(refined)}), using original")
        return original_query
    
    # Avoid identical query
    if refined.lower() == original_query.lower():
        print(f"  ⚠️  Refined query identical to original, using original")
        return original_query
    
    return refined


async def check_sufficiency(
    query: str,
    results: List[Tuple[dict, float]],
    llm_provider,
    llm_config: dict,
    max_docs: int = 10
) -> Tuple[bool, str, List[str], List[str]]:
    """
    Check if retrieval results are sufficient.
    
    Args:
        query: User query
        results: Retrieval results (Top 10)
        llm_provider: LLM Provider (Memory Layer)
        llm_config: LLM configuration dict
        max_docs: Maximum number of documents to evaluate
    
    Returns:
        (is_sufficient, reasoning, missing_information, key_information_found)
    """
    try:
        # Format documents (using Episode Memory format)
        retrieved_docs = format_documents_for_llm(
            results, 
            max_docs=max_docs,
            use_episode=True
        )
        
        # Use prompt template
        prompt = SUFFICIENCY_CHECK_PROMPT.format(
            query=query,
            retrieved_docs=retrieved_docs
        )
        
        # Call LLM (using LLMProvider)
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.0,  # Low temperature for stable judgment
            max_tokens=500,
        )
        
        # Parse JSON response
        result = parse_json_response(result_text)
        
        return (
            result["is_sufficient"],
            result["reasoning"],
            result.get("missing_information", []),
            result.get("key_information_found", [])
        )
    
    except asyncio.TimeoutError:
        print(f"  ❌ Sufficiency check timeout (30s)")
        # Timeout fallback: assume sufficient
        return True, "Timeout: LLM took too long", [], []
    except Exception as e:
        print(f"  ❌ Sufficiency check failed: {e}")
        import traceback
        traceback.print_exc()
        # Conservative fallback: assume sufficient
        return True, f"Error: {str(e)}", [], []


async def generate_refined_query(
    original_query: str,
    results: List[Tuple[dict, float]],
    missing_info: List[str],
    llm_provider,
    llm_config: dict,
    key_info: Optional[List[str]] = None,
    max_docs: int = 10
) -> str:
    """
    Generate improved query.
    
    Args:
        original_query: Original query
        results: Round 1 retrieval results (Top 10)
        missing_info: List of missing information
        llm_provider: LLM Provider
        llm_config: LLM configuration
        key_info: List of key information found (optional, for future use)
        max_docs: Maximum number of documents to use
    
    Returns:
        Refined query string
    """
    try:
        # Format documents and missing info (using Episode Memory format)
        retrieved_docs = format_documents_for_llm(
            results, 
            max_docs=max_docs,
            use_episode=True
        )
        missing_info_str = ", ".join(missing_info) if missing_info else "N/A"
        
        # Use prompt template
        prompt = REFINED_QUERY_PROMPT.format(
            original_query=original_query,
            retrieved_docs=retrieved_docs,
            missing_info=missing_info_str
        )
        
        # Call LLM (using LLMProvider)
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.3,  # Higher temperature for creativity
            max_tokens=150,
        )
        
        # Parse and validate
        refined_query = parse_refined_query(result_text, original_query)
        
        return refined_query
    
    except asyncio.TimeoutError:
        print(f"  ❌ Query refinement timeout (30s)")
        # Timeout fallback: use original query
        return original_query
    except Exception as e:
        print(f"  ❌ Query refinement failed: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to original query
        return original_query


def parse_multi_query_response(response: str, original_query: str) -> Tuple[List[str], str]:
    """
    Parse multi-query generation JSON response.
    
    Args:
        response: Raw LLM response string
        original_query: Original query (for fallback)
    
    Returns:
        (queries_list, reasoning)
    """
    try:
        # Extract JSON
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = response[start_idx:end_idx]
        result = json.loads(json_str)
        
        # Validate required fields
        if "queries" not in result or not isinstance(result["queries"], list):
            raise ValueError("Missing or invalid 'queries' field")
        
        queries = result["queries"]
        reasoning = result.get("reasoning", "No reasoning provided")
        
        # Filter and validate queries
        valid_queries = []
        for q in queries:
            if isinstance(q, str) and 5 <= len(q) <= 300:
                # Avoid identical to original query
                if q.lower().strip() != original_query.lower().strip():
                    valid_queries.append(q.strip())
        
        # Return at least 1 query
        if not valid_queries:
            print(f"  ⚠️  No valid queries generated, using original")
            return [original_query], "Fallback: used original query"
        
        # Limit to maximum 3 queries
        valid_queries = valid_queries[:3]
        
        print(f"  ✅ Generated {len(valid_queries)} valid queries")
        return valid_queries, reasoning
    
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ⚠️  Failed to parse multi-query response: {e}")
        print(f"  Raw response: {response[:200]}...")
        
        # Fallback: return original query
        return [original_query], f"Parse error: {str(e)}"


async def generate_multi_queries(
    original_query: str,
    results: List[Tuple[dict, float]],
    missing_info: List[str],
    llm_provider,
    llm_config: dict,
    key_info: Optional[List[str]] = None,
    max_docs: int = 5,
    num_queries: int = 3
) -> Tuple[List[str], str]:
    """
    Generate multiple complementary queries for multi-query retrieval.
    
    Args:
        original_query: Original query
        results: Round 1 retrieval results (Top 5)
        missing_info: List of missing information
        llm_provider: LLM Provider
        llm_config: LLM configuration
        key_info: List of key information found (for better query refinement)
        max_docs: Maximum number of documents to use (default 5)
        num_queries: Expected number of queries to generate (default 3, may be fewer)
    
    Returns:
        (queries_list, reasoning)
        queries_list: Generated query list (1-3 queries)
        reasoning: LLM generation strategy explanation
    """
    try:
        # Format documents and missing info (using Episode Memory format)
        retrieved_docs = format_documents_for_llm(
            results, 
            max_docs=max_docs,
            use_episode=True
        )
        missing_info_str = ", ".join(missing_info) if missing_info else "N/A"
        key_info_str = ", ".join(key_info) if key_info else "N/A"
        
        # Use prompt template
        prompt = MULTI_QUERY_GENERATION_PROMPT.format(
            original_query=original_query,
            retrieved_docs=retrieved_docs,
            missing_info=missing_info_str,
            key_info=key_info_str
        )
        
        # Call LLM (using LLMProvider)
        result_text = await llm_provider.generate(
            prompt=prompt,
            temperature=0.4,  # Higher temperature for query diversity
            max_tokens=300,  # Increased tokens to support multiple queries
        )
        
        # Parse and validate
        queries, reasoning = parse_multi_query_response(result_text, original_query)
        
        print(f"  [Multi-Query] Generated {len(queries)} queries:")
        for i, q in enumerate(queries, 1):
            print(f"    Query {i}: {q[:80]}{'...' if len(q) > 80 else ''}")
        print(f"  [Multi-Query] Strategy: {reasoning}")
        
        return queries, reasoning
    
    except asyncio.TimeoutError:
        print(f"  ❌ Multi-query generation timeout (30s)")
        return [original_query], "Timeout: used original query"
    except Exception as e:
        print(f"  ❌ Multi-query generation failed: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to original query
        return [original_query], f"Error: {str(e)}"

