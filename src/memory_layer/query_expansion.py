"""LLM-based query expansion for improved memory retrieval.

Generates paraphrase variants of a query so that retrieval can surface
memories that use different vocabulary than the original query.  Results
from each variant are merged (union, deduplicated by memory id) using
Reciprocal Rank Fusion.

Inspired by RAG-Fusion / HyDE: instead of a single fixed query string,
we produce 2-3 semantically equivalent rewordings and fuse their recall
sets.  This directly addresses the "vocabulary mismatch" failure mode
where stored memories use different terms than the retrieval query
(e.g. "rescue inhaler protocol" stored vs "gym bag" queried, or vice
versa).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_PARAPHRASE_PROMPT = """\
You are a query expansion assistant helping a memory retrieval system.

Given the user query below, generate {n} paraphrase variants that capture
the same intent using different vocabulary.  The variants should:
- Cover synonyms and related terms the user might not have typed
- Use both formal and informal phrasings when applicable
- Stay concise (under 120 characters each)
- NOT repeat the original query verbatim

Original query:
{query}

Reply with ONLY a JSON object in this exact format (no markdown fences):
{{
  "variants": [
    "paraphrase 1",
    "paraphrase 2",
    "paraphrase 3"
  ]
}}
"""


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------


async def expand_query(
    query: str,
    llm_provider: Any,
    n_variants: int = 2,
    temperature: float = 0.6,
) -> List[str]:
    """Generate paraphrase variants for *query* using the LLM.

    Args:
        query: The original retrieval query.
        llm_provider: Any object exposing ``async generate(prompt, temperature,
            max_tokens) -> str``; the ``LLMProvider`` from
            ``memory_layer.llm.llm_provider`` satisfies this interface.
        n_variants: How many paraphrases to request (default 2; capped at 3).
        temperature: Sampling temperature for the LLM call.  Higher values
            produce more diverse paraphrases.

    Returns:
        A list of paraphrase strings.  On any failure the list is empty so
        callers can fall back to the original query without crashing.
    """
    # Guard: don't bother for very short queries or if provider absent
    if not query or not query.strip() or llm_provider is None:
        return []

    n_variants = max(1, min(n_variants, 3))

    prompt = _PARAPHRASE_PROMPT.format(query=query.strip(), n=n_variants)

    try:
        raw = await llm_provider.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=256,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "query_expansion: LLM call failed, skipping expansion: %s", exc
        )
        return []

    return _parse_variants(raw, query, n_variants)


def _parse_variants(
    raw: str,
    original_query: str,
    max_variants: int,
) -> List[str]:
    """Extract and validate the paraphrase list from the LLM response."""
    try:
        # Strip any accidental markdown fences
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # drop first and last fence lines
            text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in LLM response")

        parsed: Dict[str, Any] = json.loads(text[start:end])
        variants_raw: List[Any] = parsed.get("variants", [])

        if not isinstance(variants_raw, list):
            raise ValueError("'variants' is not a list")

        variants: List[str] = []
        orig_lower = original_query.lower().strip()
        for item in variants_raw[:max_variants]:
            if not isinstance(item, str):
                continue
            item = item.strip()
            # Drop empty strings or verbatim copies of the original query
            if item and item.lower() != orig_lower:
                variants.append(item)

        logger.debug("query_expansion: generated %d variants: %s", len(variants), variants)
        return variants

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "query_expansion: failed to parse LLM response, skipping: %s", exc
        )
        return []


# ---------------------------------------------------------------------------
# Merge helper (union dedup by dict key "id")
# ---------------------------------------------------------------------------


def merge_hits_by_id(
    hits_per_query: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Union-merge multiple hit lists, deduplicating by the ``id`` field.

    The first occurrence of each ``id`` wins (preserves the score from the
    query whose results are listed first — typically the original query).

    Args:
        hits_per_query: One list of hit dicts per query (original + variants).

    Returns:
        A flat deduplicated list of hit dicts.
    """
    seen_ids: set[str] = set()
    merged: List[Dict[str, Any]] = []

    for hits in hits_per_query:
        for hit in hits:
            hit_id: str = hit.get("id", "")
            if hit_id and hit_id in seen_ids:
                continue
            if hit_id:
                seen_ids.add(hit_id)
            merged.append(hit)

    return merged
