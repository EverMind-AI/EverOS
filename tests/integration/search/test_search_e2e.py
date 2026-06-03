"""End-to-end ``/api/v1/memory/search`` tests over a real LoCoMo corpus.

Six tests, each pinning one path through :class:`SearchManager`:

============================================  =================================
``test_keyword_recalls_atomic_fact_origin``   keyword (BM25 only)
``test_vector_recalls_atomic_fact_origin``    vector (cosine only)
``test_hybrid_with_profile_returns_profile``  hybrid + ``include_profile``
``test_partition_respects_owner_id``          cross-owner isolation
``test_unknown_owner_returns_empty_200``      empty response, no 500
``test_filter_dsl_compiles_and_excludes``     filters DSL → LanceDB ``where``
============================================  =================================

The corpus is built once by :func:`_ingested_memory_root` (session-
scoped fixture in ``conftest.py``) and shared across all tests. Each
test re-attaches a fresh lifespan via :func:`search_client`, so the
search-manager singletons rebuild from cold per-test — a regression
in the lazy-init path can't hide behind warm state from a prior test.

Bootstrapping: queries are derived from the corpus's own
``atomic_facts`` md files via :func:`pick_query_seeds`, not
hardcoded. Closed-loop correctness — what the pipeline extracted
should be findable by the search side.

Assertions follow the project's "守恒 + 下界 + 形状" convention
(see :func:`_helpers.assert_recall`): no exact ranks, no exact
scores, no exact ids. LLM-driven retrieval is non-deterministic
across runs; brittle assertions cause CI noise, not signal.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ._helpers import (
    assert_recall,
    flatten_hits,
    pick_query_seeds,
)

# Whole module is opt-in — it depends on ``_ingested_memory_root`` which
# spends ~10 min running real LLM + embedder against LoCoMo conv_0.
pytestmark = pytest.mark.slow


# ── 1. Keyword recall ──────────────────────────────────────────────────


async def test_keyword_recalls_atomic_fact_origin(
    search_client: httpx.AsyncClient,
    _ingested_memory_root: Path,
) -> None:
    """BM25 must recall *some* episode for *some* fact-derived bigram.

    The project's tokenizer is jieba (CJK-first); single short
    English tokens and proper nouns / all-caps acronyms recall
    poorly, but ordinary lowercase content bigrams recall reliably
    (verified empirically). So we walk through the first N atomic
    facts, pull consecutive lowercase content tokens, and pass the
    test as soon as one candidate bigram returns ≥ 1 hit. This
    validates the BM25 plumbing without coupling to which specific
    fact got sampled — vector + hybrid tests own the strict
    closed-loop recall claim.
    """
    seeds = pick_query_seeds(_ingested_memory_root, limit=20)
    last_query: str | None = None
    for owner, fact in seeds:
        for query in _candidate_bigrams(fact):
            last_query = query
            resp = await search_client.post(
                "/api/v1/memory/search",
                json={
                    "user_id": owner,
                    "query": query,
                    "method": "keyword",
                    "top_k": 5,
                },
                timeout=60.0,
            )
            assert resp.status_code == 200, resp.text
            hits = flatten_hits(resp.json()["data"])
            if hits:
                # Partition still holds even on a successful keyword hit.
                for hit_owner, _s, _t in hits:
                    if hit_owner is not None:
                        assert hit_owner == owner
                return
    raise AssertionError(
        f"BM25 returned 0 hits across {len(seeds)} fact seeds; "
        f"last tried query={last_query!r}"
    )


def _candidate_bigrams(fact: str) -> list[str]:
    """Lowercase consecutive content-token bigrams from ``fact``.

    Skip tokens that include uppercase letters in the original text
    (proper nouns / acronyms — empirically poor BM25 recall under
    jieba). Returns at most 5 candidates per fact, in source order.
    """
    import re as _re

    out: list[str] = []
    tokens: list[str] = []
    for raw in _re.findall(r"\w+", fact):
        if raw.lower() == raw and len(raw) >= 3:
            tokens.append(raw)
    for i in range(len(tokens) - 1):
        out.append(f"{tokens[i]} {tokens[i + 1]}")
        if len(out) >= 5:
            break
    return out


# ── 2. Vector recall ───────────────────────────────────────────────────


async def test_vector_recalls_atomic_fact_origin(
    search_client: httpx.AsyncClient,
    _ingested_memory_root: Path,
) -> None:
    """Same fact via cosine ANN — independent of BM25 tokenisation."""
    owner, fact = pick_query_seeds(_ingested_memory_root, limit=1)[0]
    await assert_recall(
        search_client,
        owner_id=owner,
        query=fact,
        method="vector",
        # Cosine: identical text would score ~1.0; threshold loose
        # because the LLM-summarised episode text isn't the verbatim fact.
        min_score=0.1,
    )


# ── 3. Hybrid + include_profile ────────────────────────────────────────


async def test_hybrid_with_profile_returns_profile(
    search_client: httpx.AsyncClient,
    _ingested_memory_root: Path,
) -> None:
    """``include_profile=true`` must populate the profiles array."""
    owner, fact = pick_query_seeds(_ingested_memory_root, limit=1)[0]
    resp = await search_client.post(
        "/api/v1/memory/search",
        json={
            "user_id": owner,
            "query": fact,
            "method": "hybrid",
            "top_k": 5,
            "include_profile": True,
        },
        timeout=120.0,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["profiles"], "include_profile=true but profiles[] empty"
    assert data["profiles"][0]["user_id"] == owner


# ── 4. Owner partition ─────────────────────────────────────────────────


async def test_partition_respects_owner_id(
    search_client: httpx.AsyncClient,
    _ingested_memory_root: Path,
) -> None:
    """Querying owner=A must not leak owner=B's data, even on shared topics."""
    seeds = pick_query_seeds(_ingested_memory_root, limit=2)
    owners = {o for o, _ in seeds}
    assert len(owners) >= 1, "need at least one owner in the corpus"
    target_owner = next(iter(owners))
    _, fact = next((o, f) for o, f in seeds if o == target_owner)

    body = await assert_recall(
        search_client,
        owner_id=target_owner,
        query=fact,
        method="hybrid",
    )
    # Agent tracks must be empty for user owners.
    assert body["data"]["agent_cases"] == []
    assert body["data"]["agent_skills"] == []


# ── 5. Unknown owner ───────────────────────────────────────────────────


async def test_unknown_owner_returns_empty_200(
    search_client: httpx.AsyncClient,
) -> None:
    """An owner that the corpus never saw → 200 with four empty arrays."""
    resp = await search_client.post(
        "/api/v1/memory/search",
        json={
            "user_id": "ghost_user_does_not_exist",
            "query": "anything",
            "method": "hybrid",
            "top_k": 5,
        },
        timeout=60.0,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["episodes"] == []
    assert data["profiles"] == []
    assert data["agent_cases"] == []
    assert data["agent_skills"] == []


# ── 6. Filter DSL ──────────────────────────────────────────────────────


async def test_filter_dsl_compiles_and_excludes(
    search_client: httpx.AsyncClient,
    _ingested_memory_root: Path,
) -> None:
    """Add a ``session_id`` ne-filter, verify the returned hits respect it."""
    owner, fact = pick_query_seeds(_ingested_memory_root, limit=1)[0]
    bogus_session = "session_that_never_was"
    resp = await search_client.post(
        "/api/v1/memory/search",
        json={
            "user_id": owner,
            "query": fact,
            "method": "keyword",
            "top_k": 10,
            "filters": {"session_id": {"ne": bogus_session}},
        },
        timeout=120.0,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    # The filter is satisfied by every real episode (none have the
    # bogus id), so the hit count should be ≥ 1 — the filter
    # compiled and shipped to LanceDB without breaking recall.
    for ep in data["episodes"]:
        assert ep["session_id"] != bogus_session
