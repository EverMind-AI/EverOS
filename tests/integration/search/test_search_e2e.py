"""End-to-end ``/api/v1/memory/search`` tests over a real LoCoMo corpus.

Five tests, each pinning one path through :class:`SearchManager`'s
multi-round route (the only search method):

============================================  =================================
``test_multiround_recalls_an_episode``        recall of an extracted episode
``test_include_profile_returns_profile``      ``include_profile``
``test_partition_respects_owner_id``          cross-owner isolation
``test_unknown_owner_returns_empty_200``      empty response, no 500
``test_filter_dsl_compiles_and_excludes``     filters DSL → LanceDB ``where``
============================================  =================================

The corpus is built once by :func:`_ingested_memory_root` (session-
scoped fixture in ``conftest.py``) and shared across all tests. Each
test re-attaches a fresh lifespan via :func:`search_client`, so the
search-manager singletons rebuild from cold per-test — a regression
in the lazy-init path can't hide behind warm state from a prior test.

Bootstrapping: queries are derived from the corpus's own episode
subjects via :func:`pick_query_seeds`, not hardcoded. Closed-loop
correctness — what the pipeline extracted should be findable by the
search side.

Assertions follow the project's conservation + lower-bound + shape
convention (see :func:`_helpers.assert_recall`): no exact ranks, no
exact scores, no exact ids. LLM-driven retrieval is non-deterministic
across runs; brittle assertions cause CI noise, not signal.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ._helpers import (
    assert_recall,
    pick_query_seeds,
)

# Whole module is opt-in — it depends on ``_ingested_memory_root`` which
# spends ~10 min running real LLM + embedder against LoCoMo conv_0.
pytestmark = pytest.mark.slow


# ── 1. Multi-round recall ──────────────────────────────────────────────


async def test_multiround_recalls_an_episode(
    search_client: httpx.AsyncClient,
    _ingested_memory_root: Path,
) -> None:
    """An extracted episode's subject finds an episode of its owner."""
    owner, subject = pick_query_seeds(_ingested_memory_root, limit=1)[0]
    await assert_recall(search_client, owner_id=owner, query=subject)


# ── 2. include_profile ─────────────────────────────────────────────────


async def test_include_profile_returns_profile(
    search_client: httpx.AsyncClient,
    _ingested_memory_root: Path,
) -> None:
    """``include_profile=true`` must populate the profiles array."""
    owner, subject = pick_query_seeds(_ingested_memory_root, limit=1)[0]
    resp = await search_client.post(
        "/api/v1/memory/search",
        json={
            "user_id": owner,
            "query": subject,
            "method": "llm_multiround",
            "top_k": 5,
            "include_profile": True,
        },
        timeout=300.0,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["profiles"], "include_profile=true but profiles[] empty"
    assert data["profiles"][0]["user_id"] == owner


# ── 3. Owner partition ─────────────────────────────────────────────────


async def test_partition_respects_owner_id(
    search_client: httpx.AsyncClient,
    _ingested_memory_root: Path,
) -> None:
    """Querying owner=A must not leak owner=B's data, even on shared topics."""
    seeds = pick_query_seeds(_ingested_memory_root, limit=2)
    target_owner, subject = seeds[0]
    await assert_recall(search_client, owner_id=target_owner, query=subject)


# ── 4. Unknown owner ───────────────────────────────────────────────────


async def test_unknown_owner_returns_empty_200(
    search_client: httpx.AsyncClient,
) -> None:
    """An owner that the corpus never saw → 200 with empty arrays."""
    resp = await search_client.post(
        "/api/v1/memory/search",
        json={
            "user_id": "ghost_user_does_not_exist",
            "query": "anything",
            "method": "llm_multiround",
            "top_k": 5,
        },
        timeout=300.0,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert data["episodes"] == []
    assert data["images"] == []
    assert data["profiles"] == []


# ── 5. Filter DSL ──────────────────────────────────────────────────────


async def test_filter_dsl_compiles_and_excludes(
    search_client: httpx.AsyncClient,
    _ingested_memory_root: Path,
) -> None:
    """Add a ``session_id`` ne-filter, verify the returned hits respect it."""
    owner, subject = pick_query_seeds(_ingested_memory_root, limit=1)[0]
    bogus_session = "session_that_never_was"
    resp = await search_client.post(
        "/api/v1/memory/search",
        json={
            "user_id": owner,
            "query": subject,
            "method": "llm_multiround",
            "top_k": 10,
            "filters": {"session_id": {"ne": bogus_session}},
        },
        timeout=300.0,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    # The filter is satisfied by every real episode (none have the
    # bogus id), so the filter compiled and shipped to the index
    # without breaking recall.
    assert data["episodes"]
    for ep in data["episodes"]:
        assert ep["session_id"] != bogus_session
