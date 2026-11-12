"""Tier 1 (LLM only) end-to-end acceptance test: a deployment without an
embedding model.

Pins the behavior promised to a user who has configured only an LLM:
writes still work (markdown, cascade rows with no vector, the user profile),
memory search recalls by keyword (BM25) alone through the same multi-round
loop, ``/health`` reports vector recall disabled, and the background
strategy engine registers every strategy, the ones that need an embedder
guarding their own bodies.
"""

from __future__ import annotations

import asyncio

from httpx import AsyncClient

from everos.service import Application

from .conftest import add_and_flush, episode_rows, tier_storage

# ---------------------------------------------------------------------------
# 1. POST /memory/add -> md write + LanceDB vector=NULL
# ---------------------------------------------------------------------------


async def test_add_memory_writes_null_vector(tier1_runtime: AsyncClient) -> None:
    body = await add_and_flush(tier1_runtime, session_id="s_tier1_add")
    assert body["data"]["status"] == "extracted"

    rows = await episode_rows("u_alice")
    assert rows, "expected cascade to index the new episode into LanceDB"
    assert rows[0]["vector"] is None, "Tier 1 (no embed) must write vector=NULL"


# ---------------------------------------------------------------------------
# 2. Memory search recalls by keyword (BM25) alone
# ---------------------------------------------------------------------------


async def test_search_recalls_by_keyword_without_an_embedder(
    tier1_runtime: AsyncClient,
) -> None:
    await add_and_flush(tier1_runtime, session_id="s_tier1_search")

    resp = await tier1_runtime.post(
        "/api/v2/memory/search",
        json={"user_id": "u_alice", "query": "hiking", "method": "llm_multiround"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()["data"]
    assert [e["subject"] for e in data["episodes"]] == ["Hiking"]
    assert data["degraded"] == []

    # A word the episode does not hold recalls nothing: there is no vector leg
    # to bring back a near miss.
    resp = await tier1_runtime.post(
        "/api/v2/memory/search",
        json={"user_id": "u_alice", "query": "zebra", "method": "llm_multiround"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["episodes"] == []


async def test_the_profile_takes_its_direct_path(tier1_runtime: AsyncClient) -> None:
    """Without an embedder the profile is still written from the episode."""
    await add_and_flush(tier1_runtime, session_id="s_tier1_profile")

    from everos.infra.persistence.index import eq

    rows: list[object] = []
    for _ in range(200):
        rows = await tier_storage().repos.user_profile.scan(eq("owner_id", "u_alice"))
        if rows:
            break
        await asyncio.sleep(0.1)
    assert rows, "the profile strategy wrote no profile row"


# ---------------------------------------------------------------------------
# 3. /health reflects the Tier 1 capability matrix
# ---------------------------------------------------------------------------


async def test_health_reports_tier1_capabilities(tier1_runtime: AsyncClient) -> None:
    resp = await tier1_runtime.get("/health")
    assert resp.status_code == 200
    body = resp.json()

    caps = body["capabilities"]
    assert "llm" not in caps
    assert caps["embed"] is False

    # multimodal_llm / parser are orthogonal to the tiers: this suite
    # configures no multimodal model, so the multimodal upload may read
    # disabled next to vector recall.
    disabled = set(body["disabled_features"])
    assert "vector_recall" in disabled
    assert "memory_search" not in disabled
    assert disabled <= {"vector_recall", "multimodal_upload"}


# ---------------------------------------------------------------------------
# 4. OME registers every strategy in Tier 1 (body-guard, not registration-guard)
# ---------------------------------------------------------------------------


async def test_ome_registers_all_strategies_in_tier1(
    tier1_runtime: AsyncClient,
    tier_application: Application,
) -> None:
    """Registration is unconditional; embed-requiring strategies body-guard.

    OME's registry is frozen after ``engine.start()`` (scheduler snapshots
    the strategy set for Cron / Idle job creation), so a registration-time
    gate cannot adapt to a tier change. Every spec in ``STRATEGY_SPECS`` is
    registered up-front; the embed-requiring ones return no events when
    ``deps.embedding`` is ``None`` — see the per-strategy unit tests for
    that half of the contract.
    """
    engine = tier_application._engine
    registered_names = {meta.name for meta in engine._registry.all()}

    from everos.memory.strategies import STRATEGY_SPECS

    all_strategy_names = {spec.name for spec in STRATEGY_SPECS}
    assert all_strategy_names <= registered_names
