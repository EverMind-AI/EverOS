"""Tier 1 (LLM only) end-to-end acceptance test.

Pins the behavior every Task in the embed-soft-dependency refactor
promises for a user who has configured only an LLM: writes still work,
keyword search still works, everything that needs embedding/rerank
fails fast with a 422 instead of a silent 200 or a 500, and the
background strategy engine never registers a strategy it cannot run.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from everos.infra.persistence.lancedb import Episode, get_table

from .conftest import add_and_flush


async def _episode_rows(owner_id: str) -> list[dict]:
    table = await get_table(Episode.TABLE_NAME, Episode)
    return await table.query().where(f"owner_id = '{owner_id}'").to_list()


# ---------------------------------------------------------------------------
# 1. POST /memory/add -> md write + LanceDB vector=NULL
# ---------------------------------------------------------------------------


async def test_add_memory_writes_null_vector(tier1_runtime: AsyncClient) -> None:
    body = await add_and_flush(tier1_runtime, session_id="s_tier1_add")
    assert body["data"]["status"] == "extracted"

    rows = await _episode_rows("u_alice")
    assert rows, "expected cascade to index the new episode into LanceDB"
    assert rows[0]["vector"] is None, "Tier 1 (no embed) must write vector=NULL"


# ---------------------------------------------------------------------------
# 2. KEYWORD_ONLY search works even with vector=NULL rows
# ---------------------------------------------------------------------------


async def test_keyword_search_returns_results(tier1_runtime: AsyncClient) -> None:
    await add_and_flush(tier1_runtime, session_id="s_tier1_kw")

    resp = await tier1_runtime.post(
        "/api/v1/memory/search",
        json={
            "user_id": "u_alice",
            "query": "hiking",
            "method": "keyword",
        },
    )
    assert resp.status_code == 200, resp.text
    episodes = resp.json()["data"]["episodes"]
    assert episodes, "BM25 keyword search must find the vector=NULL episode"


# ---------------------------------------------------------------------------
# 3-5. VECTOR / HYBRID (user) / AGENTIC all 422 with the right feature tag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,expected_feature",
    [
        ("vector", "vector"),
        ("hybrid", "user_hybrid"),
        ("agentic", "agentic_search"),
    ],
)
async def test_embed_dependent_search_methods_422(
    tier1_runtime: AsyncClient, method: str, expected_feature: str
) -> None:
    resp = await tier1_runtime.post(
        "/api/v1/memory/search",
        json={"user_id": "u_alice", "query": "hiking", "method": method},
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "PROVIDER_NOT_CONFIGURED"
    assert "embedding" in body["error"]["message"]
    assert expected_feature in body["error"]["message"]


# ---------------------------------------------------------------------------
# 6. Knowledge endpoints all gated off
# ---------------------------------------------------------------------------


async def test_knowledge_create_document_422(tier1_runtime: AsyncClient) -> None:
    resp = await tier1_runtime.post(
        "/api/v1/knowledge/documents",
        data={"title": "Some Title"},
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "PROVIDER_NOT_CONFIGURED"
    assert "knowledge" in body["error"]["message"]


async def test_knowledge_search_422(tier1_runtime: AsyncClient) -> None:
    resp = await tier1_runtime.post(
        "/api/v1/knowledge/search", json={"query": "anything"}
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 7. /health reflects the Tier 1 capability matrix
# ---------------------------------------------------------------------------


async def test_health_reports_tier1_capabilities(tier1_runtime: AsyncClient) -> None:
    resp = await tier1_runtime.get("/health")
    assert resp.status_code == 200
    body = resp.json()

    caps = body["capabilities"]
    assert caps["llm"] is True
    assert caps["embed"] is False
    assert caps["rerank"] is False

    disabled = set(body["disabled_features"])
    expected_subset = {
        "vector_search",
        "hybrid_search",
        "agentic_search",
        "reflection",
        "skill_extraction",
        "knowledge",
    }
    assert expected_subset <= disabled


# ---------------------------------------------------------------------------
# 8. OME registers every strategy in Tier 1 (body-guard, not registration-guard)
# ---------------------------------------------------------------------------


async def test_ome_registers_all_strategies_in_tier1(
    tier1_runtime: AsyncClient,
) -> None:
    """Registration is unconditional; embed-requiring strategies body-guard.

    OME's registry is frozen after ``engine.start()`` (scheduler snapshots
    the strategy set for Cron / Idle job creation), so a registration-time
    gate cannot adapt to a runtime tier upgrade. All eight strategies are
    registered up-front; the four embed-requiring ones check
    :func:`get_embedding_capability` at body entry and no-op when the
    capability is unavailable — see the per-strategy unit tests for that
    half of the contract.
    """
    import importlib

    svc = importlib.import_module("everos.service.memorize")
    engine = svc._get_engine()
    registered_names = {meta.name for meta in engine._registry.all()}

    from everos.memory.strategies import (
        extract_agent_case,
        extract_agent_skill,
        extract_atomic_facts,
        extract_foresight,
        extract_user_profile,
        reflect_episodes,
        trigger_profile_clustering,
        trigger_skill_clustering,
    )

    all_strategy_names = {
        strategy.meta.name
        for strategy in (
            extract_atomic_facts,
            extract_foresight,
            extract_agent_case,
            extract_user_profile,
            trigger_profile_clustering,
            trigger_skill_clustering,
            extract_agent_skill,
            reflect_episodes,
        )
    }
    assert all_strategy_names <= registered_names
