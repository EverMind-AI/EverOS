"""Tier 2 (LLM + embed, no rerank) end-to-end acceptance test.

Pins the behavior promised for a user who has configured LLM + embed
but not rerank: writes get real vectors, VECTOR/HYBRID (user) search
work, AGENTIC and knowledge stay gated on rerank, agent HYBRID has two
lanes gated independently by ``enable_llm_rerank``, and the embed-
dependent OME strategies (reflection / clustering / skill extraction)
are now registered.
"""

from __future__ import annotations

import importlib

import pytest
from httpx import AsyncClient

from everos.infra.persistence.lancedb import Episode, get_table

from .conftest import add_and_flush, seed_atomic_fact_for_episode

_STUB_VECTOR = [0.1] * 1024


async def _episode_rows(owner_id: str) -> list[dict]:
    table = await get_table(Episode.TABLE_NAME, Episode)
    return await table.query().where(f"owner_id = '{owner_id}'").to_list()


# ---------------------------------------------------------------------------
# 1. Memory add + KEYWORD / VECTOR / HYBRID (user) all succeed
# ---------------------------------------------------------------------------


async def test_add_memory_writes_real_vector(tier2_runtime: AsyncClient) -> None:
    await add_and_flush(tier2_runtime, session_id="s_tier2_add")

    rows = await _episode_rows("u_alice")
    assert rows, "expected cascade to index the new episode into LanceDB"
    assert rows[0]["vector"] is not None, "Tier 2 (embed available) must embed"
    assert len(rows[0]["vector"]) == 1024


@pytest.mark.parametrize("method", ["keyword", "vector", "hybrid"])
async def test_user_search_methods_succeed(
    tier2_runtime: AsyncClient, method: str
) -> None:
    await add_and_flush(tier2_runtime, session_id=f"s_tier2_{method}")
    # VECTOR recalls via atomic_fact MaxSim, not a direct episode ANN scan
    # -- see seed_atomic_fact_for_episode's docstring.
    episode_row = (await _episode_rows("u_alice"))[0]
    await seed_atomic_fact_for_episode(episode_row, vector=_STUB_VECTOR)

    resp = await tier2_runtime.post(
        "/api/v1/memory/search",
        json={"user_id": "u_alice", "query": "hiking", "method": method},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["episodes"], f"{method} search must find the episode"


# ---------------------------------------------------------------------------
# 2. AGENTIC still 422 (needs rerank)
# ---------------------------------------------------------------------------


async def test_agentic_search_422(tier2_runtime: AsyncClient) -> None:
    resp = await tier2_runtime.post(
        "/api/v1/memory/search",
        json={"user_id": "u_alice", "query": "hiking", "method": "agentic"},
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "PROVIDER_NOT_CONFIGURED"
    assert "rerank" in body["error"]["message"]
    assert "agentic_search" in body["error"]["message"]


# ---------------------------------------------------------------------------
# 3-4. Agent HYBRID: cross-encoder lane 422, LLM lane 200
# ---------------------------------------------------------------------------


async def test_agent_hybrid_cross_encoder_lane_422(tier2_runtime: AsyncClient) -> None:
    resp = await tier2_runtime.post(
        "/api/v1/memory/search",
        json={
            "agent_id": "a_bob",
            "query": "hiking",
            "method": "hybrid",
            "enable_llm_rerank": False,
        },
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "PROVIDER_NOT_CONFIGURED"
    assert "rerank" in body["error"]["message"]
    assert "agent_hybrid" in body["error"]["message"]
    assert "enable_llm_rerank=true" in body["error"]["message"]


async def test_agent_hybrid_llm_rerank_lane_200(tier2_runtime: AsyncClient) -> None:
    resp = await tier2_runtime.post(
        "/api/v1/memory/search",
        json={
            "agent_id": "a_bob",
            "query": "hiking",
            "method": "hybrid",
            "enable_llm_rerank": True,
        },
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# 5. Knowledge still 422 (needs rerank; embed alone isn't enough)
# ---------------------------------------------------------------------------


async def test_knowledge_create_document_422(tier2_runtime: AsyncClient) -> None:
    resp = await tier2_runtime.post(
        "/api/v1/knowledge/documents",
        data={"title": "Some Title"},
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert "rerank" in body["error"]["message"]
    assert "knowledge" in body["error"]["message"]


# ---------------------------------------------------------------------------
# 6. /health reflects the Tier 2 capability matrix
# ---------------------------------------------------------------------------


async def test_health_reports_tier2_capabilities(tier2_runtime: AsyncClient) -> None:
    resp = await tier2_runtime.get("/health")
    assert resp.status_code == 200
    body = resp.json()

    caps = body["capabilities"]
    assert caps["llm"] is True
    assert caps["embed"] is True
    assert caps["rerank"] is False

    disabled = set(body["disabled_features"])
    assert "agentic_search" in disabled
    assert "knowledge" in disabled
    assert "vector_search" not in disabled
    assert "hybrid_search" not in disabled
    assert "reflection" not in disabled
    assert "skill_extraction" not in disabled


# ---------------------------------------------------------------------------
# 7. Embed-dependent OME strategies are registered now that embed is available
# ---------------------------------------------------------------------------


async def test_ome_registers_embed_dependent_strategies(
    tier2_runtime: AsyncClient,
) -> None:
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
