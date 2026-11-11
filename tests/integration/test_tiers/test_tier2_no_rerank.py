"""Tier 2 (LLM + embed, no rerank) end-to-end acceptance test.

Pins the behavior promised for a user who has configured LLM + embed
but not rerank: writes get real vectors, memory search (multi-round
retrieval, which needs no rerank) works, knowledge stays gated on rerank,
and the embed-dependent OME strategies (reflection / clustering / skill
extraction) are now registered.
"""

from __future__ import annotations

from httpx import AsyncClient

from everos.service import Application

from .conftest import add_and_flush, episode_rows

# ---------------------------------------------------------------------------
# 1. Memory add + multi-round search succeed
# ---------------------------------------------------------------------------


async def test_add_memory_writes_real_vector(tier2_runtime: AsyncClient) -> None:
    await add_and_flush(tier2_runtime, session_id="s_tier2_add")

    rows = await episode_rows("u_alice")
    assert rows, "expected cascade to index the new episode into LanceDB"
    assert rows[0]["vector"] is not None, "Tier 2 (embed available) must embed"
    assert len(rows[0]["vector"]) == 1024


async def test_user_search_succeeds_without_rerank(tier2_runtime: AsyncClient) -> None:
    await add_and_flush(tier2_runtime, session_id="s_tier2_search")

    resp = await tier2_runtime.post(
        "/api/v1/memory/search",
        json={"user_id": "u_alice", "query": "hiking", "method": "llm_multiround"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["episodes"], "multi-round search must find it"


# ---------------------------------------------------------------------------
# 2. Knowledge still 422 (needs rerank; embed alone isn't enough)
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
# 3. /health reflects the Tier 2 capability matrix
# ---------------------------------------------------------------------------


async def test_health_reports_tier2_capabilities(tier2_runtime: AsyncClient) -> None:
    resp = await tier2_runtime.get("/health")
    assert resp.status_code == 200
    body = resp.json()

    caps = body["capabilities"]
    assert "llm" not in caps
    assert caps["embed"] is True
    assert caps["rerank"] is False

    disabled = set(body["disabled_features"])
    assert "knowledge" in disabled
    assert "memory_search" not in disabled
    assert "reflection" not in disabled
    assert "skill_extraction" not in disabled


# ---------------------------------------------------------------------------
# 4. Embed-dependent OME strategies are registered now that embed is available
# ---------------------------------------------------------------------------


async def test_ome_registers_embed_dependent_strategies(
    tier2_runtime: AsyncClient,
    tier_application: Application,
) -> None:

    engine = tier_application._engine
    registered_names = {meta.name for meta in engine._registry.all()}

    from everos.memory.strategies import STRATEGY_SPECS

    all_strategy_names = {spec.name for spec in STRATEGY_SPECS}
    assert all_strategy_names <= registered_names
