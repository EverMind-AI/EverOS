"""Tier 1 (LLM only) end-to-end acceptance test.

Pins the behavior every Task in the embed-soft-dependency refactor
promises for a user who has configured only an LLM: writes still work,
everything that needs embedding/rerank -- memory search included, since
multi-round retrieval embeds every sub-query -- fails fast with a 422
instead of a silent 200 or a 500, and the background strategy engine never
registers a strategy it cannot run.
"""

from __future__ import annotations

from httpx import AsyncClient

from everos.service import Application

from .conftest import add_and_flush, episode_rows

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
# 2. Memory search needs the embedder: 422 with the right feature tag
# ---------------------------------------------------------------------------


async def test_search_without_embedding_422(tier1_runtime: AsyncClient) -> None:
    await add_and_flush(tier1_runtime, session_id="s_tier1_search")

    resp = await tier1_runtime.post(
        "/api/v1/memory/search",
        json={"user_id": "u_alice", "query": "hiking", "method": "llm_multiround"},
    )
    assert resp.status_code == 422, resp.text
    body = resp.json()
    assert body["error"]["code"] == "PROVIDER_NOT_CONFIGURED"
    assert "embedding" in body["error"]["message"]
    assert "llm_multiround" in body["error"]["message"]


# ---------------------------------------------------------------------------
# 3. Knowledge endpoints all gated off
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
# 4. /health reflects the Tier 1 capability matrix
# ---------------------------------------------------------------------------


async def test_health_reports_tier1_capabilities(tier1_runtime: AsyncClient) -> None:
    resp = await tier1_runtime.get("/health")
    assert resp.status_code == 200
    body = resp.json()

    caps = body["capabilities"]
    assert "llm" not in caps
    assert caps["embed"] is False
    assert caps["rerank"] is False

    disabled = set(body["disabled_features"])
    expected_subset = {
        "memory_search",
        "reflection",
        "skill_extraction",
        "knowledge",
    }
    assert expected_subset <= disabled


# ---------------------------------------------------------------------------
# 5. OME registers every strategy in Tier 1 (body-guard, not registration-guard)
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
