"""Tier 2 (LLM + embed) end-to-end acceptance test: the full deployment.

Pins the behavior promised to a user who has configured an LLM and an
embedding model -- every provider the chat memory uses: writes get real
vectors, memory search (multi-round retrieval) succeeds with nothing
degraded, both deletion routes take their episodes out of the index,
``/health`` reports the embedder available and no feature disabled but the
multimodal upload this suite does not configure, and every OME strategy is
registered.
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


async def test_user_search_succeeds(tier2_runtime: AsyncClient) -> None:
    await add_and_flush(tier2_runtime, session_id="s_tier2_search")

    resp = await tier2_runtime.post(
        "/api/v1/memory/search",
        json={"user_id": "u_alice", "query": "hiking", "method": "llm_multiround"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["episodes"], "multi-round search must find it"
    assert resp.json()["data"]["degraded"] == []

    # /health counts the round the search asked of the decider. The fake client
    # has no raw call, so the startup probe is skipped (reachable null).
    decider = (await tier2_runtime.get("/health")).json()["decider"]
    assert decider["rounds"] >= 1
    assert decider["failed_rounds"] == 0
    assert decider["reachable"] is None
    assert decider["model"]


# ---------------------------------------------------------------------------
# 2. Deletions take their index rows
# ---------------------------------------------------------------------------


async def test_deletions_take_their_index_rows(tier2_runtime: AsyncClient) -> None:
    """Both deletion routes -- forgets in 2.0 -- take the episodes out of the
    index before they answer, on either backend (the Milvus job runs this with
    ``index=milvus``). The entries stay in markdown, marked, so a later
    projection may write a row back: never as a current one."""
    # One sender each: a second write into the same md file does not move the
    # cascade ledger that add_and_flush waits on.
    owners = {"s_tier2_del_session": "u_del_a", "s_tier2_del_id": "u_del_b"}
    owners["s_tier2_keep"] = "u_del_c"
    for sid, owner in owners.items():
        await add_and_flush(tier2_runtime, session_id=sid, sender_id=owner)

    async def rows() -> dict[str, list[str]]:
        return {
            o: [r["id"] for r in await episode_rows(o) if r["deprecated_by"] is None]
            for o in owners.values()
        }

    before = await rows()
    assert all(before.values()), before

    resp = await tier2_runtime.post(
        "/api/v2/memory/delete", json={"session_id": "s_tier2_del_session"}
    )
    assert resp.status_code == 200, resp.text
    resp = await tier2_runtime.post(
        "/api/v2/memory/delete_by_ids",
        json={"memory_type": "episode", "memory_ids": before["u_del_b"]},
    )
    assert resp.status_code == 200, resp.text

    assert await rows() == {"u_del_a": [], "u_del_b": [], "u_del_c": before["u_del_c"]}


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

    # multimodal_llm / parser are orthogonal to the tiers: this suite
    # configures no multimodal model, so the multimodal upload is the only
    # feature that may read disabled.
    disabled = set(body["disabled_features"])
    assert "vector_recall" not in disabled
    assert "memory_search" not in disabled
    assert disabled <= {"multimodal_upload"}

    # Cascade readiness block is present on a real app that ran the cascade
    # lifespan. This pins the wiring, which is stringly-typed on both ends
    # (lifespan provider name → lifespan_data key); a rename would silently
    # drop the block from production /health while the route's own unit tests,
    # which hand-build lifespan_data, stayed green.
    cascade = body["cascade"]
    assert cascade is not None, "cascade block missing — lifespan_data wiring broke"
    assert cascade["healthy"] is True
    assert cascade["reasons"] == []


# ---------------------------------------------------------------------------
# 4. Every OME strategy is registered, the embed-dependent ones included
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
