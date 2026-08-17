"""Tier 3 (LLM + embed + rerank) end-to-end acceptance test.

Full regression: every endpoint Tier 1/2 rejected with a 422 now returns
200 given proper input, and ``/health`` reports every capability
available with no disabled features (multimodal is orthogonal to this
refactor and is skipped per the task brief).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from everalgo.rank.protocols import AgenticDecision
from everalgo.types import KnowledgeMemory
from httpx import AsyncClient

from everos.infra.persistence.lancedb import Episode, get_table

from .conftest import add_and_flush, cascade_progress, seed_atomic_fact_for_episode

_STUB_VECTOR = [0.1] * 1024


async def _episode_rows(owner_id: str) -> list[dict]:
    table = await get_table(Episode.TABLE_NAME, Episode)
    return await table.query().where(f"owner_id = '{owner_id}'").to_list()


# ---------------------------------------------------------------------------
# 1. Memory + every search method -> 200
# ---------------------------------------------------------------------------


async def test_add_memory_writes_real_vector(tier3_runtime: AsyncClient) -> None:
    await add_and_flush(tier3_runtime, session_id="s_tier3_add")

    rows = await _episode_rows("u_alice")
    assert rows and rows[0]["vector"] is not None


@pytest.mark.parametrize("method", ["keyword", "vector", "hybrid"])
async def test_user_search_methods_succeed(
    tier3_runtime: AsyncClient, method: str
) -> None:
    await add_and_flush(tier3_runtime, session_id=f"s_tier3_{method}")
    episode_row = (await _episode_rows("u_alice"))[0]
    await seed_atomic_fact_for_episode(episode_row, vector=_STUB_VECTOR)

    resp = await tier3_runtime.post(
        "/api/v1/memory/search",
        json={"user_id": "u_alice", "query": "hiking", "method": method},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["data"]["episodes"], f"{method} search must find the episode"


async def test_agentic_search_succeeds(
    tier3_runtime: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENTIC now clears the rerank gate (Task 12).

    The everalgo agentic retrieval algorithm (multi-query refinement,
    round-2 expansion) has its own dedicated unit coverage in
    ``tests/unit/test_memory/test_search/test_agentic.py``; this test's
    job is only to prove the HTTP/capability wiring reaches it, so
    ``aagentic_retrieve`` is patched the same way that unit suite does
    (see its ``fake_aagentic``) rather than re-driving everalgo's real
    multi-query LLM loop end-to-end.
    """

    async def fake_aagentic_retrieve(query: str, **kwargs: Any) -> tuple[list, Any]:
        return [], AgenticDecision(is_multi_round=False)

    monkeypatch.setattr(
        "everos.memory.search.agentic.aagentic_retrieve", fake_aagentic_retrieve
    )

    resp = await tier3_runtime.post(
        "/api/v1/memory/search",
        json={"user_id": "u_alice", "query": "hiking", "method": "agentic"},
    )
    assert resp.status_code == 200, resp.text


async def test_agent_hybrid_cross_encoder_lane_succeeds(
    tier3_runtime: AsyncClient,
) -> None:
    resp = await tier3_runtime.post(
        "/api/v1/memory/search",
        json={
            "agent_id": "a_bob",
            "query": "hiking",
            "method": "hybrid",
            "enable_llm_rerank": False,
        },
    )
    assert resp.status_code == 200, resp.text


# ---------------------------------------------------------------------------
# 2. Knowledge upload + search -> 200
# ---------------------------------------------------------------------------


def _make_extractor_stub() -> AsyncMock:
    memories = [
        KnowledgeMemory(
            doc_id="d_tier3stub01",
            topic_index=0,
            topic="Hiking Guide",
            summary="Overview of mountain hiking trails.",
            content="",
            depth=0,
            category_id="Outdoors",
            topic_path="Hiking Guide",
        ),
        KnowledgeMemory(
            doc_id="d_tier3stub01",
            topic_index=1,
            topic="Trail Safety",
            summary="Safety tips for mountain hikers.",
            content="Always carry water and check the weather forecast.",
            depth=1,
            parent_index=0,
            children_index=[],
            topic_path="Hiking Guide > Trail Safety",
            category_id="Outdoors",
        ),
    ]
    extractor = AsyncMock()
    extractor.aextract.return_value = memories
    return extractor


async def test_knowledge_upload_and_search_succeed(
    tier3_runtime: AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from everos.entrypoints.api.routes import knowledge as knowledge_routes

    monkeypatch.setattr(
        knowledge_routes, "_build_extractor", lambda: _make_extractor_stub()
    )
    # Multimodal parsing is orthogonal to this refactor (brief: "skip
    # unless spec demands") -- force the plain UTF-8 decode path so a
    # real everos[multimodal] install in this environment doesn't pull
    # in a genuinely unconfigured multimodal LLM for a .txt upload.
    monkeypatch.setattr("everos.component.parser.parser_available", lambda: False)

    async with cascade_progress() as wait_drained:
        resp = await tier3_runtime.post(
            "/api/v1/knowledge/documents",
            data={"title": "Hiking Guide"},
            files={
                "file": ("guide.txt", b"Mountain hiking safety guide.", "text/plain")
            },
        )
        assert resp.status_code == 201, resp.text
        await wait_drained(deadline_seconds=40.0)

    search_resp = await tier3_runtime.post(
        "/api/v1/knowledge/search", json={"query": "trail safety", "method": "keyword"}
    )
    assert search_resp.status_code == 200, search_resp.text
    assert search_resp.json()["data"]["hits"], "expected the indexed topic to be found"


# ---------------------------------------------------------------------------
# 3. /health reports full capability matrix + no disabled features
# ---------------------------------------------------------------------------


async def test_health_reports_tier3_capabilities(tier3_runtime: AsyncClient) -> None:
    resp = await tier3_runtime.get("/health")
    assert resp.status_code == 200
    body = resp.json()

    caps = body["capabilities"]
    assert caps["llm"] is True
    assert caps["embed"] is True
    assert caps["rerank"] is True

    # multimodal_llm / parser are orthogonal to this refactor (brief:
    # "skip unless spec demands") -- only assert the three capabilities
    # this suite actually configures.
    disabled = set(body["disabled_features"])
    assert disabled.isdisjoint(
        {
            "vector_search",
            "hybrid_search",
            "agentic_search",
            "reflection",
            "skill_extraction",
            "knowledge",
        }
    )

    # Cascade readiness block is present on a real app that ran the cascade
    # lifespan. This pins the wiring, which is stringly-typed on both ends
    # (lifespan provider name → lifespan_data key); a rename would silently
    # drop the block from production /health while the route's own unit tests,
    # which hand-build lifespan_data, stayed green.
    cascade = body["cascade"]
    assert cascade is not None, "cascade block missing — lifespan_data wiring broke"
    assert cascade["healthy"] is True
    assert cascade["reasons"] == []
