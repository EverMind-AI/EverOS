"""Knowledge HTTP e2e — all 9 endpoints via real LLM + full HTTP stack.

Drives every knowledge API endpoint end-to-end with a **real LLM**:

    GET    /categories               → taxonomy auto-generation
    POST   /documents                → real LLM extraction → md → cascade
    GET    /documents/{doc_id}       → document detail with topics
    GET    /topics/{topic_id}        → topic content
    POST   /search (keyword)         → BM25 retrieval
    POST   /search (vector)          → ANN retrieval
    POST   /search (hybrid)          → RRF fusion
    POST   /search (include_content) → content enrichment
    PATCH  /documents/{doc_id}       → metadata update
    GET    /documents                → paginated listing
    PUT    /documents/{doc_id}       → replace (delete + recreate)
    DELETE /documents/{doc_id}       → cleanup

Uses ``httpx.AsyncClient`` against ``create_app()`` with full lifespan
(SQLite + LanceDB + Cascade + OME).

Marked ``live_llm`` + ``slow`` — requires ``EVEROS_LLM__*`` +
``EVEROS_EMBEDDING__*`` credentials in ``.env``.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

# ---------------------------------------------------------------------------
# Test document — 3 clear sections for predictable topic extraction.
# ---------------------------------------------------------------------------

_TEST_DOCUMENT = """\
# 2028 Los Angeles Olympics Budget Plan

## Venue Construction
The venue construction program covers 12 new facilities and 8 renovated \
sites across greater Los Angeles. Total venue construction budget is \
estimated at $5.3 billion, with the Intuit Dome as the centerpiece for \
basketball events. Temporary overlay structures account for $800 million.

## Transportation Infrastructure
A comprehensive transportation plan connects all venue clusters via \
dedicated Olympic lanes. The LAX-to-Downtown express shuttle operates \
24 hours during Games time. Budget allocation for transportation is \
$1.2 billion including temporary bus fleet leases.

## Security Operations
Multi-agency security coordination involves LAPD, FBI, and DHS. \
Cybersecurity operations center monitors digital threats 24/7. \
Drone detection perimeter extends 30 miles around the Olympic Village. \
Security budget is $2.1 billion.
"""

_TEST_TITLE = "2028 LA Olympics Budget Plan"
_PREFIX = "/api/v1/knowledge"


# ---------------------------------------------------------------------------
# Fixtures — reuse the shared e2e conftest (async_client + lifespan)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.live_llm
@pytest.mark.slow
async def test_knowledge_full_http_lifecycle(
    async_client: httpx.AsyncClient,
    cascade_done_poll,
) -> None:
    """Full HTTP lifecycle: create → get → search → list → delete.

    Validates every HTTP endpoint with real LLM extraction output,
    checking response shapes, status codes, and semantic correctness
    against the known input document.
    """
    client = async_client

    # ── 1. GET /categories — taxonomy auto-generated ──────────────
    resp = await client.get(f"{_PREFIX}/categories")
    assert resp.status_code == 200
    cats = resp.json()["data"]["categories"]
    assert len(cats) >= 10, f"Expected >=10 default categories, got {len(cats)}"
    cat_ids = {c["id"] for c in cats}
    assert "Sports" in cat_ids
    assert "Others" in cat_ids

    # ── 2. POST /documents — create with real LLM extraction ─────
    resp = await client.post(
        f"{_PREFIX}/documents",
        data={"title": _TEST_TITLE, "source_type": "file"},
        files={"file": ("budget.md", _TEST_DOCUMENT.encode(), "text/plain")},
    )
    assert resp.status_code == 201, f"Create failed: {resp.text}"
    create_data = resp.json()["data"]
    doc_id = create_data["doc_id"]
    assert doc_id.startswith("d_")
    assert create_data["topic_count"] >= 2, (
        f"3-section doc should produce >= 2 topics, got {create_data['topic_count']}"
    )
    assert create_data["category_id"], "LLM should assign a category"
    assert create_data["source_name"] == "budget.md"

    # ── 3. Wait for cascade to index all topics ─────────────────
    # The watcher needs time to detect new files, then the worker
    # processes them. Poll the GET endpoint until topics appear.
    expected_topics = create_data["topic_count"]
    async with asyncio.timeout(60.0):
        while True:
            resp = await client.get(f"{_PREFIX}/documents/{doc_id}")
            if resp.status_code == 200:
                detail = resp.json()["data"]
                if len(detail.get("topics", [])) >= expected_topics:
                    break
            await asyncio.sleep(1.0)
    # Extra settle for LanceDB FTS index.
    await asyncio.sleep(1.0)

    # ── 4. GET /documents/{doc_id} — document detail ─────────────
    assert detail["doc_id"] == doc_id
    assert detail["title"] == _TEST_TITLE
    assert len(detail["summary"]) > 30, "Document summary should be meaningful"
    assert len(detail["topics"]) >= 2

    # Verify topic names relate to the input document sections.
    topic_names = [t["topic_name"].lower() for t in detail["topics"]]
    found_sections = {
        "venue": any("venue" in n for n in topic_names),
        "transport": any("transport" in n for n in topic_names),
        "security": any("security" in n for n in topic_names),
    }
    assert sum(found_sections.values()) >= 2, (
        f"Expected >= 2 of 3 sections in topic names, got: {topic_names}"
    )

    # All topics have topic_path and summary.
    for t in detail["topics"]:
        assert t["topic_path"], f"topic_path empty for {t['topic_id']}"
        assert len(t["summary"]) > 10, f"summary too short for {t['topic_id']}"

    # ── 5. GET /topics/{topic_id} — topic detail with content ────
    first_topic = detail["topics"][0]
    topic_id = first_topic["topic_id"]
    resp = await client.get(f"{_PREFIX}/topics/{topic_id}")
    assert resp.status_code == 200
    topic_detail = resp.json()["data"]
    assert topic_detail["topic_id"] == topic_id
    assert topic_detail["doc_id"] == doc_id
    assert len(topic_detail["content"]) > 20, "Topic should have content body"
    assert topic_detail["category_id"] == create_data["category_id"]

    # ── 6. POST /search — keyword search ─────────────────────────
    resp = await client.post(
        f"{_PREFIX}/search",
        json={"query": "budget", "method": "keyword", "top_k": 10},
    )
    assert resp.status_code == 200
    search_data = resp.json()["data"]
    assert search_data["total"] >= 1, "Keyword search 'budget' should find hits, got 0"
    assert search_data["took_ms"] > 0

    hit = search_data["hits"][0]
    assert hit["score"] > 0
    assert hit["topic_id"]
    assert hit["document"]["doc_id"] == doc_id
    assert hit["document"]["title"] == _TEST_TITLE

    # ── 7. POST /search — include_content=true ───────────────────
    resp = await client.post(
        f"{_PREFIX}/search",
        json={
            "query": "venue construction",
            "method": "keyword",
            "top_k": 5,
            "include_content": True,
        },
    )
    assert resp.status_code == 200
    hits_with_content = resp.json()["data"]["hits"]
    if hits_with_content:
        assert hits_with_content[0]["content"], (
            "include_content=true should populate content field"
        )

    # ── 8. POST /search — vector search ─────────────────────────
    resp = await client.post(
        f"{_PREFIX}/search",
        json={"query": "Olympic venue stadium", "method": "vector", "top_k": 5},
    )
    assert resp.status_code == 200
    vector_data = resp.json()["data"]
    assert vector_data["total"] >= 1, "Vector search should find hits"
    assert vector_data["hits"][0]["retrieval_method"] == "vector"

    # ── 9. POST /search — hybrid search ──────────────────────────
    resp = await client.post(
        f"{_PREFIX}/search",
        json={"query": "security operations", "method": "hybrid", "top_k": 5},
    )
    assert resp.status_code == 200
    hybrid_data = resp.json()["data"]
    assert hybrid_data["total"] >= 1, "Hybrid search should find hits"

    # ── 10. PATCH /documents/{doc_id} — update metadata ──────────
    new_title = "Updated Olympics Budget 2028"
    resp = await client.patch(
        f"{_PREFIX}/documents/{doc_id}",
        json={"title": new_title},
    )
    assert resp.status_code == 200
    patch_data = resp.json()["data"]
    assert patch_data["doc_id"] == doc_id
    assert "title" in patch_data["updated_fields"]

    # Verify title change persisted via GET.
    resp = await client.get(f"{_PREFIX}/documents/{doc_id}")
    assert resp.json()["data"]["title"] == new_title

    # ── 11. GET /documents — paginated listing ───────────────────
    resp = await client.get(f"{_PREFIX}/documents")
    assert resp.status_code == 200
    list_data = resp.json()["data"]
    assert list_data["total"] >= 1
    assert any(d["doc_id"] == doc_id for d in list_data["documents"])
    our_doc = next(d for d in list_data["documents"] if d["doc_id"] == doc_id)
    assert our_doc["title"] == new_title
    assert our_doc["topic_count"] >= 2

    # ── 12. PUT /documents/{doc_id} — replace with new content ───
    replacement_doc = (
        "# Revised 2028 LA Olympics Plan\n\n"
        "## Athlete Village\n"
        "The athlete village will house 15,000 athletes in UCLA campus "
        "facilities. Total village budget is $1.8 billion.\n"
    )
    resp = await client.put(
        f"{_PREFIX}/documents/{doc_id}",
        data={"title": "Revised Olympics Plan"},
        files={"file": ("revised.md", replacement_doc.encode(), "text/plain")},
    )
    assert resp.status_code == 200, f"Replace failed: {resp.text}"
    replace_data = resp.json()["data"]
    assert replace_data["doc_id"] == doc_id, "PUT should preserve doc_id"
    assert replace_data["topic_count"] >= 1

    # Wait for cascade to fully cycle: old topics deleted + new topics indexed.
    # Poll until GET returns new title AND topic names from the replacement doc.
    async with asyncio.timeout(60.0):
        while True:
            resp = await client.get(f"{_PREFIX}/documents/{doc_id}")
            if resp.status_code == 200:
                d = resp.json()["data"]
                names = [t["topic_name"].lower() for t in d.get("topics", [])]
                has_new = any("village" in n or "athlete" in n for n in names)
                has_old = any("venue" in n or "security" in n for n in names)
                if d["title"] == "Revised Olympics Plan" and has_new and not has_old:
                    break
            await asyncio.sleep(1.0)

    # Verify replaced content.
    resp = await client.get(f"{_PREFIX}/documents/{doc_id}")
    replaced = resp.json()["data"]
    assert replaced["title"] == "Revised Olympics Plan"

    # ── 13. DELETE /documents/{doc_id} ────────────────────────────
    resp = await client.delete(f"{_PREFIX}/documents/{doc_id}")
    assert resp.status_code == 200
    del_data = resp.json()["data"]
    assert del_data["doc_id"] == doc_id
    assert del_data["deleted_topics"] >= 1

    # Note: cascade cleanup of SQLite/LanceDB rows is eventually
    # consistent (scanner interval + FK constraint retry). The service
    # DELETE removes md files immediately; the cascade handler catches
    # up on subsequent scan passes. Integration tests verify cascade
    # cleanup; this e2e test verifies the HTTP contract only.
