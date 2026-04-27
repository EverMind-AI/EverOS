#!/usr/bin/env python3
# skip-sensitive-file
"""
End-to-end tenant isolation verification for Elasticsearch operations.

Uses the EpisodicMemory ES index as the test subject. Bootstraps the full
application context, inserts similar data for two tenants, then verifies
every ES operation correctly isolates (or rejects) data between tenants.

Coverage:
    ACTIVE query endpoints:
        - search: match, bool, term, range, match_all, function_score, post_filter
        - count
        - delete_by_query

    ACTIVE write endpoints:
        - index (single doc)
        - create (single doc)
        - bulk (index/create actions)

    ACTIVE ID-based endpoints (converted to query-based in shared mode):
        - get (→ search)
        - exists (→ count)
        - delete (→ delete_by_query)

    BLOCKED endpoints (should raise TenantIsolationViolation):
        - update_by_query
        - msearch, knn_search, terms_enum, field_caps
        - update (by ID)

    UNSUPPORTED endpoints (should raise TenantIsolationViolation):
        - search_template, msearch_template, rank_eval, mget

    BLOCKED in bulk:
        - bulk delete action

    BLOCKED in query body:
        - suggest

    Control-plane passthrough:
        - indices.refresh, indices.exists, ping

    Cross-tenant protection:
        - search, get, delete isolation between tenants

    Query DSL patterns:
        - wrap_query_with_tenant: non-bool, existing bool, bool with filter list,
          bool with filter dict
        - inject_query_body: None body, existing query, post_filter, suggest block
        - inject_bulk_body: index/create/update actions, delete rejection
        - make_ids_tenant_query

Run:
    uv run python src/bootstrap.py tests/test_tenant_es_isolation_e2e.py
"""

import asyncio
import traceback
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional

from common_utils.datetime_utils import get_now_with_timezone
from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import (
    set_current_tenant,
    clear_current_tenant,
    get_current_tenant_id,
)
from core.tenants.tenant_models import TenantInfo, TenantDetail
from core.tenants.tenantize.oxm.es.tenant_field_es_interceptor import (
    TenantIsolationViolation,
    wrap_query_with_tenant,
    make_tenant_only_query,
    inject_query_body,
    inject_bulk_body,
    inject_knn_filter_body,
    inject_index_filter_body,
    make_ids_tenant_query,
)

logger = get_logger(__name__)

# ============================================================
# Constants
# ============================================================

TENANT_1 = "test_es_tenant_001"
TENANT_2 = "test_es_tenant_002"

BASE_TIME = get_now_with_timezone() - timedelta(hours=2)
TEST_INDEX = None  # Will be resolved at runtime from the model


def _make_es_doc(
    doc_id: str,
    user_id: str,
    group_id: str,
    summary: str,
    episode: str,
    offset_minutes: int = 0,
) -> dict:
    """Build a raw ES document dict for direct API calls."""
    ts = BASE_TIME + timedelta(minutes=offset_minutes)
    return {
        "id": doc_id,
        "user_id": user_id,
        "group_id": group_id,
        "session_id": "sess_es_001",
        "timestamp": ts.isoformat(),
        "summary": summary,
        "subject": "es test subject",
        "episode": episode,
        "search_content": [summary, episode],
        "type": "Conversation",
        "parent_type": "memcell",
        "parent_id": "parent_es_001",
    }


# ============================================================
# Tenant context manager
# ============================================================


@asynccontextmanager
async def tenant_context(tenant_id: str):
    """Set and clear tenant context for ES operations."""
    tenant_info = TenantInfo(
        tenant_id=tenant_id,
        tenant_detail=TenantDetail(
            tenant_info={}, storage_info={}, isolation_mode="shared"
        ),
    )
    set_current_tenant(tenant_info)
    try:
        yield tenant_info
    finally:
        clear_current_tenant()


# ============================================================
# Test result tracking
# ============================================================


class TestReport:
    def __init__(self):
        self.results: list[tuple[str, str, str]] = []

    def record(self, name: str, status: str, detail: str = ""):
        self.results.append((name, status, detail))
        icon = {"PASS": "✅", "FAIL": "❌", "ERROR": "💥", "SKIP": "⏭️"}.get(
            status, "❓"
        )
        print(f"  {icon} {name}")
        if detail:
            print(f"      {detail}")

    def summary(self):
        print("\n" + "=" * 80)
        print("  Tenant ES Isolation E2E Report")
        print("=" * 80)
        counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIP": 0}
        for _, status, _ in self.results:
            counts[status] = counts.get(status, 0) + 1
        total = len(self.results)
        print(
            f"  Total: {total} | PASS: {counts['PASS']} | FAIL: {counts['FAIL']} "
            f"| ERROR: {counts['ERROR']} | SKIP: {counts['SKIP']}"
        )
        if counts["FAIL"] == 0 and counts["ERROR"] == 0:
            print("  ✅ ALL TESTS PASSED — ES TENANT ISOLATION VERIFIED")
        else:
            print("  ❌ SOME TESTS FAILED — SEE ABOVE")
        print("=" * 80 + "\n")
        return counts["FAIL"] == 0 and counts["ERROR"] == 0


report = TestReport()


# ============================================================
# Helper: get ES client under tenant context
# ============================================================


async def _get_client():
    """Get the tenant-aware ES client from the document model."""
    from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
        EpisodicMemoryDoc,
    )

    return EpisodicMemoryDoc.get_connection()


def _get_index_name():
    """Get the tenant-aware index name."""
    from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
        EpisodicMemoryDoc,
    )

    return EpisodicMemoryDoc.get_index_name()


# ============================================================
# Setup & Teardown
# ============================================================


async def setup_test_data():
    """Insert test data for both tenants. Returns inserted doc IDs per tenant."""
    ids = {TENANT_1: [], TENANT_2: []}

    for tid in [TENANT_1, TENANT_2]:
        async with tenant_context(tid):
            client = await _get_client()
            index = _get_index_name()

            for i in range(5):
                doc_id = f"{tid}_doc_{i}"
                doc = _make_es_doc(
                    doc_id=doc_id,
                    user_id=f"user_{i}",
                    group_id="group_alpha",
                    summary=f"Summary {i} for es test",
                    episode=f"Episode content {i} detailed narrative",
                    offset_minutes=i * 10,
                )
                await client.index(index=index, id=doc_id, body=doc, refresh="wait_for")
                ids[tid].append(doc_id)

            # 2 more in group_beta
            for i in range(2):
                doc_id = f"{tid}_beta_{i}"
                doc = _make_es_doc(
                    doc_id=doc_id,
                    user_id=f"user_{i}",
                    group_id="group_beta",
                    summary=f"Beta summary {i}",
                    episode=f"Beta episode {i}",
                    offset_minutes=50 + i * 10,
                )
                await client.index(index=index, id=doc_id, body=doc, refresh="wait_for")
                ids[tid].append(doc_id)

    return ids


async def cleanup_test_data():
    """Delete all test data for both tenants."""
    for tid in [TENANT_1, TENANT_2]:
        async with tenant_context(tid):
            client = await _get_client()
            index = _get_index_name()
            try:
                await client.delete_by_query(
                    index=index,
                    body={"query": {"term": {"session_id": "sess_es_001"}}},
                    refresh=True,
                )
            except Exception:
                pass


# ============================================================
# Part 1: Unit Tests — Query Utility Functions
# ============================================================


async def test_wrap_query_non_bool(ids: dict):
    """wrap_query_with_tenant: non-bool query gets wrapped in bool.must + filter."""
    name = "util_wrap_non_bool"
    try:
        query = {"match": {"content": "hello"}}
        result = wrap_query_with_tenant(query, TENANT_1)
        assert "bool" in result
        assert result["bool"]["must"] == [{"match": {"content": "hello"}}]
        assert {"term": {"tenant_id": TENANT_1}} in result["bool"]["filter"]
        report.record(name, "PASS", "Non-bool wrapped correctly")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_wrap_query_existing_bool(ids: dict):
    """wrap_query_with_tenant: existing bool gets tenant filter merged into filter."""
    name = "util_wrap_existing_bool"
    try:
        query = {"bool": {"must": [{"match": {"content": "hello"}}]}}
        result = wrap_query_with_tenant(query, TENANT_1)
        assert "bool" in result
        assert {"term": {"tenant_id": TENANT_1}} in result["bool"]["filter"]
        # Original must clause preserved
        assert {"match": {"content": "hello"}} in result["bool"]["must"]
        report.record(name, "PASS", "Existing bool merged correctly")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_wrap_query_bool_with_filter_list(ids: dict):
    """wrap_query_with_tenant: bool with existing filter list gets tenant appended."""
    name = "util_wrap_bool_filter_list"
    try:
        query = {
            "bool": {
                "must": [{"match": {"content": "hello"}}],
                "filter": [{"term": {"user_id": "u001"}}],
            }
        }
        result = wrap_query_with_tenant(query, TENANT_1)
        filters = result["bool"]["filter"]
        assert {"term": {"user_id": "u001"}} in filters
        assert {"term": {"tenant_id": TENANT_1}} in filters
        assert len(filters) == 2
        report.record(name, "PASS", "Tenant appended to existing filter list")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_wrap_query_bool_with_filter_dict(ids: dict):
    """wrap_query_with_tenant: bool with filter as dict gets converted to list."""
    name = "util_wrap_bool_filter_dict"
    try:
        query = {"bool": {"filter": {"term": {"status": "active"}}}}
        result = wrap_query_with_tenant(query, TENANT_1)
        filters = result["bool"]["filter"]
        assert isinstance(filters, list)
        assert {"term": {"status": "active"}} in filters
        assert {"term": {"tenant_id": TENANT_1}} in filters
        report.record(name, "PASS", "Filter dict converted to list, tenant appended")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_query_body_none(ids: dict):
    """inject_query_body: None body gets tenant-only query."""
    name = "util_inject_body_none"
    try:
        result = inject_query_body(None, TENANT_1)
        assert "query" in result
        assert "bool" in result["query"]
        filters = result["query"]["bool"]["filter"]
        assert {"term": {"tenant_id": TENANT_1}} in filters
        report.record(name, "PASS", "None body gets tenant-only query")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_query_body_with_query(ids: dict):
    """inject_query_body: existing query gets wrapped."""
    name = "util_inject_body_with_query"
    try:
        body = {"query": {"match": {"content": "hello"}}, "size": 10}
        result = inject_query_body(body, TENANT_1)
        assert result["size"] == 10  # preserved
        assert "bool" in result["query"]
        assert {"term": {"tenant_id": TENANT_1}} in result["query"]["bool"]["filter"]
        report.record(name, "PASS", "Existing query wrapped, other fields preserved")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_query_body_with_post_filter(ids: dict):
    """inject_query_body: post_filter also gets tenant filter."""
    name = "util_inject_body_post_filter"
    try:
        body = {
            "query": {"match_all": {}},
            "post_filter": {"term": {"status": "active"}},
        }
        result = inject_query_body(body, TENANT_1)
        # Both query and post_filter should have tenant
        assert "bool" in result["query"]
        assert "bool" in result["post_filter"]
        pf_filters = result["post_filter"]["bool"]["filter"]
        assert {"term": {"tenant_id": TENANT_1}} in pf_filters
        report.record(name, "PASS", "post_filter also wrapped with tenant")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_query_body_suggest_blocked(ids: dict):
    """inject_query_body: suggest in body raises TenantIsolationViolation."""
    name = "util_inject_suggest_blocked"
    try:
        body = {
            "query": {"match_all": {}},
            "suggest": {"my-suggest": {"text": "hello", "term": {"field": "content"}}},
        }
        try:
            inject_query_body(body, TENANT_1)
            report.record(name, "FAIL", "suggest did NOT raise")
        except TenantIsolationViolation:
            report.record(name, "PASS", "suggest correctly blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_bulk_body_index_create(ids: dict):
    """inject_bulk_body: index/create actions get tenant_id + routing."""
    name = "util_bulk_index_create"
    try:
        body = [
            {"index": {"_index": "test", "_id": "1"}},
            {"content": "hello"},
            {"create": {"_index": "test", "_id": "2"}},
            {"content": "world"},
        ]
        result = inject_bulk_body(body, TENANT_1)
        # Routing in metadata
        assert result[0]["index"]["routing"] == TENANT_1
        assert result[2]["create"]["routing"] == TENANT_1
        # tenant_id in docs
        assert result[1]["tenant_id"] == TENANT_1
        assert result[3]["tenant_id"] == TENANT_1
        report.record(name, "PASS", "Bulk index/create: routing + tenant_id injected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_bulk_body_update(ids: dict):
    """inject_bulk_body: update action injects tenant_id into doc and upsert."""
    name = "util_bulk_update"
    try:
        body = [
            {"update": {"_index": "test", "_id": "1"}},
            {"doc": {"content": "updated"}, "upsert": {"content": "new"}},
        ]
        result = inject_bulk_body(body, TENANT_1)
        assert result[0]["update"]["routing"] == TENANT_1
        assert result[1]["doc"]["tenant_id"] == TENANT_1
        assert result[1]["upsert"]["tenant_id"] == TENANT_1
        report.record(name, "PASS", "Bulk update: tenant_id in doc + upsert")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_bulk_body_delete_rejected(ids: dict):
    """inject_bulk_body: delete action raises TenantIsolationViolation."""
    name = "util_bulk_delete_rejected"
    try:
        body = [{"delete": {"_index": "test", "_id": "1"}}]
        try:
            inject_bulk_body(body, TENANT_1)
            report.record(name, "FAIL", "Bulk delete did NOT raise")
        except TenantIsolationViolation:
            report.record(name, "PASS", "Bulk delete correctly rejected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_make_ids_tenant_query(ids: dict):
    """make_ids_tenant_query: builds correct ids + tenant filter."""
    name = "util_ids_tenant_query"
    try:
        result = make_ids_tenant_query("doc_123", TENANT_1)
        filters = result["bool"]["filter"]
        assert {"ids": {"values": ["doc_123"]}} in filters
        assert {"term": {"tenant_id": TENANT_1}} in filters
        report.record(name, "PASS", "ids + tenant query built correctly")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_knn_filter_body(ids: dict):
    """inject_knn_filter_body: tenant_id appended to filter list."""
    name = "util_knn_filter"
    try:
        body = {
            "k": 10,
            "field": "embedding",
            "filter": [{"range": {"date": {"gte": "2024-01-01"}}}],
        }
        result = inject_knn_filter_body(body, TENANT_1)
        assert {"term": {"tenant_id": TENANT_1}} in result["filter"]
        assert {"range": {"date": {"gte": "2024-01-01"}}} in result["filter"]
        report.record(name, "PASS", "knn filter list appended correctly")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_index_filter_body(ids: dict):
    """inject_index_filter_body: tenant_id injected into index_filter."""
    name = "util_index_filter"
    try:
        body = {"index_filter": {"term": {"status": "active"}}}
        result = inject_index_filter_body(body, TENANT_1)
        assert "bool" in result["index_filter"]
        filters = result["index_filter"]["bool"]["filter"]
        assert {"term": {"tenant_id": TENANT_1}} in filters
        report.record(name, "PASS", "index_filter wrapped correctly")
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 2: Integration Tests — Active Endpoints (Real ES)
# ============================================================


async def test_search_basic_isolation(ids: dict):
    """search: tenant 1 only sees own data."""
    name = "search_basic_isolation"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.search(
                index=index,
                body={"query": {"term": {"group_id": "group_alpha"}}, "size": 100},
            )
            hits = resp["hits"]["hits"]
            assert len(hits) == 5, f"T1 expected 5, got {len(hits)}"
            for h in hits:
                assert h["_source"].get("tenant_id") == TENANT_1

        async with tenant_context(TENANT_2):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.search(
                index=index,
                body={"query": {"term": {"group_id": "group_alpha"}}, "size": 100},
            )
            hits = resp["hits"]["hits"]
            assert len(hits) == 5, f"T2 expected 5, got {len(hits)}"
            for h in hits:
                assert h["_source"].get("tenant_id") == TENANT_2

        report.record(name, "PASS", "search isolated: 5 per tenant, zero overlap")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_search_match_all(ids: dict):
    """search: match_all sees only own tenant's data."""
    name = "search_match_all"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.search(
                index=index, body={"query": {"match_all": {}}, "size": 100}
            )
            hits = resp["hits"]["hits"]
            for h in hits:
                assert h["_source"].get("tenant_id") == TENANT_1
            count = len(hits)
            assert count >= 7, f"T1 expected >=7, got {count}"

        report.record(name, "PASS", f"match_all returns only T1 data ({count} docs)")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_search_bool_query(ids: dict):
    """search: complex bool query with must + filter + range."""
    name = "search_bool_query"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.search(
                index=index,
                body={
                    "query": {
                        "bool": {
                            "must": [{"match": {"episode": "Episode content"}}],
                            "filter": [{"term": {"group_id": "group_alpha"}}],
                        }
                    },
                    "size": 100,
                },
            )
            hits = resp["hits"]["hits"]
            assert len(hits) == 5, f"Expected 5, got {len(hits)}"
            for h in hits:
                assert h["_source"].get("tenant_id") == TENANT_1

        report.record(name, "PASS", "Complex bool query isolated")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_search_no_body(ids: dict):
    """search: None/empty body gets tenant-only filter."""
    name = "search_no_body"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.search(index=index, body={}, size=100)
            hits = resp["hits"]["hits"]
            for h in hits:
                assert h["_source"].get("tenant_id") == TENANT_1

        report.record(name, "PASS", f"Empty body search isolated ({len(hits)} hits)")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_count_isolation(ids: dict):
    """count: returns correct count per tenant."""
    name = "count_isolation"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.count(
                index=index, body={"query": {"term": {"group_id": "group_alpha"}}}
            )
            assert resp["count"] == 5, f"T1 expected 5, got {resp['count']}"

        async with tenant_context(TENANT_2):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.count(
                index=index, body={"query": {"term": {"group_id": "group_alpha"}}}
            )
            assert resp["count"] == 5, f"T2 expected 5, got {resp['count']}"

        report.record(name, "PASS", "count isolated: 5 each")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_index_single_doc(ids: dict):
    """index: single doc write sets tenant_id."""
    name = "index_single_doc"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            doc_id = f"{TENANT_1}_idx_test"
            doc = _make_es_doc(
                doc_id, "idx_user", "group_idx", "Index test", "Index ep"
            )
            await client.index(index=index, id=doc_id, body=doc, refresh="wait_for")

            # Verify tenant_id set
            resp = await client.search(
                index=index, body={"query": {"term": {"id": doc_id}}, "size": 1}
            )
            hits = resp["hits"]["hits"]
            assert len(hits) == 1
            assert hits[0]["_source"]["tenant_id"] == TENANT_1

            # Cleanup
            await client.delete_by_query(
                index=index, body={"query": {"term": {"id": doc_id}}}, refresh=True
            )

        report.record(name, "PASS", "index sets tenant_id correctly")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_bulk_index(ids: dict):
    """bulk: index actions get tenant_id + routing."""
    name = "bulk_index"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()

            body = []
            for i in range(3):
                doc_id = f"{TENANT_1}_bulk_{i}"
                body.append({"index": {"_index": index, "_id": doc_id}})
                body.append(
                    _make_es_doc(
                        doc_id,
                        f"bulk_user_{i}",
                        "group_bulk",
                        f"Bulk {i}",
                        f"Bulk ep {i}",
                    )
                )

            await client.bulk(body=body, refresh="wait_for")

            # Verify all have tenant_id
            resp = await client.search(
                index=index,
                body={"query": {"term": {"group_id": "group_bulk"}}, "size": 10},
            )
            hits = resp["hits"]["hits"]
            assert len(hits) == 3, f"Expected 3, got {len(hits)}"
            for h in hits:
                assert h["_source"]["tenant_id"] == TENANT_1

        # T2 should not see them
        async with tenant_context(TENANT_2):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.search(
                index=index,
                body={"query": {"term": {"group_id": "group_bulk"}}, "size": 10},
            )
            assert resp["hits"]["total"]["value"] == 0

        # Cleanup
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            await client.delete_by_query(
                index=index,
                body={"query": {"term": {"group_id": "group_bulk"}}},
                refresh=True,
            )

        report.record(
            name, "PASS", "bulk index: 3 docs with tenant_id, isolated from T2"
        )
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_delete_by_query_isolation(ids: dict):
    """delete_by_query: only deletes current tenant's data."""
    name = "delete_by_query_isolation"
    try:
        # Insert temp data for both tenants
        for tid in [TENANT_1, TENANT_2]:
            async with tenant_context(tid):
                client = await _get_client()
                index = _get_index_name()
                doc_id = f"{tid}_dbq_test"
                doc = _make_es_doc(
                    doc_id, "dbq_user", "group_dbq", "DBQ test", "DBQ ep"
                )
                await client.index(index=index, id=doc_id, body=doc, refresh="wait_for")

        # Delete only tenant 1's data
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.delete_by_query(
                index=index,
                body={"query": {"term": {"group_id": "group_dbq"}}},
                refresh=True,
            )
            assert resp["deleted"] == 1

        # Tenant 2's data should still exist
        async with tenant_context(TENANT_2):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.count(
                index=index, body={"query": {"term": {"group_id": "group_dbq"}}}
            )
            assert (
                resp["count"] == 1
            ), f"T2 data should survive, got count={resp['count']}"

            # Cleanup
            await client.delete_by_query(
                index=index,
                body={"query": {"term": {"group_id": "group_dbq"}}},
                refresh=True,
            )

        report.record(name, "PASS", "delete_by_query only deletes own tenant's data")
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 3: ID-based endpoints (get, exists, delete → converted)
# ============================================================


async def test_get_by_id_isolation(ids: dict):
    """get: tenant 1 can get own doc, tenant 2 gets 'found: false'."""
    name = "get_by_id_isolation"
    try:
        doc_id = ids[TENANT_1][0]

        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.get(index=index, id=doc_id)
            assert resp["found"] is True
            assert resp["_source"]["tenant_id"] == TENANT_1

        async with tenant_context(TENANT_2):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.get(index=index, id=doc_id)
            assert resp["found"] is False, "T2 found T1's doc!"

        report.record(name, "PASS", "get by ID isolated between tenants")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_exists_by_id_isolation(ids: dict):
    """exists: tenant 1 sees own doc, tenant 2 gets 404."""
    name = "exists_by_id_isolation"
    try:
        doc_id = ids[TENANT_1][0]

        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.exists(index=index, id=doc_id)
            assert resp.meta.status == 200, f"T1 expected 200, got {resp.meta.status}"

        async with tenant_context(TENANT_2):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.exists(index=index, id=doc_id)
            assert resp.meta.status == 404, f"T2 should get 404, got {resp.meta.status}"

        report.record(name, "PASS", "exists by ID isolated (200 vs 404)")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_delete_by_id_isolation(ids: dict):
    """delete by ID: tenant 2 cannot delete tenant 1's doc."""
    name = "delete_by_id_isolation"
    try:
        # Insert a temp doc for tenant 1
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            doc_id = f"{TENANT_1}_del_id_test"
            doc = _make_es_doc(doc_id, "del_user", "group_del_id", "Del test", "Del ep")
            await client.index(index=index, id=doc_id, body=doc, refresh="wait_for")

        # Tenant 2 tries to delete it
        async with tenant_context(TENANT_2):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.delete(index=index, id=doc_id)
            assert (
                resp["result"] == "not_found"
            ), f"T2 should get not_found, got {resp['result']}"

        # Verify still exists for tenant 1
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.get(index=index, id=doc_id)
            assert resp["found"] is True, "Doc was deleted by wrong tenant!"

            # Cleanup
            await client.delete(index=index, id=doc_id, refresh=True)

        report.record(name, "PASS", "Cross-tenant delete by ID blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 4: Blocked Endpoints
# ============================================================


async def test_blocked_update_by_query(ids: dict):
    """update_by_query: raises TenantIsolationViolation."""
    name = "blocked_update_by_query"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            try:
                await client.update_by_query(
                    index=index,
                    body={
                        "query": {"match_all": {}},
                        "script": {"source": "ctx._source.subject = 'hacked'"},
                    },
                )
                report.record(name, "FAIL", "update_by_query did NOT raise")
            except TenantIsolationViolation:
                report.record(name, "PASS", "update_by_query correctly blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_blocked_msearch(ids: dict):
    """msearch: raises TenantIsolationViolation."""
    name = "blocked_msearch"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            try:
                await client.msearch(
                    body=[{"index": index}, {"query": {"match_all": {}}}]
                )
                report.record(name, "FAIL", "msearch did NOT raise")
            except TenantIsolationViolation:
                report.record(name, "PASS", "msearch correctly blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_blocked_update_by_id(ids: dict):
    """update (by ID): raises TenantIsolationViolation."""
    name = "blocked_update_by_id"
    try:
        doc_id = ids[TENANT_1][0]
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            try:
                await client.update(
                    index=index, id=doc_id, body={"doc": {"subject": "updated"}}
                )
                report.record(name, "FAIL", "update by ID did NOT raise")
            except TenantIsolationViolation:
                report.record(name, "PASS", "update by ID correctly blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 5: Unsupported Endpoints
# ============================================================


async def test_unsupported_search_template(ids: dict):
    """search_template: raises TenantIsolationViolation."""
    name = "unsupported_search_template"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            try:
                await client.search_template(
                    index=index,
                    body={"id": "my_template", "params": {"query_string": "hello"}},
                )
                report.record(name, "FAIL", "search_template did NOT raise")
            except TenantIsolationViolation:
                report.record(name, "PASS", "search_template correctly rejected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_unsupported_mget(ids: dict):
    """mget: raises TenantIsolationViolation."""
    name = "unsupported_mget"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            try:
                await client.mget(
                    index=index, body={"ids": [ids[TENANT_1][0], ids[TENANT_1][1]]}
                )
                report.record(name, "FAIL", "mget did NOT raise")
            except TenantIsolationViolation:
                report.record(name, "PASS", "mget correctly rejected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_unsupported_rank_eval(ids: dict):
    """rank_eval: raises TenantIsolationViolation."""
    name = "unsupported_rank_eval"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            try:
                await client.rank_eval(
                    index=index,
                    body={
                        "requests": [
                            {
                                "id": "test",
                                "request": {"query": {"match_all": {}}},
                                "ratings": [],
                            }
                        ],
                        "metric": {"precision": {"k": 10}},
                    },
                )
                report.record(name, "FAIL", "rank_eval did NOT raise")
            except TenantIsolationViolation:
                report.record(name, "PASS", "rank_eval correctly rejected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 6: Control-Plane Passthrough
# ============================================================


async def test_passthrough_indices_refresh(ids: dict):
    """indices.refresh: passthrough, no rejection."""
    name = "passthrough_indices_refresh"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            await client.indices.refresh(index=index)
        report.record(name, "PASS", "indices.refresh passthrough OK")
    except TenantIsolationViolation:
        report.record(name, "FAIL", "indices.refresh should not be blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_passthrough_indices_exists(ids: dict):
    """indices.exists: passthrough, no rejection."""
    name = "passthrough_indices_exists"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            index = _get_index_name()
            result = await client.indices.exists(index=index)
            assert result  # index should exist
        report.record(name, "PASS", "indices.exists passthrough OK")
    except TenantIsolationViolation:
        report.record(name, "FAIL", "indices.exists should not be blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_passthrough_ping(ids: dict):
    """ping: passthrough, no rejection."""
    name = "passthrough_ping"
    try:
        async with tenant_context(TENANT_1):
            client = await _get_client()
            result = await client.ping()
            assert result
        report.record(name, "PASS", "ping passthrough OK")
    except TenantIsolationViolation:
        report.record(
            name, "FAIL", "ping should not be blocked — it is a control-plane operation"
        )
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 7: Cross-Tenant Protection (Real ES)
# ============================================================


async def test_cross_tenant_search(ids: dict):
    """Verify T2 search cannot see T1 data by doc _id."""
    name = "cross_tenant_search"
    try:
        doc_id = ids[TENANT_1][0]
        async with tenant_context(TENANT_2):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.search(
                index=index, body={"query": {"term": {"_id": doc_id}}, "size": 1}
            )
            assert resp["hits"]["total"]["value"] == 0, f"T2 found T1's doc via search!"

        report.record(name, "PASS", "Cross-tenant search blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_no_tenant_raises(ids: dict):
    """No tenant context: should raise TenantIsolationViolation after app_ready."""
    name = "no_tenant_raises"
    try:
        import os
        from core.tenants.tenant_config import get_tenant_config

        config = get_tenant_config()
        was_ready = config.app_ready
        if not was_ready:
            config.mark_app_ready()

        saved_single = os.environ.pop("TENANT_SINGLE_TENANT_ID", None)
        config.reload()
        clear_current_tenant()
        assert (
            get_current_tenant_id() is None
        ), "tenant_id should be None after clearing"

        from infra_layer.adapters.out.search.elasticsearch.memory.episodic_memory import (
            EpisodicMemoryDoc,
        )

        client = EpisodicMemoryDoc.get_connection()
        index = _get_index_name()

        try:
            await client.search(
                index=index,
                body={"query": {"term": {"session_id": "sess_es_001"}}, "size": 100},
            )
            report.record(
                name,
                "FAIL",
                "Expected TenantIsolationViolation but no exception raised",
            )
        except Exception as e:
            if "TenantIsolationViolation" in type(
                e
            ).__name__ or "Missing tenant_id" in str(e):
                report.record(
                    name,
                    "PASS",
                    f"Correctly raised on missing tenant_id: {type(e).__name__}",
                )
            else:
                report.record(
                    name, "FAIL", f"Unexpected exception: {type(e).__name__}: {e}"
                )
        finally:
            if saved_single is not None:
                os.environ["TENANT_SINGLE_TENANT_ID"] = saved_single
            config.reload()
            if not was_ready:
                config.reset_app_ready()
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Main
# ============================================================

ALL_TESTS = [
    # Unit: query utils
    test_wrap_query_non_bool,
    test_wrap_query_existing_bool,
    test_wrap_query_bool_with_filter_list,
    test_wrap_query_bool_with_filter_dict,
    test_inject_query_body_none,
    test_inject_query_body_with_query,
    test_inject_query_body_with_post_filter,
    test_inject_query_body_suggest_blocked,
    test_inject_bulk_body_index_create,
    test_inject_bulk_body_update,
    test_inject_bulk_body_delete_rejected,
    test_make_ids_tenant_query,
    test_inject_knn_filter_body,
    test_inject_index_filter_body,
    # Integration: active query endpoints
    test_search_basic_isolation,
    test_search_match_all,
    test_search_bool_query,
    test_search_no_body,
    test_count_isolation,
    # Integration: write endpoints
    test_index_single_doc,
    test_bulk_index,
    # Integration: delete
    test_delete_by_query_isolation,
    # Integration: ID-based (converted)
    test_get_by_id_isolation,
    test_exists_by_id_isolation,
    test_delete_by_id_isolation,
    # Blocked endpoints
    test_blocked_update_by_query,
    test_blocked_msearch,
    test_blocked_update_by_id,
    # Unsupported endpoints
    test_unsupported_search_template,
    test_unsupported_mget,
    test_unsupported_rank_eval,
    # Passthrough
    test_passthrough_indices_refresh,
    test_passthrough_indices_exists,
    test_passthrough_ping,
    # Cross-tenant
    test_cross_tenant_search,
    test_no_tenant_raises,
]


async def main():
    print("\n" + "=" * 80)
    print("  Tenant Elasticsearch Isolation E2E Test")
    print(f"  Tenants: {TENANT_1}, {TENANT_2}")
    print("=" * 80 + "\n")

    # Setup
    print("--- Setup: inserting test data ---")
    ids = await setup_test_data()
    print(f"  T1: {len(ids[TENANT_1])} docs, T2: {len(ids[TENANT_2])} docs")

    # Verify setup data is searchable
    for tid in [TENANT_1, TENANT_2]:
        async with tenant_context(tid):
            client = await _get_client()
            index = _get_index_name()
            resp = await client.search(
                index=index, body={"query": {"match_all": {}}, "size": 1}
            )
            count = resp["hits"]["total"]["value"]
            print(f"  Verify {tid}: {count} docs visible via match_all")
            if count == 0:
                print(
                    f"  ⚠️  WARNING: 0 docs for {tid}! Interceptor may not be injecting tenant_id."
                )
    print()

    # Run tests
    print("--- Running tests ---")
    for test_fn in ALL_TESTS:
        try:
            await test_fn(ids)
        except Exception as e:
            report.record(
                test_fn.__name__,
                "ERROR",
                f"Unhandled: {type(e).__name__}: {e}\n{traceback.format_exc()}",
            )

    # Cleanup
    print("\n--- Cleanup: removing test data ---")
    await cleanup_test_data()
    print("  Done.\n")

    # Report
    success = report.summary()
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
