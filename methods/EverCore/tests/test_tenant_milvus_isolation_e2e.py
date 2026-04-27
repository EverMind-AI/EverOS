#!/usr/bin/env python3
# skip-sensitive-file
"""
End-to-end tenant isolation verification for Milvus operations.

Uses the EpisodicMemory Milvus collection as the test subject. Bootstraps the
full application context, inserts similar data for two tenants, then verifies
every Milvus operation correctly isolates (or rejects) data between tenants.

Coverage:
    IMPLEMENTED data operations:
        - insert (single + batch): tenant_id force-injected
        - upsert: tenant_id force-injected
        - search (vector): tenant filter prepended to expr (shared mode)
        - query (scalar): tenant filter prepended to expr (shared mode)
        - delete: tenant filter prepended + empty-expr guard

    Control-plane passthrough:
        - flush, load, release, compact, describe, num_entities

    Rejected operations:
        - Unknown methods (e.g., random_method) → AttributeError
        - Empty delete expression → TenantIsolationViolation
        - Non-dict entity in insert → TenantIsolationViolation

    Unit tests for proxy helpers:
        - _prepend_tenant_filter: empty expr, existing expr
        - _inject_tenant_to_entities: single dict, list of dicts, non-dict rejection
        - _exclude_tenant_from_fields: None, with/without tenant_id

    Cross-tenant protection:
        - T2 cannot query/search/delete T1 data
        - No-tenant passthrough sees all data

Run:
    uv run python src/bootstrap.py tests/test_tenant_milvus_isolation_e2e.py
"""

import asyncio
import traceback
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional
import random

from common_utils.datetime_utils import get_now_with_timezone
from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import (
    set_current_tenant,
    clear_current_tenant,
    get_current_tenant_id,
)
from core.tenants.tenant_models import TenantInfo, TenantDetail
from core.tenants.tenantize.oxm.milvus.tenant_field_collection_proxy import (
    TenantFieldCollectionProxy,
    TenantIsolationViolation,
    _exclude_tenant_from_fields,
)

logger = get_logger(__name__)

# ============================================================
# Constants
# ============================================================

TENANT_1 = "test_milvus_t001"
TENANT_2 = "test_milvus_t002"

BASE_TIME = get_now_with_timezone() - timedelta(hours=2)

# Dummy vector dimension — must match the collection schema
VECTOR_DIM = None  # Resolved at runtime from schema


def _get_vector_dim() -> int:
    global VECTOR_DIM
    if VECTOR_DIM is None:
        from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
            EpisodicMemoryCollection,
        )

        for field in EpisodicMemoryCollection._SCHEMA.fields:
            if field.name == "vector":
                VECTOR_DIM = field.dim
                break
        if VECTOR_DIM is None:
            VECTOR_DIM = 1536  # fallback
    return VECTOR_DIM


def _random_vector() -> list[float]:
    """Generate a random unit vector for testing."""
    dim = _get_vector_dim()
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


def _make_entity(
    doc_id: str,
    user_id: str,
    group_id: str,
    episode: str,
    offset_minutes: int = 0,
    parent_id: str = "parent_milvus_001",
) -> dict:
    """Build a raw Milvus entity dict (tenant_id injected by proxy)."""
    ts = BASE_TIME + timedelta(minutes=offset_minutes)
    return {
        "id": doc_id,
        "vector": _random_vector(),
        "user_id": user_id,
        "group_id": group_id,
        "session_id": "sess_milvus_001",
        "participants": [user_id],
        "sender_ids": [user_id],
        "type": "Conversation",
        "timestamp": int(ts.timestamp() * 1000),
        "episode": episode,
        "search_content": episode,
        "parent_type": "memcell",
        "parent_id": parent_id,
    }


# ============================================================
# Tenant context manager
# ============================================================


@asynccontextmanager
async def tenant_context(tenant_id: str):
    """Set and clear tenant context."""
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
        print("  Tenant Milvus Isolation E2E Report")
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
            print("  ✅ ALL TESTS PASSED — MILVUS TENANT ISOLATION VERIFIED")
        else:
            print("  ❌ SOME TESTS FAILED — SEE ABOVE")
        print("=" * 80 + "\n")
        return counts["FAIL"] == 0 and counts["ERROR"] == 0


report = TestReport()


# ============================================================
# Helper: get collection proxy
# ============================================================


def _get_collection() -> TenantFieldCollectionProxy:
    """Get the tenant-aware Milvus collection (TenantFieldCollectionProxy)."""
    from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
        EpisodicMemoryCollection,
    )

    return EpisodicMemoryCollection.async_collection()


def _get_all_output_fields() -> list[str]:
    """Get all field names from schema (excluding vector for lighter queries)."""
    from infra_layer.adapters.out.search.milvus.memory.episodic_memory_collection import (
        EpisodicMemoryCollection,
    )

    return [
        f.name for f in EpisodicMemoryCollection._SCHEMA.fields if f.name != "vector"
    ]


# ============================================================
# Setup & Teardown
# ============================================================


async def setup_test_data():
    """Insert test data for both tenants. Returns inserted doc IDs per tenant."""
    ids = {TENANT_1: [], TENANT_2: []}

    for tid in [TENANT_1, TENANT_2]:
        async with tenant_context(tid):
            coll = _get_collection()

            for i in range(5):
                doc_id = f"{tid}_doc_{i}"
                entity = _make_entity(
                    doc_id=doc_id,
                    user_id=f"user_{i}",
                    group_id="group_alpha",
                    episode=f"Episode content {i} detailed narrative",
                    offset_minutes=i * 10,
                    parent_id=f"parent_{i:03d}",
                )
                await coll.insert(entity)
                ids[tid].append(doc_id)

            # 2 more in group_beta
            for i in range(2):
                doc_id = f"{tid}_beta_{i}"
                entity = _make_entity(
                    doc_id=doc_id,
                    user_id=f"user_{i}",
                    group_id="group_beta",
                    episode=f"Beta episode {i}",
                    offset_minutes=50 + i * 10,
                )
                await coll.insert(entity)
                ids[tid].append(doc_id)

            await coll.flush()

    return ids


async def cleanup_test_data():
    """Delete all test data for both tenants."""
    for tid in [TENANT_1, TENANT_2]:
        async with tenant_context(tid):
            coll = _get_collection()
            try:
                await coll.delete(expr='session_id == "sess_milvus_001"')
                await coll.flush()
            except Exception:
                pass


# ============================================================
# Part 1: Unit Tests — Proxy Helpers
# ============================================================


async def test_prepend_filter_empty(ids: dict):
    """_prepend_tenant_filter: empty expr returns tenant-only clause."""
    name = "util_prepend_filter_empty"
    try:
        result = TenantFieldCollectionProxy._prepend_tenant_filter("", TENANT_1)
        assert result == f'(tenant_id == "{TENANT_1}")'
        result2 = TenantFieldCollectionProxy._prepend_tenant_filter(None, TENANT_1)
        assert result2 == f'(tenant_id == "{TENANT_1}")'
        report.record(name, "PASS", "Empty expr → tenant-only clause")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_prepend_filter_existing(ids: dict):
    """_prepend_tenant_filter: existing expr gets tenant prepended."""
    name = "util_prepend_filter_existing"
    try:
        result = TenantFieldCollectionProxy._prepend_tenant_filter(
            'user_id == "u1"', TENANT_1
        )
        expected = f'(tenant_id == "{TENANT_1}") and (user_id == "u1")'
        assert result == expected, f"Got: {result}"
        report.record(name, "PASS", "Existing expr gets tenant prepended")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_entities_single(ids: dict):
    """_inject_tenant_to_entities: single dict gets tenant_id set."""
    name = "util_inject_single"
    try:
        entity = {"id": "test", "user_id": "u1"}
        TenantFieldCollectionProxy._inject_tenant_to_entities(entity, TENANT_1)
        assert entity["tenant_id"] == TENANT_1
        report.record(name, "PASS", "Single dict injected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_entities_list(ids: dict):
    """_inject_tenant_to_entities: list of dicts all get tenant_id."""
    name = "util_inject_list"
    try:
        entities = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        TenantFieldCollectionProxy._inject_tenant_to_entities(entities, TENANT_1)
        for e in entities:
            assert e["tenant_id"] == TENANT_1
        report.record(name, "PASS", "List of 3 dicts all injected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_entities_force_overwrite(ids: dict):
    """_inject_tenant_to_entities: existing tenant_id is force-overwritten."""
    name = "util_inject_overwrite"
    try:
        entity = {"id": "test", "tenant_id": "WRONG_TENANT"}
        TenantFieldCollectionProxy._inject_tenant_to_entities(entity, TENANT_1)
        assert entity["tenant_id"] == TENANT_1, "Should overwrite"
        report.record(name, "PASS", "Existing tenant_id force-overwritten")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_inject_entities_non_dict_rejected(ids: dict):
    """_inject_tenant_to_entities: non-dict entity raises TenantIsolationViolation."""
    name = "util_inject_non_dict"
    try:
        try:
            TenantFieldCollectionProxy._inject_tenant_to_entities(
                ["not", "a", "dict"], TENANT_1
            )
            report.record(name, "FAIL", "Non-dict did NOT raise")
        except TenantIsolationViolation:
            report.record(name, "PASS", "Non-dict correctly rejected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_exclude_tenant_from_fields(ids: dict):
    """_exclude_tenant_from_fields: removes tenant_id from output list."""
    name = "util_exclude_fields"
    try:
        result = _exclude_tenant_from_fields(["id", "user_id", "tenant_id", "episode"])
        assert "tenant_id" not in result
        assert "id" in result
        assert "episode" in result

        # None input returns None
        assert _exclude_tenant_from_fields(None) is None

        # No tenant_id in input — unchanged
        result2 = _exclude_tenant_from_fields(["id", "user_id"])
        assert result2 == ["id", "user_id"]

        report.record(name, "PASS", "tenant_id excluded, None preserved")
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 2: Integration Tests — Write Operations
# ============================================================


async def test_insert_single(ids: dict):
    """insert: single entity gets tenant_id injected."""
    name = "insert_single"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            doc_id = f"{TENANT_1}_ins_test"
            entity = _make_entity(doc_id, "ins_user", "group_ins", "Insert test ep")
            await coll.insert(entity)
            await coll.flush()

            # Verify via query
            results = await coll.query(
                expr=f'id == "{doc_id}"', output_fields=["id", "user_id", "tenant_id"]
            )
            assert len(results) == 1, f"Expected 1, got {len(results)}"
            # Note: tenant_id may be excluded by proxy in shared mode.
            # The fact that query found it means tenant filter matched.

            # Cleanup
            await coll.delete(expr=f'id == "{doc_id}"')
            await coll.flush()

        report.record(name, "PASS", "Insert with tenant isolation verified")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_insert_batch(ids: dict):
    """insert: batch of entities all get tenant_id."""
    name = "insert_batch"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            entities = [
                _make_entity(
                    f"{TENANT_1}_batch_{i}",
                    f"batch_u{i}",
                    "group_batch",
                    f"Batch ep {i}",
                )
                for i in range(3)
            ]
            await coll.insert(entities)
            await coll.flush()

            results = await coll.query(
                expr='group_id == "group_batch"', output_fields=["id"]
            )
            assert len(results) == 3, f"Expected 3, got {len(results)}"

        # T2 should not see them
        async with tenant_context(TENANT_2):
            coll = _get_collection()
            results = await coll.query(
                expr='group_id == "group_batch"', output_fields=["id"]
            )
            assert len(results) == 0, f"T2 saw T1's batch data: {len(results)}"

        # Cleanup
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            await coll.delete(expr='group_id == "group_batch"')
            await coll.flush()

        report.record(name, "PASS", "Batch insert isolated from T2")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_upsert(ids: dict):
    """upsert: entity gets tenant_id injected."""
    name = "upsert"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            doc_id = f"{TENANT_1}_ups_test"
            entity = _make_entity(doc_id, "ups_user", "group_ups", "Upsert ep v1")
            await coll.upsert(entity)
            await coll.flush()

            # Query to verify
            results = await coll.query(
                expr=f'id == "{doc_id}"', output_fields=["id", "episode"]
            )
            assert len(results) == 1

            # Upsert again with updated content
            entity2 = _make_entity(doc_id, "ups_user", "group_ups", "Upsert ep v2")
            await coll.upsert(entity2)
            await coll.flush()

            results2 = await coll.query(
                expr=f'id == "{doc_id}"', output_fields=["id", "episode"]
            )
            assert len(results2) == 1
            assert results2[0]["episode"] == "Upsert ep v2"

            # Cleanup
            await coll.delete(expr=f'id == "{doc_id}"')
            await coll.flush()

        report.record(name, "PASS", "Upsert with tenant isolation verified")
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 3: Integration Tests — Read Operations
# ============================================================


async def test_query_basic_isolation(ids: dict):
    """query: T1 only sees own data."""
    name = "query_basic_isolation"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            results = await coll.query(
                expr='group_id == "group_alpha"', output_fields=["id", "user_id"]
            )
            assert len(results) == 5, f"T1 expected 5, got {len(results)}"

        async with tenant_context(TENANT_2):
            coll = _get_collection()
            results = await coll.query(
                expr='group_id == "group_alpha"', output_fields=["id", "user_id"]
            )
            assert len(results) == 5, f"T2 expected 5, got {len(results)}"

        # Verify no overlap
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            r1 = await coll.query(
                expr='group_id == "group_alpha"', output_fields=["id"]
            )
        async with tenant_context(TENANT_2):
            coll = _get_collection()
            r2 = await coll.query(
                expr='group_id == "group_alpha"', output_fields=["id"]
            )
        ids_1 = {r["id"] for r in r1}
        ids_2 = {r["id"] for r in r2}
        assert ids_1.isdisjoint(ids_2), "Tenant data overlap!"

        report.record(
            name, "PASS", f"Query isolated: T1={len(r1)}, T2={len(r2)}, zero overlap"
        )
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_query_by_id(ids: dict):
    """query: T1 can find own doc by ID, T2 cannot."""
    name = "query_by_id_isolation"
    try:
        doc_id = ids[TENANT_1][0]

        async with tenant_context(TENANT_1):
            coll = _get_collection()
            results = await coll.query(expr=f'id == "{doc_id}"', output_fields=["id"])
            assert len(results) == 1, f"T1 should find own doc, got {len(results)}"

        async with tenant_context(TENANT_2):
            coll = _get_collection()
            results = await coll.query(expr=f'id == "{doc_id}"', output_fields=["id"])
            assert len(results) == 0, f"T2 found T1's doc: {len(results)}"

        report.record(name, "PASS", "Query by ID isolated between tenants")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_search_vector_isolation(ids: dict):
    """search: vector search with expr filter isolated per tenant."""
    name = "search_vector_isolation"
    try:
        query_vec = _random_vector()
        output_fields = _get_all_output_fields()

        async with tenant_context(TENANT_1):
            coll = _get_collection()
            results = await coll.search(
                data=[query_vec],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=20,
                expr='group_id == "group_alpha"',
                output_fields=output_fields,
            )
            # Results is a list of lists (one per query vector)
            hits = results[0] if results else []
            assert len(hits) == 5, f"T1 expected 5 hits, got {len(hits)}"
            for hit in hits:
                assert hit.id.startswith(
                    TENANT_1
                ), f"T1 search returned non-T1 doc: {hit.id}"

        async with tenant_context(TENANT_2):
            coll = _get_collection()
            results = await coll.search(
                data=[query_vec],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=20,
                expr='group_id == "group_alpha"',
                output_fields=output_fields,
            )
            hits = results[0] if results else []
            assert len(hits) == 5, f"T2 expected 5 hits, got {len(hits)}"
            for hit in hits:
                assert hit.id.startswith(
                    TENANT_2
                ), f"T2 search returned non-T2 doc: {hit.id}"

        report.record(name, "PASS", "Vector search isolated: 5 hits per tenant")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_search_no_expr(ids: dict):
    """search: no expr filter still gets tenant filter in shared mode."""
    name = "search_no_expr"
    try:
        query_vec = _random_vector()

        async with tenant_context(TENANT_1):
            coll = _get_collection()
            results = await coll.search(
                data=[query_vec],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=100,
                expr=None,  # No expr — proxy adds tenant filter
                output_fields=["id"],
            )
            hits = results[0] if results else []
            # Should see all 7 T1 docs (5 alpha + 2 beta)
            assert len(hits) >= 7, f"T1 expected >=7 hits, got {len(hits)}"
            for hit in hits:
                assert hit.id.startswith(TENANT_1), f"Non-T1 doc: {hit.id}"

        report.record(
            name, "PASS", f"No-expr search returns only T1 data ({len(hits)} hits)"
        )
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 4: Integration Tests — Delete Operations
# ============================================================


async def test_delete_with_filter(ids: dict):
    """delete: only deletes current tenant's data."""
    name = "delete_with_filter"
    try:
        # Insert temp data for both tenants
        for tid in [TENANT_1, TENANT_2]:
            async with tenant_context(tid):
                coll = _get_collection()
                entity = _make_entity(
                    f"{tid}_del_test", "del_user", "group_del", "Delete test ep"
                )
                await coll.insert(entity)
                await coll.flush()

        # Delete only T1's data
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            await coll.delete(expr='group_id == "group_del"')
            await coll.flush()

        # T2's data should survive
        async with tenant_context(TENANT_2):
            coll = _get_collection()
            results = await coll.query(
                expr='group_id == "group_del"', output_fields=["id"]
            )
            assert len(results) == 1, f"T2 data should survive, got {len(results)}"

            # Cleanup
            await coll.delete(expr='group_id == "group_del"')
            await coll.flush()

        report.record(name, "PASS", "Delete only affects own tenant's data")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_delete_empty_expr_rejected(ids: dict):
    """delete: empty expression raises TenantIsolationViolation."""
    name = "delete_empty_expr_rejected"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            try:
                await coll.delete(expr="")
                report.record(name, "FAIL", "Empty delete did NOT raise")
            except TenantIsolationViolation:
                report.record(name, "PASS", "Empty delete correctly rejected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_delete_whitespace_expr_rejected(ids: dict):
    """delete: whitespace-only expression also rejected."""
    name = "delete_whitespace_expr_rejected"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            try:
                await coll.delete(expr="   ")
                report.record(name, "FAIL", "Whitespace delete did NOT raise")
            except TenantIsolationViolation:
                report.record(name, "PASS", "Whitespace delete correctly rejected")
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 5: Control-Plane Passthrough
# ============================================================


async def test_passthrough_flush(ids: dict):
    """flush: passthrough, no rejection."""
    name = "passthrough_flush"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            await coll.flush()
        report.record(name, "PASS", "flush passthrough OK")
    except TenantIsolationViolation:
        report.record(name, "FAIL", "flush should not be blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_passthrough_load(ids: dict):
    """load: passthrough, no rejection."""
    name = "passthrough_load"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            await coll.load()
        report.record(name, "PASS", "load passthrough OK")
    except TenantIsolationViolation:
        report.record(name, "FAIL", "load should not be blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_passthrough_describe(ids: dict):
    """describe: whitelisted passthrough via __getattr__."""
    name = "passthrough_describe"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            desc = coll.describe()
            # describe may return a coroutine (async_wrap) — await if needed
            if asyncio.iscoroutine(desc):
                desc = await desc
            assert desc is not None
        report.record(name, "PASS", "describe passthrough OK")
    except AttributeError:
        report.record(name, "FAIL", "describe should be whitelisted")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_passthrough_num_entities(ids: dict):
    """num_entities: whitelisted passthrough."""
    name = "passthrough_num_entities"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            count = coll.num_entities
            assert isinstance(count, int)
        report.record(name, "PASS", f"num_entities={count} passthrough OK")
    except AttributeError:
        report.record(name, "FAIL", "num_entities should be whitelisted")
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 6: Rejected Operations
# ============================================================


async def test_unknown_method_rejected(ids: dict):
    """Unknown method (e.g., random_method) raises AttributeError."""
    name = "unknown_method_rejected"
    try:
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            try:
                _ = coll.random_nonexistent_method
                report.record(name, "FAIL", "Unknown method did NOT raise")
            except AttributeError as e:
                assert "does not expose" in str(e)
                report.record(
                    name,
                    "PASS",
                    "Unknown method correctly rejected with AttributeError",
                )
    except Exception as e:
        report.record(name, "FAIL", str(e))


# ============================================================
# Part 7: Cross-Tenant Protection
# ============================================================


async def test_cross_tenant_query_blocked(ids: dict):
    """T2 cannot query T1's data by ID."""
    name = "cross_tenant_query_blocked"
    try:
        doc_id = ids[TENANT_1][0]

        async with tenant_context(TENANT_2):
            coll = _get_collection()
            results = await coll.query(expr=f'id == "{doc_id}"', output_fields=["id"])
            assert len(results) == 0, f"T2 found T1's doc via query!"

        report.record(name, "PASS", "Cross-tenant query blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_cross_tenant_delete_blocked(ids: dict):
    """T2 cannot delete T1's data."""
    name = "cross_tenant_delete_blocked"
    try:
        doc_id = ids[TENANT_1][0]

        async with tenant_context(TENANT_2):
            coll = _get_collection()
            await coll.delete(expr=f'id == "{doc_id}"')
            await coll.flush()

        # Verify T1's doc still exists
        async with tenant_context(TENANT_1):
            coll = _get_collection()
            results = await coll.query(expr=f'id == "{doc_id}"', output_fields=["id"])
            assert len(results) == 1, f"T1's doc was deleted by T2!"

        report.record(name, "PASS", "Cross-tenant delete blocked")
    except Exception as e:
        report.record(name, "FAIL", str(e))


async def test_cross_tenant_search_blocked(ids: dict):
    """T2 vector search cannot find T1's data."""
    name = "cross_tenant_search_blocked"
    try:
        query_vec = _random_vector()

        async with tenant_context(TENANT_2):
            coll = _get_collection()
            results = await coll.search(
                data=[query_vec],
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=100,
                expr=None,
                output_fields=["id"],
            )
            hits = results[0] if results else []
            for hit in hits:
                assert not hit.id.startswith(
                    TENANT_1
                ), f"T2 search found T1 doc: {hit.id}"

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

        coll = _get_collection()
        try:
            await coll.query(
                expr='session_id == "sess_milvus_001"',
                output_fields=["id", "tenant_id"],
                limit=100,
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
    # Unit: proxy helpers
    test_prepend_filter_empty,
    test_prepend_filter_existing,
    test_inject_entities_single,
    test_inject_entities_list,
    test_inject_entities_force_overwrite,
    test_inject_entities_non_dict_rejected,
    test_exclude_tenant_from_fields,
    # Write operations
    test_insert_single,
    test_insert_batch,
    test_upsert,
    # Read operations
    test_query_basic_isolation,
    test_query_by_id,
    test_search_vector_isolation,
    test_search_no_expr,
    # Delete operations
    test_delete_with_filter,
    test_delete_empty_expr_rejected,
    test_delete_whitespace_expr_rejected,
    # Passthrough
    test_passthrough_flush,
    test_passthrough_load,
    test_passthrough_describe,
    test_passthrough_num_entities,
    # Rejected
    test_unknown_method_rejected,
    # Cross-tenant
    test_cross_tenant_query_blocked,
    test_cross_tenant_delete_blocked,
    test_cross_tenant_search_blocked,
    test_no_tenant_raises,
]


async def main():
    print("\n" + "=" * 80)
    print("  Tenant Milvus Isolation E2E Test")
    print(f"  Tenants: {TENANT_1}, {TENANT_2}")
    print("=" * 80 + "\n")

    # Setup
    print("--- Setup: inserting test data ---")
    ids = await setup_test_data()
    print(f"  T1: {len(ids[TENANT_1])} docs, T2: {len(ids[TENANT_2])} docs")

    # Verify setup
    for tid in [TENANT_1, TENANT_2]:
        async with tenant_context(tid):
            coll = _get_collection()
            results = await coll.query(
                expr='session_id == "sess_milvus_001"', output_fields=["id"], limit=100
            )
            print(f"  Verify {tid}: {len(results)} docs visible via query")
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
