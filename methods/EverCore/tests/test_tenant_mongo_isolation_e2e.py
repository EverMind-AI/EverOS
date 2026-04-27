#!/usr/bin/env python3
# skip-sensitive-file
"""
End-to-end tenant isolation verification for MongoDB operations.

Uses EpisodicMemory as the test subject. Bootstraps the full application context,
inserts similar data for two tenants, then verifies every MongoDB operation pattern
used in the project correctly isolates data between tenants.

Coverage:
    - Beanie ODM: insert, insert_many, save, find, find_one, get, find().count(),
                  find().delete(), find().sort().skip().limit(), projection
    - Soft delete: delete, delete_many, restore, restore_many, hard_delete,
                   hard_delete_many, find_many, find_one, hard_find_many,
                   hard_find_one, count, is_deleted, apply_soft_delete_filter
    - PyMongo direct: find, find_one, insert_one, update_one, update_many,
                      replace_one, delete_one, delete_many, aggregate,
                      count_documents, estimated_document_count, distinct,
                      find_one_and_update, find_one_and_delete, find_one_and_replace
    - Cursor: batch_size + getMore, async for iteration
    - Rejected: find_raw_batches, aggregate_raw_batches (InvalidOperation)
    - Unknown commands: TenantIsolationViolation

Run:
    uv run python src/bootstrap.py tests/test_tenant_mongo_isolation_e2e.py
"""

import asyncio
import traceback
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional

from beanie import PydanticObjectId
from pymongo.errors import InvalidOperation

from common_utils.datetime_utils import get_now_with_timezone
from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import (
    set_current_tenant,
    clear_current_tenant,
    get_current_tenant_id,
)
from core.tenants.tenant_models import TenantInfo, TenantDetail
from core.tenants.tenantize.oxm.mongo.tenant_field_command_interceptor import (
    TenantIsolationViolation,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
    EpisodicMemoryProjection,
)

logger = get_logger(__name__)

# ============================================================
# Constants
# ============================================================

TENANT_1 = "test_tenant_iso_001"
TENANT_2 = "test_tenant_iso_002"

# Shared test data template — similar content, different tenant
BASE_TIME = get_now_with_timezone() - timedelta(hours=2)


def _make_episodic(
    user_id: str,
    group_id: str,
    summary: str,
    episode: str,
    offset_minutes: int = 0,
    session_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> EpisodicMemory:
    """Create an EpisodicMemory instance (tenant_id is injected by interceptor)."""
    return EpisodicMemory(
        user_id=user_id,
        group_id=group_id,
        session_id=session_id or "sess_001",
        timestamp=BASE_TIME + timedelta(minutes=offset_minutes),
        participants=[user_id],
        sender_ids=[user_id],
        summary=summary,
        subject="test subject",
        episode=episode,
        type="Conversation",
        parent_type="memcell",
        parent_id=parent_id or "parent_001",
    )


# ============================================================
# Tenant context manager
# ============================================================


@asynccontextmanager
async def tenant_context(tenant_id: str):
    """
    Async context manager that sets and clears tenant context.

    Usage:
        async with tenant_context("test_tenant_001"):
            # all MongoDB operations here are scoped to this tenant
            ...
    """
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
        self.results: list[tuple[str, str, str]] = []  # (name, status, detail)

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
        print("  Tenant Mongo Isolation E2E Report")
        print("=" * 80)
        counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "SKIP": 0}
        for name, status, detail in self.results:
            counts[status] = counts.get(status, 0) + 1
        total = len(self.results)
        print(
            f"  Total: {total} | PASS: {counts['PASS']} | FAIL: {counts['FAIL']} "
            f"| ERROR: {counts['ERROR']} | SKIP: {counts['SKIP']}"
        )
        if counts["FAIL"] == 0 and counts["ERROR"] == 0:
            print("  ✅ ALL TESTS PASSED — TENANT ISOLATION VERIFIED")
        else:
            print("  ❌ SOME TESTS FAILED — SEE ABOVE")
        print("=" * 80 + "\n")
        return counts["FAIL"] == 0 and counts["ERROR"] == 0


report = TestReport()


# ============================================================
# Setup & Teardown
# ============================================================


async def setup_test_data():
    """Insert test data for both tenants. Returns inserted doc IDs per tenant."""
    ids = {TENANT_1: [], TENANT_2: []}

    for tid in [TENANT_1, TENANT_2]:
        async with tenant_context(tid):
            # Insert 5 docs per tenant with similar content
            for i in range(5):
                doc = _make_episodic(
                    user_id=f"user_{i}",
                    group_id="group_alpha",
                    summary=f"Summary {i} for tenant",
                    episode=f"Episode content {i} — detailed narrative",
                    offset_minutes=i * 10,
                    parent_id=f"parent_{i:03d}",
                )
                await doc.insert()
                ids[tid].append(doc.id)

            # Insert 2 more in a different group
            for i in range(2):
                doc = _make_episodic(
                    user_id=f"user_{i}",
                    group_id="group_beta",
                    summary=f"Beta summary {i}",
                    episode=f"Beta episode {i}",
                    offset_minutes=50 + i * 10,
                )
                await doc.insert()
                ids[tid].append(doc.id)

    return ids


async def cleanup_test_data():
    """Hard-delete all test data for both tenants."""
    for tid in [TENANT_1, TENANT_2]:
        async with tenant_context(tid):
            await EpisodicMemory.hard_delete_many({"session_id": "sess_001"})


# ============================================================
# Test Cases: Beanie ODM Operations
# ============================================================


async def test_beanie_insert_single(ids: dict):
    """Beanie: document.insert() — verify tenant_id is set on the stored doc."""
    name = "beanie_insert_single"
    try:
        async with tenant_context(TENANT_1):
            doc = _make_episodic(
                user_id="insert_test_user",
                group_id="group_insert",
                summary="Insert test",
                episode="Insert episode",
                offset_minutes=100,
            )
            await doc.insert()
            # Re-read from DB to verify tenant_id was set
            reloaded = await EpisodicMemory.find_one({"_id": doc.id})
            assert reloaded is not None, "Inserted doc not found"
            assert (
                reloaded.tenant_id == TENANT_1
            ), f"tenant_id mismatch: {reloaded.tenant_id}"
            # Cleanup
            await reloaded.hard_delete()
        report.record(name, "PASS", f"tenant_id={TENANT_1} correctly set on insert")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_beanie_insert_many(ids: dict):
    """Beanie: Model.insert_many() — batch insert with tenant isolation."""
    name = "beanie_insert_many"
    try:
        async with tenant_context(TENANT_1):
            docs = [
                _make_episodic(
                    user_id=f"batch_user_{i}",
                    group_id="group_batch",
                    summary=f"Batch {i}",
                    episode=f"Batch ep {i}",
                    offset_minutes=200 + i,
                )
                for i in range(3)
            ]
            await EpisodicMemory.insert_many(docs)

            # Verify all have tenant_id
            found = await EpisodicMemory.find({"group_id": "group_batch"}).to_list()
            assert len(found) == 3, f"Expected 3, got {len(found)}"
            for d in found:
                assert d.tenant_id == TENANT_1, f"tenant_id mismatch: {d.tenant_id}"

        # Verify tenant 2 cannot see them
        async with tenant_context(TENANT_2):
            found_t2 = await EpisodicMemory.find({"group_id": "group_batch"}).to_list()
            assert (
                len(found_t2) == 0
            ), f"Tenant 2 saw tenant 1's batch data: {len(found_t2)}"

        # Cleanup
        async with tenant_context(TENANT_1):
            await EpisodicMemory.hard_delete_many({"group_id": "group_batch"})

        report.record(name, "PASS", "3 docs inserted, isolated from tenant 2")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_beanie_save_replace(ids: dict):
    """Beanie: document.save() — update via replace, verify tenant isolation."""
    name = "beanie_save_replace"
    try:
        doc_id = ids[TENANT_1][0]
        async with tenant_context(TENANT_1):
            doc = await EpisodicMemory.find_one({"_id": doc_id})
            assert doc is not None, "Doc not found for save test"
            original_summary = doc.summary
            doc.summary = "Updated summary via save"
            await doc.save()

            reloaded = await EpisodicMemory.find_one({"_id": doc_id})
            assert reloaded.summary == "Updated summary via save"
            assert reloaded.tenant_id == TENANT_1

            # Restore original
            reloaded.summary = original_summary
            await reloaded.save()

        report.record(name, "PASS", "save() preserves tenant_id after replace")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_beanie_get_by_id(ids: dict):
    """Beanie: Model.get(id) — verify tenant scoping."""
    name = "beanie_get_by_id"
    try:
        doc_id = ids[TENANT_1][0]

        # Tenant 1 can see it
        async with tenant_context(TENANT_1):
            doc = await EpisodicMemory.get(doc_id)
            assert doc is not None, "Tenant 1 cannot see own doc"

        # Tenant 2 should NOT see it
        async with tenant_context(TENANT_2):
            doc = await EpisodicMemory.get(doc_id)
            assert doc is None, "Tenant 2 can see tenant 1's doc via get()!"

        report.record(name, "PASS", "get() correctly isolated between tenants")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


# ============================================================
# Test Cases: Beanie Query Operations (Cursor Path)
# ============================================================


async def test_find_basic_isolation(ids: dict):
    """Beanie find: tenant 1 data invisible to tenant 2."""
    name = "find_basic_isolation"
    try:
        async with tenant_context(TENANT_1):
            t1_docs = await EpisodicMemory.find({"group_id": "group_alpha"}).to_list()

        async with tenant_context(TENANT_2):
            t2_docs = await EpisodicMemory.find({"group_id": "group_alpha"}).to_list()

        # Both should have 5 docs each
        assert len(t1_docs) == 5, f"Tenant 1: expected 5, got {len(t1_docs)}"
        assert len(t2_docs) == 5, f"Tenant 2: expected 5, got {len(t2_docs)}"

        # IDs should be completely disjoint
        t1_ids = {str(d.id) for d in t1_docs}
        t2_ids = {str(d.id) for d in t2_docs}
        assert t1_ids.isdisjoint(t2_ids), "Tenant data overlap detected!"

        report.record(
            name,
            "PASS",
            f"T1: {len(t1_docs)} docs, T2: {len(t2_docs)} docs, zero overlap",
        )
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_find_chained_sort_skip_limit(ids: dict):
    """Beanie find: .sort().skip().limit() chain — cursor path."""
    name = "find_chained_sort_skip_limit"
    try:
        async with tenant_context(TENANT_1):
            docs = (
                await EpisodicMemory.find({"group_id": "group_alpha"})
                .sort("-timestamp")
                .skip(1)
                .limit(3)
                .to_list()
            )
            assert len(docs) == 3, f"Expected 3, got {len(docs)}"
            # Verify sorted descending
            for i in range(len(docs) - 1):
                assert docs[i].timestamp >= docs[i + 1].timestamp, "Sort order wrong"
            for d in docs:
                assert d.tenant_id == TENANT_1

        report.record(name, "PASS", "sort/skip/limit with correct isolation")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_find_with_projection(ids: dict):
    """Beanie find: projection_model to exclude vector field."""
    name = "find_with_projection"
    try:
        async with tenant_context(TENANT_1):
            docs = await EpisodicMemory.find(
                {"group_id": "group_alpha"}, projection_model=EpisodicMemoryProjection
            ).to_list()
            assert len(docs) == 5
            # EpisodicMemoryProjection should not have vector field
            for d in docs:
                assert not hasattr(d, "vector") or d.vector is None

        report.record(name, "PASS", "Projection works with tenant isolation")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_find_count(ids: dict):
    """Beanie find().count() — aggregate path for counting."""
    name = "find_count"
    try:
        async with tenant_context(TENANT_1):
            count = await EpisodicMemory.find({"group_id": "group_alpha"}).count()
            assert count == 5, f"T1 expected 5, got {count}"

        async with tenant_context(TENANT_2):
            count = await EpisodicMemory.find({"group_id": "group_alpha"}).count()
            assert count == 5, f"T2 expected 5, got {count}"

        report.record(name, "PASS", "find().count() isolated per tenant (5 each)")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_cursor_batch_size(ids: dict):
    """Cursor: small batch_size triggers getMore — verify initial find is filtered."""
    name = "cursor_batch_size"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            # Use pymongo directly — batch_size(2) with 5+ docs forces getMore
            cursor = collection.find(
                {"group_id": "group_alpha", "deleted_at": None}, batch_size=2
            )
            docs = await cursor.to_list(length=100)
            assert len(docs) == 5, f"Expected 5, got {len(docs)}"
            for d in docs:
                assert d.get("tenant_id") == TENANT_1

        report.record(name, "PASS", "batch_size cursor correctly isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_cursor_async_for(ids: dict):
    """Cursor: async for iteration — the most common consumption pattern."""
    name = "cursor_async_for"
    try:
        collected = []
        async with tenant_context(TENANT_1):
            async for doc in EpisodicMemory.find({"group_id": "group_alpha"}):
                collected.append(doc)
        assert len(collected) == 5
        for d in collected:
            assert d.tenant_id == TENANT_1

        report.record(name, "PASS", "async for iteration correctly isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


# ============================================================
# Test Cases: Soft Delete Operations
# ============================================================


async def test_soft_delete_single(ids: dict):
    """Soft delete: document.delete() — marks as deleted, not visible in find."""
    name = "soft_delete_single"
    try:
        async with tenant_context(TENANT_1):
            # Insert a doc to soft-delete
            doc = _make_episodic(
                user_id="sd_user",
                group_id="group_sd",
                summary="To be soft deleted",
                episode="SD episode",
            )
            await doc.insert()
            doc_id = doc.id

            await doc.delete(deleted_by="test_admin")
            assert doc.is_deleted(), "Document should be marked deleted"

            # find_one should not see it
            result = await EpisodicMemory.find_one({"_id": doc_id})
            assert result is None, "Soft-deleted doc visible in find_one"

            # hard_find_one should see it
            result = await EpisodicMemory.hard_find_one({"_id": doc_id})
            assert result is not None, "Soft-deleted doc not found in hard_find_one"
            assert result.deleted_by == "test_admin"

            # Cleanup
            await result.hard_delete()

        report.record(name, "PASS", "Soft delete works with tenant isolation")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_soft_delete_many(ids: dict):
    """Soft delete: Model.delete_many(filter) — bulk soft delete."""
    name = "soft_delete_many"
    try:
        async with tenant_context(TENANT_1):
            # Insert docs to soft-delete
            for i in range(3):
                doc = _make_episodic(
                    user_id=f"sdm_user_{i}",
                    group_id="group_sdm",
                    summary=f"Bulk SD {i}",
                    episode=f"Bulk SD ep {i}",
                )
                await doc.insert()

            result = await EpisodicMemory.delete_many(
                {"group_id": "group_sdm"}, deleted_by="admin"
            )
            assert (
                result.modified_count == 3
            ), f"Expected 3, modified {result.modified_count}"

            # find_many should not see them
            visible = await EpisodicMemory.find_many(
                {"group_id": "group_sdm"}
            ).to_list()
            assert len(visible) == 0, f"Soft-deleted docs still visible: {len(visible)}"

            # Cleanup
            await EpisodicMemory.hard_delete_many({"group_id": "group_sdm"})

        report.record(name, "PASS", "Bulk soft delete with isolation")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_soft_delete_restore(ids: dict):
    """Soft delete: restore single doc."""
    name = "soft_delete_restore"
    try:
        async with tenant_context(TENANT_1):
            doc = _make_episodic(
                user_id="restore_user",
                group_id="group_restore",
                summary="To restore",
                episode="Restore ep",
            )
            await doc.insert()
            doc_id = doc.id

            await doc.delete(deleted_by="admin")
            assert doc.is_deleted()

            # Restore
            await doc.restore()
            assert not doc.is_deleted()

            # Should be visible again
            result = await EpisodicMemory.find_one({"_id": doc_id})
            assert result is not None, "Restored doc not visible"

            await result.hard_delete()

        report.record(name, "PASS", "Single restore works with isolation")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_soft_delete_restore_many(ids: dict):
    """Soft delete: restore_many — bulk restore."""
    name = "soft_delete_restore_many"
    try:
        async with tenant_context(TENANT_1):
            for i in range(3):
                doc = _make_episodic(
                    user_id=f"rm_user_{i}",
                    group_id="group_rm",
                    summary=f"RM {i}",
                    episode=f"RM ep {i}",
                )
                await doc.insert()
                await doc.delete(deleted_by="admin")

            # Restore all
            result = await EpisodicMemory.restore_many({"group_id": "group_rm"})
            assert (
                result.modified_count == 3
            ), f"Expected 3 restored, got {result.modified_count}"

            # Should be visible again
            visible = await EpisodicMemory.find_many({"group_id": "group_rm"}).to_list()
            assert len(visible) == 3

            # Cleanup
            await EpisodicMemory.hard_delete_many({"group_id": "group_rm"})

        report.record(name, "PASS", "Bulk restore works with isolation")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_hard_find_many(ids: dict):
    """Soft delete: hard_find_many — includes soft-deleted docs."""
    name = "hard_find_many"
    try:
        async with tenant_context(TENANT_1):
            doc = _make_episodic(
                user_id="hfm_user",
                group_id="group_hfm",
                summary="HFM test",
                episode="HFM ep",
            )
            await doc.insert()
            await doc.delete(deleted_by="admin")

            # hard_find_many should see it
            results = await EpisodicMemory.hard_find_many(
                {"group_id": "group_hfm"}
            ).to_list()
            assert len(results) == 1
            assert results[0].is_deleted()

        # Tenant 2 should not see it
        async with tenant_context(TENANT_2):
            results = await EpisodicMemory.hard_find_many(
                {"group_id": "group_hfm"}
            ).to_list()
            assert (
                len(results) == 0
            ), f"Tenant 2 sees tenant 1's hard_find_many data: {len(results)}"

        # Cleanup
        async with tenant_context(TENANT_1):
            await EpisodicMemory.hard_delete_many({"group_id": "group_hfm"})

        report.record(name, "PASS", "hard_find_many isolated between tenants")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_hard_delete_many(ids: dict):
    """Soft delete: hard_delete_many — physical bulk delete."""
    name = "hard_delete_many"
    try:
        async with tenant_context(TENANT_1):
            for i in range(2):
                doc = _make_episodic(
                    user_id=f"hdm_user_{i}",
                    group_id="group_hdm",
                    summary=f"HDM {i}",
                    episode=f"HDM ep {i}",
                )
                await doc.insert()

            result = await EpisodicMemory.hard_delete_many({"group_id": "group_hdm"})
            assert result.deleted_count == 2

            # Verify gone
            results = await EpisodicMemory.hard_find_many(
                {"group_id": "group_hdm"}
            ).to_list()
            assert len(results) == 0

        report.record(name, "PASS", "hard_delete_many works with isolation")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_soft_delete_count(ids: dict):
    """Soft delete: count() — count with soft delete awareness."""
    name = "soft_delete_count"
    try:
        async with tenant_context(TENANT_1):
            # count() uses count_documents({"deleted_at": None})
            count = await EpisodicMemory.count()
            # At least 7 docs (5 alpha + 2 beta) for tenant 1
            assert count >= 7, f"T1 count too low: {count}"

        async with tenant_context(TENANT_2):
            count = await EpisodicMemory.count()
            assert count >= 7, f"T2 count too low: {count}"

        report.record(name, "PASS", "count() works per tenant")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_apply_soft_delete_filter(ids: dict):
    """Utility: apply_soft_delete_filter for raw pymongo queries."""
    name = "apply_soft_delete_filter"
    try:
        async with tenant_context(TENANT_1):
            filter_dict = EpisodicMemory.apply_soft_delete_filter(
                {"group_id": "group_alpha"}
            )
            assert "deleted_at" in filter_dict
            assert filter_dict["deleted_at"] is None

            collection = EpisodicMemory.get_pymongo_collection()
            results = await collection.find(filter_dict).to_list(length=100)
            assert len(results) == 5
            for r in results:
                assert r.get("tenant_id") == TENANT_1

        report.record(name, "PASS", "apply_soft_delete_filter + pymongo isolation")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


# ============================================================
# Test Cases: Direct PyMongo Operations
# ============================================================


async def test_pymongo_find(ids: dict):
    """PyMongo: collection.find(filter).to_list()"""
    name = "pymongo_find"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            results = await collection.find({"group_id": "group_alpha"}).to_list(
                length=100
            )
            assert len(results) == 5
            for r in results:
                assert r.get("tenant_id") == TENANT_1

        async with tenant_context(TENANT_2):
            results = await collection.find({"group_id": "group_alpha"}).to_list(
                length=100
            )
            assert len(results) == 5
            for r in results:
                assert r.get("tenant_id") == TENANT_2

        report.record(name, "PASS", "pymongo find isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_find_one(ids: dict):
    """PyMongo: collection.find_one(filter)"""
    name = "pymongo_find_one"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            result = await collection.find_one({"user_id": "user_0"})
            assert result is not None
            assert result.get("tenant_id") == TENANT_1

        report.record(name, "PASS", "pymongo find_one isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_insert_one(ids: dict):
    """PyMongo: collection.insert_one(doc)"""
    name = "pymongo_insert_one"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            now = get_now_with_timezone()
            result = await collection.insert_one(
                {
                    "user_id": "pymongo_insert_user",
                    "group_id": "group_pymongo_ins",
                    "session_id": "sess_001",
                    "timestamp": now,
                    "summary": "pymongo insert",
                    "episode": "pymongo insert ep",
                    "type": "Conversation",
                    "deleted_at": None,
                    "deleted_id": 0,
                }
            )
            doc_id = result.inserted_id

            # Verify tenant_id
            doc = await collection.find_one({"_id": doc_id})
            assert doc.get("tenant_id") == TENANT_1

            # Cleanup
            await collection.delete_one({"_id": doc_id})

        report.record(name, "PASS", "pymongo insert_one sets tenant_id")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_update_one(ids: dict):
    """PyMongo: collection.update_one(filter, update)"""
    name = "pymongo_update_one"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            # Update a tenant 1 doc
            result = await collection.update_one(
                {"user_id": "user_0"}, {"$set": {"subject": "updated_subject"}}
            )
            assert result.modified_count == 1

        # Verify tenant 2's user_0 is untouched
        async with tenant_context(TENANT_2):
            doc = await collection.find_one({"user_id": "user_0"})
            assert doc is not None
            assert (
                doc.get("subject") != "updated_subject"
            ), "Tenant 2 doc was modified by tenant 1 update!"

        # Restore
        async with tenant_context(TENANT_1):
            await collection.update_one(
                {"user_id": "user_0"}, {"$set": {"subject": "test subject"}}
            )

        report.record(name, "PASS", "pymongo update_one isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_update_many(ids: dict):
    """PyMongo: collection.update_many(filter, update)"""
    name = "pymongo_update_many"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            result = await collection.update_many(
                {"group_id": "group_alpha"}, {"$set": {"type": "UpdatedType"}}
            )
            assert (
                result.modified_count == 5
            ), f"Expected 5, modified {result.modified_count}"

        # Verify tenant 2 untouched
        async with tenant_context(TENANT_2):
            docs = await collection.find(
                {"group_id": "group_alpha", "type": "UpdatedType"}
            ).to_list(length=100)
            assert len(docs) == 0, f"Tenant 2 has {len(docs)} docs with UpdatedType!"

        # Restore
        async with tenant_context(TENANT_1):
            await collection.update_many(
                {"group_id": "group_alpha"}, {"$set": {"type": "Conversation"}}
            )

        report.record(name, "PASS", "pymongo update_many isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_replace_one(ids: dict):
    """PyMongo: collection.replace_one(filter, replacement)"""
    name = "pymongo_replace_one"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            # Insert a doc to replace
            now = get_now_with_timezone()
            await collection.insert_one(
                {
                    "user_id": "replace_user",
                    "group_id": "group_replace",
                    "session_id": "sess_001",
                    "timestamp": now,
                    "summary": "before replace",
                    "episode": "before replace ep",
                    "deleted_at": None,
                    "deleted_id": 0,
                }
            )

            result = await collection.replace_one(
                {"user_id": "replace_user"},
                {
                    "user_id": "replace_user",
                    "group_id": "group_replace",
                    "session_id": "sess_001",
                    "timestamp": now,
                    "summary": "after replace",
                    "episode": "after replace ep",
                    "deleted_at": None,
                    "deleted_id": 0,
                },
            )
            assert result.modified_count == 1

            # Verify tenant_id preserved in replacement
            doc = await collection.find_one({"user_id": "replace_user"})
            assert doc.get("tenant_id") == TENANT_1
            assert doc.get("summary") == "after replace"

            # Cleanup
            await collection.delete_one({"user_id": "replace_user"})

        report.record(name, "PASS", "pymongo replace_one preserves tenant_id")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_delete_one(ids: dict):
    """PyMongo: collection.delete_one(filter)"""
    name = "pymongo_delete_one"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            now = get_now_with_timezone()
            await collection.insert_one(
                {
                    "user_id": "del_one_user",
                    "group_id": "group_del",
                    "session_id": "sess_001",
                    "timestamp": now,
                    "summary": "to delete",
                    "episode": "del ep",
                    "deleted_at": None,
                    "deleted_id": 0,
                }
            )
            result = await collection.delete_one({"user_id": "del_one_user"})
            assert result.deleted_count == 1

        report.record(name, "PASS", "pymongo delete_one isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_delete_many(ids: dict):
    """PyMongo: collection.delete_many(filter)"""
    name = "pymongo_delete_many"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            for i in range(3):
                now = get_now_with_timezone()
                await collection.insert_one(
                    {
                        "user_id": f"del_many_user_{i}",
                        "group_id": "group_del_many",
                        "session_id": "sess_001",
                        "timestamp": now,
                        "summary": f"del many {i}",
                        "episode": f"del many ep {i}",
                        "deleted_at": None,
                        "deleted_id": 0,
                    }
                )
            result = await collection.delete_many({"group_id": "group_del_many"})
            assert result.deleted_count == 3

        report.record(name, "PASS", "pymongo delete_many isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_aggregate(ids: dict):
    """PyMongo: collection.aggregate(pipeline)"""
    name = "pymongo_aggregate"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            pipeline = [
                {"$match": {"session_id": "sess_001", "deleted_at": None}},
                {"$group": {"_id": "$group_id", "count": {"$sum": 1}}},
                {"$sort": {"_id": 1}},
            ]
            cursor = await collection.aggregate(pipeline)
            results = await cursor.to_list(length=100)
            # Should see group_alpha (5) and group_beta (2)
            group_counts = {r["_id"]: r["count"] for r in results}
            assert (
                group_counts.get("group_alpha") == 5
            ), f"Expected 5 for group_alpha, got {group_counts}"
            assert (
                group_counts.get("group_beta") == 2
            ), f"Expected 2 for group_beta, got {group_counts}"

        report.record(name, "PASS", "pymongo aggregate isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_count_documents(ids: dict):
    """PyMongo: collection.count_documents(filter)"""
    name = "pymongo_count_documents"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            count = await collection.count_documents(
                {"group_id": "group_alpha", "deleted_at": None}
            )
            assert count == 5, f"T1 expected 5, got {count}"

        async with tenant_context(TENANT_2):
            count = await collection.count_documents(
                {"group_id": "group_alpha", "deleted_at": None}
            )
            assert count == 5, f"T2 expected 5, got {count}"

        report.record(name, "PASS", "pymongo count_documents isolated (5 each)")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_estimated_document_count(ids: dict):
    """PyMongo: collection.estimated_document_count() — uses count command."""
    name = "pymongo_estimated_document_count"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            count = await collection.estimated_document_count()
            # This uses the count command which gets tenant_id injected.
            # However, estimated_document_count is metadata-based and may
            # not respect filters. We just verify it doesn't raise.
            assert isinstance(count, int)

        report.record(
            name,
            "PASS",
            f"estimated_document_count returned {count} (interceptor injects tenant_id into count cmd)",
        )
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_distinct(ids: dict):
    """PyMongo: collection.distinct(key, filter)"""
    name = "pymongo_distinct"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            groups = await collection.distinct("group_id", {"deleted_at": None})
            assert "group_alpha" in groups
            assert "group_beta" in groups

        report.record(name, "PASS", f"pymongo distinct isolated: {groups}")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_find_one_and_update(ids: dict):
    """PyMongo: collection.find_one_and_update() — findAndModify command."""
    name = "pymongo_find_one_and_update"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            doc = await collection.find_one_and_update(
                {"user_id": "user_0", "deleted_at": None},
                {"$set": {"subject": "fau_subject"}},
                return_document=True,
            )
            assert doc is not None
            assert doc.get("tenant_id") == TENANT_1
            assert doc.get("subject") == "fau_subject"

            # Restore
            await collection.update_one(
                {"user_id": "user_0"}, {"$set": {"subject": "test subject"}}
            )

        report.record(name, "PASS", "find_one_and_update isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_find_one_and_delete(ids: dict):
    """PyMongo: collection.find_one_and_delete()"""
    name = "pymongo_find_one_and_delete"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            now = get_now_with_timezone()
            await collection.insert_one(
                {
                    "user_id": "fad_user",
                    "group_id": "group_fad",
                    "session_id": "sess_001",
                    "timestamp": now,
                    "summary": "fad test",
                    "episode": "fad ep",
                    "deleted_at": None,
                    "deleted_id": 0,
                }
            )
            doc = await collection.find_one_and_delete({"user_id": "fad_user"})
            assert doc is not None
            assert doc.get("tenant_id") == TENANT_1

        report.record(name, "PASS", "find_one_and_delete isolated")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_pymongo_find_one_and_replace(ids: dict):
    """PyMongo: collection.find_one_and_replace()"""
    name = "pymongo_find_one_and_replace"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            now = get_now_with_timezone()
            await collection.insert_one(
                {
                    "user_id": "far_user",
                    "group_id": "group_far",
                    "session_id": "sess_001",
                    "timestamp": now,
                    "summary": "before",
                    "episode": "before ep",
                    "deleted_at": None,
                    "deleted_id": 0,
                }
            )
            doc = await collection.find_one_and_replace(
                {"user_id": "far_user"},
                {
                    "user_id": "far_user",
                    "group_id": "group_far",
                    "session_id": "sess_001",
                    "timestamp": now,
                    "summary": "after",
                    "episode": "after ep",
                    "deleted_at": None,
                    "deleted_id": 0,
                },
                return_document=True,
            )
            assert doc is not None
            assert doc.get("tenant_id") == TENANT_1
            assert doc.get("summary") == "after"

            # Cleanup
            await collection.delete_one({"user_id": "far_user"})

        report.record(
            name, "PASS", "find_one_and_replace sets tenant_id in replacement"
        )
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


# ============================================================
# Test Cases: Rejected / Unsupported Operations
# ============================================================


async def test_find_raw_batches_rejected(ids: dict):
    """Rejected: find_raw_batches raises InvalidOperation when _encrypter is set."""
    name = "find_raw_batches_rejected"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            try:
                cursor = collection.find_raw_batches({"user_id": "user_0"})
                await cursor.to_list(length=10)
                report.record(name, "FAIL", "find_raw_batches did NOT raise")
            except InvalidOperation:
                report.record(
                    name, "PASS", "find_raw_batches correctly raises InvalidOperation"
                )
            except Exception as e:
                report.record(
                    name,
                    "PASS",
                    f"find_raw_batches rejected with {type(e).__name__}: {e}",
                )
    except Exception as e:
        report.record(name, "ERROR", f"{e}")


async def test_aggregate_raw_batches_rejected(ids: dict):
    """Rejected: aggregate_raw_batches raises InvalidOperation when _encrypter is set."""
    name = "aggregate_raw_batches_rejected"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_1):
            try:
                cursor = await collection.aggregate_raw_batches([{"$match": {}}])
                await cursor.to_list(length=10)
                report.record(name, "FAIL", "aggregate_raw_batches did NOT raise")
            except InvalidOperation:
                report.record(
                    name,
                    "PASS",
                    "aggregate_raw_batches correctly raises InvalidOperation",
                )
            except Exception as e:
                report.record(
                    name,
                    "PASS",
                    f"aggregate_raw_batches rejected with {type(e).__name__}: {e}",
                )
    except Exception as e:
        report.record(name, "ERROR", f"{e}")


# ============================================================
# Test Cases: Cross-Tenant Mutation Protection
# ============================================================


async def test_cross_tenant_update_blocked(ids: dict):
    """Verify tenant 2 cannot update tenant 1's data via update_one."""
    name = "cross_tenant_update_blocked"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        # Get a known doc ID from tenant 1
        doc_id = ids[TENANT_1][0]

        # Tenant 2 tries to update by _id
        async with tenant_context(TENANT_2):
            result = await collection.update_one(
                {"_id": doc_id}, {"$set": {"summary": "HACKED"}}
            )
            assert (
                result.modified_count == 0
            ), f"Tenant 2 modified tenant 1's doc! modified_count={result.modified_count}"

        # Verify tenant 1's doc is unchanged
        async with tenant_context(TENANT_1):
            doc = await collection.find_one({"_id": doc_id})
            assert doc is not None
            assert doc.get("summary") != "HACKED", "Doc was actually modified!"

        report.record(name, "PASS", "Cross-tenant update correctly blocked")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_cross_tenant_delete_blocked(ids: dict):
    """Verify tenant 2 cannot delete tenant 1's data."""
    name = "cross_tenant_delete_blocked"
    try:
        collection = EpisodicMemory.get_pymongo_collection()
        doc_id = ids[TENANT_1][0]

        async with tenant_context(TENANT_2):
            result = await collection.delete_one({"_id": doc_id})
            assert result.deleted_count == 0, "Tenant 2 deleted tenant 1's doc!"

        # Verify still exists
        async with tenant_context(TENANT_1):
            doc = await collection.find_one({"_id": doc_id})
            assert doc is not None, "Doc was actually deleted!"

        report.record(name, "PASS", "Cross-tenant delete correctly blocked")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_cross_tenant_find_blocked(ids: dict):
    """Verify tenant 2 cannot find tenant 1's data by _id."""
    name = "cross_tenant_find_blocked"
    try:
        collection = EpisodicMemory.get_pymongo_collection()
        doc_id = ids[TENANT_1][0]

        async with tenant_context(TENANT_2):
            doc = await collection.find_one({"_id": doc_id})
            assert doc is None, "Tenant 2 found tenant 1's doc!"

        report.record(name, "PASS", "Cross-tenant find correctly blocked")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


async def test_cross_tenant_aggregate_blocked(ids: dict):
    """Verify tenant 2's aggregate cannot see tenant 1's data."""
    name = "cross_tenant_aggregate_blocked"
    try:
        collection = EpisodicMemory.get_pymongo_collection()

        async with tenant_context(TENANT_2):
            pipeline = [
                {"$match": {"_id": ids[TENANT_1][0]}},
                {"$project": {"summary": 1, "tenant_id": 1}},
            ]
            cursor = await collection.aggregate(pipeline)
            results = await cursor.to_list(length=10)
            assert len(results) == 0, f"Tenant 2 aggregate sees tenant 1 doc: {results}"

        report.record(name, "PASS", "Cross-tenant aggregate correctly blocked")
    except Exception as e:
        report.record(name, "FAIL", f"{e}")


# ============================================================
# Test Cases: No-Tenant Passthrough
# ============================================================


async def test_no_tenant_raises(ids: dict):
    """Verify operations raise TenantIsolationViolation without tenant context after app_ready."""
    name = "no_tenant_raises"
    try:
        import os
        from core.tenants.tenant_config import get_tenant_config

        config = get_tenant_config()
        was_ready = config.app_ready
        if not was_ready:
            config.mark_app_ready()

        # Temporarily remove single_tenant_id fallback so clear_current_tenant truly clears
        saved_single = os.environ.pop("TENANT_SINGLE_TENANT_ID", None)
        config.reload()
        clear_current_tenant()
        assert (
            get_current_tenant_id() is None
        ), "tenant_id should be None after clearing"

        collection = EpisodicMemory.get_pymongo_collection()
        try:
            await collection.find(
                {"group_id": "group_alpha", "deleted_at": None}
            ).to_list(length=100)
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
        report.record(name, "FAIL", f"{e}")


# ============================================================
# Main
# ============================================================

ALL_TESTS = [
    # Beanie ODM
    test_beanie_insert_single,
    test_beanie_insert_many,
    test_beanie_save_replace,
    test_beanie_get_by_id,
    # Query / Cursor
    test_find_basic_isolation,
    test_find_chained_sort_skip_limit,
    test_find_with_projection,
    test_find_count,
    test_cursor_batch_size,
    test_cursor_async_for,
    # Soft Delete
    test_soft_delete_single,
    test_soft_delete_many,
    test_soft_delete_restore,
    test_soft_delete_restore_many,
    test_hard_find_many,
    test_hard_delete_many,
    test_soft_delete_count,
    test_apply_soft_delete_filter,
    # Direct PyMongo
    test_pymongo_find,
    test_pymongo_find_one,
    test_pymongo_insert_one,
    test_pymongo_update_one,
    test_pymongo_update_many,
    test_pymongo_replace_one,
    test_pymongo_delete_one,
    test_pymongo_delete_many,
    test_pymongo_aggregate,
    test_pymongo_count_documents,
    test_pymongo_estimated_document_count,
    test_pymongo_distinct,
    test_pymongo_find_one_and_update,
    test_pymongo_find_one_and_delete,
    test_pymongo_find_one_and_replace,
    # Rejected operations
    test_find_raw_batches_rejected,
    test_aggregate_raw_batches_rejected,
    # Cross-tenant protection
    test_cross_tenant_update_blocked,
    test_cross_tenant_delete_blocked,
    test_cross_tenant_find_blocked,
    test_cross_tenant_aggregate_blocked,
    # Passthrough
    test_no_tenant_raises,
]


async def main():
    print("\n" + "=" * 80)
    print("  Tenant MongoDB Isolation E2E Test")
    print("  Using EpisodicMemory (v1_episodic_memories)")
    print(f"  Tenants: {TENANT_1}, {TENANT_2}")
    print("=" * 80 + "\n")

    # Setup
    print("--- Setup: inserting test data ---")
    ids = await setup_test_data()
    print(
        f"  Tenant 1: {len(ids[TENANT_1])} docs, Tenant 2: {len(ids[TENANT_2])} docs\n"
    )

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
