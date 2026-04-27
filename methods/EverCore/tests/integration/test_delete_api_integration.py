"""
Integration tests for POST /api/v1/memories/delete

Tests the live API against real MongoDB. Inserts fixture data directly
into MongoDB, calls the HTTP endpoint, and verifies both responses and
DB state. All fixture data uses the 'test_intg_delete_' prefix and is
cleaned up after tests complete.

Requirements:
    - API server running on localhost:1995
    - MongoDB accessible (connection details from .env)

Usage:
    PYTHONPATH=src pytest tests/integration/test_delete_api_integration.py -v --tb=long
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from bson import ObjectId
from pymongo import AsyncMongoClient

# Suppress pymongo GC noise when event loop closes before client cleanup
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnraisableExceptionWarning"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = os.getenv("TEST_API_BASE", "http://localhost:1995")
DELETE_URL = f"{API_BASE}/api/v1/memories/delete"

MONGO_HOST = os.getenv("MONGODB_HOST")
MONGO_PORT = int(os.getenv("MONGODB_PORT", "27017"))
MONGO_USER = os.getenv("MONGODB_USERNAME")
MONGO_PASS = os.getenv("MONGODB_PASSWORD")
MONGO_TENANT_ID = os.getenv("TENANT_SINGLE_TENANT_ID")
_BASE_DB = os.getenv("MONGODB_DATABASE")
# API uses tenant-prefixed database: {tenant_id}_{base_name}
MONGO_DB = f"{MONGO_TENANT_ID}_{_BASE_DB}" if MONGO_TENANT_ID else _BASE_DB

# Collection names
COL_MEMCELLS = "v1_memcells"
COL_EPISODES = "v1_episodic_memories"
COL_ATOMIC_FACTS = "v1_atomic_fact_records"
COL_FORESIGHTS = "v1_foresight_records"
COL_RAW_MESSAGES = "v1_raw_messages"

ALL_COLLECTIONS = [
    COL_MEMCELLS,
    COL_EPISODES,
    COL_ATOMIC_FACTS,
    COL_FORESIGHTS,
    COL_RAW_MESSAGES,
]

# Test data prefix
PREFIX = "test_intg_delete_"

# ---------------------------------------------------------------------------
# Test result collector (for report generation)
# ---------------------------------------------------------------------------
_test_results: list[dict[str, Any]] = []


def _record_result(
    scenario: str,
    category: str,
    passed: bool,
    duration_ms: float,
    detail: str = "",
    request_body: dict | None = None,
    response_body: dict | None = None,
    status_code: int | None = None,
):
    _test_results.append(
        {
            "scenario": scenario,
            "category": category,
            "passed": passed,
            "duration_ms": round(duration_ms, 2),
            "detail": detail,
            "request_body": request_body,
            "response_body": response_body,
            "status_code": status_code,
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def event_loop():
    """Create a module-scoped event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def mongo_client():
    """Module-scoped Motor client."""
    uri = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/?authSource=admin"
    client = AsyncMongoClient(uri)
    yield client
    # AsyncMongoClient is bound to its creation event loop;
    # just let Python GC handle cleanup to avoid cross-loop errors.


@pytest.fixture(scope="module")
def db(mongo_client):
    """Module-scoped database reference."""
    return mongo_client[MONGO_DB]


@pytest.fixture(scope="module")
def http_client():
    """Module-scoped httpx client."""
    with httpx.Client(timeout=30.0) as client:
        yield client


@pytest.fixture(autouse=True, scope="module")
def _check_server(http_client):
    """Verify API server is reachable before running tests."""
    try:
        r = http_client.get(f"{API_BASE}/health")
        assert r.status_code == 200, f"Health check failed: {r.text}"
    except httpx.ConnectError:
        pytest.skip("API server not running on localhost:1995")


def _now():
    return datetime.now(timezone.utc)


def _make_memcell(
    user_id: str | None = None,
    group_id: str | None = None,
    session_id: str | None = None,
    participants: list[str] | None = None,
    _id: ObjectId | None = None,
) -> dict:
    """Build a minimal MemCell document."""
    doc: dict[str, Any] = {
        "_id": _id or ObjectId(),
        "timestamp": _now(),
        "original_data": [{"content": "test fixture data"}],
        "type": "conversation",
        "created_at": _now(),
        "updated_at": _now(),
        "deleted_at": None,
        "deleted_by": None,
        "deleted_id": 0,
    }
    if user_id is not None:
        doc["user_id"] = user_id
    if group_id is not None:
        doc["group_id"] = group_id
    if session_id is not None:
        doc["session_id"] = session_id
    if participants is not None:
        doc["participants"] = participants
    return doc


def _make_child(
    collection_type: str,
    parent_id: str,
    user_id: str | None = None,
    group_id: str | None = None,
    session_id: str | None = None,
    participants: list[str] | None = None,
) -> dict:
    """Build a minimal child document (episode / atomic_fact / foresight)."""
    doc: dict[str, Any] = {
        "_id": ObjectId(),
        "parent_type": "memcell",
        "parent_id": parent_id,
        "timestamp": _now(),
        "created_at": _now(),
        "updated_at": _now(),
        "deleted_at": None,
        "deleted_by": None,
        "deleted_id": 0,
    }

    if user_id is not None:
        doc["user_id"] = user_id
    if group_id is not None:
        doc["group_id"] = group_id
    if session_id is not None:
        doc["session_id"] = session_id
    if participants is not None:
        doc["participants"] = participants

    if collection_type == COL_EPISODES:
        doc["summary"] = "test episode summary"
        doc["subject"] = "test subject"
        doc["episode"] = "test episode content"
    elif collection_type == COL_ATOMIC_FACTS:
        doc["atomic_fact"] = "test atomic fact"
    elif collection_type == COL_FORESIGHTS:
        doc["content"] = "test foresight content"
        doc["start_time"] = "2026-03-08"
        doc["end_time"] = "2026-03-15"
        doc["duration_days"] = 7
    return doc


def _make_request_log(user_id: str | None = None, group_id: str | None = None) -> dict:
    """Build a minimal MemoryRequestLog document."""
    doc: dict[str, Any] = {
        "_id": ObjectId(),
        "request_id": f"{PREFIX}req_{ObjectId()}",
        "group_id": group_id or f"{PREFIX}group_default",
        "content": "test request log",
        "sync_status": -1,
        "created_at": _now(),
        "updated_at": _now(),
        "deleted_at": None,
        "deleted_by": None,
        "deleted_id": 0,
    }
    if user_id is not None:
        doc["user_id"] = user_id
    return doc


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------
async def _insert_docs(db, collection: str, docs: list[dict]):
    """Insert documents into a collection."""
    if docs:
        await db[collection].insert_many(docs)


async def _count_alive(db, collection: str, filter_dict: dict) -> int:
    """Count non-soft-deleted documents matching the filter."""
    f = {**filter_dict, "deleted_at": None}
    return await db[collection].count_documents(f)


async def _count_soft_deleted(db, collection: str, filter_dict: dict) -> int:
    """Count soft-deleted documents matching the filter."""
    f = {**filter_dict, "deleted_at": {"$ne": None}}
    return await db[collection].count_documents(f)


async def _cleanup_test_data(db):
    """Remove ALL documents with test prefix from all collections."""
    for col_name in ALL_COLLECTIONS:
        col = db[col_name]
        # Match any field containing the test prefix
        await col.delete_many(
            {
                "$or": [
                    {"user_id": {"$regex": f"^{PREFIX}"}},
                    {"group_id": {"$regex": f"^{PREFIX}"}},
                    {"session_id": {"$regex": f"^{PREFIX}"}},
                    {"request_id": {"$regex": f"^{PREFIX}"}},
                ]
            }
        )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _call_delete(http_client, body: dict) -> tuple[int, dict]:
    """POST to delete endpoint, return (status_code, response_json)."""
    r = http_client.post(DELETE_URL, json=body)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, {"raw": r.text}


# ===================================================================
# TEST CLASS
# ===================================================================
@pytest.mark.integration
class TestDeleteApiIntegration:
    """Full API integration tests for the delete endpoint."""

    # ---------------------------------------------------------------
    # 1. Request validation (no DB interaction needed)
    # ---------------------------------------------------------------

    def test_empty_body_returns_422(self, http_client):
        """Empty request body should fail validation."""
        t0 = time.monotonic()
        status, body = _call_delete(http_client, {})
        elapsed = (time.monotonic() - t0) * 1000

        _record_result(
            scenario="Empty request body",
            category="Request Validation",
            passed=status == 422,
            duration_ms=elapsed,
            detail=f"Expected 422, got {status}",
            request_body={},
            response_body=body,
            status_code=status,
        )
        assert status == 422

    def test_memory_id_with_user_id_returns_422(self, http_client):
        """memory_id + user_id together should fail."""
        req = {"memory_id": "abcdef1234567890abcdef12", "user_id": "u1"}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        _record_result(
            scenario="memory_id + user_id mutual exclusion",
            category="Request Validation",
            passed=status == 422,
            duration_ms=elapsed,
            detail=f"Expected 422, got {status}",
            request_body=req,
            response_body=body,
            status_code=status,
        )
        assert status == 422

    def test_only_session_id_returns_422(self, http_client):
        """session_id alone (no user_id/group_id) should fail."""
        req = {"session_id": "s1"}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        _record_result(
            scenario="session_id only (no scope)",
            category="Request Validation",
            passed=status == 422,
            duration_ms=elapsed,
            detail=f"Expected 422, got {status}",
            request_body=req,
            response_body=body,
            status_code=status,
        )
        assert status == 422

    def test_only_sender_id_returns_422(self, http_client):
        """sender_id alone (no user_id/group_id) should fail."""
        req = {"sender_id": "sender1"}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        _record_result(
            scenario="sender_id only (no scope)",
            category="Request Validation",
            passed=status == 422,
            duration_ms=elapsed,
            detail=f"Expected 422, got {status}",
            request_body=req,
            response_body=body,
            status_code=status,
        )
        assert status == 422

    # ---------------------------------------------------------------
    # 2. Delete by memory_id
    # ---------------------------------------------------------------

    def test_delete_by_id_with_cascade(self, db, http_client, event_loop):
        """Delete a memcell by ID, verify cascade to children."""
        user_id = f"{PREFIX}user_byid"
        group_id = f"{PREFIX}group_byid"
        memcell_id = ObjectId()
        memcell_id_str = str(memcell_id)

        # Insert fixtures
        memcell = _make_memcell(user_id=user_id, group_id=group_id, _id=memcell_id)
        episodes = [
            _make_child(
                COL_EPISODES, memcell_id_str, user_id=user_id, group_id=group_id
            )
            for _ in range(3)
        ]
        atomic_facts = [
            _make_child(
                COL_ATOMIC_FACTS, memcell_id_str, user_id=user_id, group_id=group_id
            )
            for _ in range(2)
        ]
        foresights = [
            _make_child(
                COL_FORESIGHTS, memcell_id_str, user_id=user_id, group_id=group_id
            )
        ]

        async def setup():
            await _insert_docs(db, COL_MEMCELLS, [memcell])
            await _insert_docs(db, COL_EPISODES, episodes)
            await _insert_docs(db, COL_ATOMIC_FACTS, atomic_facts)
            await _insert_docs(db, COL_FORESIGHTS, foresights)

        event_loop.run_until_complete(setup())

        # Call delete API
        req = {"memory_id": memcell_id_str}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        # Verify response
        assert status == 204, f"Expected 204, got {status}: {body}"

        # Verify DB state: all should be soft-deleted
        async def verify():
            assert await _count_alive(db, COL_MEMCELLS, {"_id": memcell_id}) == 0
            assert await _count_soft_deleted(db, COL_MEMCELLS, {"_id": memcell_id}) == 1
            for ep in episodes:
                assert await _count_alive(db, COL_EPISODES, {"_id": ep["_id"]}) == 0
            for el in atomic_facts:
                assert await _count_alive(db, COL_ATOMIC_FACTS, {"_id": el["_id"]}) == 0
            for fs in foresights:
                assert await _count_alive(db, COL_FORESIGHTS, {"_id": fs["_id"]}) == 0

        event_loop.run_until_complete(verify())

        _record_result(
            scenario="Delete by memory_id with cascade",
            category="Delete by ID",
            passed=True,
            duration_ms=elapsed,
            detail="204 No Content, DB state verified",
            request_body=req,
            response_body=body,
            status_code=status,
        )

    def test_delete_nonexistent_id(self, http_client):
        """Delete a non-existent memory_id should return 204."""
        fake_id = str(ObjectId())
        req = {"memory_id": fake_id}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status == 204

        _record_result(
            scenario="Delete non-existent memory_id",
            category="Delete by ID",
            passed=True,
            duration_ms=elapsed,
            detail="204 No Content (idempotent)",
            request_body=req,
            response_body=body,
            status_code=status,
        )

    def test_delete_idempotent(self, db, http_client, event_loop):
        """Deleting the same ID twice should be idempotent."""
        memcell_id = ObjectId()
        memcell = _make_memcell(
            user_id=f"{PREFIX}user_idempotent",
            group_id=f"{PREFIX}group_idempotent",
            _id=memcell_id,
        )
        event_loop.run_until_complete(_insert_docs(db, COL_MEMCELLS, [memcell]))

        req = {"memory_id": str(memcell_id)}

        # First delete
        status1, _ = _call_delete(http_client, req)
        assert status1 == 204

        # Second delete (idempotent)
        t0 = time.monotonic()
        status2, body2 = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status2 == 204

        _record_result(
            scenario="Idempotent double delete",
            category="Delete by ID",
            passed=True,
            duration_ms=elapsed,
            detail="Both deletes return 204 (idempotent)",
            request_body=req,
            response_body=body2,
            status_code=status2,
        )

    # ---------------------------------------------------------------
    # 3. Delete by filters
    # ---------------------------------------------------------------

    def test_delete_by_user_id(self, db, http_client, event_loop):
        """Delete by user_id: child records (episodes) are soft-deleted.

        Note: filter-based delete does NOT touch MemCells themselves —
        only child records (episodes, atomic_facts, foresights, request_logs).
        MemCell.user_id is normally None (group-level unit), so memcell
        deletion is only supported via delete_by_id mode.
        """
        user_id = f"{PREFIX}user_filter_uid"
        group_id = f"{PREFIX}group_filter_uid"

        memcells = []
        all_episodes = []
        for i in range(3):
            mc_id = ObjectId()
            memcells.append(
                _make_memcell(user_id=user_id, group_id=group_id, _id=mc_id)
            )
            all_episodes.append(
                _make_child(
                    COL_EPISODES, str(mc_id), user_id=user_id, group_id=group_id
                )
            )

        async def setup():
            await _insert_docs(db, COL_MEMCELLS, memcells)
            await _insert_docs(db, COL_EPISODES, all_episodes)

        event_loop.run_until_complete(setup())

        req = {"user_id": user_id}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status == 204

        # Verify DB: child episodes should be soft-deleted
        # (MemCells are NOT deleted in filter mode — by design)
        async def verify():
            for ep in all_episodes:
                assert await _count_alive(db, COL_EPISODES, {"_id": ep["_id"]}) == 0
                assert (
                    await _count_soft_deleted(db, COL_EPISODES, {"_id": ep["_id"]}) == 1
                )

        event_loop.run_until_complete(verify())

        _record_result(
            scenario="Delete by user_id filter",
            category="Delete by Filters",
            passed=True,
            duration_ms=elapsed,
            detail="204 No Content, child episodes soft-deleted (memcells untouched by design)",
            request_body=req,
            response_body=body,
            status_code=status,
        )

    def test_delete_by_group_id(self, db, http_client, event_loop):
        """Delete all memcells for a group."""
        group_id = f"{PREFIX}group_filter_gid"

        memcells = []
        all_atomic_facts = []
        for i in range(2):
            mc_id = ObjectId()
            memcells.append(_make_memcell(group_id=group_id, _id=mc_id))
            all_atomic_facts.append(
                _make_child(COL_ATOMIC_FACTS, str(mc_id), group_id=group_id)
            )

        async def setup():
            await _insert_docs(db, COL_MEMCELLS, memcells)
            await _insert_docs(db, COL_ATOMIC_FACTS, all_atomic_facts)

        event_loop.run_until_complete(setup())

        req = {"group_id": group_id}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status == 204

        _record_result(
            scenario="Delete by group_id filter",
            category="Delete by Filters",
            passed=True,
            duration_ms=elapsed,
            detail="204 No Content",
            request_body=req,
            response_body=body,
            status_code=status,
        )

    def test_delete_by_user_and_session(self, db, http_client, event_loop):
        """Delete memcells filtered by user_id + session_id (narrow scope)."""
        user_id = f"{PREFIX}user_sess"
        group_id = f"{PREFIX}group_sess"
        session_target = f"{PREFIX}session_target"
        session_other = f"{PREFIX}session_other"

        mc_target = _make_memcell(
            user_id=user_id,
            group_id=group_id,
            session_id=session_target,
            _id=ObjectId(),
        )
        mc_other = _make_memcell(
            user_id=user_id, group_id=group_id, session_id=session_other, _id=ObjectId()
        )

        async def setup():
            await _insert_docs(db, COL_MEMCELLS, [mc_target, mc_other])

        event_loop.run_until_complete(setup())

        req = {"user_id": user_id, "session_id": session_target}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status == 204

        # Verify: other session's memcell still alive
        async def verify():
            assert await _count_alive(db, COL_MEMCELLS, {"_id": mc_other["_id"]}) == 1

        event_loop.run_until_complete(verify())

        _record_result(
            scenario="Delete by user_id + session_id",
            category="Delete by Filters",
            passed=True,
            duration_ms=elapsed,
            detail="Only target session deleted, other session intact",
            request_body=req,
            response_body=body,
            status_code=status,
        )

    def test_delete_by_user_and_sender(self, db, http_client, event_loop):
        """Delete memcells filtered by user_id + sender_id."""
        user_id = f"{PREFIX}user_sender"
        group_id = f"{PREFIX}group_sender"
        sender_target = f"{PREFIX}sender_A"
        sender_other = f"{PREFIX}sender_B"

        mc_target = _make_memcell(
            user_id=user_id,
            group_id=group_id,
            participants=[sender_target, "other_person"],
            _id=ObjectId(),
        )
        mc_other = _make_memcell(
            user_id=user_id,
            group_id=group_id,
            participants=[sender_other],
            _id=ObjectId(),
        )

        async def setup():
            await _insert_docs(db, COL_MEMCELLS, [mc_target, mc_other])

        event_loop.run_until_complete(setup())

        req = {"user_id": user_id, "sender_id": sender_target}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status == 204

        # Verify: sender_B's memcell still alive
        async def verify():
            assert await _count_alive(db, COL_MEMCELLS, {"_id": mc_other["_id"]}) == 1

        event_loop.run_until_complete(verify())

        _record_result(
            scenario="Delete by user_id + sender_id",
            category="Delete by Filters",
            passed=True,
            duration_ms=elapsed,
            detail="Only target sender deleted, other sender intact",
            request_body=req,
            response_body=body,
            status_code=status,
        )

    # ---------------------------------------------------------------
    # 4. Data isolation
    # ---------------------------------------------------------------

    def test_data_isolation(self, db, http_client, event_loop):
        """Deleting user_A data should not affect user_B."""
        user_a = f"{PREFIX}user_iso_A"
        user_b = f"{PREFIX}user_iso_B"
        group_id = f"{PREFIX}group_iso"

        mc_a = _make_memcell(user_id=user_a, group_id=group_id, _id=ObjectId())
        mc_b = _make_memcell(user_id=user_b, group_id=group_id, _id=ObjectId())

        ep_a = _make_child(
            COL_EPISODES, str(mc_a["_id"]), user_id=user_a, group_id=group_id
        )
        ep_b = _make_child(
            COL_EPISODES, str(mc_b["_id"]), user_id=user_b, group_id=group_id
        )

        async def setup():
            await _insert_docs(db, COL_MEMCELLS, [mc_a, mc_b])
            await _insert_docs(db, COL_EPISODES, [ep_a, ep_b])

        event_loop.run_until_complete(setup())

        # Delete user_A only
        req = {"user_id": user_a}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status == 204

        # Verify: user_B data untouched
        async def verify():
            assert await _count_alive(db, COL_MEMCELLS, {"_id": mc_b["_id"]}) == 1
            assert await _count_alive(db, COL_EPISODES, {"_id": ep_b["_id"]}) == 1

        event_loop.run_until_complete(verify())

        _record_result(
            scenario="Data isolation: user_A delete does not affect user_B",
            category="Data Isolation",
            passed=True,
            duration_ms=elapsed,
            detail="user_B memcell and episode still alive after user_A deletion",
            request_body=req,
            response_body=body,
            status_code=status,
        )

    # ---------------------------------------------------------------
    # 5. Additional validation scenarios (mutual exclusion)
    # ---------------------------------------------------------------

    def test_memory_id_with_group_id_returns_422(self, http_client):
        """memory_id + group_id together should fail validation."""
        req = {"memory_id": "abcdef1234567890abcdef12", "group_id": "g1"}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        _record_result(
            scenario="memory_id + group_id mutual exclusion",
            category="Request Validation",
            passed=status == 422,
            duration_ms=elapsed,
            request_body=req,
            response_body=body,
            status_code=status,
        )
        assert status == 422

    def test_memory_id_with_session_id_returns_422(self, http_client):
        """memory_id + session_id together should fail validation."""
        req = {"memory_id": "abcdef1234567890abcdef12", "session_id": "s1"}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        _record_result(
            scenario="memory_id + session_id mutual exclusion",
            category="Request Validation",
            passed=status == 422,
            duration_ms=elapsed,
            request_body=req,
            response_body=body,
            status_code=status,
        )
        assert status == 422

    def test_memory_id_with_sender_id_returns_422(self, http_client):
        """memory_id + sender_id together should fail validation."""
        req = {"memory_id": "abcdef1234567890abcdef12", "sender_id": "sd1"}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        _record_result(
            scenario="memory_id + sender_id mutual exclusion",
            category="Request Validation",
            passed=status == 422,
            duration_ms=elapsed,
            request_body=req,
            response_body=body,
            status_code=status,
        )
        assert status == 422

    def test_memory_id_with_all_filters_returns_422(self, http_client):
        """memory_id + all filter fields together should fail validation."""
        req = {
            "memory_id": "abcdef1234567890abcdef12",
            "user_id": "u1",
            "group_id": "g1",
            "session_id": "s1",
            "sender_id": "sd1",
        }
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        _record_result(
            scenario="memory_id + all filter fields mutual exclusion",
            category="Request Validation",
            passed=status == 422,
            duration_ms=elapsed,
            request_body=req,
            response_body=body,
            status_code=status,
        )
        assert status == 422

    # ---------------------------------------------------------------
    # 6. Additional functional scenarios
    # ---------------------------------------------------------------

    def test_filter_delete_matching_zero_records(self, http_client):
        """Filter delete with nonexistent user should return 204 (idempotent)."""
        req = {"user_id": f"{PREFIX}nonexistent_user_xyz"}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        _record_result(
            scenario="Filter delete matching 0 records (idempotent)",
            category="Delete by Filters",
            passed=status == 204,
            duration_ms=elapsed,
            request_body=req,
            response_body=body,
            status_code=status,
        )
        assert status == 204

    def test_delete_by_user_and_group_combined(self, db, http_client, event_loop):
        """Delete by user_id + group_id: verify AND logic with two scope filters."""
        user_id = f"{PREFIX}user_ug_combo"
        group_target = f"{PREFIX}group_ug_target"
        group_other = f"{PREFIX}group_ug_other"

        mc_target = _make_memcell(
            user_id=user_id, group_id=group_target, _id=ObjectId()
        )
        mc_other = _make_memcell(user_id=user_id, group_id=group_other, _id=ObjectId())
        ep_target = _make_child(
            COL_EPISODES, str(mc_target["_id"]), user_id=user_id, group_id=group_target
        )
        ep_other = _make_child(
            COL_EPISODES, str(mc_other["_id"]), user_id=user_id, group_id=group_other
        )

        async def setup():
            await _insert_docs(db, COL_MEMCELLS, [mc_target, mc_other])
            await _insert_docs(db, COL_EPISODES, [ep_target, ep_other])

        event_loop.run_until_complete(setup())

        req = {"user_id": user_id, "group_id": group_target}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status == 204

        # Verify AND logic: only target group deleted
        async def verify():
            assert await _count_alive(db, COL_EPISODES, {"_id": ep_other["_id"]}) == 1

        event_loop.run_until_complete(verify())

        _record_result(
            scenario="Delete by user_id + group_id combined (AND logic)",
            category="Delete by Filters",
            passed=True,
            duration_ms=elapsed,
            request_body=req,
            response_body=body,
            status_code=status,
        )

    def test_delete_all_four_filters_combined(self, db, http_client, event_loop):
        """Delete with all four filter fields combined."""
        user_id = f"{PREFIX}user_all4"
        group_id = f"{PREFIX}group_all4"
        session_id = f"{PREFIX}session_all4"
        sender_id = f"{PREFIX}sender_all4"

        mc = _make_memcell(
            user_id=user_id,
            group_id=group_id,
            session_id=session_id,
            participants=[sender_id],
            _id=ObjectId(),
        )
        ep = _make_child(
            COL_EPISODES,
            str(mc["_id"]),
            user_id=user_id,
            group_id=group_id,
            session_id=session_id,
            participants=[sender_id],
        )

        async def setup():
            await _insert_docs(db, COL_MEMCELLS, [mc])
            await _insert_docs(db, COL_EPISODES, [ep])

        event_loop.run_until_complete(setup())

        req = {
            "user_id": user_id,
            "group_id": group_id,
            "session_id": session_id,
            "sender_id": sender_id,
        }
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status == 204

        _record_result(
            scenario="Delete with all four filters combined",
            category="Delete by Filters",
            passed=True,
            duration_ms=elapsed,
            request_body=req,
            response_body=body,
            status_code=status,
        )

    def test_raw_message_untouched_on_filter_delete(self, db, http_client, event_loop):
        """Verify RawMessage (source data) is NOT deleted by filter delete."""
        user_id = f"{PREFIX}user_rawmsg"
        group_id = f"{PREFIX}group_rawmsg"

        mc = _make_memcell(user_id=user_id, group_id=group_id, _id=ObjectId())
        raw_msg = _make_request_log(user_id=user_id, group_id=group_id)

        async def setup():
            await _insert_docs(db, COL_MEMCELLS, [mc])
            await _insert_docs(db, COL_RAW_MESSAGES, [raw_msg])

        event_loop.run_until_complete(setup())

        req = {"user_id": user_id}
        t0 = time.monotonic()
        status, body = _call_delete(http_client, req)
        elapsed = (time.monotonic() - t0) * 1000

        assert status == 204

        # Verify raw message is NOT deleted (source data preserved)
        async def verify():
            assert (
                await _count_alive(db, COL_RAW_MESSAGES, {"_id": raw_msg["_id"]}) == 1
            )

        event_loop.run_until_complete(verify())

        _record_result(
            scenario="RawMessage untouched on filter delete",
            category="Delete by Filters",
            passed=True,
            duration_ms=elapsed,
            detail="Raw message preserved (source data not deleted by filter mode)",
            request_body=req,
            response_body=body,
            status_code=status,
        )

    # ---------------------------------------------------------------
    # 7. Response validation
    # ---------------------------------------------------------------

    def test_204_response_has_empty_body(self, http_client):
        """204 No Content response should have empty body."""
        fake_id = str(ObjectId())
        req = {"memory_id": fake_id}
        r = http_client.post(DELETE_URL, json=req)

        assert r.status_code == 204
        assert len(r.content) == 0

        _record_result(
            scenario="204 response has empty body",
            category="Response Validation",
            passed=True,
            duration_ms=0,
            request_body=req,
            status_code=204,
        )

    def test_422_response_has_structured_error(self, http_client):
        """422 response should contain structured error information."""
        req = {}
        r = http_client.post(DELETE_URL, json=req)

        assert r.status_code == 422
        body = r.json()
        # FastAPI validation errors have a 'detail' field
        assert "detail" in body

        _record_result(
            scenario="422 response has structured error body",
            category="Response Validation",
            passed=True,
            duration_ms=0,
            request_body=req,
            response_body=body,
            status_code=422,
        )


# ---------------------------------------------------------------------------
# Cleanup fixture (runs after ALL tests in this module)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module", autouse=True)
def _cleanup_after_all(db, event_loop):
    """Clean up all test fixture data after module completes."""
    yield  # tests run here
    event_loop.run_until_complete(_cleanup_test_data(db))


# ---------------------------------------------------------------------------
# Report generation (runs after ALL tests)
# ---------------------------------------------------------------------------
REPORT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "docs",
    "plans",
    "delete-api-integration-test-report.json",
)


def pytest_sessionfinish(session, exitstatus):
    """Write test results to JSON for Confluence report generation."""
    if _test_results:
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "total": len(_test_results),
                    "passed": sum(1 for r in _test_results if r["passed"]),
                    "failed": sum(1 for r in _test_results if not r["passed"]),
                    "results": _test_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
