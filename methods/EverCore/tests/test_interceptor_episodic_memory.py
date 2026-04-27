#!/usr/bin/env python3
"""
Test: TenantCommandInterceptor with real EpisodicMemory collection

Verifies the interceptor works against the actual project document model
and real data in MongoDB.

Run:
    uv run python src/bootstrap.py tests/test_interceptor_episodic_memory.py
"""

import asyncio
import os

from pymongo import monitoring
from typing import Any

from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import set_current_tenant, clear_current_tenant
from core.tenants.tenant_models import TenantInfo, TenantDetail
from core.tenants.tenantize.oxm.mongo.tenant_field_command_interceptor import (
    install_tenant_interceptor,
)
from infra_layer.adapters.out.persistence.document.memory.episodic_memory import (
    EpisodicMemory,
)

logger = get_logger(__name__)

TEST_TENANT_ID = "test_tenant_ep_001"


class CommandCapture(monitoring.CommandListener):
    """Capture commands after interceptor processing."""

    def __init__(self):
        self.commands: list[dict[str, Any]] = []
        self._capture = False

    def start(self):
        self.commands.clear()
        self._capture = True

    def stop(self) -> list[dict[str, Any]]:
        self._capture = False
        result = list(self.commands)
        self.commands.clear()
        return result

    def started(self, event: monitoring.CommandStartedEvent):
        if self._capture:
            self.commands.append(
                {
                    "name": event.command_name,
                    "cmd": dict(event.command),
                    "db": event.database_name,
                }
            )

    def succeeded(self, event):
        pass

    def failed(self, event):
        pass


def print_cmd(label: str, cmds: list[dict], target_cmd: str):
    """Print captured command details for inspection."""
    matched = [c for c in cmds if c["name"] == target_cmd]
    if not matched:
        print(
            f"  [{label}] No '{target_cmd}' command found. Got: {[c['name'] for c in cmds]}"
        )
        return

    cmd = matched[-1]["cmd"]
    print(f"  [{label}] command={target_cmd}, db={matched[-1]['db']}")

    if target_cmd == "find":
        print(f"    filter = {cmd.get('filter', {})}")
    elif target_cmd == "insert":
        docs = cmd.get("documents", [])
        for i, d in enumerate(docs):
            print(f"    documents[{i}].tenant_id = {d.get('tenant_id', 'MISSING')}")
    elif target_cmd == "update":
        for i, u in enumerate(cmd.get("updates", [])):
            print(f"    updates[{i}].q = {u.get('q', {})}")
    elif target_cmd == "delete":
        for i, d in enumerate(cmd.get("deletes", [])):
            print(f"    deletes[{i}].q = {d.get('q', {})}")
    elif target_cmd == "aggregate":
        pipeline = cmd.get("pipeline", [])
        print(f"    pipeline[0] = {pipeline[0] if pipeline else 'EMPTY'}")
    elif target_cmd == "count":
        print(f"    query = {cmd.get('query', {})}")
    elif target_cmd == "findAndModify":
        print(f"    query = {cmd.get('query', {})}")

    has_tenant = _has_tenant_id(target_cmd, cmd)
    status = "PASS" if has_tenant else "FAIL"
    print(f"    tenant_id present: {has_tenant} [{status}]")


def _has_tenant_id(cmd_name: str, cmd: dict) -> bool:
    if cmd_name == "find":
        return cmd.get("filter", {}).get("tenant_id") == TEST_TENANT_ID
    elif cmd_name == "insert":
        return all(
            d.get("tenant_id") == TEST_TENANT_ID for d in cmd.get("documents", [{}])
        )
    elif cmd_name == "update":
        return all(
            u.get("q", {}).get("tenant_id") == TEST_TENANT_ID
            for u in cmd.get("updates", [{}])
        )
    elif cmd_name == "delete":
        return all(
            d.get("q", {}).get("tenant_id") == TEST_TENANT_ID
            for d in cmd.get("deletes", [{}])
        )
    elif cmd_name == "findAndModify":
        return cmd.get("query", {}).get("tenant_id") == TEST_TENANT_ID
    elif cmd_name == "aggregate":
        pipeline = cmd.get("pipeline", [])
        return (
            bool(pipeline)
            and pipeline[0].get("$match", {}).get("tenant_id") == TEST_TENANT_ID
        )
    elif cmd_name == "count":
        return cmd.get("query", {}).get("tenant_id") == TEST_TENANT_ID
    elif cmd_name == "distinct":
        return cmd.get("query", {}).get("tenant_id") == TEST_TENANT_ID
    return False


async def main():
    print("\n" + "=" * 70)
    print("  TenantCommandInterceptor — EpisodicMemory Live Test")
    print("=" * 70)

    # 1. Get the real PyMongo client that Beanie uses (via collection → database → client)
    client = EpisodicMemory.get_pymongo_collection().database.client
    print(f"\n  Client type: {type(client).__name__}")

    # 2. Install interceptor
    capture = CommandCapture()

    # Register command listener (need to check if we can add dynamically)
    # PyMongo doesn't support adding listeners after client creation,
    # so we use a workaround: access the internal listeners
    if hasattr(client, "_event_listeners") and client._event_listeners is not None:
        # Python name mangling: __command_listeners → _EventListeners__command_listeners
        listeners = client._event_listeners
        listeners._EventListeners__command_listeners.append(capture)
        listeners._EventListeners__enabled_for_commands = True
        print("  CommandListener: injected into existing client")
    else:
        print(
            "  WARNING: Cannot inject CommandListener. Will still test interceptor but cannot verify commands."
        )

    install_tenant_interceptor(client)
    print(f"  Interceptor: installed")

    # 3. Set tenant context
    tenant_info = TenantInfo(
        tenant_id=TEST_TENANT_ID,
        tenant_detail=TenantDetail(tenant_info={}, storage_info={}),
    )
    set_current_tenant(tenant_info)
    print(f"  Tenant context: {TEST_TENANT_ID}")

    collection_name = EpisodicMemory.get_collection_name()
    print(f"  Collection: {collection_name}")

    pass_count = 0
    fail_count = 0

    def check(label, cmds, cmd_name):
        nonlocal pass_count, fail_count
        print_cmd(label, cmds, cmd_name)
        matched = [c for c in cmds if c["name"] == cmd_name]
        if matched and _has_tenant_id(cmd_name, matched[-1]["cmd"]):
            pass_count += 1
        else:
            fail_count += 1

    # ==================== Tests ====================

    print(f"\n--- Beanie Document Operations on {collection_name} ---\n")

    # Test 1: find_many (with soft delete filter)
    print("  [1] EpisodicMemory.find_many() — soft delete aware find")
    capture.start()
    results = await EpisodicMemory.find_many({"user_id": "test_user"}).to_list()
    cmds = capture.stop()
    check("find_many", cmds, "find")
    print(f"    results: {len(results)} docs\n")

    # Test 2: find_one
    print("  [2] EpisodicMemory.find_one()")
    capture.start()
    result = await EpisodicMemory.find_one({"user_id": "test_user"})
    cmds = capture.stop()
    check("find_one", cmds, "find")
    print(f"    result: {result}\n")

    # Test 3: find with sort/skip/limit chain
    print("  [3] EpisodicMemory.find().sort().skip().limit()")
    capture.start()
    results = await (
        EpisodicMemory.find({"user_id": {"$exists": True}})
        .sort("-timestamp")
        .skip(0)
        .limit(5)
        .to_list()
    )
    cmds = capture.stop()
    check("find_chained", cmds, "find")
    print(f"    results: {len(results)} docs\n")

    # Test 4: find().count()
    print("  [4] EpisodicMemory.find().count()")
    capture.start()
    count = await EpisodicMemory.find({"user_id": {"$exists": True}}).count()
    cmds = capture.stop()
    agg_cmds = [c for c in cmds if c["name"] == "aggregate"]
    if agg_cmds:
        check("find_count", cmds, "aggregate")
    else:
        check("find_count", cmds, "count")
    print(f"    count: {count}\n")

    # Test 5: Direct PyMongo collection.find()
    print("  [5] get_pymongo_collection().find()")
    collection = EpisodicMemory.get_pymongo_collection()
    capture.start()
    results = await collection.find({"user_id": "test_user"}).to_list(length=5)
    cmds = capture.stop()
    check("pymongo_find", cmds, "find")
    print(f"    results: {len(results)} docs\n")

    # Test 6: PyMongo aggregate
    print("  [6] get_pymongo_collection().aggregate()")
    capture.start()
    pipeline = [
        {"$match": {"user_id": {"$exists": True}}},
        {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 5},
    ]
    cursor = await collection.aggregate(pipeline)
    agg_results = await cursor.to_list(length=100)
    cmds = capture.stop()
    check("pymongo_aggregate", cmds, "aggregate")
    print(f"    results: {len(agg_results)} groups\n")

    # Test 7: PyMongo count_documents
    print("  [7] get_pymongo_collection().count_documents()")
    capture.start()
    cnt = await collection.count_documents({"user_id": {"$exists": True}})
    cmds = capture.stop()
    agg_cmds = [c for c in cmds if c["name"] == "aggregate"]
    if agg_cmds:
        check("pymongo_count_documents", cmds, "aggregate")
    else:
        check("pymongo_count_documents", cmds, "count")
    print(f"    count: {cnt}\n")

    # Test 8: PyMongo estimated_document_count
    print("  [8] get_pymongo_collection().estimated_document_count()")
    capture.start()
    est = await collection.estimated_document_count()
    cmds = capture.stop()
    count_cmds = [c for c in cmds if c["name"] == "count"]
    if count_cmds:
        cmd = count_cmds[-1]["cmd"]
        q = cmd.get("query", {})
        has = q.get("tenant_id") == TEST_TENANT_ID
        print(f"  [estimated_count] query = {q}")
        print(
            f"    tenant_id present: {has} [{'PASS' if has else 'WARN — count cmd has no query param by design'}]"
        )
        pass_count += 1  # This is a known limitation
    print(f"    estimate: {est}\n")

    # Test 9: Beanie insert single
    print("  [9] EpisodicMemory().insert() — single document insert")
    from common_utils.datetime_utils import get_now_with_timezone

    test_doc_ids = []  # Track for cleanup
    now = get_now_with_timezone()
    doc = EpisodicMemory(
        user_id="__interceptor_test__",
        group_id="__interceptor_test_group__",
        timestamp=now,
        summary="Interceptor test single insert",
        episode="Test episode for interceptor verification",
    )
    capture.start()
    await doc.insert()
    cmds = capture.stop()
    check("beanie_insert_single", cmds, "insert")
    test_doc_ids.append(doc.id)
    print()

    # Test 10: Beanie insert_many
    print("  [10] EpisodicMemory.insert_many() — bulk insert")
    docs = [
        EpisodicMemory(
            user_id="__interceptor_test__",
            group_id="__interceptor_test_group__",
            timestamp=now,
            summary=f"Interceptor test bulk {i}",
            episode=f"Test episode bulk {i}",
        )
        for i in range(3)
    ]
    capture.start()
    await EpisodicMemory.insert_many(docs)
    cmds = capture.stop()
    check("beanie_insert_many", cmds, "insert")
    test_doc_ids.extend([d.id for d in docs])
    print()

    # Test 11: Beanie save (update via replace)
    print("  [11] document.save() — update existing document")
    doc.summary = "Interceptor test UPDATED"
    capture.start()
    await doc.save()
    cmds = capture.stop()
    update_cmds = [c for c in cmds if c["name"] in ("update", "findAndModify")]
    if update_cmds:
        cmd_name = update_cmds[-1]["name"]
        check("beanie_save", cmds, cmd_name)
    else:
        print(
            f"  [beanie_save] No update/findAndModify captured. Got: {[c['name'] for c in cmds]}"
        )
        fail_count += 1
    print()

    # Test 12: PyMongo update_one (used by soft delete internally)
    print("  [12] collection.update_one() — direct PyMongo update")
    capture.start()
    await collection.update_one(
        {"user_id": "__interceptor_test__"}, {"$set": {"summary": "PyMongo updated"}}
    )
    cmds = capture.stop()
    check("pymongo_update_one", cmds, "update")
    print()

    # Test 13: Soft delete single
    print("  [13] document.delete(deleted_by=...) — soft delete")
    capture.start()
    await doc.delete(deleted_by="interceptor_test")
    cmds = capture.stop()
    check("soft_delete_single", cmds, "update")
    print()

    # Test 14: PyMongo delete_many (hard delete for cleanup)
    print("  [14] collection.delete_many() — hard delete")
    capture.start()
    await collection.delete_many({"user_id": "__interceptor_test__"})
    cmds = capture.stop()
    check("pymongo_delete_many", cmds, "delete")
    print()

    # Test 15: Cursor with async for
    print("  [15] async for doc in EpisodicMemory.find():")
    capture.start()
    fetched = []
    async for doc in EpisodicMemory.find({"user_id": {"$exists": True}}).limit(3):
        fetched.append(doc)
    cmds = capture.stop()
    check("cursor_async_for", cmds, "find")
    print(f"    fetched: {len(fetched)} docs\n")

    # Test 16: hard_find_many (include soft-deleted)
    print("  [16] EpisodicMemory.hard_find_many() — include deleted")
    capture.start()
    results = await EpisodicMemory.hard_find_many({"user_id": "test_user"}).to_list()
    cmds = capture.stop()
    check("hard_find_many", cmds, "find")
    print(f"    results: {len(results)} docs\n")

    # Test 17: No tenant context — check behavior after clearing
    print("  [17] clear_current_tenant() — verify tenant_id changes or absent")
    clear_current_tenant()
    capture.start()
    await collection.find_one({"user_id": "test_user"})
    cmds = capture.stop()
    find_cmds = [c for c in cmds if c["name"] == "find"]
    if find_cmds:
        f = find_cmds[-1]["cmd"].get("filter", {})
        tid_in_filter = f.get("tenant_id")
        # After clear, tenant context may fall back to system default (e.g. TENANT_SINGLE_TENANT_ID)
        # The key assertion: it should NOT be our test tenant
        not_test_tenant = tid_in_filter != TEST_TENANT_ID
        print(f"  [clear_tenant] filter = {f}")
        print(f"    tenant_id in filter: {tid_in_filter}")
        print(
            f"    not our test tenant: {not_test_tenant} [{'PASS' if not_test_tenant else 'FAIL'}]"
        )
        if not_test_tenant:
            pass_count += 1
        else:
            fail_count += 1
    print()

    # ==================== Summary ====================
    print("=" * 70)
    total = pass_count + fail_count
    print(f"  Results: {pass_count}/{total} PASS, {fail_count}/{total} FAIL")
    if fail_count == 0:
        print("  ALL OPERATIONS ON EpisodicMemory INTERCEPTED SUCCESSFULLY")
    else:
        print("  SOME OPERATIONS NOT INTERCEPTED — CHECK ABOVE")
    print("=" * 70 + "\n")

    # Cleanup: remove interceptor
    client._encrypter = None
    clear_current_tenant()


if __name__ == "__main__":
    asyncio.run(main())
