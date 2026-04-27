#!/usr/bin/env python3
"""
Test: TenantCommandInterceptor coverage verification

Verifies that ALL MongoDB operation patterns used in this project
are intercepted by the TenantCommandInterceptor at the network layer.

Strategy:
    1. Connect to real MongoDB with interceptor + CommandListener installed
    2. Set tenant context
    3. Execute every operation pattern found in the codebase
    4. CommandListener captures the FINAL command sent to MongoDB
    5. Assert tenant_id is present in every captured command

Run:
    uv run python src/bootstrap.py tests/test_tenant_field_command_interceptor.py
"""

import asyncio
import os
from datetime import timedelta
from typing import Any, Optional
from collections import defaultdict

from bson import ObjectId
from pymongo import monitoring
from pymongo.asynchronous.mongo_client import AsyncMongoClient

from beanie import init_beanie

from common_utils.datetime_utils import get_now_with_timezone
from core.observation.logger import get_logger
from core.tenants.tenant_contextvar import set_current_tenant, clear_current_tenant
from core.tenants.tenant_models import TenantInfo, TenantDetail
from core.tenants.tenantize.oxm.mongo.tenant_field_command_interceptor import (
    TenantCommandInterceptor,
    install_tenant_interceptor,
)

logger = get_logger(__name__)

# Test constants
TEST_TENANT_ID = "test_tenant_interceptor_001"
TEST_DB_NAME = "test_interceptor_db"
TEST_COLLECTION = "test_interceptor_docs"

# ==================== Command Capture ====================


class CommandCapture(monitoring.CommandListener):
    """
    Captures all commands sent to MongoDB AFTER the interceptor processes them.
    This proves the interceptor is in the pipeline.
    """

    def __init__(self):
        self.commands: list[dict[str, Any]] = []
        self._capture = False

    def start_capture(self):
        self.commands.clear()
        self._capture = True

    def stop_capture(self) -> list[dict[str, Any]]:
        self._capture = False
        result = list(self.commands)
        self.commands.clear()
        return result

    def started(self, event: monitoring.CommandStartedEvent):
        if self._capture:
            self.commands.append(
                {
                    "command_name": event.command_name,
                    "command": dict(event.command),
                    "database_name": event.database_name,
                }
            )

    def succeeded(self, event):
        pass

    def failed(self, event):
        pass


# ==================== Test Document (Beanie) ====================

from pydantic import Field
from core.oxm.mongo.document_base import DocumentBase
from core.oxm.mongo.document_base_with_soft_delete import DocumentBaseWithSoftDelete
from core.oxm.mongo.audit_base import AuditBase


class TestDoc(DocumentBaseWithSoftDelete, AuditBase):
    """Test document with soft delete + audit, mimics real project documents."""

    user_id: str = ""
    name: str = ""
    status: str = "active"
    tags: list[str] = Field(default_factory=list)
    score: int = 0

    class Settings:
        name = TEST_COLLECTION
        use_state_management = True


# ==================== Test Runner ====================


class InterceptorTestRunner:
    def __init__(self):
        self.client: Optional[AsyncMongoClient] = None
        self.capture = CommandCapture()
        self.results: dict[str, dict] = {}

    async def setup(self):
        """Create a fresh client with interceptor + command listener."""
        host = os.getenv("MONGODB_HOST", "localhost")
        port = int(os.getenv("MONGODB_PORT", "27017"))
        username = os.getenv("MONGODB_USERNAME", "")
        password = os.getenv("MONGODB_PASSWORD", "")

        conn_kwargs = {
            "host": host,
            "port": port,
            "event_listeners": [self.capture],
            "serverSelectionTimeoutMS": 5000,
        }
        if username and password:
            from urllib.parse import quote_plus

            uri = (
                f"mongodb://{quote_plus(username)}:{quote_plus(password)}@{host}:{port}"
            )
            self.client = AsyncMongoClient(
                uri,
                **{k: v for k, v in conn_kwargs.items() if k != "host" and k != "port"},
            )
        else:
            self.client = AsyncMongoClient(**conn_kwargs)

        # Install interceptor
        install_tenant_interceptor(
            self.client, excluded_collections={"__excluded_test__"}
        )

        # Init beanie with test document
        db = self.client[TEST_DB_NAME]
        await init_beanie(database=db, document_models=[TestDoc])

        # Clean up test collection
        await db[TEST_COLLECTION].delete_many({})

        # Set tenant context
        tenant_info = TenantInfo(
            tenant_id=TEST_TENANT_ID,
            tenant_detail=TenantDetail(tenant_info={}, storage_info={}),
        )
        set_current_tenant(tenant_info)

        logger.info("Setup complete. Interceptor installed, tenant=%s", TEST_TENANT_ID)

    async def teardown(self):
        """Cleanup."""
        clear_current_tenant()
        if self.client:
            await self.client[TEST_DB_NAME].drop_collection(TEST_COLLECTION)
            await self.client.close()

    def _check_command(
        self,
        test_name: str,
        commands: list[dict],
        expected_cmd_name: str,
        check_fn: Any = None,
    ) -> bool:
        """Verify a captured command has tenant_id injected."""
        matched = [c for c in commands if c["command_name"] == expected_cmd_name]
        if not matched:
            self.results[test_name] = {
                "status": "FAIL",
                "reason": f"No '{expected_cmd_name}' command captured. Got: {[c['command_name'] for c in commands]}",
            }
            return False

        cmd = matched[-1]["command"]  # Take the last one

        if check_fn:
            ok, reason = check_fn(cmd)
        else:
            ok, reason = self._default_tenant_check(expected_cmd_name, cmd)

        self.results[test_name] = {
            "status": "PASS" if ok else "FAIL",
            "reason": reason,
            "command": cmd,
        }
        return ok

    def _default_tenant_check(self, cmd_name: str, cmd: dict) -> tuple[bool, str]:
        """Default check: verify tenant_id in the expected location."""
        if cmd_name == "insert":
            docs = cmd.get("documents", [])
            if not docs:
                return False, "No documents in insert command"
            for i, doc in enumerate(docs):
                if doc.get("tenant_id") != TEST_TENANT_ID:
                    return False, f"documents[{i}] missing tenant_id"
            return True, f"tenant_id injected into {len(docs)} document(s)"

        elif cmd_name == "update":
            updates = cmd.get("updates", [])
            for i, u in enumerate(updates):
                q = u.get("q", {})
                if q.get("tenant_id") != TEST_TENANT_ID:
                    return False, f"updates[{i}].q missing tenant_id"
            return True, f"tenant_id injected into {len(updates)} update filter(s)"

        elif cmd_name == "delete":
            deletes = cmd.get("deletes", [])
            for i, d in enumerate(deletes):
                q = d.get("q", {})
                if q.get("tenant_id") != TEST_TENANT_ID:
                    return False, f"deletes[{i}].q missing tenant_id"
            return True, f"tenant_id injected into {len(deletes)} delete filter(s)"

        elif cmd_name == "find":
            f = cmd.get("filter", {})
            if f.get("tenant_id") != TEST_TENANT_ID:
                return False, f"filter missing tenant_id. Got: {f}"
            return True, "tenant_id in filter"

        elif cmd_name == "findAndModify":
            q = cmd.get("query", {})
            if q.get("tenant_id") != TEST_TENANT_ID:
                return False, f"query missing tenant_id. Got: {q}"
            return True, "tenant_id in query"

        elif cmd_name == "aggregate":
            pipeline = cmd.get("pipeline", [])
            if not pipeline:
                return False, "Empty pipeline"
            first_stage = pipeline[0]
            match = first_stage.get("$match", {})
            if match.get("tenant_id") != TEST_TENANT_ID:
                return (
                    False,
                    f"First $match stage missing tenant_id. Got: {first_stage}",
                )
            return True, "tenant_id in first $match stage"

        elif cmd_name == "count":
            q = cmd.get("query", {})
            if q.get("tenant_id") != TEST_TENANT_ID:
                return False, f"query missing tenant_id. Got: {q}"
            return True, "tenant_id in count query"

        elif cmd_name == "distinct":
            q = cmd.get("query", {})
            if q.get("tenant_id") != TEST_TENANT_ID:
                return False, f"query missing tenant_id. Got: {q}"
            return True, "tenant_id in distinct query"

        return False, f"Unknown command type: {cmd_name}"

    # ==================== Test Cases ====================

    async def test_beanie_insert_single(self):
        """Beanie: document.insert() — used by BaseRepository.create()"""
        doc = TestDoc(user_id="u1", name="Alice", status="active")
        self.capture.start_capture()
        await doc.insert()
        cmds = self.capture.stop_capture()
        self._check_command("beanie_insert_single", cmds, "insert")

    async def test_beanie_insert_many(self):
        """Beanie: Model.insert_many() — used by BaseRepository.create_batch()"""
        docs = [
            TestDoc(user_id="u2", name="Bob"),
            TestDoc(user_id="u3", name="Charlie"),
        ]
        self.capture.start_capture()
        await TestDoc.insert_many(docs)
        cmds = self.capture.stop_capture()
        self._check_command("beanie_insert_many", cmds, "insert")

    async def test_beanie_save(self):
        """Beanie: document.save() — used by BaseRepository.update()"""
        doc = TestDoc(user_id="u4", name="Dave")
        await doc.insert()

        doc.name = "Dave Updated"
        self.capture.start_capture()
        await doc.save()
        cmds = self.capture.stop_capture()
        # save() internally does replace_one → update command
        # OR findAndModify depending on Beanie version
        update_cmds = [
            c for c in cmds if c["command_name"] in ("update", "findAndModify")
        ]
        if update_cmds:
            cmd_name = update_cmds[-1]["command_name"]
            self._check_command("beanie_save", cmds, cmd_name)
        else:
            self.results["beanie_save"] = {
                "status": "FAIL",
                "reason": f"No update/findAndModify captured. Got: {[c['command_name'] for c in cmds]}",
            }

    async def test_beanie_find(self):
        """Beanie: Model.find(query) — used widely in repositories"""
        self.capture.start_capture()
        await TestDoc.find({"user_id": "u1"}).to_list()
        cmds = self.capture.stop_capture()
        self._check_command("beanie_find", cmds, "find")

    async def test_beanie_find_one(self):
        """Beanie: Model.find_one(query)"""
        self.capture.start_capture()
        await TestDoc.find_one({"user_id": "u1"})
        cmds = self.capture.stop_capture()
        self._check_command("beanie_find_one", cmds, "find")

    async def test_beanie_find_chained(self):
        """Beanie: Model.find(query).sort().skip().limit() — used in repository queries"""
        self.capture.start_capture()
        await TestDoc.find({"status": "active"}).sort("name").skip(0).limit(
            10
        ).to_list()
        cmds = self.capture.stop_capture()
        self._check_command("beanie_find_chained", cmds, "find")

    async def test_beanie_find_count(self):
        """Beanie: Model.find(query).count() — used in repository count queries"""
        self.capture.start_capture()
        await TestDoc.find({"status": "active"}).count()
        cmds = self.capture.stop_capture()
        # .count() internally uses aggregate with $match + $group
        # Check if aggregate pipeline has tenant_id
        agg_cmds = [c for c in cmds if c["command_name"] == "aggregate"]
        if agg_cmds:
            self._check_command("beanie_find_count", cmds, "aggregate")
        else:
            # Fallback: might use count command
            self._check_command("beanie_find_count", cmds, "count")

    async def test_cursor_getmore(self):
        """Cursor: verify find goes through interceptor, getMore is passthrough.

        When cursor batch_size < total docs, PyMongo sends:
          1. find command (with filter) → should have tenant_id
          2. getMore command (no filter) → passthrough, no tenant_id needed

        This tests the CURSOR path:
          cursor._refresh() → _send_message(_Query) → client._run_operation()
            → server.run_operation() → server.operation_to_command()
              → _encrypter.encrypt()  ← interceptor hook
        """
        # Insert enough docs to trigger getMore (batch_size=2, insert 5)
        for i in range(5):
            await TestDoc(user_id=f"u_cursor_{i}", name=f"Cursor{i}").insert()

        self.capture.start_capture()
        # Small batch_size forces getMore
        cursor = TestDoc.find({"user_id": {"$regex": "^u_cursor_"}}).batch_size(2)
        results = await cursor.to_list()
        cmds = self.capture.stop_capture()

        # Verify find command has tenant_id
        find_cmds = [c for c in cmds if c["command_name"] == "find"]
        getmore_cmds = [c for c in cmds if c["command_name"] == "getMore"]

        if not find_cmds:
            self.results["cursor_getmore"] = {
                "status": "FAIL",
                "reason": f"No find command captured. Got: {[c['command_name'] for c in cmds]}",
            }
            return

        find_filter = find_cmds[0]["command"].get("filter", {})
        find_ok = find_filter.get("tenant_id") == TEST_TENANT_ID

        self.results["cursor_getmore"] = {
            "status": "PASS" if find_ok else "FAIL",
            "reason": (
                f"find filter has tenant_id: {find_ok}, "
                f"getMore commands: {len(getmore_cmds)} (passthrough, no filter needed), "
                f"total docs fetched: {len(results)}"
            ),
        }

    async def test_cursor_async_for(self):
        """Cursor: async for iteration — the most common cursor consumption pattern."""
        self.capture.start_capture()
        results = []
        async for doc in TestDoc.find({"status": "active"}):
            results.append(doc)
            if len(results) >= 3:
                break
        cmds = self.capture.stop_capture()
        self._check_command("cursor_async_for", cmds, "find")

    async def test_beanie_find_delete(self):
        """Beanie: Model.find(query).delete() — used for bulk delete via query chain"""
        # Insert a doc to delete
        doc = TestDoc(user_id="u_del_chain", name="ToDeleteChain")
        await doc.insert()

        self.capture.start_capture()
        await TestDoc.find({"user_id": "u_del_chain"}).delete()
        cmds = self.capture.stop_capture()
        self._check_command("beanie_find_delete", cmds, "delete")

    async def test_beanie_get_by_id(self):
        """Beanie: Model.get(object_id) — used by BaseRepository.get_by_id()"""
        doc = TestDoc(user_id="u_get", name="GetById")
        await doc.insert()

        self.capture.start_capture()
        await TestDoc.get(doc.id)
        cmds = self.capture.stop_capture()
        self._check_command("beanie_get_by_id", cmds, "find")

    async def test_beanie_delete_instance(self):
        """Beanie: document.delete() — hard delete on instance"""
        doc = TestDoc(user_id="u_hard_del", name="HardDel")
        await doc.insert()

        self.capture.start_capture()
        await doc.delete()
        cmds = self.capture.stop_capture()
        self._check_command("beanie_delete_instance", cmds, "delete")

    # ---- Soft Delete Operations ----

    async def test_soft_delete_single(self):
        """SoftDelete: document.delete(deleted_by=...) — used in all repositories"""
        doc = TestDoc(user_id="u_soft", name="SoftDel")
        await doc.insert()

        self.capture.start_capture()
        await doc.delete(deleted_by="test_admin")
        cmds = self.capture.stop_capture()
        # Soft delete uses PyMongo update_one internally
        self._check_command("soft_delete_single", cmds, "update")

    async def test_soft_delete_many(self):
        """SoftDelete: Model.delete_many(filter) — bulk soft delete"""
        await TestDoc(user_id="u_bulk_sd_1", name="BulkSD1").insert()
        await TestDoc(user_id="u_bulk_sd_2", name="BulkSD2").insert()

        self.capture.start_capture()
        await TestDoc.delete_many(
            {"user_id": {"$regex": "^u_bulk_sd"}}, deleted_by="admin"
        )
        cmds = self.capture.stop_capture()
        self._check_command("soft_delete_many", cmds, "update")

    async def test_soft_delete_find_many(self):
        """SoftDelete: Model.find_many(query) — auto-filters deleted_at=None"""
        self.capture.start_capture()
        await TestDoc.find_many({"status": "active"}).to_list()
        cmds = self.capture.stop_capture()
        self._check_command("soft_delete_find_many", cmds, "find")

    async def test_soft_delete_find_one(self):
        """SoftDelete: Model.find_one(query) — auto-filters deleted_at=None"""
        self.capture.start_capture()
        await TestDoc.find_one({"user_id": "u1"})
        cmds = self.capture.stop_capture()
        self._check_command("soft_delete_find_one", cmds, "find")

    async def test_hard_find_many(self):
        """SoftDelete: Model.hard_find_many(query) — include deleted"""
        self.capture.start_capture()
        await TestDoc.hard_find_many({"status": "active"}).to_list()
        cmds = self.capture.stop_capture()
        self._check_command("hard_find_many", cmds, "find")

    async def test_hard_delete_many(self):
        """SoftDelete: Model.hard_delete_many(filter) — physical bulk delete"""
        await TestDoc(user_id="u_hd_1", name="HD1").insert()

        self.capture.start_capture()
        await TestDoc.hard_delete_many({"user_id": "u_hd_1"})
        cmds = self.capture.stop_capture()
        self._check_command("hard_delete_many", cmds, "delete")

    async def test_soft_delete_count(self):
        """SoftDelete: Model.count() — count with soft delete awareness"""
        self.capture.start_capture()
        await TestDoc.count()
        cmds = self.capture.stop_capture()
        # count() uses count_documents or estimated_document_count
        count_cmds = [c for c in cmds if c["command_name"] in ("count", "aggregate")]
        if count_cmds:
            self._check_command(
                "soft_delete_count", cmds, count_cmds[-1]["command_name"]
            )
        else:
            self.results["soft_delete_count"] = {
                "status": "WARN",
                "reason": f"estimated_document_count does not support filter. Got: {[c['command_name'] for c in cmds]}",
            }

    # ---- Direct PyMongo Operations ----

    async def test_pymongo_find(self):
        """PyMongo: collection.find(filter) — used in demo/debug scripts"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.find({"status": "active"}).to_list(length=10)
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_find", cmds, "find")

    async def test_pymongo_find_one(self):
        """PyMongo: collection.find_one(filter) — used in debug scripts"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.find_one({"status": "active"})
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_find_one", cmds, "find")

    async def test_pymongo_insert_one(self):
        """PyMongo: collection.insert_one(doc)"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.insert_one(
            {"user_id": "pymongo_u1", "name": "PyInsert", "status": "active"}
        )
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_insert_one", cmds, "insert")

    async def test_pymongo_update_one(self):
        """PyMongo: collection.update_one(filter, update) — used in soft delete impl"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.update_one(
            {"user_id": "pymongo_u1"}, {"$set": {"name": "PyUpdated"}}
        )
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_update_one", cmds, "update")

    async def test_pymongo_update_many(self):
        """PyMongo: collection.update_many(filter, update) — used in raw_message_repository"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.update_many({"status": "active"}, {"$set": {"score": 100}})
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_update_many", cmds, "update")

    async def test_pymongo_replace_one(self):
        """PyMongo: collection.replace_one(filter, replacement)"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.replace_one(
            {"user_id": "pymongo_u1"},
            {"user_id": "pymongo_u1", "name": "PyReplaced", "status": "replaced"},
        )
        cmds = self.capture.stop_capture()

        def check_replace(cmd):
            updates = cmd.get("updates", [])
            if not updates:
                return False, "No updates in command"
            u = updates[0]
            q = u.get("q", {})
            replacement = u.get("u", {})
            q_ok = q.get("tenant_id") == TEST_TENANT_ID
            r_ok = replacement.get("tenant_id") == TEST_TENANT_ID
            if not q_ok:
                return False, f"Filter missing tenant_id: {q}"
            if not r_ok:
                return False, f"Replacement missing tenant_id: {replacement}"
            return True, "tenant_id in both filter and replacement"

        self._check_command("pymongo_replace_one", cmds, "update", check_replace)

    async def test_pymongo_delete_one(self):
        """PyMongo: collection.delete_one(filter)"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.delete_one({"user_id": "pymongo_u1"})
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_delete_one", cmds, "delete")

    async def test_pymongo_delete_many(self):
        """PyMongo: collection.delete_many(filter) — used in clear_all_data, hard_delete"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.delete_many({"status": "replaced"})
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_delete_many", cmds, "delete")

    async def test_pymongo_aggregate(self):
        """PyMongo: collection.aggregate(pipeline) — used in memory_utils, demo scripts"""
        collection = TestDoc.get_pymongo_collection()
        pipeline = [
            {"$match": {"status": "active"}},
            {"$group": {"_id": "$user_id", "count": {"$sum": 1}}},
        ]
        self.capture.start_capture()
        cursor = collection.aggregate(pipeline)
        await cursor.to_list(length=100)
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_aggregate", cmds, "aggregate")

    async def test_pymongo_count_documents(self):
        """PyMongo: collection.count_documents(filter) — used in soft delete count"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.count_documents({"status": "active"})
        cmds = self.capture.stop_capture()
        # count_documents uses aggregate internally in newer PyMongo
        agg_cmds = [c for c in cmds if c["command_name"] == "aggregate"]
        if agg_cmds:
            self._check_command("pymongo_count_documents", cmds, "aggregate")
        else:
            self._check_command("pymongo_count_documents", cmds, "count")

    async def test_pymongo_estimated_document_count(self):
        """PyMongo: collection.estimated_document_count() — no filter support"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.estimated_document_count()
        cmds = self.capture.stop_capture()
        # estimated_document_count uses the 'count' command without a query filter
        count_cmds = [c for c in cmds if c["command_name"] == "count"]
        if count_cmds:
            cmd = count_cmds[-1]["command"]
            q = cmd.get("query", {})
            has_tenant = q.get("tenant_id") == TEST_TENANT_ID
            self.results["pymongo_estimated_document_count"] = {
                "status": "PASS" if has_tenant else "WARN",
                "reason": f"estimated_document_count query: {q}. "
                + (
                    "tenant_id injected"
                    if has_tenant
                    else "No query param — count cmd gets tenant_id but result is still global estimate"
                ),
            }
        else:
            self.results["pymongo_estimated_document_count"] = {
                "status": "WARN",
                "reason": f"No count command captured. Got: {[c['command_name'] for c in cmds]}",
            }

    async def test_pymongo_distinct(self):
        """PyMongo: collection.distinct(key, filter)"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.distinct("status", {"user_id": "u1"})
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_distinct", cmds, "distinct")

    async def test_pymongo_find_one_and_update(self):
        """PyMongo: collection.find_one_and_update() — findAndModify command"""
        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.find_one_and_update(
            {"user_id": "u1"}, {"$set": {"score": 999}}
        )
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_find_one_and_update", cmds, "findAndModify")

    async def test_pymongo_find_one_and_delete(self):
        """PyMongo: collection.find_one_and_delete()"""
        # Insert a disposable doc
        collection = TestDoc.get_pymongo_collection()
        await collection.insert_one({"user_id": "u_fad", "name": "FindAndDel"})

        self.capture.start_capture()
        await collection.find_one_and_delete({"user_id": "u_fad"})
        cmds = self.capture.stop_capture()
        self._check_command("pymongo_find_one_and_delete", cmds, "findAndModify")

    async def test_pymongo_find_one_and_replace(self):
        """PyMongo: collection.find_one_and_replace()"""
        collection = TestDoc.get_pymongo_collection()
        await collection.insert_one({"user_id": "u_far", "name": "FindAndReplace"})

        self.capture.start_capture()
        await collection.find_one_and_replace(
            {"user_id": "u_far"},
            {"user_id": "u_far", "name": "Replaced", "status": "new"},
        )
        cmds = self.capture.stop_capture()

        def check_far(cmd):
            q = cmd.get("query", {})
            update = cmd.get("update", {})
            q_ok = q.get("tenant_id") == TEST_TENANT_ID
            # replacement doc (no $ operators) should get tenant_id
            u_ok = update.get("tenant_id") == TEST_TENANT_ID
            if not q_ok:
                return False, f"query missing tenant_id: {q}"
            if not u_ok:
                return False, f"replacement missing tenant_id: {update}"
            return True, "tenant_id in both query and replacement"

        self._check_command(
            "pymongo_find_one_and_replace", cmds, "findAndModify", check_far
        )

    # ---- Excluded Collection Test ----

    async def test_excluded_collection_passthrough(self):
        """Verify excluded collections are NOT intercepted."""
        db = self.client[TEST_DB_NAME]
        excluded_coll = db["__excluded_test__"]

        self.capture.start_capture()
        await excluded_coll.insert_one(
            {"user_id": "excluded", "name": "ShouldNotHaveTenant"}
        )
        cmds = self.capture.stop_capture()

        insert_cmds = [c for c in cmds if c["command_name"] == "insert"]
        if insert_cmds:
            doc = insert_cmds[-1]["command"].get("documents", [{}])[0]
            has_tenant = "tenant_id" in doc
            self.results["excluded_collection_passthrough"] = {
                "status": "PASS" if not has_tenant else "FAIL",
                "reason": (
                    "Excluded collection correctly skipped"
                    if not has_tenant
                    else f"tenant_id should NOT be in excluded collection doc: {doc}"
                ),
            }
        else:
            self.results["excluded_collection_passthrough"] = {
                "status": "FAIL",
                "reason": "No insert command captured",
            }

        # Cleanup
        await excluded_coll.drop()

    # ---- No Tenant Context Test ----

    async def test_no_tenant_passthrough(self):
        """Verify commands pass through unmodified when no tenant context."""
        clear_current_tenant()

        collection = TestDoc.get_pymongo_collection()
        self.capture.start_capture()
        await collection.find_one({"user_id": "no_tenant"})
        cmds = self.capture.stop_capture()

        find_cmds = [c for c in cmds if c["command_name"] == "find"]
        if find_cmds:
            f = find_cmds[-1]["command"].get("filter", {})
            has_tenant = "tenant_id" in f
            self.results["no_tenant_passthrough"] = {
                "status": "PASS" if not has_tenant else "FAIL",
                "reason": (
                    "No tenant_id when context empty"
                    if not has_tenant
                    else f"Unexpected tenant_id in filter: {f}"
                ),
            }
        else:
            self.results["no_tenant_passthrough"] = {
                "status": "FAIL",
                "reason": "No find command captured",
            }

        # Restore tenant context for remaining tests
        tenant_info = TenantInfo(
            tenant_id=TEST_TENANT_ID,
            tenant_detail=TenantDetail(tenant_info={}, storage_info={}),
        )
        set_current_tenant(tenant_info)

    # ==================== Run All ====================

    async def run_all(self):
        """Execute all tests and print report."""
        await self.setup()

        test_methods = [
            m for m in dir(self) if m.startswith("test_") and callable(getattr(self, m))
        ]
        test_methods.sort()

        for method_name in test_methods:
            method = getattr(self, method_name)
            try:
                await method()
            except Exception as e:
                self.results[method_name] = {
                    "status": "ERROR",
                    "reason": f"{type(e).__name__}: {e}",
                }

        await self.teardown()
        self._print_report()

    def _print_report(self):
        """Print test results summary."""
        print("\n" + "=" * 80)
        print("  TenantCommandInterceptor Coverage Report")
        print("=" * 80)

        pass_count = 0
        fail_count = 0
        warn_count = 0
        error_count = 0

        for name, result in sorted(self.results.items()):
            status = result["status"]
            reason = result["reason"]

            if status == "PASS":
                icon = "✅"
                pass_count += 1
            elif status == "WARN":
                icon = "⚠️ "
                warn_count += 1
            elif status == "ERROR":
                icon = "💥"
                error_count += 1
            else:
                icon = "❌"
                fail_count += 1

            print(f"  {icon} {name}")
            print(f"      {reason}")

        print("\n" + "-" * 80)
        print(
            f"  Total: {len(self.results)} | PASS: {pass_count} | WARN: {warn_count} | FAIL: {fail_count} | ERROR: {error_count}"
        )

        if fail_count == 0 and error_count == 0:
            print("  Result: ALL OPERATIONS INTERCEPTED SUCCESSFULLY")
        else:
            print("  Result: SOME OPERATIONS NOT INTERCEPTED — SEE ABOVE")
        print("=" * 80 + "\n")


# ==================== Entry Point ====================


async def main():
    runner = InterceptorTestRunner()
    await runner.run_all()


if __name__ == "__main__":
    asyncio.run(main())
