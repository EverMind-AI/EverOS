"""
Seed mock vendor history into EverMemOS for the AP Memory Agent demo.

Two modes:
  --direct (default): Insert pre-built episodic memories directly into MongoDB + ES.
                      No LLM credits needed. Fast and reliable.
  --via-api:          POST messages through the EverMemOS API pipeline (requires LLM).

Usage:
    python seed_memory.py           # Direct seed (recommended)
    python seed_memory.py --clear   # Clear first, then seed
    python seed_memory.py --via-api # Seed through API pipeline (needs LLM credits)

Requires: EVERMEM_GROUP_ID=eb6618c4d52d3bf9_group in .env (matches ap_agent_system)
"""

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("EVERMEM_BASE_URL", "http://localhost:1995")
SEED_JSON = Path(__file__).parent / "data" / "ap_agent_seed.json"
GROUP_ID = "eb6618c4d52d3bf9_group"  # hash(ap_agent_system)_group
GROUP_NAME = "AP Agent Vendor History"

# Delay between messages (seconds) - for --via-api mode
MESSAGE_DELAY_SEC = 0.5

# Common English stopwords to filter from search_content
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "shall", "should", "may", "might", "must", "can", "could",
    "not", "no", "nor", "so", "if", "then", "than", "that", "this",
    "these", "those", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "we", "our", "you", "your", "my", "am", "up", "out",
}


def _tokenize_for_search(*texts: str) -> list[str]:
    """Tokenize text into search terms (mimics jieba + stopword filtering for English)."""
    combined = " ".join(t for t in texts if t)
    words = re.findall(r"[a-zA-Z0-9]+(?:[-][a-zA-Z0-9]+)*", combined)
    return [w.lower() for w in words if len(w) >= 2 and w.lower() not in _STOPWORDS]


# ─────────────────────────────────────────────
# PRE-BUILT EPISODIC MEMORIES (no LLM needed)
# ─────────────────────────────────────────────

EPISODIC_MEMORIES = [
    {
        "subject": "Acme Corp Invoices INV-001 to INV-005 Approved — Clean Payment History January 2025",
        "episode": (
            "On January 15, 2025 at 10:00 AM UTC, the user inquired about the status of "
            "multiple Acme Corp invoices. Invoice INV-001 for $500 was approved with Net 30 terms "
            "and paid on time, covering office supplies. Invoice INV-002 for $450 was approved "
            "with Net 30 terms and paid within terms for software licenses. Invoices INV-003 ($550), "
            "INV-004 ($480), and INV-005 ($520) were all approved with Net 30 terms and clean "
            "payment history. Acme Corp is assessed as low-risk: 5 invoices approved in the past "
            "6 months, amounts ranging $400-$600, with no disputes or late payments."
        ),
        "timestamp": "2025-01-15T10:02:30",
        "participants": ["ap_agent_user", "ap_agent_system"],
    },
    {
        "subject": "Globex Supplies Invoices INV-101 to INV-104 — Dispute and Late Payment History January 2025",
        "episode": (
            "On January 16, 2025 at 10:05 AM UTC, the user inquired about Globex Supplies "
            "invoices INV-101 and INV-102. The assistant confirmed INV-101 for $1,200 was approved "
            "with Net 30 terms and paid on time, and INV-102 for $980 was similarly approved and "
            "paid on time. The user then asked about any disputes with Globex Supplies. The "
            "assistant reported that Invoice INV-103 for $2,000 was disputed in March 2025 due to "
            "shipment delay, resolved by issuing partial credit. The user asked about late payments "
            "from Globex Supplies. The assistant stated that INV-104 for $1,500 was paid 15 days "
            "late under Net 30 terms. Globex Supplies has a prior dispute and late payment history, "
            "assessed as moderate risk."
        ),
        "timestamp": "2025-01-16T10:07:30",
        "participants": ["ap_agent_user", "ap_agent_system"],
    },
    {
        "subject": "Shadow LLC Duplicate Invoice INV-777 Rejected — High Risk Vendor January 2025",
        "episode": (
            "On January 17, 2025 at 10:10 AM UTC, the user inquired about the status of "
            "invoice INV-777 from Shadow LLC. The assistant reported that invoice INV-777 for "
            "$3,000 was rejected due to a duplicate invoice number — the same invoice number was "
            "submitted twice. Shadow LLC was flagged for the duplicate attempt and assessed as "
            "high risk. The assistant advised to always verify invoice numbers against history "
            "when processing Shadow LLC invoices."
        ),
        "timestamp": "2025-01-17T10:10:30",
        "participants": ["ap_agent_user", "ap_agent_system"],
    },
]


def clear_group_data():
    """Clear all MongoDB and ES data for the group."""
    from pymongo import MongoClient as _MongoClient

    print("  Clearing MongoDB...")
    _client = _MongoClient(
        "mongodb://admin:memsys123@localhost:27017/memsys?authSource=admin"
    )
    _db = _client["memsys"]
    for coll_name in _db.list_collection_names():
        r = _db[coll_name].delete_many({"group_id": GROUP_ID})
        if r.deleted_count > 0:
            print(f"    {coll_name}: {r.deleted_count} deleted")
    _client.close()

    # Clear ES
    try:
        import requests
        resp = requests.post(
            "http://localhost:19200/episodic-memory-memsys-*/_delete_by_query",
            json={"query": {"term": {"group_id": GROUP_ID}}},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.ok:
            deleted = resp.json().get("deleted", 0)
            if deleted:
                print(f"    Elasticsearch: {deleted} deleted")
    except Exception:
        pass
    print("  Cleared.")


def seed_direct(clear_first: bool):
    """Insert pre-built episodic memories directly into MongoDB + Elasticsearch."""
    from pymongo import MongoClient as _MongoClient

    if clear_first:
        clear_group_data()

    # 1. Save conversation meta via API
    print("  Saving conversation-meta via API...")
    import requests
    meta_payload = {
        "version": "1.0",
        "scene": "assistant",
        "scene_desc": {"description": "AP Agent vendor history for demo"},
        "name": GROUP_NAME,
        "description": "Vendor invoice history: Acme Corp (clean), Globex Supplies (dispute), Shadow LLC (duplicate).",
        "group_id": GROUP_ID,
        "created_at": "2025-01-15T00:00:00Z",
        "default_timezone": "UTC",
        "user_details": {
            "ap_agent_user": {"full_name": "AP User", "role": "user", "extra": {}},
            "ap_agent_system": {"full_name": "AP Agent", "role": "assistant", "extra": {}},
        },
        "tags": ["AP", "Vendor", "Invoice"],
    }
    try:
        resp = requests.post(
            f"{BASE_URL}/api/v1/memories/conversation-meta",
            json=meta_payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code == 200:
            print("  Conversation-meta saved.")
        else:
            print(f"  Warning: conversation-meta HTTP {resp.status_code}")
    except Exception as e:
        print(f"  Warning: conversation-meta failed: {e}")

    # 2. Insert episodic memories directly into MongoDB
    print("  Inserting episodic memories into MongoDB...")
    _client = _MongoClient(
        "mongodb://admin:memsys123@localhost:27017/memsys?authSource=admin"
    )
    _db = _client["memsys"]
    now = datetime.now(timezone.utc)

    inserted = 0
    for mem in EPISODIC_MEMORIES:
        ts = datetime.fromisoformat(mem["timestamp"].replace("Z", "+00:00"))
        # Insert group episode + personal episodes (for each participant)
        for user_id in [None] + mem["participants"]:
            doc = {
                "created_at": now,
                "updated_at": now,
                "deleted_at": None,
                "deleted_by": None,
                "deleted_id": 0,
                "user_id": user_id,
                "memory_type": "episodic_memory",
                "subject": mem["subject"],
                "summary": mem["episode"][:200],
                "episode": mem["episode"],
                "timestamp": ts,
                "group_id": GROUP_ID,
                "group_name": GROUP_NAME,
                "participants": mem["participants"],
                "type": "Conversation",
                "extend": None,
                "memcell_event_id_list": [],
                "ori_event_id_list": [],
                "keywords": [],
            }
            _db.episodic_memories.insert_one(doc)
            inserted += 1
    _client.close()
    print(f"  Inserted {inserted} episodic memories ({len(EPISODIC_MEMORIES)} episodes x 3 copies).")

    # 3. Index into Elasticsearch
    print("  Indexing into Elasticsearch...")
    try:
        _client2 = _MongoClient(
            "mongodb://admin:memsys123@localhost:27017/memsys?authSource=admin"
        )
        _db2 = _client2["memsys"]
        es_docs = []
        for doc in _db2.episodic_memories.find({"group_id": GROUP_ID}):
            doc_id = str(doc["_id"])
            subject = doc.get("subject", "")
            summary = doc.get("summary", "")
            episode = doc.get("episode", "")
            search_content = _tokenize_for_search(subject, summary, episode)
            es_doc = {
                "_id": doc_id,
                "event_id": doc_id,
                "search_content": search_content,
                "title": subject,
                "subject": subject,
                "summary": summary,
                "episode": episode,
                "group_id": doc.get("group_id", ""),
                "group_name": doc.get("group_name", ""),
                "user_id": doc.get("user_id"),
                "participants": doc.get("participants", []),
                "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None,
                "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
                "type": doc.get("type", "Conversation"),
            }
            es_docs.append(es_doc)

        # Find the ES index name
        resp = requests.get("http://localhost:19200/_cat/indices?format=json", timeout=5)
        es_index = None
        if resp.ok:
            for idx in resp.json():
                if "episodic-memory-memsys" in idx.get("index", ""):
                    es_index = idx["index"]
                    break

        if es_index and es_docs:
            # Bulk index
            bulk_body = ""
            for es_doc in es_docs:
                doc_id = es_doc.pop("_id")
                bulk_body += json.dumps({"index": {"_index": es_index, "_id": doc_id}}) + "\n"
                bulk_body += json.dumps(es_doc) + "\n"
            resp = requests.post(
                f"http://localhost:19200/_bulk",
                data=bulk_body,
                headers={"Content-Type": "application/x-ndjson"},
                timeout=10,
            )
            if resp.ok:
                result = resp.json()
                errors = result.get("errors", False)
                print(f"  ES indexed {len(es_docs)} docs (errors={errors}).")
            else:
                print(f"  ES bulk index failed: HTTP {resp.status_code}")
        else:
            print(f"  Warning: ES index not found or no docs to index.")
        _client2.close()
    except Exception as e:
        print(f"  Warning: ES indexing failed: {e}")

    print(f"\nDone. {len(EPISODIC_MEMORIES)} vendor episodes seeded.")
    print("Run: uv run python demo/ap-memory-agent/test_demo.py")


# ─────────────────────────────────────────────
# ORIGINAL API-BASED SEEDING (requires LLM)
# ─────────────────────────────────────────────

def load_conversation() -> tuple:
    """Load conversation from JSON. Returns (messages, group_id, group_name, meta)."""
    if not SEED_JSON.exists():
        raise FileNotFoundError(f"Seed data not found: {SEED_JSON}")
    with open(SEED_JSON, encoding="utf-8") as f:
        data = json.load(f)
    messages = data.get("conversation_list", [])
    meta = data.get("conversation_meta", {})
    group_id = meta.get("group_id", GROUP_ID)
    group_name = meta.get("name", "AP Agent Vendor History")
    user_details = meta.get("user_details", {})
    for msg in messages:
        msg["group_id"] = group_id
        msg["group_name"] = group_name
        sender = msg.get("sender")
        if sender and sender in user_details:
            msg["role"] = user_details[sender].get("role", "user")
        else:
            msg["role"] = "assistant" if "system" in str(sender or "").lower() else "user"
    return messages, group_id, group_name, meta


async def upsert_conversation_meta(
    client: httpx.AsyncClient,
    meta: dict,
    messages: list,
    group_id: str,
    group_name: str,
) -> None:
    """Save conversation-meta (required for extraction scene)."""
    created_at = meta.get("created_at") or (
        messages[0].get("create_time") if messages else None
    ) or datetime.now(timezone.utc).isoformat()

    user_details = meta.get("user_details") or {}
    if not user_details:
        for m in messages:
            s = m.get("sender")
            if s:
                user_details[s] = {
                    "full_name": m.get("sender_name", s),
                    "role": "user" if m.get("role") == "user" else "assistant",
                    "extra": {},
                }

    payload = {
        "version": meta.get("version", "1.0"),
        "scene": "assistant",
        "scene_desc": meta.get("scene_desc", {}),
        "name": group_name,
        "description": meta.get("description", ""),
        "group_id": group_id,
        "created_at": created_at,
        "default_timezone": meta.get("default_timezone", "UTC"),
        "user_details": user_details,
        "tags": meta.get("tags", []),
    }

    resp = await client.post(
        f"{BASE_URL}/api/v1/memories/conversation-meta",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    if resp.status_code != 200:
        print(f"  Warning: conversation-meta failed HTTP {resp.status_code}")
    else:
        print("  Conversation-meta saved.")


async def seed_messages(
    client: httpx.AsyncClient,
    messages: list,
    clear_first: bool,
) -> None:
    """POST each message through the API pipeline (requires LLM credits)."""
    if clear_first:
        clear_group_data()

    url = f"{BASE_URL}/api/v1/memories?sync_mode=false"
    extracted = 0
    accumulated = 0

    for idx, msg in enumerate(messages, 1):
        try:
            resp = await client.post(
                url, json=msg, headers={"Content-Type": "application/json"}, timeout=600
            )
            if resp.status_code == 200:
                result = resp.json()
                status = result.get("result", {}).get("status_info", "")
                count = result.get("result", {}).get("count", 0)
                if count > 0 or "extracted" in status.lower():
                    extracted += 1
                    print(f"  [{idx}/{len(messages)}] Extracted")
                else:
                    accumulated += 1
                    print(f"  [{idx}/{len(messages)}] Queued")
            elif resp.status_code == 202:
                extracted += 1
                print(f"  [{idx}/{len(messages)}] Processing (202)")
            else:
                print(f"  [{idx}/{len(messages)}] HTTP {resp.status_code}")
        except httpx.ReadTimeout:
            print(f"  [{idx}/{len(messages)}] Timeout (continuing)")
        except Exception as e:
            print(f"  [{idx}/{len(messages)}] Error: {e}")

        if idx < len(messages):
            await asyncio.sleep(MESSAGE_DELAY_SEC)

    print(f"\n  Summary: {extracted} extracted, {accumulated} queued")


async def main_api(clear: bool) -> None:
    messages, group_id, group_name, meta = load_conversation()
    print(f"Loaded {len(messages)} messages from {SEED_JSON}")
    print(f"Group ID: {group_id}")

    async with httpx.AsyncClient(timeout=600) as client:
        await upsert_conversation_meta(client, meta, messages, group_id, group_name)
        await seed_messages(client, messages, clear)

    print("\nDone.")
    print("Run: uv run python demo/ap-memory-agent/test_demo.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed vendor history for AP Memory Agent demo")
    parser.add_argument("--clear", action="store_true", help="Delete group memories before seeding")
    parser.add_argument("--via-api", action="store_true", help="Seed via API pipeline (requires LLM credits)")
    args = parser.parse_args()

    if args.via_api:
        print("Seeding via API pipeline (requires LLM credits)...")
        asyncio.run(main_api(args.clear))
    else:
        print("Seeding directly into MongoDB + Elasticsearch (no LLM needed)...")
        seed_direct(args.clear)


if __name__ == "__main__":
    main()
