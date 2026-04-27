import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
import httpx
from api_specs.memory_types import ScenarioType
from demo.tools.clear_all_data import clear_all_memories
from common_utils.language_utils import get_prompt_language


def load_conversation_data(file_path: str) -> tuple:
    """Load conversation data from JSON file

    Returns:
        tuple: (messages, group_id, group_name, session_meta)
    """
    data_file = Path(file_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract message list and metadata
    messages = data.get('conversation_list', [])
    session_meta = data.get('session_meta', {})
    group_id = session_meta.get('group_id', 'unknown_group')
    group_name = session_meta.get('name', 'unknown')

    print(f"Loaded {len(messages)} messages from {file_path}")
    print(f"group_id: {group_id}")
    print(f"group_name: {group_name}")

    return messages, group_id, group_name, session_meta


def _parse_create_time_to_unix_ms(create_time: str) -> int:
    """Convert ISO 8601 create_time string to unix milliseconds."""
    dt = datetime.fromisoformat(create_time.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def _infer_role(sender: str, session_meta: dict) -> str:
    """Infer role (user/assistant) from sender id and session_meta."""
    user_details = session_meta.get("user_details", {})
    detail = user_details.get(sender, {})
    role = detail.get("role", "")
    if role in ("user", "assistant"):
        return role
    # Heuristic: robot/assistant in sender name
    sender_lower = sender.lower()
    if "robot" in sender_lower or "assistant" in sender_lower or "bot" in sender_lower:
        return "assistant"
    return "user"


def convert_to_v1_message(msg: dict, session_meta: dict) -> dict:
    """Convert old-format message to V1 MessageItem format.

    Old: { message_id, create_time, sender, sender_name, type, content, ... }
    New: { message_id, sender_id, sender_name, role, timestamp, type, text: {content} }
    """
    role = _infer_role(msg.get("sender", ""), session_meta)
    timestamp_ms = _parse_create_time_to_unix_ms(
        msg.get("create_time", "2025-01-01T00:00:00Z")
    )

    return {
        "message_id": msg.get("message_id"),
        "sender_id": msg.get("sender"),
        "sender_name": msg.get("sender_name"),
        "role": role,
        "timestamp": timestamp_ms,
        "type": msg.get("type", "text"),
        "text": {"content": msg.get("content", "")},
    }


async def init_settings(
    client: httpx.AsyncClient, base_url: str, session_meta: dict
) -> None:
    """Initialize global settings via V1 Settings API."""
    payload = {}

    url = f"{base_url}/api/v1/settings"
    resp = await client.put(
        url, json=payload, headers={"Content-Type": "application/json"}
    )
    if resp.status_code != 200:
        print(f"⚠️  Failed to init settings: HTTP {resp.status_code}")
        print(resp.text[:300])
    else:
        print("✓ settings initialized")


def prompt_clear_data() -> bool:
    """Prompt user whether to clear existing data before extraction

    Returns:
        bool: True if user wants to clear data, False otherwise
    """
    print()
    print("=" * 60)
    print("⚠️  Clear existing data before extraction?")
    print("=" * 60)
    print()
    print("This will delete ALL existing memories from:")
    print("  • MongoDB (memcells, episodic_memories, etc.)")
    print("  • Elasticsearch (episodic-memory, atomic-fact, foresight)")
    print("  • Milvus (vector collections)")
    print()

    while True:
        choice = input("Clear all existing data? [Y/N]: ").strip().upper()
        if choice == 'Y':
            print()
            return True
        elif choice == 'N':
            print()
            print("✓ Keeping existing data, will append new memories")
            print()
            return False
        else:
            print("Please enter Y (yes) or N (no)")


async def test_memorize_api():
    """Test V1 API /memories endpoint (single message storage)"""

    # Ask user whether to clear existing data
    should_clear = prompt_clear_data()
    if should_clear:
        await clear_all_memories()

    base_url = "http://localhost:1995"

    print("=" * 100)
    print("🧪 Testing V1 API HTTP Interface - Memory Storage")
    print("=" * 100)

    # Load conversation data based on language setting
    language = get_prompt_language()
    print(f"\n📌 Language setting: MEMORY_LANGUAGE={language}")
    print(
        f"   (Set via environment variable, affects both data file and server prompts)"
    )

    # ===== Scene selection =====
    # SOLO: personal conversation (1 user + assistant)
    # TEAM: group conversation (multiple users), runs with 2 group_ids to verify isolation
    scene = ScenarioType.SOLO.value
    # scene = ScenarioType.TEAM.value

    if language == "zh":
        data_file = "data/solo_chat_zh.json" if scene == ScenarioType.SOLO.value else "data/team_chat_zh.json"
    else:
        data_file = "data/solo_chat_en.json" if scene == ScenarioType.SOLO.value else "data/team_chat_en.json"

    try:
        test_messages, group_id, group_name, session_meta = load_conversation_data(
            data_file
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return False

    is_solo = scene == ScenarioType.SOLO.value

    # For TEAM scene, run the same data with 2 different group_ids to verify multi-group isolation
    if is_solo:
        group_ids = [None]  # personal endpoint uses user_id, not group_id
    else:
        group_ids = [group_id, f"{group_id}_2"]

    # Determine user_id from session_meta (used for SOLO endpoint)
    user_id = None
    for uid, detail in session_meta.get("user_details", {}).items():
        if detail.get("role") == "user":
            user_id = uid
            break
    if not user_id:
        user_id = "user_001"

    async with httpx.AsyncClient(timeout=500.0) as client:
        # Initialize settings
        await init_settings(client=client, base_url=base_url, session_meta=session_meta)

        for gid in group_ids:
            # SOLO uses personal endpoint, TEAM uses group endpoint
            if is_solo:
                memorize_url = f"{base_url}/api/v1/memories?sync_mode=false"
            else:
                memorize_url = f"{base_url}/api/v1/memories/group?sync_mode=false"

            print()
            print("=" * 100)
            if is_solo:
                print(f"📤 [{scene.upper()}] Sending {len(test_messages)} messages")
                print(f"   URL: {memorize_url}")
                print(f"   user_id: {user_id}")
            else:
                print(f"📤 [{scene.upper()}] Sending {len(test_messages)} messages to group: {gid}")
                print(f"   URL: {memorize_url}")
                print(f"   group_id: {gid}")
            print("=" * 100)
            print()
            print("ℹ️  How it works:")
            print("   • Messages accumulate in Redis until boundary condition is met")
            print("   • '⏳ Queued' = Message stored, waiting for boundary detection")
            print("   • '🔄 Processing' = Boundary detected, submitted to background worker")
            print()

            total_accumulated = 0
            total_processing = 0

            for idx, message in enumerate(test_messages, 1):
                print(
                    f"[{idx}/{len(test_messages)}] {message.get('sender', '?')}: {message.get('content', '')[:40]}..."
                )

                # Convert to V1 format
                v1_msg = convert_to_v1_message(message, session_meta)
                if is_solo:
                    payload = {"user_id": user_id, "messages": [v1_msg]}
                else:
                    payload = {"group_id": gid, "messages": [v1_msg]}

                try:
                    response = await client.post(
                        memorize_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )

                    if response.status_code == 200:
                        result = response.json()
                        data = result.get("data", {})
                        status = data.get("status_info") or data.get("status", "unknown")
                        saved_count = data.get("count", 0)

                        if status in ("accumulated", "queued"):
                            total_accumulated += 1
                            print(f"   ⏳ Queued")
                        elif status in ("extracted", "processing"):
                            total_processing += 1
                            print(f"   ✅ Extracted {saved_count} memories")
                        else:
                            print(f"   ✗ Unexpected status: {status}")
                            print(f"      Response: {response.text}")
                    elif response.status_code == 202:
                        result = response.json()
                        total_processing += 1
                        request_id = result.get("request_id", "")
                        print(f"   🔄 Processing (request_id: {request_id[:8]})")
                    else:
                        print(f"   ✗ Failed: HTTP {response.status_code}")
                        print(f"      {response.text[:200]}")

                except httpx.ConnectError:
                    print(f"   ✗ Connection failed: Unable to connect to {base_url}")
                    print(f"      Ensure V1 API service is running:")
                    print(f"      uv run python src/bootstrap.py src/run.py")
                    return False
                except httpx.ReadTimeout:
                    print(f"   ⚠ Timeout: Processing exceeded 500s")
                    print(f"      Skipping message and continuing...")
                    continue
                except Exception as e:
                    print(f"   ✗ Error: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    return False

            effective_id = user_id if is_solo else gid
            print(f"\n📊 [{effective_id}] Summary:")
            print(f"   Total messages:    {len(test_messages)}")
            print(f"   Queued:            {total_accumulated}")
            print(f"   Processing:        {total_processing}")

            if total_processing > 0:
                print("\n🔄 Background processing in progress:")
                print("   • MemCells are being extracted and saved by background workers")
                print("   • Episode memories, foresights, and atomic facts are being generated")
                print("   • Check worker logs for progress")

    if not is_solo and len(group_ids) > 1:
        print("\n" + "=" * 100)
        print(f"✓ TEAM data sent to {len(group_ids)} group_ids: {group_ids}")
        print("  Profiles will be extracted independently for each group.")
        print("  Use test_v1api_search.py or the E2E test to verify isolation.")
        print("=" * 100)

    print("\n📝 Next steps:")
    print("   Run chat demo: uv run python src/bootstrap.py demo/chat_with_memory.py")
    print("=" * 100)

    return True


if __name__ == "__main__":
    asyncio.run(test_memorize_api())
