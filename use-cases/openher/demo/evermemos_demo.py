#!/usr/bin/env python3
"""
OpenHer × EverCore Integration Demo

Demonstrates how EverCore provides long-term memory to the
AI Being persona engine. Shows session context loading, memory
storage, search, and relationship vector evolution.

Usage:
    # With EverCore Cloud
    export EVERMEMOS_BASE_URL=https://api.evermind.ai/v1
    export EVERMEMOS_API_KEY=your_key
    python demo/evermemos_demo.py

    # With self-hosted EverCore
    export EVERMEMOS_BASE_URL=http://localhost:1995/api/v1
    python demo/evermemos_demo.py
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from typing import Optional

# ──────────────────────────────────────────────
# EverCore Client (minimal standalone version)
# ──────────────────────────────────────────────

try:
    import httpx
except ImportError:
    print("❌ httpx not installed. Run: pip install httpx")
    sys.exit(1)


class EverCoreClient:
    """Minimal EverCore client for demo purposes."""

    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=10.0)
        self.available = bool(base_url)

    async def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def health_check(self) -> bool:
        """Check if EverCore is reachable."""
        try:
            # Try the health endpoint (remove /api/v1 suffix)
            health_url = self.base_url.replace("/api/v1", "") + "/health"
            resp = await self._client.get(health_url, headers=await self._headers())
            return resp.status_code == 200
        except Exception:
            return False

    async def store_turn(
        self,
        user_id: str,
        persona_id: str,
        persona_name: str,
        user_name: str,
        group_id: str,
        user_message: str,
        agent_reply: str,
    ) -> dict:
        """Store a conversation turn as memory."""
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S+00:00")
        messages = [
            {
                "message_id": f"msg_{hash(user_message) & 0xFFFF:04x}_u",
                "create_time": now,
                "sender": user_id,
                "sender_name": user_name,
                "content": user_message,
            },
            {
                "message_id": f"msg_{hash(agent_reply) & 0xFFFF:04x}_a",
                "create_time": now,
                "sender": persona_id,
                "sender_name": persona_name,
                "content": agent_reply,
            },
        ]
        resp = await self._client.post(
            f"{self.base_url}/memories",
            json={"messages": messages, "group_id": group_id},
            headers=await self._headers(),
        )
        return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def search(
        self,
        query: str,
        user_id: str,
        group_id: str,
        top_k: int = 5,
    ) -> dict:
        """Search for relevant memories."""
        resp = await self._client.get(
            f"{self.base_url}/memories/search",
            params={
                "query": query,
                "user_id": user_id,
                "group_id": group_id,
                "top_k": top_k,
                "retrieve_method": "hybrid",
            },
            headers=await self._headers(),
        )
        return resp.json() if resp.status_code == 200 else {"error": resp.text}

    async def get_user_profile(self, user_id: str) -> dict:
        """Get user profile (accumulated from conversations)."""
        resp = await self._client.get(
            f"{self.base_url}/users/{user_id}/profile",
            headers=await self._headers(),
        )
        return resp.json() if resp.status_code == 200 else {}

    async def close(self):
        await self._client.aclose()


# ──────────────────────────────────────────────
# Relationship Vector (from EverCore session)
# ──────────────────────────────────────────────

def compute_relationship_vector(profile_data: dict) -> dict:
    """
    Extract 4D relationship vector from EverCore profile data.

    These 4 dimensions expand the persona engine's neural network
    from 8D to 12D input, allowing it to differentiate behavior
    between strangers and old friends.
    """
    return {
        "relationship_depth": min(1.0, profile_data.get("interaction_count", 0) / 50),
        "emotional_valence": profile_data.get("sentiment_avg", 0.0),
        "trust_level": min(1.0, profile_data.get("trust_score", 0.0)),
        "pending_foresight": 1.0 if profile_data.get("foresight") else 0.0,
    }


def apply_relationship_ema(
    prior: dict,
    delta: dict,
    conversation_depth: float,
    prev_ema: Optional[dict] = None,
) -> dict:
    """
    Semi-emergent relationship update (Step 2.5 of ChatAgent lifecycle).

    Blends EverCore prior with LLM-judged delta through EMA:
      - alpha modulated by conversation depth (deeper = trust LLM more)
      - Clips to valid ranges
      - Preserves momentum through prev_ema
    """
    if prev_ema is None:
        prev_ema = dict(prior)

    alpha = max(0.15, min(0.65, 0.15 + 0.5 * conversation_depth))

    ema = {}
    for k in prior:
        lo = -1.0 if k == "emotional_valence" else 0.0
        posterior = max(lo, min(1.0, prior[k] + delta.get(k, 0.0)))
        prev = prev_ema.get(k, prior[k])
        ema[k] = round(alpha * posterior + (1 - alpha) * prev, 4)

    return ema


# ──────────────────────────────────────────────
# Demo
# ──────────────────────────────────────────────

async def main():
    base_url = os.getenv("EVERMEMOS_BASE_URL", "")
    api_key = os.getenv("EVERMEMOS_API_KEY", "")

    if not base_url:
        print("=" * 60)
        print("OpenHer × EverCore Integration Demo")
        print("=" * 60)
        print()
        print("⚠️  EVERMEMOS_BASE_URL not set.")
        print()
        print("To run this demo, set up EverCore:")
        print()
        print("  Option A — Cloud:")
        print("    export EVERMEMOS_BASE_URL=https://api.evermind.ai/v1")
        print("    export EVERMEMOS_API_KEY=your_key")
        print()
        print("  Option B — Self-hosted:")
        print("    cd vendor/EverCore && docker compose up -d")
        print("    uv run python src/run.py")
        print("    export EVERMEMOS_BASE_URL=http://localhost:1995/api/v1")
        print()
        print("Get your API key: https://console.evermind.ai/")
        print()
        print("Running in simulation mode...\n")
        await demo_simulation()
        return

    client = EverCoreClient(base_url, api_key)

    print("=" * 60)
    print("OpenHer × EverCore Integration Demo")
    print("=" * 60)
    print(f"\n📡 EverCore: {base_url}")

    # Health check
    healthy = await client.health_check()
    if not healthy:
        print("❌ EverCore is not reachable. Check your URL and try again.")
        await client.close()
        return
    print("✅ EverCore is healthy\n")

    # ── Demo conversation ──
    user_id = "demo_user"
    persona_id = "luna"
    persona_name = "Luna (陆暖)"
    user_name = "Demo User"
    group_id = f"{persona_id}__{user_id}"

    conversations = [
        ("My name is Alex, I'm a software engineer", "Nice to meet you Alex! What kind of software do you work on?"),
        ("I love hiking in the mountains on weekends", "That sounds wonderful! There's something about being up high that makes everything else feel small."),
        ("I drink my coffee black, no sugar", "Noted! A purist. I respect that."),
    ]

    print("📝 Storing conversation memories...\n")
    for user_msg, agent_reply in conversations:
        result = await client.store_turn(
            user_id=user_id,
            persona_id=persona_id,
            persona_name=persona_name,
            user_name=user_name,
            group_id=group_id,
            user_message=user_msg,
            agent_reply=agent_reply,
        )
        status = "✅" if "error" not in result else "❌"
        print(f"  {status} User: \"{user_msg[:50]}...\"")

    # Wait for indexing
    print("\n⏳ Waiting for memory indexing (3s)...")
    await asyncio.sleep(3)

    # Search
    print("\n🔍 Searching for relevant memories...\n")
    queries = [
        "What does Alex like to do on weekends?",
        "How does Alex take their coffee?",
        "What is Alex's occupation?",
    ]

    for query in queries:
        result = await client.search(
            query=query,
            user_id=user_id,
            group_id=group_id,
        )
        memories = result.get("result", {}).get("memories", [])
        print(f"  Q: \"{query}\"")
        if memories:
            for mem in memories[:2]:
                content = str(mem)[:100]
                print(f"     → {content}")
        else:
            print("     → (no results yet — indexing may still be in progress)")
        print()

    # Relationship vector
    print("📊 Relationship Vector Evolution:\n")
    prior = {"relationship_depth": 0.0, "emotional_valence": 0.0, "trust_level": 0.0, "pending_foresight": 0.0}
    deltas = [
        {"relationship_depth": 0.1, "emotional_valence": 0.2, "trust_level": 0.05},
        {"relationship_depth": 0.05, "emotional_valence": 0.1, "trust_level": 0.1},
        {"relationship_depth": 0.08, "emotional_valence": 0.15, "trust_level": 0.12},
    ]

    ema = None
    for i, delta in enumerate(deltas):
        ema = apply_relationship_ema(prior, delta, conversation_depth=0.2 * (i + 1), prev_ema=ema)
        print(f"  Turn {i+1}: depth={ema['relationship_depth']:.3f} "
              f"valence={ema['emotional_valence']:.3f} "
              f"trust={ema['trust_level']:.3f}")
        prior = ema

    print(f"\n  → After 3 turns: no longer a stranger (depth={ema['relationship_depth']:.3f})")
    print(f"  → Neural network now produces warmer, more familiar behavioral signals\n")

    await client.close()
    print("✅ Demo complete!")


async def demo_simulation():
    """Run demo in simulation mode (no EverCore connection)."""
    print("📊 Simulating Relationship Vector Evolution:\n")
    print("   This shows how the 4D EverCore relationship vector")
    print("   deepens over multiple conversation turns.\n")

    prior = {"relationship_depth": 0.0, "emotional_valence": 0.0, "trust_level": 0.0, "pending_foresight": 0.0}

    # Simulate 10 turns of conversation
    simulated_deltas = [
        (0.3, {"relationship_depth": 0.10, "emotional_valence": 0.15, "trust_level": 0.05}),
        (0.4, {"relationship_depth": 0.08, "emotional_valence": 0.10, "trust_level": 0.08}),
        (0.5, {"relationship_depth": 0.05, "emotional_valence": 0.20, "trust_level": 0.12}),
        (0.6, {"relationship_depth": 0.06, "emotional_valence": -0.10, "trust_level": 0.03}),
        (0.7, {"relationship_depth": 0.04, "emotional_valence": 0.08, "trust_level": 0.10}),
        (0.7, {"relationship_depth": 0.03, "emotional_valence": 0.12, "trust_level": 0.08}),
        (0.8, {"relationship_depth": 0.02, "emotional_valence": 0.05, "trust_level": 0.06}),
        (0.8, {"relationship_depth": 0.03, "emotional_valence": 0.10, "trust_level": 0.05}),
        (0.9, {"relationship_depth": 0.01, "emotional_valence": 0.08, "trust_level": 0.04}),
        (0.9, {"relationship_depth": 0.02, "emotional_valence": 0.06, "trust_level": 0.03}),
    ]

    ema = None
    for i, (depth, delta) in enumerate(simulated_deltas, 1):
        alpha = max(0.15, min(0.65, 0.15 + 0.5 * depth))
        ema = apply_relationship_ema(prior, delta, conversation_depth=depth, prev_ema=ema)
        bar_d = "█" * int(ema["relationship_depth"] * 20)
        bar_v = "█" * int(max(0, ema["emotional_valence"]) * 20)
        bar_t = "█" * int(ema["trust_level"] * 20)
        print(f"  Turn {i:2d} (α={alpha:.2f}): "
              f"depth={ema['relationship_depth']:.3f} {bar_d}")
        print(f"                     "
              f"valence={ema['emotional_valence']:+.3f} {bar_v}")
        print(f"                     "
              f"trust={ema['trust_level']:.3f} {bar_t}")
        print()
        prior = ema

    print("  ──────────────────────────────────")
    print(f"  Final state: depth={ema['relationship_depth']:.3f}, "
          f"valence={ema['emotional_valence']:+.3f}, "
          f"trust={ema['trust_level']:.3f}")
    print()
    print("  Turn 4 shows a negative emotional event (valence delta = -0.10),")
    print("  but the EMA smoothing prevents overreaction — the relationship")
    print("  continues to deepen because trust was already building.")
    print()
    print("  This vector feeds into the 25D neural network input,")
    print("  producing different behavioral signals for strangers vs. friends:")
    print("  - Higher warmth, vulnerability, and initiative for trusted users")
    print("  - More guarded, formal signals for new users")
    print()
    print("✅ Simulation complete!")


if __name__ == "__main__":
    asyncio.run(main())
