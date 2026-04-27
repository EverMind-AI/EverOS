"""
End-to-End Test: Unified Profile Extraction (SOLO + GROUP)

Tests the full pipeline after the "unify GROUP/SOLO profile extraction" refactor:
1. Send 5 conversation rounds (each flush creates 1 memcell)
2. Profile extraction triggers at memcell #5 (interval=5)
3. Verify profile is created in new unified format (explicit_info + implicit_traits)
4. Verify profile is searchable and retrievable
5. Verify old GROUP format fields are absent

Prerequisites:
    Service running on http://localhost:1995
    (tmux attach -t EverOS)

Run:
    cd /Users/admin/Documents/Projects/evermemos-opensource
    PYTHONPATH=src .venv/bin/python tests/test_profile_e2e.py
"""

import asyncio
import json
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

# ─── Config ───────────────────────────────────────────────────────────
BASE_URL = "http://localhost:1995"
TIMEOUT = httpx.Timeout(180.0)

# Use unique IDs to avoid collision with existing data
TEST_USER_ID = "e2e_profile_test_user_005"
TEST_GROUP_ID_GROUP = "e2e_profile_test_group_005a"
TEST_GROUP_USER_A = "e2e_group_user_charlie_5"
TEST_GROUP_USER_B = "e2e_group_user_diana_5"

# Second group (different topic & users) for multi-group isolation test
TEST_GROUP_ID_GROUP2 = "e2e_profile_test_group_005b"
TEST_GROUP2_USER_A = "e2e_group_user_evan_5"
TEST_GROUP2_USER_B = "e2e_group_user_fiona_5"

# Profile extraction interval from memorize_config.py
PROFILE_EXTRACTION_INTERVAL = 5

# Old GROUP format fields that should NOT exist in the new profile
OLD_GROUP_FIELDS = [
    "hard_skills", "soft_skills", "personality", "interests",
    "values", "communication_style", "decision_making",
    "work_style", "emotional_traits", "social_behavior",
    "knowledge_domains", "life_experiences", "goals_aspirations",
    "cultural_background",
]


# ─── Helpers ──────────────────────────────────────────────────────────

def ts_ms(round_idx: int, offset_minutes: int = 0) -> int:
    """Generate unix timestamp in milliseconds. Each round is 1 day apart."""
    dt = datetime(2026, 3, 20, 10, 0, 0, tzinfo=timezone.utc) + timedelta(days=round_idx, minutes=offset_minutes)
    return int(dt.timestamp() * 1000)


def make_msg(role: str, text: str, sender_id: str, sender_name: str, round_idx: int, offset: int, msg_id: str) -> Dict:
    return {
        "message_id": msg_id,
        "sender_id": sender_id,
        "sender_name": sender_name,
        "role": role,
        "timestamp": ts_ms(round_idx, offset),
        "content": [{"type": "text", "text": text}],
    }


class TestResult:
    def __init__(self):
        self.checks: List[Tuple[str, bool, str]] = []

    def check(self, name: str, condition: bool, detail: str = ""):
        self.checks.append((name, condition, detail))
        icon = "PASS" if condition else "FAIL"
        print(f"  [{icon}] {name}" + (f" -- {detail}" if detail else ""))

    def summary(self) -> bool:
        total = len(self.checks)
        passed = sum(1 for _, ok, _ in self.checks if ok)
        failed = total - passed
        print(f"\n{'='*70}")
        print(f"  TOTAL: {total}  |  PASSED: {passed}  |  FAILED: {failed}")
        if failed:
            print(f"\n  Failed checks:")
            for name, ok, detail in self.checks:
                if not ok:
                    print(f"    - {name}: {detail}")
        print(f"{'='*70}")
        return failed == 0


# ─── API Wrappers ────────────────────────────────────────────────────

async def api_add_personal(client: httpx.AsyncClient, user_id: str, messages: List[Dict]) -> Dict:
    resp = await client.post(f"{BASE_URL}/api/v1/memories", json={"user_id": user_id, "messages": messages})
    resp.raise_for_status()
    return resp.json()


async def api_add_group(client: httpx.AsyncClient, group_id: str, messages: List[Dict]) -> Dict:
    resp = await client.post(f"{BASE_URL}/api/v1/memories/group", json={"group_id": group_id, "messages": messages})
    resp.raise_for_status()
    return resp.json()


async def api_flush_personal(client: httpx.AsyncClient, user_id: str) -> Dict:
    resp = await client.post(f"{BASE_URL}/api/v1/memories/flush", json={"user_id": user_id})
    resp.raise_for_status()
    return resp.json()


async def api_flush_group(client: httpx.AsyncClient, group_id: str) -> Dict:
    resp = await client.post(f"{BASE_URL}/api/v1/memories/group/flush", json={"group_id": group_id})
    resp.raise_for_status()
    return resp.json()


async def api_get_profiles(client: httpx.AsyncClient, user_id: str = None, group_id: str = None) -> Dict:
    filters = {}
    if user_id:
        filters["user_id"] = user_id
    if group_id:
        filters["group_id"] = group_id
    resp = await client.post(f"{BASE_URL}/api/v1/memories/get", json={
        "memory_type": "profile", "filters": filters, "page": 1, "page_size": 50,
    })
    resp.raise_for_status()
    return resp.json()


async def api_search_profiles(client: httpx.AsyncClient, query: str, method: str = "hybrid", user_id: str = None, group_id: str = None) -> Dict:
    filters = {}
    if user_id:
        filters["user_id"] = user_id
    if group_id:
        filters["group_id"] = group_id
    resp = await client.post(f"{BASE_URL}/api/v1/memories/search", json={
        "query": query, "method": method, "memory_types": ["profile"],
        "filters": filters, "top_k": 10,
    })
    resp.raise_for_status()
    return resp.json()


async def api_get_episodes(client: httpx.AsyncClient, user_id: str = None, group_id: str = None) -> Dict:
    filters = {}
    if user_id:
        filters["user_id"] = user_id
    if group_id:
        filters["group_id"] = group_id
    resp = await client.post(f"{BASE_URL}/api/v1/memories/get", json={
        "memory_type": "episodic_memory", "filters": filters, "page": 1, "page_size": 50,
    })
    resp.raise_for_status()
    return resp.json()


async def api_init_settings(client: httpx.AsyncClient):
    resp = await client.put(f"{BASE_URL}/api/v1/settings", json={})
    resp.raise_for_status()


# ─── 5 Conversation Rounds (each becomes 1 memcell on flush) ────────

def build_solo_rounds() -> List[List[Dict]]:
    """Build 5 separate conversation rounds with distinct topics."""
    uid = TEST_USER_ID
    aid = "assistant_001"

    rounds = [
        # Round 1: Career background
        [
            make_msg("user", "Hi! I'm a software engineer at Google, working on the search ranking team. I've been there for 3 years.", uid, "TestUser", 0, 0, "r1m1"),
            make_msg("assistant", "That's fascinating! What technologies do you work with?", aid, "Assistant", 0, 1, "r1m2"),
            make_msg("user", "Mainly Python and Go for the backend. I also use TensorFlow for ML feature engineering. Before Google I was at Microsoft doing C# for 2 years.", uid, "TestUser", 0, 2, "r1m3"),
            make_msg("assistant", "Impressive stack! Do you enjoy the ML or infrastructure side more?", aid, "Assistant", 0, 3, "r1m4"),
            make_msg("user", "I love infrastructure challenges - scaling systems to handle billions of requests excites me the most. I'm the tech lead for our data pipeline processing 10TB daily.", uid, "TestUser", 0, 4, "r1m5"),
        ],
        # Round 2: Hobbies and outdoor activities
        [
            make_msg("user", "I wanted to ask about outdoor activities. I'm really into rock climbing and hiking.", uid, "TestUser", 1, 0, "r2m1"),
            make_msg("assistant", "Great hobbies! Where do you usually go?", aid, "Assistant", 1, 1, "r2m2"),
            make_msg("user", "I go to Yosemite almost every month. My golden retriever Max comes with me on all hiking trips - he's 4 years old and the best hiking buddy.", uid, "TestUser", 1, 2, "r2m3"),
            make_msg("assistant", "A golden retriever hiking companion sounds wonderful!", aid, "Assistant", 1, 3, "r2m4"),
            make_msg("user", "He really is! I also play guitar in my free time - mostly blues and jazz. Music helps me unwind after intense coding sessions.", uid, "TestUser", 1, 4, "r2m5"),
        ],
        # Round 3: Food and travel interests
        [
            make_msg("user", "I visited Tokyo last year and completely fell in love with the city.", uid, "TestUser", 2, 0, "r3m1"),
            make_msg("assistant", "Tokyo is amazing! What did you enjoy most?", aid, "Assistant", 2, 1, "r3m2"),
            make_msg("user", "The food scene is incredible. I'm a huge ramen fan - especially tonkotsu style. I actually started making ramen at home, the broth takes 12 hours.", uid, "TestUser", 2, 2, "r3m3"),
            make_msg("assistant", "That's serious dedication to cooking!", aid, "Assistant", 2, 3, "r3m4"),
            make_msg("user", "I'm also learning Japanese because I want to live in Tokyo someday. I'm originally from Sichuan, China, so I love spicy food too - hot pot is my absolute favorite.", uid, "TestUser", 2, 4, "r3m5"),
        ],
        # Round 4: Career goals and leadership
        [
            make_msg("user", "I've been thinking about transitioning to an engineering manager role in the next year or two.", uid, "TestUser", 3, 0, "r4m1"),
            make_msg("assistant", "That's a great career goal! What leadership experience do you have?", aid, "Assistant", 3, 1, "r4m2"),
            make_msg("user", "I've been mentoring two junior engineers and led a cross-team project last quarter coordinating 4 teams. My manager says I have good communication skills.", uid, "TestUser", 3, 2, "r4m3"),
            make_msg("assistant", "Those are excellent foundations for an EM transition!", aid, "Assistant", 3, 3, "r4m4"),
            make_msg("user", "Thanks! I'm reading 'The Manager's Path' by Camille Fournier. I also contribute to Apache Beam on weekends - open source leadership is good practice.", uid, "TestUser", 3, 4, "r4m5"),
        ],
        # Round 5: Health and fitness
        [
            make_msg("user", "I've been working on improving my health lately. My BMI is around 29.8 which isn't great.", uid, "TestUser", 4, 0, "r5m1"),
            make_msg("assistant", "That's good that you're aware of it! What are you doing about it?", aid, "Assistant", 4, 1, "r5m2"),
            make_msg("user", "I started running 3 times a week and doing yoga on rest days. Combined with the hiking and climbing, I'm hoping to get down to 80kg from 86kg.", uid, "TestUser", 4, 2, "r5m3"),
            make_msg("assistant", "That sounds like a solid fitness plan!", aid, "Assistant", 4, 3, "r5m4"),
            make_msg("user", "My girlfriend is a fitness instructor so she helps me with nutrition plans. We cook healthy meals together - mostly Mediterranean diet with some Asian fusion.", uid, "TestUser", 4, 4, "r5m5"),
        ],
    ]
    return rounds


def build_group_rounds() -> List[List[Dict]]:
    """Build 5 group conversation rounds."""
    ua = TEST_GROUP_USER_A
    ub = TEST_GROUP_USER_B

    rounds = [
        # Round 1: Project kickoff
        [
            make_msg("user", "Hey Diana, I finished the API design doc for the new payment service. Can you review it?", ua, "Charlie", 0, 0, "g1m1"),
            make_msg("user", "Sure! I have experience with payment systems from my last company where we used Stripe extensively.", ub, "Diana", 0, 1, "g1m2"),
            make_msg("user", "Great. The REST endpoints look solid but I think we need webhook support for async payment notifications.", ua, "Charlie", 0, 3, "g1m3"),
            make_msg("user", "Agreed. We also need idempotency keys for retry safety - I've seen duplicate charge issues before.", ub, "Diana", 0, 4, "g1m4"),
        ],
        # Round 2: Architecture discussion
        [
            make_msg("user", "Diana, I've been thinking about the database schema. I'm proposing PostgreSQL with ACID transactions for the ledger.", ua, "Charlie", 1, 0, "g2m1"),
            make_msg("user", "Makes sense. I've been studying distributed transactions - maybe we should use the saga pattern for cross-service consistency.", ub, "Diana", 1, 1, "g2m2"),
            make_msg("user", "I was thinking the same thing! I'll prepare some architecture diagrams for our meeting.", ua, "Charlie", 1, 2, "g2m3"),
            make_msg("user", "Perfect. I'll bring my notes on error handling patterns. I'm based in London so let's find a timezone-friendly slot.", ub, "Diana", 1, 3, "g2m4"),
        ],
        # Round 3: Implementation details
        [
            make_msg("user", "I started implementing the core ledger service. Using Domain-Driven Design with event sourcing.", ua, "Charlie", 2, 0, "g3m1"),
            make_msg("user", "Nice! I've set up the Stripe integration with their sandbox. I prefer integration tests over unit tests for payment flows.", ub, "Diana", 2, 1, "g3m2"),
            make_msg("user", "Good call. I'm also adding comprehensive audit logging - every transaction gets a full audit trail.", ua, "Charlie", 2, 2, "g3m3"),
            make_msg("user", "That's essential for financial compliance. I'll add PCI DSS compliance checks to the CI pipeline.", ub, "Diana", 2, 3, "g3m4"),
        ],
        # Round 4: Testing strategy
        [
            make_msg("user", "Diana, we need to discuss our testing strategy. I've been writing property-based tests for the ledger.", ua, "Charlie", 3, 0, "g4m1"),
            make_msg("user", "I love property-based testing! I use Hypothesis in Python. For the Stripe integration I'm using their test clock API for time-based scenarios.", ub, "Diana", 3, 1, "g4m2"),
            make_msg("user", "Smart approach. I also want to add chaos engineering tests - simulating network failures between our service and Stripe.", ua, "Charlie", 3, 2, "g4m3"),
            make_msg("user", "Excellent idea. I've used Toxiproxy before for that. Let me set it up in our staging environment.", ub, "Diana", 3, 3, "g4m4"),
        ],
        # Round 5: Deployment planning
        [
            make_msg("user", "We're almost ready to deploy. I think we should use canary releases for the payment service.", ua, "Charlie", 4, 0, "g5m1"),
            make_msg("user", "Absolutely. I'll configure the feature flags so we can gradually roll out. I'm very cautious with payment systems.", ub, "Diana", 4, 1, "g5m2"),
            make_msg("user", "Good plan. I've also set up real-time monitoring dashboards in Grafana for transaction success rates.", ua, "Charlie", 4, 2, "g5m3"),
            make_msg("user", "Perfect. I'll add alerting rules - if the failure rate exceeds 1% we should auto-rollback. Safety first for financial transactions.", ub, "Diana", 4, 3, "g5m4"),
        ],
    ]
    return rounds


def build_group2_rounds() -> List[List[Dict]]:
    """Build 5 rounds for a SECOND group with completely different topic (cooking/restaurant)."""
    ua = TEST_GROUP2_USER_A
    ub = TEST_GROUP2_USER_B

    rounds = [
        [
            make_msg("user", "Fiona, I'm opening a French restaurant downtown. I studied at Le Cordon Bleu in Paris for 2 years.", ua, "Evan", 0, 0, "h1m1"),
            make_msg("user", "That's amazing! I'm a sommelier - I specialize in Burgundy wines. I'd love to curate the wine list.", ub, "Fiona", 0, 1, "h1m2"),
            make_msg("user", "Perfect! My signature dish is duck confit with truffle sauce. We need wines that pair well with rich dishes.", ua, "Evan", 0, 2, "h1m3"),
            make_msg("user", "For duck confit I'd recommend a Pinot Noir from Cote de Nuits. I have connections with several Burgundy producers.", ub, "Fiona", 0, 3, "h1m4"),
        ],
        [
            make_msg("user", "Fiona, I've been experimenting with Japanese-French fusion. My miso-glazed foie gras was a hit at the tasting event.", ua, "Evan", 1, 0, "h2m1"),
            make_msg("user", "That sounds incredible! For fusion dishes, sake pairings could be interesting. I've been studying sake certification - the WSET Level 3 in Sake.", ub, "Fiona", 1, 1, "h2m2"),
            make_msg("user", "Great idea. I'm also planning a farm-to-table concept. I've partnered with 3 local organic farms for seasonal ingredients.", ua, "Evan", 1, 2, "h2m3"),
            make_msg("user", "Love the sustainability angle. I'll focus on natural wines and biodynamic producers to match that philosophy.", ub, "Fiona", 1, 3, "h2m4"),
        ],
        [
            make_msg("user", "The interior design is almost done. I went with a modern minimalist style - lots of natural wood and warm lighting.", ua, "Evan", 2, 0, "h3m1"),
            make_msg("user", "Sounds beautiful. For the wine cellar, I'm thinking temperature-controlled display cases that double as a visual feature.", ub, "Fiona", 2, 1, "h3m2"),
            make_msg("user", "I love that. The kitchen will have an open layout so guests can see the cooking process. Transparency builds trust.", ua, "Evan", 2, 2, "h3m3"),
            make_msg("user", "Agree. I'll set up a tasting corner near the entrance where guests can sample wines before ordering.", ub, "Fiona", 2, 3, "h3m4"),
        ],
        [
            make_msg("user", "I'm hiring my kitchen team. Looking for a sous chef with pastry experience - my desserts need work.", ua, "Evan", 3, 0, "h4m1"),
            make_msg("user", "I know someone! My friend Marie was a pastry chef at a Michelin-starred restaurant in Lyon.", ub, "Fiona", 3, 1, "h4m2"),
            make_msg("user", "That would be perfect. I also need someone for the raw bar - we'll feature fresh oysters and sashimi.", ua, "Evan", 3, 2, "h4m3"),
            make_msg("user", "For the raw bar, Champagne pairings are essential. I'll prepare a dedicated sparkling wine section.", ub, "Fiona", 3, 3, "h4m4"),
        ],
        [
            make_msg("user", "Grand opening is in 3 weeks! The menu has 28 dishes. I'm most excited about my bouillabaisse recipe.", ua, "Evan", 4, 0, "h5m1"),
            make_msg("user", "I've finalized the wine list - 120 labels across 8 countries. The markup is 2.5x, which is fair for fine dining.", ub, "Fiona", 4, 1, "h5m2"),
            make_msg("user", "Great pricing. I've set up our reservation system and we already have 50 bookings for opening week.", ua, "Evan", 4, 2, "h5m3"),
            make_msg("user", "Wonderful! I'll organize a VIP preview evening for food critics and wine journalists the night before.", ub, "Fiona", 4, 3, "h5m4"),
        ],
    ]
    return rounds


# ─── Core Test Logic ─────────────────────────────────────────────────

async def send_rounds_and_wait(
    client: httpx.AsyncClient,
    rounds: List[List[Dict]],
    send_fn,
    flush_fn,
    entity_id: str,
    get_episodes_fn,
    label: str,
) -> int:
    """Send N rounds of messages, flushing after each. Returns total episodes found."""
    total_episodes = 0

    for i, msgs in enumerate(rounds, 1):
        print(f"    Round {i}/{len(rounds)}: Sending {len(msgs)} messages...")
        await send_fn(client, entity_id, msgs)

        print(f"    Round {i}/{len(rounds)}: Flushing...")
        flush_resp = await flush_fn(client, entity_id)
        flush_status = flush_resp.get("data", {}).get("status", "unknown")
        print(f"      flush status: {flush_status}")

        # Wait for this round's episode to be created
        max_wait = 90
        poll = 5
        waited = 0
        prev_count = total_episodes

        while waited < max_wait:
            await asyncio.sleep(poll)
            waited += poll
            ep_resp = await get_episodes_fn(client)
            ep_count = ep_resp.get("data", {}).get("total_count", 0)
            if ep_count > prev_count:
                total_episodes = ep_count
                print(f"      Episode #{ep_count} created after {waited}s")
                break
        else:
            # Check one more time
            ep_resp = await get_episodes_fn(client)
            ep_count = ep_resp.get("data", {}).get("total_count", 0)
            total_episodes = ep_count
            if ep_count <= prev_count:
                print(f"      WARNING: No new episode after {max_wait}s (total={ep_count})")

        # Small pause between rounds
        if i < len(rounds):
            await asyncio.sleep(2)

    return total_episodes


async def test_solo_profile_extraction(client: httpx.AsyncClient, result: TestResult):
    """Test 1: SOLO profile extraction - send 5 rounds to trigger profile."""
    print("\n" + "=" * 70)
    print("  TEST 1: SOLO Profile Extraction")
    print(f"  (Need {PROFILE_EXTRACTION_INTERVAL} memcells to trigger profile)")
    print("=" * 70)

    rounds = build_solo_rounds()

    async def get_ep(c):
        return await api_get_episodes(c, user_id=TEST_USER_ID)

    total_episodes = await send_rounds_and_wait(
        client, rounds,
        send_fn=api_add_personal,
        flush_fn=api_flush_personal,
        entity_id=TEST_USER_ID,
        get_episodes_fn=get_ep,
        label="SOLO",
    )

    result.check(
        "SOLO: Created enough episodes",
        total_episodes >= PROFILE_EXTRACTION_INTERVAL,
        f"episodes={total_episodes}, need>={PROFILE_EXTRACTION_INTERVAL}",
    )

    # Now wait for profile extraction (triggered by the 5th clustering)
    print(f"\n  Waiting for profile extraction after {total_episodes} memcells...")
    max_wait = 120
    poll = 5
    waited = 0
    profile_found = False

    while waited < max_wait:
        await asyncio.sleep(poll)
        waited += poll
        prof_resp = await api_get_profiles(client, user_id=TEST_USER_ID)
        prof_count = prof_resp.get("data", {}).get("total_count", 0)
        if prof_count > 0:
            profile_found = True
            print(f"    Profile found after {waited}s!")
            break
        if waited % 15 == 0:
            print(f"    [{waited}s] profiles={prof_count}...")

    result.check("SOLO: Profile created", profile_found, f"after {waited}s" if profile_found else f"not found after {max_wait}s")

    if not profile_found:
        print("    WARNING: Profile not created. Possible causes:")
        print("      - Profile extraction still in progress (LLM call can be slow)")
        print("      - Clustering may not have assigned all memcells to clusters")
        print("      - Check server logs for errors")
        return

    # Validate profile format
    print("\n  Validating profile format...")
    prof_resp = await api_get_profiles(client, user_id=TEST_USER_ID)
    profiles = prof_resp.get("data", {}).get("profiles", [])
    profile = profiles[0]
    profile_data = profile.get("profile_data", {})
    explicit_info = profile_data.get("explicit_info", [])
    implicit_traits = profile_data.get("implicit_traits", [])

    result.check("SOLO: Has explicit_info", "explicit_info" in profile_data, f"{len(explicit_info)} items")
    result.check("SOLO: Has implicit_traits", "implicit_traits" in profile_data, f"{len(implicit_traits)} items")
    result.check("SOLO: Has content", len(explicit_info) + len(implicit_traits) > 0, f"explicit={len(explicit_info)}, implicit={len(implicit_traits)}")

    # Validate item structures
    if explicit_info:
        item = explicit_info[0]
        result.check("SOLO: explicit_info has 'category'", "category" in item, f"keys={list(item.keys())}")
        result.check("SOLO: explicit_info has 'description'", "description" in item, f"{str(item.get('description', ''))[:60]}")
        result.check("SOLO: explicit_info has 'sources'", "sources" in item, f"sources={item.get('sources', 'N/A')}")

    if implicit_traits:
        trait = implicit_traits[0]
        result.check("SOLO: implicit_traits has 'trait'", "trait" in trait, f"keys={list(trait.keys())}")
        result.check("SOLO: implicit_traits has 'description'", "description" in trait, f"{str(trait.get('description', ''))[:60]}")

    # Verify NO old GROUP fields
    found_old = [f for f in OLD_GROUP_FIELDS if f in profile_data]
    result.check("SOLO: No old GROUP fields", len(found_old) == 0, f"Found: {found_old}" if found_old else "Clean")

    # Verify processed_episode_ids
    result.check("SOLO: Has processed_episode_ids", "processed_episode_ids" in profile_data,
                 f"count={len(profile_data.get('processed_episode_ids', []))}")

    # Print profile summary
    print(f"\n  Profile content ({len(explicit_info)} explicit + {len(implicit_traits)} implicit):")
    for i, info in enumerate(explicit_info[:5], 1):
        print(f"    E{i}. [{info.get('category', '?')}] {str(info.get('description', ''))[:70]}")
    for i, t in enumerate(implicit_traits[:3], 1):
        print(f"    I{i}. [{t.get('trait', '?')}] {str(t.get('description', ''))[:70]}")


async def test_group_profile_extraction(client: httpx.AsyncClient, result: TestResult):
    """Test 2: GROUP profile extraction - also uses unified format."""
    print("\n" + "=" * 70)
    print("  TEST 2: GROUP Profile Extraction")
    print(f"  (Need {PROFILE_EXTRACTION_INTERVAL} memcells to trigger profile)")
    print("=" * 70)

    rounds = build_group_rounds()

    async def get_ep(c):
        return await api_get_episodes(c, group_id=TEST_GROUP_ID_GROUP)

    total_episodes = await send_rounds_and_wait(
        client, rounds,
        send_fn=api_add_group,
        flush_fn=api_flush_group,
        entity_id=TEST_GROUP_ID_GROUP,
        get_episodes_fn=get_ep,
        label="GROUP",
    )

    result.check("GROUP: Created enough episodes", total_episodes >= PROFILE_EXTRACTION_INTERVAL,
                 f"episodes={total_episodes}")

    # Wait for profile extraction - need BOTH users to have profiles
    print(f"\n  Waiting for profile extraction (both users)...")
    max_wait = 120
    poll = 5
    waited = 0
    both_found = False

    while waited < max_wait:
        await asyncio.sleep(poll)
        waited += poll
        prof_a = await api_get_profiles(client, user_id=TEST_GROUP_USER_A, group_id=TEST_GROUP_ID_GROUP)
        prof_b = await api_get_profiles(client, user_id=TEST_GROUP_USER_B, group_id=TEST_GROUP_ID_GROUP)
        ca = prof_a.get("data", {}).get("total_count", 0)
        cb = prof_b.get("data", {}).get("total_count", 0)
        if ca > 0 and cb > 0:
            both_found = True
            print(f"    Both profiles found after {waited}s! (Charlie={ca}, Diana={cb})")
            break
        if waited % 15 == 0:
            print(f"    [{waited}s] Charlie={ca}, Diana={cb}...")

    result.check("GROUP: Both users have profiles", both_found,
                 f"after {waited}s" if both_found else f"not both found after {max_wait}s")

    if not both_found:
        return

    # Validate both users' profiles
    for label, uid in [("Charlie", TEST_GROUP_USER_A), ("Diana", TEST_GROUP_USER_B)]:
        prof_resp = await api_get_profiles(client, user_id=uid, group_id=TEST_GROUP_ID_GROUP)
        profiles = prof_resp.get("data", {}).get("profiles", [])

        result.check(f"GROUP({label}): Profile exists", len(profiles) > 0)
        if not profiles:
            continue

        pd = profiles[0].get("profile_data", {})
        ei = pd.get("explicit_info", [])
        it = pd.get("implicit_traits", [])

        result.check(f"GROUP({label}): Unified format", "explicit_info" in pd,
                     f"explicit={len(ei)}, implicit={len(it)}")
        result.check(f"GROUP({label}): Has content", len(ei) + len(it) > 0,
                     f"explicit={len(ei)}, implicit={len(it)}")
        found_old = [f for f in OLD_GROUP_FIELDS if f in pd]
        result.check(f"GROUP({label}): No old fields", len(found_old) == 0,
                     f"Found: {found_old}" if found_old else "Clean")

        print(f"    {label}: {len(ei)} explicit + {len(it)} implicit items")


async def test_profile_search(client: httpx.AsyncClient, result: TestResult):
    """Test 3: Profile search & retrieval.

    Note: Search API returns individual profile items from Milvus (each explicit_info
    and implicit_trait stored as separate vector entries), NOT the full profile document.
    profile_data in search results contains {item_type, embed_text}.
    """
    print("\n" + "=" * 70)
    print("  TEST 3: Profile Search & Retrieval")
    print("=" * 70)

    prof_resp = await api_get_profiles(client, user_id=TEST_USER_ID)
    if prof_resp.get("data", {}).get("total_count", 0) == 0:
        print("    SKIP: No SOLO profile available")
        return

    queries = [
        ("rock climbing hiking outdoor", "vector"),
        ("what programming languages does the user know", "hybrid"),
    ]

    for query, method in queries:
        print(f"\n  Search: '{query}' ({method})...")
        try:
            resp = await api_search_profiles(client, query, method, user_id=TEST_USER_ID)
            profiles = resp.get("data", {}).get("profiles", [])
            result.check(f"Search({method}): Returns results", len(profiles) > 0,
                         f"{len(profiles)} profile item(s)")
            if profiles:
                p = profiles[0]
                pd = p.get("profile_data", {})
                # Search returns individual items: item_type + embed_text
                item_type = pd.get("item_type", "")
                embed_text = pd.get("embed_text", "")
                valid_type = item_type in ("explicit_info", "implicit_trait")
                result.check(f"Search({method}): item_type is valid",
                             valid_type,
                             f"item_type={item_type}, text={embed_text[:60]}")
                result.check(f"Search({method}): user_id matches",
                             p.get("user_id") == TEST_USER_ID,
                             f"user_id={p.get('user_id')}")
                result.check(f"Search({method}): Has score",
                             "score" in p and p["score"] > 0,
                             f"score={p.get('score', 0):.4f}")
        except Exception as e:
            result.check(f"Search({method}): No error", False, str(e))


async def test_multi_group_isolation(client: httpx.AsyncClient, result: TestResult):
    """Test 5: Multi-group isolation - two groups should have independent profiles."""
    print("\n" + "=" * 70)
    print("  TEST 5: Multi-Group Profile Isolation")
    print(f"  Group A: {TEST_GROUP_ID_GROUP} (payment service - from Test 2)")
    print(f"  Group B: {TEST_GROUP_ID_GROUP2} (restaurant/cooking)")
    print("=" * 70)

    # Group A was already created in Test 2.
    # Now create Group B with completely different content.
    rounds = build_group2_rounds()

    async def get_ep(c):
        return await api_get_episodes(c, group_id=TEST_GROUP_ID_GROUP2)

    total_episodes = await send_rounds_and_wait(
        client, rounds,
        send_fn=api_add_group,
        flush_fn=api_flush_group,
        entity_id=TEST_GROUP_ID_GROUP2,
        get_episodes_fn=get_ep,
        label="GROUP2",
    )

    result.check("ISOLATION: Group B created enough episodes",
                 total_episodes >= PROFILE_EXTRACTION_INTERVAL,
                 f"episodes={total_episodes}")

    # Wait for Group B profiles - need BOTH users
    print(f"\n  Waiting for Group B profile extraction (both users)...")
    max_wait = 120
    poll = 5
    waited = 0
    both_found = False

    while waited < max_wait:
        await asyncio.sleep(poll)
        waited += poll
        prof_a = await api_get_profiles(client, user_id=TEST_GROUP2_USER_A, group_id=TEST_GROUP_ID_GROUP2)
        prof_b = await api_get_profiles(client, user_id=TEST_GROUP2_USER_B, group_id=TEST_GROUP_ID_GROUP2)
        ca = prof_a.get("data", {}).get("total_count", 0)
        cb = prof_b.get("data", {}).get("total_count", 0)
        if ca > 0 and cb > 0:
            both_found = True
            print(f"    Both profiles found after {waited}s! (Evan={ca}, Fiona={cb})")
            break
        if ca > 0 or cb > 0:
            if waited % 10 == 0:
                print(f"    [{waited}s] Partial: Evan={ca}, Fiona={cb}, waiting for both...")
        elif waited % 15 == 0:
            print(f"    [{waited}s] Evan={ca}, Fiona={cb}...")

    result.check("ISOLATION: Group B both users have profiles", both_found,
                 f"after {waited}s" if both_found else f"not both found after {max_wait}s")

    if not both_found:
        return

    # Now the key isolation checks:
    # 1. Group A profiles should NOT contain Group B content (cooking/restaurant/wine)
    print("\n  Checking isolation: Group A profiles should not contain Group B content...")
    group_a_profiles = []
    for uid in [TEST_GROUP_USER_A, TEST_GROUP_USER_B]:
        resp = await api_get_profiles(client, user_id=uid, group_id=TEST_GROUP_ID_GROUP)
        for p in resp.get("data", {}).get("profiles", []):
            group_a_profiles.append(p)

    group_a_text = ""
    for p in group_a_profiles:
        pd = p.get("profile_data", {})
        for item in pd.get("explicit_info", []) + pd.get("implicit_traits", []):
            group_a_text += " " + str(item.get("description", ""))

    # Group B keywords that should NOT appear in Group A
    b_keywords = ["restaurant", "sommelier", "wine", "duck confit", "Le Cordon Bleu", "Burgundy", "bouillabaisse"]
    leaked_b_in_a = [kw for kw in b_keywords if kw.lower() in group_a_text.lower()]
    result.check("ISOLATION: Group A has no Group B content",
                 len(leaked_b_in_a) == 0,
                 f"Leaked keywords: {leaked_b_in_a}" if leaked_b_in_a else "Clean")

    # 2. Group B profiles should NOT contain Group A content (payment/Stripe/PostgreSQL)
    print("  Checking isolation: Group B profiles should not contain Group A content...")
    group_b_profiles = []
    for uid in [TEST_GROUP2_USER_A, TEST_GROUP2_USER_B]:
        resp = await api_get_profiles(client, user_id=uid, group_id=TEST_GROUP_ID_GROUP2)
        for p in resp.get("data", {}).get("profiles", []):
            group_b_profiles.append(p)

    group_b_text = ""
    for p in group_b_profiles:
        pd = p.get("profile_data", {})
        for item in pd.get("explicit_info", []) + pd.get("implicit_traits", []):
            group_b_text += " " + str(item.get("description", ""))

    # Group A keywords that should NOT appear in Group B
    a_keywords = ["payment", "Stripe", "PostgreSQL", "idempotency", "saga pattern", "canary release"]
    leaked_a_in_b = [kw for kw in a_keywords if kw.lower() in group_b_text.lower()]
    result.check("ISOLATION: Group B has no Group A content",
                 len(leaked_a_in_b) == 0,
                 f"Leaked keywords: {leaked_a_in_b}" if leaked_a_in_b else "Clean")

    # 3. Group B profile format is also unified
    for p in group_b_profiles:
        pd = p.get("profile_data", {})
        result.check(f"ISOLATION: Group B ({p.get('user_id','?')}) unified format",
                     "explicit_info" in pd, f"keys={list(pd.keys())[:5]}")
        break  # check one is enough

    # Print summary
    print(f"\n  Group A ({TEST_GROUP_ID_GROUP}): {len(group_a_profiles)} profile(s)")
    print(f"  Group B ({TEST_GROUP_ID_GROUP2}): {len(group_b_profiles)} profile(s)")
    if group_b_profiles:
        pd = group_b_profiles[0].get("profile_data", {})
        ei = pd.get("explicit_info", [])
        for i, info in enumerate(ei[:3], 1):
            print(f"    B-E{i}. [{info.get('category', '?')}] {str(info.get('description', ''))[:60]}")


async def test_api_contract(client: httpx.AsyncClient, result: TestResult):
    """Test 4: API response structure checks."""
    print("\n" + "=" * 70)
    print("  TEST 4: API Contract Checks")
    print("=" * 70)

    # GET response shape
    resp = await api_get_profiles(client, user_id=TEST_USER_ID)
    data = resp.get("data", {})
    result.check("API: GET has 'profiles'", "profiles" in data, f"keys={list(data.keys())}")
    result.check("API: GET has 'total_count'", "total_count" in data, f"total_count={data.get('total_count')}")

    # Search response shape
    sr = await api_search_profiles(client, "test", "keyword", user_id=TEST_USER_ID)
    sd = sr.get("data", {})
    result.check("API: Search has 'profiles'", "profiles" in sd)
    result.check("API: Search has 'episodes'", "episodes" in sd)

    # Flush for non-existent user doesn't crash
    fr = await api_flush_personal(client, "nonexistent_user_contract_test")
    result.check("API: Flush for empty user OK", True, f"status={fr.get('data', {}).get('status')}")

    # Profile item structure
    profiles = data.get("profiles", [])
    if profiles:
        p = profiles[0]
        has_fields = {"id", "user_id", "profile_data"}.issubset(set(p.keys()))
        result.check("API: Profile item has required fields", has_fields, f"fields={sorted(p.keys())}")


# ─── Main ────────────────────────────────────────────────────────────

async def main():
    print("=" * 70)
    print("  E2E TEST: Unified Profile Extraction")
    print(f"  Target: {BASE_URL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Profile extraction interval: {PROFILE_EXTRACTION_INTERVAL} memcells")
    print("=" * 70)

    result = TestResult()

    # Connectivity check
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            r = await client.get(f"{BASE_URL}/api/v1/memories/get", timeout=5.0)
            print(f"\n  API reachable (HTTP {r.status_code})")
    except Exception as e:
        print(f"\n  FATAL: Cannot connect to {BASE_URL}: {e}")
        sys.exit(1)

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            await api_init_settings(client)
            print("  Settings initialized.\n")
        except Exception as e:
            print(f"  Settings init warning: {e}\n")

        for test_fn in [test_solo_profile_extraction, test_group_profile_extraction, test_multi_group_isolation, test_profile_search, test_api_contract]:
            try:
                await test_fn(client, result)
            except Exception as e:
                result.check(f"{test_fn.__name__} completed", False, f"{type(e).__name__}: {e}")
                traceback.print_exc()

    all_passed = result.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
