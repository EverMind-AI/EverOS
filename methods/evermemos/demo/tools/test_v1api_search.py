"""V1 API Retrieval Test Script

Test V1 API retrieval functions:
- GET /api/v1/memories - Fetch memories (supports profile type)
- GET /api/v1/memories/search - Search memories (keyword/vector/hybrid)

Usage:
    # Ensure API server is running
    uv run python src/bootstrap.py src/run.py --port 1995
    
    # Run tests
    uv run python demo/tools/test_v1api_search.py
"""

import httpx
import asyncio
import json
import os
from typing import Dict, Any
import dotenv

dotenv.load_dotenv()
# Get language setting from environment variable
MEMORY_LANGUAGE = os.getenv('MEMORY_LANGUAGE').lower()

# Query words based on language setting
QUERY_WORDS = {
    'en': {'default': 'What sports do I like', 'travel': 'travel'},
    'zh': {'default': '我喜欢什么运动', 'travel': '旅游'},  # skip-i18n-check
}


def get_query_word(key: str = 'default') -> str:
    """Get query word based on MEMORY_LANGUAGE setting"""
    lang = MEMORY_LANGUAGE
    return QUERY_WORDS[lang].get(key, QUERY_WORDS[lang]['default'])


class V1APITester:
    """V1 API Tester"""

    def __init__(self, base_url: str = "http://localhost:1995"):
        self.base_url = base_url
        self.results = []

    async def test_fetch_memories(
        self, user_id: str, memory_type: str = "profile", limit: int = 5
    ) -> Dict[str, Any]:
        """Test GET /api/v1/memories"""
        url = f"{self.base_url}/api/v1/memories"
        params = {"user_id": user_id, "memory_type": memory_type, "limit": limit}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(url, params=params)
            return response.json()

    async def test_search_memories(
        self,
        user_id: str = None,
        query: str = None,
        method: str = "keyword",
        top_k: int = 5,
        memory_types: list = None,
        group_id: str = None,
    ) -> Dict[str, Any]:
        """Test GET /api/v1/memories/search (RESTful query params)

        Supports all data sources: episodic_memory, foresight, atomic_fact
        user_id: User ID, required for personal memories
        group_id: Group ID, required for group memories
        """
        url = f"{self.base_url}/api/v1/memories/search"
        params = {"query": query, "method": method, "top_k": top_k}
        # user_id and group_id are mutually exclusive
        if user_id:
            params["user_id"] = user_id
        if group_id:
            params["group_id"] = group_id
        # memory_types passed as comma-separated values
        if memory_types:
            params["memory_types"] = ",".join(memory_types)

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(url, params=params)
            return response.json()

    def print_result(
        self,
        name: str,
        result: Dict[str, Any],
        verbose: bool = False,
        raw: bool = False,
    ):
        """Print test results"""
        if raw:
            print(f"\n📤 {name} Raw response:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        status = result.get("status", "unknown")
        memories = result.get("result", {}).get("memories", [])
        scores = result.get("result", {}).get("scores", [])
        count = len(memories) if memories else 0

        # Calculate total record count
        total_records = 0

        for mem in memories:
            if "profile_data" in mem:
                total_records += 1
            elif "profile" in mem:
                total_records += 1
            elif "score" in mem:
                total_records += 1
            elif (
                "summary" in mem
                or "title" in mem
                or "atomic_fact" in mem
                or "content" in mem
            ):
                # Fetch returns episodic/atomic_fact/foresight types
                total_records += 1
            elif isinstance(mem, dict):
                # V1 Search result: {group_id: [records]} structure
                for group_id, records in mem.items():
                    if isinstance(records, list):
                        total_records += len(records)

        status_icon = "✅" if status == "ok" else "❌"
        print(
            f"{status_icon} {name}: status={status}, groups={count}, records={total_records}"
        )

        # Print detailed content
        if verbose and memories:
            print("-" * 40)
            for i, mem in enumerate(memories[:5]):  # Show at most 5
                # Print key fields based on different types
                if "profile_data" in mem:
                    # V1 Profile type
                    print(f"  📝 Profile:")
                    print(f"     user_id: {mem.get('user_id')}")
                    print(f"     group_id: {mem.get('group_id')}")
                    print(f"     scenario: {mem.get('scenario')}")
                    print(f"     version: {mem.get('version')}")
                    profile = mem.get('profile_data', {})
                    print(f"     profile: {mem}")
                    if profile:
                        print(
                            f"     explicit_info: {len(profile.get('explicit_info', []))} items"
                        )
                        print(
                            f"     implicit_traits: {len(profile.get('implicit_traits', []))} items"
                        )
                elif "profile" in mem:
                    # V1 Profile type
                    print(f"  📝 Profile:")
                    print(f"     user_id: {mem.get('user_id')}")
                    print(f"     group_id: {mem.get('group_id')}")
                    print(f"     scenario: {mem.get('scenario')}")
                elif "score" in mem:
                    # Select display field based on data source
                    content = (
                        mem.get('foresight')  # foresight
                        or mem.get('atomic_fact')  # atomic_fact
                        or mem.get('subject')  # episode
                        or 'N/A'
                    )
                    content = (
                        content[:50] + '...'
                        if content and len(content) > 50
                        else content
                    )
                    print(f"  [{i+1}] score={mem.get('score', 0):.4f} | {content}")
                elif (
                    "summary" in mem
                    or "title" in mem
                    or "atomic_fact" in mem
                    or "content" in mem
                ):
                    # Fetch returns episodic/atomic_fact/foresight types
                    content = (
                        mem.get('summary')
                        or mem.get('content')
                        or mem.get('foresight')
                        or mem.get('atomic_fact')
                        or mem.get('title')
                        or 'N/A'
                    )
                    content = (
                        content[:60] + '...'
                        if content and len(content) > 60
                        else content
                    )
                    print(f"  [{i+1}] {content}")
                else:
                    # V1 Search result type: {group_id: [records]}
                    for group_id, records in mem.items():
                        if isinstance(records, list):
                            # Get corresponding score list
                            group_scores = []
                            for s in scores:
                                if isinstance(s, dict) and group_id in s:
                                    group_scores = s[group_id]
                                    break
                            print(f"  📁 Group: {group_id}, records: {len(records)}")
                            for j, r in enumerate(records[:3]):
                                if isinstance(r, dict):
                                    content = (
                                        r.get('atomic_fact')
                                        or r.get('foresight')
                                        or r.get('episode')
                                        or r.get('subject')
                                        or 'N/A'
                                    )
                                else:
                                    content = str(r)
                                content = (
                                    content[:40] + '...'
                                    if content and len(content) > 40
                                    else content
                                )
                                score_val = (
                                    group_scores[j] if j < len(group_scores) else 0
                                )
                                if isinstance(score_val, (int, float)):
                                    print(f"     [{j+1}] {score_val:.2f} | {content}")
                                else:
                                    print(f"     [{j+1}] {score_val} | {content}")
                            if len(records) > 3:
                                print(f"     ... {len(records) - 3} more")
            if len(memories) > 5:
                print(f"  ... {len(memories) - 5} more")
            print("-" * 40)

        self.results.append(
            {
                "name": name,
                "status": status,
                "count": total_records,
                "success": status == "ok",
            }
        )

    async def run_all_tests(
        self,
        user_id: str = "user_001",
        query: str = None,
        verbose: bool = True,
        raw: bool = False,
    ):
        # Use language-aware default query if not specified
        if query is None:
            query = get_query_word('default')
        """Run all tests: 3 memory types × 3 retrieval methods = 9 combinations"""
        print("=" * 60)
        print("V1 API Retrieval Test (Full Combinations)")
        print("=" * 60)
        print(f"User ID: {user_id}")
        print(f"Query: {query}")
        print("-" * 60)

        # Test Fetch (KV method)
        print("\n📦 Test Fetch")
        print("-" * 40)
        fetch_types = ["profile", "episodic_memory", "foresight", "atomic_fact"]
        for mem_type in fetch_types:
            result = await self.test_fetch_memories(user_id, mem_type, 5)
            self.print_result(f"fetch {mem_type}", result, verbose, raw)

        # Full combination test of memory types and retrieval methods
        memory_types = ["episodic_memory", "foresight", "atomic_fact"]
        # memory_types = ["foresight"]
        retrieval_methods = ["keyword", "vector", "hybrid", "rrf", "agentic"]
        # retrieval_methods = ["vector"]
        # retrieval_methods = ["hybrid", "vector"]
        # retrieval_methods = ["keyword"]
        icons = {"episodic_memory": "🎬", "foresight": "🔮", "atomic_fact": "📋"}

        # Group memory test (only group_id, no user_id) - only episodic_memory
        group_id = "chat_user_001_assistant"
        print(f"\n🏢 Group Memory Test (group_id={group_id}, only episodic_memory)")
        print("=" * 50)
        print(f"\n{icons['episodic_memory']} Test episodic_memory")
        print("-" * 40)
        for method in retrieval_methods:
            result = await self.test_search_memories(
                None, query, method, 5, ["episodic_memory"], group_id
            )
            self.print_result(f"Group episodic_memory + {method}", result, verbose, raw)

        # Personal memory test (no group_id)
        print(f"\n👤 Personal Memory Test (no group_id)")
        print("=" * 50)
        for mem_type in memory_types:
            print(f"\n{icons[mem_type]} Test {mem_type}")
            print("-" * 40)
            for method in retrieval_methods:
                result = await self.test_search_memories(
                    user_id, query, method, 5, [mem_type]
                )
                self.print_result(
                    f"Personal {mem_type} + {method}", result, verbose, raw
                )

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)

        total = len(self.results)
        passed = sum(1 for r in self.results if r["success"])

        for r in self.results:
            icon = "✅" if r["success"] else "❌"
            print(f"  {icon} {r['name']}: {r['count']} memories")

        print("-" * 60)
        print(f"Passed: {passed}/{total}")

        return passed == total


async def main():
    tester = V1APITester("http://localhost:1995")

    # Modify parameters here
    user_id = "user_001"
    query = get_query_word('travel')  # Auto-select based on MEMORY_LANGUAGE env var
    raw = False  # Don't print raw API output

    print(f"🌐 Language: {MEMORY_LANGUAGE.upper()}")
    success = await tester.run_all_tests(user_id, query, verbose=True, raw=raw)

    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed")


if __name__ == "__main__":
    asyncio.run(main())
