"""Search Agent Memory Demo - Web Search Trajectory + Agent Skill

Demonstrates how the agent memory system works with a search/research agent:
1. Send two similar search agent trajectories (OpenAI chat completion format)
2. Wait for memory extraction (MemCell segmentation + AgentCase extraction)
3. The second trajectory triggers MemScene clustering -> AgentSkill extraction
4. Fetch extracted agent experiences AND agent skills
5. Hybrid search over agent_case and agent_skill (method=hybrid)

The mock trajectories simulate a research agent that:
- Trajectory 1: Compares Python web frameworks, then deploys FastAPI with Docker
- Trajectory 2: Compares databases for a real-time app, then sets up ClickHouse

Both follow the same research pattern (search -> compare -> recommend -> deploy),
so they cluster together and trigger AgentSkill extraction.

Prerequisites:
    Start the API server first (in another terminal):
    uv run python src/run.py

Run the demo:
    uv run python src/bootstrap.py demo/search_agent_demo.py
"""

import asyncio
import json

from demo.utils.agent_demo_helpers import (
    AgentDemoRunner,
    MEMORY_TYPE_PRINTERS,
    print_separator,
    print_search_case_results,
    print_search_skill_results,
)


# ==================== Mock Search Agent Trajectory ====================

def build_mock_trajectory():
    """Build a mock search agent trajectory in OpenAI chat completion format.

    Simulates a two-turn interaction:
    Turn 1: User asks about Python web frameworks -> Agent searches and compares
    Turn 2: User asks a follow-up about FastAPI deployment -> Agent searches
    """
    messages = [
        # ===== Turn 1: Python web frameworks comparison =====
        {
            "role": "user",
            "content": [{"type": "text", "text": "Compare the top 3 Python web frameworks by GitHub stars and performance benchmarks"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I'll search for information about the top Python web frameworks and compare them."}],
            "tool_calls": [
                {
                    "id": "call_001",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({
                            "query": "top Python web frameworks 2026 GitHub stars comparison"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": json.dumps({
                "results": [
                    {"title": "Django - 82k stars", "url": "https://github.com/django/django"},
                    {"title": "Flask - 69k stars", "url": "https://github.com/pallets/flask"},
                    {"title": "FastAPI - 78k stars", "url": "https://github.com/tiangolo/fastapi"},
                ]
            }),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Found the top frameworks. Let me now search for performance benchmarks."}],
            "tool_calls": [
                {
                    "id": "call_002",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({
                            "query": "Django vs Flask vs FastAPI performance benchmark 2026"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_002",
            "content": json.dumps({
                "results": [
                    {
                        "title": "TechEmpower Benchmark Round 23",
                        "summary": "FastAPI: 45k req/s, Flask: 12k req/s, Django: 8k req/s (JSON serialization)",
                    }
                ]
            }),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": (
                "Here's a comparison of the top 3 Python web frameworks:\n\n"
                "1. **Django** (82k stars) - Full-featured, batteries-included. Best for large applications. "
                "Performance: ~8k req/s\n"
                "2. **FastAPI** (78k stars) - Modern, async-first, auto-generated docs. "
                "Performance: ~45k req/s\n"
                "3. **Flask** (69k stars) - Lightweight, flexible, microframework. "
                "Performance: ~12k req/s\n\n"
                "FastAPI leads in raw performance due to its async architecture built on Starlette."
            )}],
        },

        # ===== Turn 2: Follow-up about FastAPI deployment =====
        {
            "role": "user",
            "content": [{"type": "text", "text": "How do I deploy a FastAPI app to production with Docker?"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me search for the best practices for deploying FastAPI with Docker."}],
            "tool_calls": [
                {
                    "id": "call_003",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({
                            "query": "FastAPI Docker deployment production best practices"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_003",
            "content": json.dumps({
                "results": [
                    {
                        "title": "FastAPI Official Docs - Deployment",
                        "summary": (
                            "Use multi-stage Docker build. Base: python:3.12-slim. "
                            "Install with pip. Run with uvicorn --workers 4. "
                            "Use gunicorn with uvicorn workers for production."
                        ),
                    },
                    {
                        "title": "Production FastAPI Setup Guide",
                        "summary": (
                            "Dockerfile: multi-stage build, non-root user, health check. "
                            "docker-compose for orchestration. Nginx reverse proxy. "
                            "Environment variables for config."
                        ),
                    },
                ]
            }),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": (
                "Here's how to deploy a FastAPI app to production with Docker:\n\n"
                "**Dockerfile (multi-stage):**\n"
                "```dockerfile\n"
                "FROM python:3.12-slim as base\n"
                "WORKDIR /app\n"
                "COPY requirements.txt .\n"
                "RUN pip install --no-cache-dir -r requirements.txt\n"
                "COPY . .\n"
                "CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\", \"--workers\", \"4\"]\n"
                "```\n\n"
                "**Key tips:**\n"
                "- Use `gunicorn` with `uvicorn.workers.UvicornWorker` for production\n"
                "- Run as non-root user for security\n"
                "- Add health check endpoint\n"
                "- Use Nginx as reverse proxy"
            )}],
        },
    ]

    return messages


def build_second_trajectory():
    """Build a second search agent trajectory to trigger clustering + skill extraction.

    Simulates a different research topic but same pattern:
    Turn 1: User asks about databases for real-time analytics -> Agent searches and compares
    Turn 2: User asks about ClickHouse Docker setup -> Agent searches

    This trajectory follows the same research pattern as the first (search -> compare ->
    recommend, then search -> deploy/setup guide), so it should cluster into the same
    MemScene and trigger AgentSkill extraction.
    """
    messages = [
        # ===== Turn 1: Database comparison for real-time analytics =====
        {
            "role": "user",
            "content": [{"type": "text", "text": "What's the best database for a real-time analytics dashboard handling 100k events/sec?"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "I'll research databases suited for real-time analytics at that scale."}],
            "tool_calls": [
                {
                    "id": "call_101",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({
                            "query": "best database real-time analytics high throughput 2026 comparison"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_101",
            "content": json.dumps({
                "results": [
                    {"title": "ClickHouse - 38k stars", "url": "https://github.com/ClickHouse/ClickHouse",
                     "summary": "Column-oriented OLAP database, 100M+ rows/sec aggregation"},
                    {"title": "TimescaleDB - 18k stars", "url": "https://github.com/timescale/timescaledb",
                     "summary": "PostgreSQL extension for time-series, familiar SQL interface"},
                    {"title": "Apache Druid - 13k stars", "url": "https://github.com/apache/druid",
                     "summary": "Real-time analytics database, sub-second OLAP queries"},
                ]
            }),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Found the top candidates. Let me get performance benchmarks for this scale."}],
            "tool_calls": [
                {
                    "id": "call_102",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({
                            "query": "ClickHouse vs TimescaleDB vs Druid benchmark throughput 2026"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_102",
            "content": json.dumps({
                "results": [
                    {
                        "title": "OLAP Database Benchmark 2026",
                        "summary": (
                            "ClickHouse: 1.2B rows/sec scan, 500k inserts/sec. "
                            "TimescaleDB: 200k inserts/sec, best for time-series with SQL joins. "
                            "Druid: 800k events/sec ingestion, sub-second queries on pre-aggregated data."
                        ),
                    }
                ]
            }),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": (
                "Here's a comparison of the top 3 databases for real-time analytics:\n\n"
                "1. **ClickHouse** (38k stars) - Column-oriented OLAP. Blazing fast aggregations "
                "over billions of rows. Best for: analytical queries, event analytics.\n"
                "   Throughput: 1.2B rows/sec scan, 500k inserts/sec\n"
                "2. **Apache Druid** (13k stars) - Real-time OLAP with sub-second queries. "
                "Best for: pre-aggregated dashboards, streaming ingestion.\n"
                "   Throughput: 800k events/sec ingestion\n"
                "3. **TimescaleDB** (18k stars) - PostgreSQL extension for time-series. "
                "Best for: teams already on PostgreSQL, complex SQL joins.\n"
                "   Throughput: 200k inserts/sec\n\n"
                "For 100k events/sec with ad-hoc analytics, **ClickHouse** is the strongest choice. "
                "If you need a simpler setup with PostgreSQL compatibility, go with TimescaleDB."
            )}],
        },

        # ===== Turn 2: Follow-up about ClickHouse Docker setup =====
        {
            "role": "user",
            "content": [{"type": "text", "text": "How do I set up ClickHouse with Docker for production?"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Let me search for ClickHouse Docker production setup best practices."}],
            "tool_calls": [
                {
                    "id": "call_103",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({
                            "query": "ClickHouse Docker production setup best practices 2026"
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_103",
            "content": json.dumps({
                "results": [
                    {
                        "title": "ClickHouse Docker Official Guide",
                        "summary": (
                            "Use clickhouse/clickhouse-server image. Mount /var/lib/clickhouse "
                            "for data persistence. Configure users.xml for auth. "
                            "Set max_memory_usage and max_threads for resource limits."
                        ),
                    },
                    {
                        "title": "Production ClickHouse Docker Deployment",
                        "summary": (
                            "docker-compose with persistent volumes. Tune MergeTree settings "
                            "for write-heavy workloads. Enable monitoring via system.metrics. "
                            "Use clickhouse-backup for automated snapshots."
                        ),
                    },
                ]
            }),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": (
                "Here's how to set up ClickHouse with Docker for production:\n\n"
                "**docker-compose.yml:**\n"
                "```yaml\n"
                "services:\n"
                "  clickhouse:\n"
                "    image: clickhouse/clickhouse-server:latest\n"
                "    ports:\n"
                "      - '8123:8123'  # HTTP\n"
                "      - '9000:9000'  # Native\n"
                "    volumes:\n"
                "      - clickhouse_data:/var/lib/clickhouse\n"
                "      - ./config/users.xml:/etc/clickhouse-server/users.xml\n"
                "    ulimits:\n"
                "      nofile: { soft: 262144, hard: 262144 }\n"
                "volumes:\n"
                "  clickhouse_data:\n"
                "```\n\n"
                "**Key tips:**\n"
                "- Mount `/var/lib/clickhouse` for data persistence across restarts\n"
                "- Set `max_memory_usage` to 80% of available RAM\n"
                "- Configure `users.xml` for authentication (don't use default user in prod)\n"
                "- Enable `system.metrics` table for monitoring with Grafana"
            )}],
        },
    ]

    return messages


# ==================== Main ====================

async def main():
    runner = AgentDemoRunner(
        group_id_prefix="search_agent_demo",
        group_name="Search Agent Demo Session",
        description="Agent Memory Demo - Search Agent",
        tags=["demo", "agent"],
        msg_prefix="search_msg",
    )

    print_separator("Search Agent Memory Demo - Web Search Trajectory + Agent Skill")

    # Step 1: Initialize conversation metadata
    print("\nStep 1: Initialize conversation metadata")
    print_separator()
    await runner.save_conversation_meta()

    # Step 2: Send first agent trajectory (Python web frameworks)
    print("\nStep 2: Send first search agent trajectory (Python frameworks)")
    print_separator()

    messages1 = build_mock_trajectory()
    print(f"  Sending {len(messages1)} messages (2 interaction turns)...\n")

    for idx, msg in enumerate(messages1):
        is_last = (idx == len(messages1) - 1)
        await runner.send_agent_message(msg, idx, flush=is_last)
        await asyncio.sleep(0.5)

    # Step 3: Wait for first extraction to complete
    print("\nStep 3: Wait for first extraction pipeline")
    print_separator()
    print("  Waiting 60 seconds for first AgentCase extraction...")
    await asyncio.sleep(60)

    # Step 4: Send second trajectory (database comparison - different topic, same pattern)
    print("\nStep 4: Send second trajectory (database comparison for analytics)")
    print_separator()
    print("  Different topic, but same research pattern (search -> compare -> deploy).")
    print("  It will cluster into the same MemScene and trigger AgentSkill extraction.\n")

    messages2 = build_second_trajectory()
    offset = len(messages1)
    print(f"  Sending {len(messages2)} messages (2 interaction turns)...\n")

    for idx, msg in enumerate(messages2):
        is_last = (idx == len(messages2) - 1)
        await runner.send_agent_message(msg, offset + idx, flush=is_last)
        await asyncio.sleep(0.5)

    # Step 5: Wait for second extraction + skill extraction
    print("\nStep 5: Wait for second extraction pipeline + AgentSkill extraction")
    print_separator()
    print("  Waiting 60 seconds for AgentCase + AgentSkill extraction...")
    await asyncio.sleep(60)

    # Step 6: Fetch and print all extracted memories
    print("\nStep 6: Fetch all extracted memories")

    for memory_type, label, printer in MEMORY_TYPE_PRINTERS:
        print(f"\n--- {label} ---")
        memories = await runner.fetch_memories(memory_type)
        print(f"  ({len(memories)} records)")
        printer(memories)

    # Step 7: Keyword search demo (BM25)
    print("\nStep 7: Keyword search (agent_memory, method=keyword)")

    search_queries = [
        "how to evaluate and compare tech solutions with benchmarks",
        "which database should I use for real-time analytics",
        "compare Python web frameworks performance",
        "best practices for technology research and production deployment",
        "how to compare and benchmark databases",
        "steps to evaluate Python frameworks by popularity and speed",
        "recommend a high-throughput database for event streaming",
    ]

    for query in search_queries:
        print(f"\n--- Keyword Search: Agent Memory ---")
        print(f"  Query: \"{query}\"")
        result = await runner.search_memories(
            query, "agent_memory", method="keyword"
        )
        cases = result.get("cases", [])
        skills = result.get("skills", [])
        print(f"  ({len(cases)} cases, {len(skills)} skills)")
        if cases:
            print("  [Cases]")
            print_search_case_results(cases)
        if skills:
            print("  [Skills]")
            print_search_skill_results(skills)

    # Step 8: Vector search demo (embedding similarity)
    print("\nStep 8: Vector search (agent_memory, method=vector)")

    for query in search_queries:
        print(f"\n--- Vector Search: Agent Memory ---")
        print(f"  Query: \"{query}\"")
        result = await runner.search_memories(
            query, "agent_memory", method="vector"
        )
        cases = result.get("cases", [])
        skills = result.get("skills", [])
        print(f"  ({len(cases)} cases, {len(skills)} skills)")
        if cases:
            print("  [Cases]")
            print_search_case_results(cases)
        if skills:
            print("  [Skills]")
            print_search_skill_results(skills)

    print_separator("Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
