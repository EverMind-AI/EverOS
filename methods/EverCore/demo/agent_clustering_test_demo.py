"""Clustering Test Demo - 4 Trajectories (3 distinct + 1 similar)

Tests whether agent memory clustering correctly separates different trajectory patterns
AND merges similar ones.

Sends 4 trajectories to the SAME group_id:
1. Code debugging: read logs -> locate bug -> write fix -> run tests (orders 500 error)
2. Data analysis: query database -> transform data -> plot chart -> summarize
3. Infrastructure provisioning: check inventory -> write config -> apply -> verify
4. Code debugging #2: read logs -> locate bug -> write fix -> run tests (auth 403 error)

Trajectory 4 follows the same pattern as Trajectory 1, so they should cluster together.
The other two should remain in separate clusters.

Expected result: 3 clusters (debugging x2, data analysis, infra).

Prerequisites:
    Start the API server first (in another terminal):
    uv run python src/run.py

Run the demo:
    uv run python src/bootstrap.py demo/agent_clustering_test_demo.py
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


# ==================== Trajectory 1: Code Debugging ====================

def build_debugging_trajectory():
    """Simulate a code debugging agent: read logs -> find bug -> fix -> test."""
    messages = [
        {
            "role": "user",
            "content": "The /api/orders endpoint is returning 500 errors in production. Can you investigate?",
        },
        {
            "role": "assistant",
            "content": "I'll check the application logs to find the root cause.",
            "tool_calls": [
                {
                    "id": "dbg_001",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({
                            "path": "/var/log/app/orders-service.log",
                            "tail": 100,
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "dbg_001",
            "content": json.dumps({
                "lines": [
                    "2026-03-15 10:23:45 ERROR OrderService.get_orders: TypeError: 'NoneType' object is not subscriptable",
                    "2026-03-15 10:23:45 ERROR   File 'order_service.py', line 42, in get_orders",
                    "2026-03-15 10:23:45 ERROR     total = order['items']['count']",
                ]
            }),
        },
        {
            "role": "assistant",
            "content": "Found it - a NoneType error on line 42. Let me read the source file.",
            "tool_calls": [
                {
                    "id": "dbg_002",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({
                            "path": "src/order_service.py",
                            "start_line": 35,
                            "end_line": 50,
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "dbg_002",
            "content": json.dumps({
                "content": (
                    "def get_orders(self, user_id):\n"
                    "    orders = self.db.query('SELECT * FROM orders WHERE user_id = ?', user_id)\n"
                    "    results = []\n"
                    "    for order in orders:\n"
                    "        items = self.db.query('SELECT * FROM order_items WHERE order_id = ?', order['id'])\n"
                    "        total = order['items']['count']  # BUG: should use len(items)\n"
                    "        results.append({'order': order, 'total': total})\n"
                    "    return results\n"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": "The bug is on line 42: `order['items']` is None because `items` is fetched separately. Applying fix.",
            "tool_calls": [
                {
                    "id": "dbg_003",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({
                            "path": "src/order_service.py",
                            "line": 42,
                            "old": "total = order['items']['count']",
                            "new": "total = len(items) if items else 0",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "dbg_003",
            "content": json.dumps({"status": "ok", "message": "Line 42 updated"}),
        },
        {
            "role": "assistant",
            "content": "Fix applied. Let me run the test suite to make sure nothing else breaks.",
            "tool_calls": [
                {
                    "id": "dbg_004",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({
                            "command": "pytest tests/test_order_service.py -v",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "dbg_004",
            "content": json.dumps({
                "exit_code": 0,
                "output": "5 passed in 1.2s",
            }),
        },
        {
            "role": "assistant",
            "content": (
                "Fixed the 500 error on /api/orders.\n\n"
                "**Root cause:** Line 42 in `order_service.py` accessed `order['items']['count']`, "
                "but `order['items']` was None because item data is fetched in a separate query.\n\n"
                "**Fix:** Changed to `len(items) if items else 0` to use the locally-fetched items list.\n\n"
                "All 5 tests pass."
            ),
        },
    ]
    return messages


# ==================== Trajectory 2: Data Analysis ====================

def build_data_analysis_trajectory():
    """Simulate a data analysis agent: query -> transform -> visualize -> summarize."""
    messages = [
        {
            "role": "user",
            "content": "Analyze our Q1 2026 sales data and create a breakdown by region and product category.",
        },
        {
            "role": "assistant",
            "content": "I'll query the sales database for Q1 2026 data first.",
            "tool_calls": [
                {
                    "id": "da_001",
                    "type": "function",
                    "function": {
                        "name": "run_sql",
                        "arguments": json.dumps({
                            "query": (
                                "SELECT region, category, SUM(revenue) as total_revenue, "
                                "COUNT(*) as order_count "
                                "FROM sales WHERE sale_date BETWEEN '2026-01-01' AND '2026-03-31' "
                                "GROUP BY region, category ORDER BY total_revenue DESC"
                            ),
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "da_001",
            "content": json.dumps({
                "rows": [
                    {"region": "North America", "category": "Electronics", "total_revenue": 4200000, "order_count": 12500},
                    {"region": "North America", "category": "Clothing", "total_revenue": 2800000, "order_count": 18000},
                    {"region": "Europe", "category": "Electronics", "total_revenue": 3100000, "order_count": 9800},
                    {"region": "Europe", "category": "Clothing", "total_revenue": 1900000, "order_count": 11200},
                    {"region": "Asia Pacific", "category": "Electronics", "total_revenue": 3800000, "order_count": 15600},
                    {"region": "Asia Pacific", "category": "Clothing", "total_revenue": 1500000, "order_count": 8900},
                ],
                "total_rows": 6,
            }),
        },
        {
            "role": "assistant",
            "content": "Got the data. Let me compute growth rates compared to Q4 2025.",
            "tool_calls": [
                {
                    "id": "da_002",
                    "type": "function",
                    "function": {
                        "name": "run_sql",
                        "arguments": json.dumps({
                            "query": (
                                "SELECT region, category, SUM(revenue) as total_revenue "
                                "FROM sales WHERE sale_date BETWEEN '2025-10-01' AND '2025-12-31' "
                                "GROUP BY region, category"
                            ),
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "da_002",
            "content": json.dumps({
                "rows": [
                    {"region": "North America", "category": "Electronics", "total_revenue": 3900000},
                    {"region": "North America", "category": "Clothing", "total_revenue": 2600000},
                    {"region": "Europe", "category": "Electronics", "total_revenue": 3000000},
                    {"region": "Europe", "category": "Clothing", "total_revenue": 1850000},
                    {"region": "Asia Pacific", "category": "Electronics", "total_revenue": 3200000},
                    {"region": "Asia Pacific", "category": "Clothing", "total_revenue": 1400000},
                ],
            }),
        },
        {
            "role": "assistant",
            "content": "Now I'll generate the visualization.",
            "tool_calls": [
                {
                    "id": "da_003",
                    "type": "function",
                    "function": {
                        "name": "run_python",
                        "arguments": json.dumps({
                            "code": (
                                "import matplotlib.pyplot as plt\n"
                                "import pandas as pd\n"
                                "# ... grouped bar chart by region and category\n"
                                "fig.savefig('q1_sales_breakdown.png', dpi=150)\n"
                                "print('Chart saved to q1_sales_breakdown.png')"
                            ),
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "da_003",
            "content": json.dumps({
                "output": "Chart saved to q1_sales_breakdown.png",
            }),
        },
        {
            "role": "assistant",
            "content": (
                "Here's the Q1 2026 sales analysis:\n\n"
                "**Revenue by Region:**\n"
                "- North America: $7.0M (+7.7% QoQ) - strongest in Electronics\n"
                "- Asia Pacific: $5.3M (+15.2% QoQ) - fastest growing region\n"
                "- Europe: $5.0M (+3.1% QoQ) - steady growth\n\n"
                "**Key Insights:**\n"
                "1. Asia Pacific Electronics grew 18.8% QoQ - driven by semiconductor demand\n"
                "2. Clothing is weaker across all regions, suggesting seasonal effect\n"
                "3. Total Q1 revenue: $17.3M, up 8.5% from Q4 2025\n\n"
                "Chart saved to `q1_sales_breakdown.png`."
            ),
        },
    ]
    return messages


# ==================== Trajectory 3: Infrastructure Provisioning ====================

def build_infra_trajectory():
    """Simulate an infra agent: check inventory -> write config -> apply -> verify."""
    messages = [
        {
            "role": "user",
            "content": "We need a new Redis cluster for the recommendation service. 3 nodes, 16GB each, in us-east-1.",
        },
        {
            "role": "assistant",
            "content": "Let me check current infrastructure inventory for available capacity.",
            "tool_calls": [
                {
                    "id": "infra_001",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({
                            "command": "terraform state list | grep 'aws_elasticache'",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "infra_001",
            "content": json.dumps({
                "output": (
                    "aws_elasticache_cluster.session_cache\n"
                    "aws_elasticache_cluster.rate_limiter\n"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": "Two existing clusters found. I'll write the Terraform config for the new one.",
            "tool_calls": [
                {
                    "id": "infra_002",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": json.dumps({
                            "path": "infra/modules/redis/recommendation.tf",
                            "content": (
                                'resource "aws_elasticache_replication_group" "recommendation" {\n'
                                '  replication_group_id = "recommendation-redis"\n'
                                '  description          = "Redis cluster for recommendation service"\n'
                                '  node_type            = "cache.r6g.xlarge"  # 16GB\n'
                                '  num_cache_clusters   = 3\n'
                                '  engine_version       = "7.0"\n'
                                '  port                 = 6379\n'
                                '  subnet_group_name    = aws_elasticache_subnet_group.main.name\n'
                                '  security_group_ids   = [aws_security_group.redis.id]\n'
                                '  at_rest_encryption_enabled = true\n'
                                '  transit_encryption_enabled = true\n'
                                "}\n"
                            ),
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "infra_002",
            "content": json.dumps({"status": "ok", "message": "File written"}),
        },
        {
            "role": "assistant",
            "content": "Config written. Running terraform plan to validate.",
            "tool_calls": [
                {
                    "id": "infra_003",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({
                            "command": "terraform plan -target=aws_elasticache_replication_group.recommendation",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "infra_003",
            "content": json.dumps({
                "output": (
                    "Plan: 1 to add, 0 to change, 0 to destroy.\n"
                    "Estimated monthly cost: $1,152.00 (3x cache.r6g.xlarge)"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": "Plan looks good. Applying now.",
            "tool_calls": [
                {
                    "id": "infra_004",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({
                            "command": "terraform apply -target=aws_elasticache_replication_group.recommendation -auto-approve",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "infra_004",
            "content": json.dumps({
                "output": (
                    "aws_elasticache_replication_group.recommendation: Creating...\n"
                    "aws_elasticache_replication_group.recommendation: Creation complete after 8m32s\n"
                    "Apply complete! Resources: 1 added, 0 changed, 0 destroyed.\n"
                    "Outputs:\n"
                    "  endpoint = recommendation-redis.abc123.use1.cache.amazonaws.com:6379"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": "Let me verify the cluster is healthy.",
            "tool_calls": [
                {
                    "id": "infra_005",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({
                            "command": "redis-cli -h recommendation-redis.abc123.use1.cache.amazonaws.com ping",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "infra_005",
            "content": json.dumps({"output": "PONG"}),
        },
        {
            "role": "assistant",
            "content": (
                "Redis cluster for recommendation service is ready.\n\n"
                "**Configuration:**\n"
                "- 3x `cache.r6g.xlarge` nodes (16GB each) in us-east-1\n"
                "- Engine: Redis 7.0 with TLS encryption\n"
                "- Endpoint: `recommendation-redis.abc123.use1.cache.amazonaws.com:6379`\n"
                "- Estimated cost: $1,152/month\n\n"
                "Encryption at rest and in transit are both enabled. "
                "The cluster is responding to PING and ready for the recommendation service to connect."
            ),
        },
    ]
    return messages


# ==================== Trajectory 4: Code Debugging #2 (similar to #1) ====================

def build_debugging_trajectory_2():
    """Simulate another code debugging session: read logs -> find bug -> fix -> test.

    Different service (auth) and different error (KeyError), but same agent pattern
    as Trajectory 1. Should cluster together with it.
    """
    messages = [
        {
            "role": "user",
            "content": "Users are getting 403 Forbidden on the /api/auth/login endpoint since this morning. Please look into it.",
        },
        {
            "role": "assistant",
            "content": "I'll pull up the authentication service logs to see what's happening.",
            "tool_calls": [
                {
                    "id": "dbg2_001",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({
                            "path": "/var/log/app/auth-service.log",
                            "tail": 80,
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "dbg2_001",
            "content": json.dumps({
                "lines": [
                    "2026-03-15 08:01:12 ERROR AuthHandler.authenticate: KeyError: 'password_hash'",
                    "2026-03-15 08:01:12 ERROR   File 'auth_handler.py', line 67, in authenticate",
                    "2026-03-15 08:01:12 ERROR     stored_hash = user_record['password_hash']",
                    "2026-03-15 08:01:12 WARN  Failed login attempts spiking: 1,240 in last hour",
                ]
            }),
        },
        {
            "role": "assistant",
            "content": "There's a KeyError on 'password_hash' at line 67. Let me read the source code.",
            "tool_calls": [
                {
                    "id": "dbg2_002",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": json.dumps({
                            "path": "src/auth_handler.py",
                            "start_line": 60,
                            "end_line": 80,
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "dbg2_002",
            "content": json.dumps({
                "content": (
                    "def authenticate(self, username, password):\n"
                    "    user_record = self.db.find_one('users', {'username': username})\n"
                    "    if not user_record:\n"
                    "        raise AuthError('User not found')\n"
                    "    # After DB migration, column renamed from 'password_hash' to 'hashed_pw'\n"
                    "    stored_hash = user_record['password_hash']  # BUG: old column name\n"
                    "    if not bcrypt.checkpw(password.encode(), stored_hash):\n"
                    "        raise AuthError('Invalid credentials')\n"
                    "    return self._create_session(user_record)\n"
                ),
            }),
        },
        {
            "role": "assistant",
            "content": (
                "Found it. After a DB migration the column was renamed from `password_hash` to "
                "`hashed_pw`, but the code still uses the old name. Fixing now."
            ),
            "tool_calls": [
                {
                    "id": "dbg2_003",
                    "type": "function",
                    "function": {
                        "name": "edit_file",
                        "arguments": json.dumps({
                            "path": "src/auth_handler.py",
                            "line": 67,
                            "old": "stored_hash = user_record['password_hash']",
                            "new": "stored_hash = user_record['hashed_pw']",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "dbg2_003",
            "content": json.dumps({"status": "ok", "message": "Line 67 updated"}),
        },
        {
            "role": "assistant",
            "content": "Fix applied. Running the auth test suite to verify.",
            "tool_calls": [
                {
                    "id": "dbg2_004",
                    "type": "function",
                    "function": {
                        "name": "run_command",
                        "arguments": json.dumps({
                            "command": "pytest tests/test_auth_handler.py -v",
                        }),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "dbg2_004",
            "content": json.dumps({
                "exit_code": 0,
                "output": "8 passed in 2.1s",
            }),
        },
        {
            "role": "assistant",
            "content": (
                "Fixed the 403 errors on /api/auth/login.\n\n"
                "**Root cause:** A recent DB migration renamed the `password_hash` column to "
                "`hashed_pw`, but `auth_handler.py` line 67 still referenced the old column name, "
                "causing a KeyError that surfaced as 403 Forbidden.\n\n"
                "**Fix:** Updated the column reference from `user_record['password_hash']` to "
                "`user_record['hashed_pw']`.\n\n"
                "All 8 auth tests pass."
            ),
        },
    ]
    return messages


# ==================== Main ====================

async def main():
    runner = AgentDemoRunner(
        group_id_prefix="clustering_test",
        group_name="Clustering Test Session",
        description="Test clustering with 3 distinct + 1 similar trajectory patterns",
        tags=["demo", "agent", "clustering-test"],
        msg_prefix="cluster_msg",
    )

    print_separator("Clustering Test Demo - 4 Trajectories (3 distinct + 1 similar)")

    # Step 1: Initialize conversation metadata
    print("\nStep 1: Initialize conversation metadata")
    print_separator()
    await runner.save_conversation_meta()

    trajectories = [
        ("Code Debugging #1 (orders 500 error: read logs -> find bug -> fix -> test)", build_debugging_trajectory),
        ("Data Analysis (query DB -> transform -> visualize -> summarize)", build_data_analysis_trajectory),
        ("Infra Provisioning (check inventory -> write config -> apply -> verify)", build_infra_trajectory),
        ("Code Debugging #2 (auth 403 error: read logs -> find bug -> fix -> test)", build_debugging_trajectory_2),
    ]

    msg_offset = 0

    for traj_idx, (label, builder) in enumerate(trajectories, 1):
        print(f"\nStep {traj_idx + 1}: Send trajectory #{traj_idx} - {label}")
        print_separator()

        messages = builder()
        print(f"  Sending {len(messages)} messages...\n")

        for idx, msg in enumerate(messages):
            is_last = (idx == len(messages) - 1)
            await runner.send_agent_message(msg, msg_offset + idx, flush=is_last)
            await asyncio.sleep(0.3)

        msg_offset += len(messages)

        if traj_idx < len(trajectories):
            print(f"\n  Waiting 5 seconds before next trajectory...")
            await asyncio.sleep(5)

    # Wait for extraction + clustering
    print(f"\nStep {len(trajectories) + 2}: Wait for extraction + clustering pipeline")
    print_separator()
    print("  Waiting 60 seconds for MemCell + AgentCase + clustering + AgentSkill...")
    await asyncio.sleep(60)

    # Fetch and print all extracted memories
    print(f"\nStep {len(trajectories) + 3}: Fetch all extracted memories")

    for memory_type, label, printer in MEMORY_TYPE_PRINTERS:
        print(f"\n--- {label} ---")
        memories = await runner.fetch_memories(memory_type)
        print(f"  ({len(memories)} records)")
        printer(memories)

    # Search to verify clustering separation
    print(f"\nStep {len(trajectories) + 4}: Search to verify clustering separation")

    search_queries = [
        ("debug production error by reading logs and fixing code", "debugging"),
        ("analyze sales data with SQL queries and create charts", "data analysis"),
        ("provision cloud infrastructure with Terraform", "infra"),
        ("how to debug production errors in a web service", "debugging"),
        ("how to analyze business data and generate reports", "data analysis"),
        ("how to provision and verify cloud infrastructure", "infra"),
    ]

    for query, label in search_queries:
        print(f"\n--- Hybrid Search: Agent Memory ({label}) ---")
        print(f'  Query: "{query}"')
        result = await runner.search_memories(query, "agent_memory")
        cases = result.get("cases", [])
        skills = result.get("skills", [])
        print(f"  ({len(cases)} cases, {len(skills)} skills)")
        if cases:
            print("  [Cases]")
            print_search_case_results(cases)
        if skills:
            print("  [Skills]")
            print_search_skill_results(skills)

    # Agentic search for agent memory
    print(f"\nStep {len(trajectories) + 5}: Agentic search for Agent Memory (LLM-guided multi-round retrieval)")

    agentic_queries = [
        ("how to debug production errors in a web service", "debugging"),
        ("how to analyze business data and generate reports", "data analysis"),
        ("how to provision and verify cloud infrastructure", "infra"),
    ]

    for query, label in agentic_queries:
        print(f"\n--- Agentic Search: Agent Memory ({label}) ---")
        print(f'  Query: "{query}"')
        result = await runner.search_memories(
            query, "agent_memory", retrieve_method="agentic"
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
