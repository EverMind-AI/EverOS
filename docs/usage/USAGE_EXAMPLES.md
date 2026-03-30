# Usage Examples

[Home](../../README.md) > [Docs](../README.md) > [Usage](.) > Usage Examples

This guide provides comprehensive examples of how to use EverMemOS in different scenarios.

---

## Table of Contents

1. [Simple Demo - Quick Start](#1-simple-demo---quick-start)
2. [Full Demo - Memory Extraction & Chat](#2-full-demo---memory-extraction--chat)
3. [Evaluation & Performance Testing](#3-evaluation--performance-testing)
4. [Direct API Usage](#4-direct-api-usage)
5. [Batch Operations](#5-batch-operations)
6. [Advanced Integration](#6-advanced-integration)

---

## Prerequisites

Before using these examples, ensure you have:

1. **Completed installation** - See [Setup Guide](../installation/SETUP.md)
2. **Started the API server**:
   ```bash
   uv run python src/run.py --port 1995
   ```
3. **Configured .env** with required API keys

---

## 1. Simple Demo - Quick Start

The fastest way to experience EverMemOS! Just 2 steps to see memory storage and retrieval in action.

### What It Does

- Stores 4 conversation messages about sports hobbies
- Waits 10 seconds for indexing
- Searches for relevant memories with 3 different queries
- Shows complete workflow with friendly explanations

### Usage

```bash
# Terminal 1: Start the API server
uv run python src/run.py --port 1995

# Terminal 2: Run the simple demo
uv run python src/bootstrap.py demo/simple_demo.py
```

### Expected Output

You'll see:
1. Messages being stored
2. Indexing progress
3. Search results for queries like "What sports does the user like?"
4. Relevant memories retrieved with scores

### Demo Code

See the complete code at [`demo/simple_demo.py`](../../demo/simple_demo.py)

### Perfect For

- First-time users
- Quick testing
- Understanding core concepts
- Verifying installation

---

## 2. Full Demo - Memory Extraction & Chat

Experience the complete EverMemOS workflow: memory extraction from conversations followed by interactive chat with memory retrieval.

### Prerequisites

**Start the API Server:**

```bash
# Terminal 1: Start the API server (required)
uv run python src/run.py --port 1995
```

> 💡 **Tip**: Keep the API server running throughout. All following operations should be performed in another terminal.

---

### Step 1: Extract Memories

Run the memory extraction script to process sample conversation data and build the memory database:

```bash
# Terminal 2: Run the extraction script
uv run python src/bootstrap.py demo/extract_memory.py
```

**What This Script Does:**

1. Calls `demo.tools.clear_all_data.clear_all_memories()` so the demo starts from an empty MongoDB/Elasticsearch/Milvus/Redis state. Ensure the dependency stack launched by `docker-compose` is running before executing the script, otherwise the wipe step will fail.

2. Loads `data/assistant_chat_zh.json`, appends `scene="assistant"` to each message, and streams every entry to `http://localhost:1995/api/v1/memories`.

3. Update the `base_url`, `data_file`, or `profile_scene` constants in `demo/extract_memory.py` if you host the API on another endpoint or want to ingest a different scenario.

4. Writes through the HTTP API only: MemCells, episodes, and profiles are created inside your databases, not under `demo/memcell_outputs/`. Inspect MongoDB (and Milvus/Elasticsearch) to verify ingestion or proceed directly to the chat demo.

> **💡 Tip**: For detailed configuration instructions and usage guide, please refer to the [Demo Documentation](../../demo/README.md).

---

### Step 2: Chat with Memory

After extracting memories, start the interactive chat demo:

```bash
# Terminal 2: Run the chat program (ensure API server is still running)
uv run python src/bootstrap.py demo/chat_with_memory.py
```

**How It Works:**

This program loads `.env` via `python-dotenv`, verifies that at least one LLM key (`LLM_API_KEY`, `OPENROUTER_API_KEY`, or `OPENAI_API_KEY`) is available, and connects to MongoDB through `demo.utils.ensure_mongo_beanie_ready` to enumerate groups that already contain MemCells.

Each user query invokes `api/v1/memories/search` unless you explicitly select the Agentic mode, in which case the orchestrator switches to agentic retrieval and warns about the additional LLM latency.

### Interactive Workflow

1. **Select Language**: Choose a zh or en terminal UI.
2. **Select Scenario Mode**: Assistant (one-on-one) or Group Chat (multi-speaker analysis).
3. **Select Conversation Group**: Groups are read live from MongoDB via `query_all_groups_from_mongodb`; run the extraction step first so the list is non-empty.
4. **Select Retrieval Mode**: `rrf`, `vector`, `keyword`, or LLM-guided Agentic retrieval.
5. **Start Chatting**: Pose questions, inspect the retrieved memories that are displayed before each response, and use `help`, `clear`, `reload`, or `exit` to manage the session.

---

## 3. Evaluation & Performance Testing

The evaluation framework provides a unified, modular way to benchmark memory systems on standard datasets (LoCoMo, LongMemEval, PersonaMem).

### Quick Test (Smoke Test)

Verify everything works with limited data:

```bash
# Default smoke test
# First conversation, first 10 messages, first 3 questions
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# Custom smoke test: 20 messages, 5 questions
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --smoke-messages 20 --smoke-questions 5

# Test different datasets
uv run python -m evaluation.cli --dataset longmemeval --system evermemos --smoke
uv run python -m evaluation.cli --dataset personamem --system evermemos --smoke

# Test specific stages (e.g., only search and answer)
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --stages search answer

# View smoke test results quickly
cat evaluation/results/locomo-evermemos-smoke/report.txt
```

### Full Evaluation

Run complete evaluation on entire datasets:

```bash
# Evaluate EvermemOS on LoCoMo benchmark
uv run python -m evaluation.cli --dataset locomo --system evermemos

# Evaluate on other datasets
uv run python -m evaluation.cli --dataset longmemeval --system evermemos
uv run python -m evaluation.cli --dataset personamem --system evermemos

# Use --run-name to distinguish multiple runs (useful for A/B testing)
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name baseline
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name experiment1

# Resume from checkpoint if interrupted (automatic)
# Just re-run the same command - it will detect and resume from checkpoint
uv run python -m evaluation.cli --dataset locomo --system evermemos
```

### View Results

```bash
# Results are saved to evaluation/results/{dataset}-{system}[-{run-name}]/
cat evaluation/results/locomo-evermemos/report.txt          # Summary metrics
cat evaluation/results/locomo-evermemos/eval_results.json   # Detailed per-question results
cat evaluation/results/locomo-evermemos/pipeline.log        # Execution logs
```

### Evaluation Pipeline

The evaluation pipeline consists of 4 stages with automatic checkpointing and resume support:

1. **Add** - Ingest conversation data into the system
2. **Search** - Retrieve relevant memories for each question
3. **Answer** - Generate answers using retrieved context
4. **Evaluate** - Score answers against ground truth

### Configuration

> **⚙️ Evaluation Configuration**:
> - **Data Preparation**: Place datasets in `evaluation/data/` (see `evaluation/README.md`)
> - **Environment**: Configure `.env` with LLM API keys (see `env.template`)
> - **Installation**: Run `uv sync --group evaluation` to install dependencies
> - **Custom Config**: Copy and modify YAML files in `evaluation/config/systems/` or `evaluation/config/datasets/`
> - **Advanced Usage**: See `evaluation/README.md` for checkpoint management, stage-specific runs, and system comparisons

---

## 4. Direct API Usage

Use the Memory API to integrate EverMemOS into your application.

### Prerequisites

**Start the API Server:**

```bash
uv run python src/run.py --port 1995
```

> 💡 **Tip**: Keep the API server running throughout. All following API calls should be performed in another terminal.

---

### Store Single Message Memory

Use the `/api/v1/memories` endpoint to store individual messages:

**Minimal Example (Required Fields Only):**

```bash
curl -X POST http://localhost:1995/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+00:00",
    "sender": "user_001",
    "content": "I love playing soccer on weekends"
  }'
```

**With Optional Fields:**

```bash
curl -X POST http://localhost:1995/api/v1/memories \
  -H "Content-Type: application/json" \
  -d '{
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+00:00",
    "sender": "user_103",
    "sender_name": "Chen",
    "content": "We need to complete the product design this week",
    "group_id": "group_001",
    "group_name": "Project Discussion Group"
  }'
```

> ℹ️ **Required fields**: `message_id`, `create_time`, `sender`, `content`
> ℹ️ **Optional fields**: `group_id`, `group_name`, `sender_name`, `role`, `refer_list`
> ℹ️ By default, all memory types are extracted and stored

### API Endpoints

- **`POST /api/v1/memories`**: Store single message memory
- **`GET /api/v1/memories/search`**: Memory retrieval (supports keyword/vector/hybrid search modes)

For complete API documentation, see [Memory API Documentation](../api_docs/memory_api.md).

---

### Retrieve Memories

EverMemOS provides two retrieval modes: **Lightweight** (fast) and **Agentic** (intelligent).

#### Lightweight Retrieval

Fast retrieval for latency-sensitive scenarios.

**Parameters:**

| Parameter | Required | Description |
|-----------|----------|-------------|
| `query` | Yes* | Natural language query (*optional for profile type) |
| `user_id` | No* | User ID |
| `group_id` | No* | Group ID |
| `memory_types` | No | `["episodic_memory"]` / `["event_log"]` / `["foresight"]` (default: `["episodic_memory"]`) |
| `retrieve_method` | No | `keyword` / `vector` / `hybrid` / `rrf` (recommended) / `agentic` |
| `current_time` | No | Filter valid foresight (format: ISO 8601) |
| `top_k` | No | Number of results (default: 40, max: 100) |

*At least one of `user_id` or `group_id` must be provided.

**Example 1: Personal Memory**

```bash
curl -X GET http://localhost:1995/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What sports does the user like?",
    "user_id": "user_001",
    "memory_types": ["episodic_memory"],
    "retrieve_method": "rrf"
  }'
```

**Example 2: Group Memory**

```bash
curl -X GET http://localhost:1995/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Discuss project progress",
    "group_id": "project_team_001",
    "memory_types": ["episodic_memory"],
    "retrieve_method": "rrf"
  }'
```

> 📖 Full Documentation: [Memory API](../api_docs/memory_api.md) | Testing Tool: `demo/tools/test_retrieval_comprehensive.py`

---

## 5. Batch Operations

Process multiple messages efficiently using batch scripts.

See the dedicated [Batch Operations Guide](BATCH_OPERATIONS.md) for complete information.

### Quick Example

```bash
# Batch store group chat messages (Chinese data)
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_zh.json \
  --api-url http://localhost:1995/api/v1/memories \
  --scene group_chat

# Or use English data
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_en.json \
  --api-url http://localhost:1995/api/v1/memories \
  --scene group_chat

# Validate file format
uv run python src/bootstrap.py src/run_memorize.py \
  --input data/group_chat_en.json \
  --scene group_chat \
  --validate-only
```

> ℹ️ **Scene Parameter Explanation**: The `scene` parameter is required and specifies the memory extraction strategy:
> - Use `assistant` for one-on-one conversations with AI assistant
> - Use `group_chat` for multi-person group discussions

For complete details, see:
- [Batch Operations Guide](BATCH_OPERATIONS.md)
- [Group Chat Format Specification](../../data_format/group_chat/group_chat_format.md)

---

## 6. Advanced Integration

### Python SDK Usage

Use EverMemOS in your Python applications:

```python
import requests

class EverMemOSClient:
    def __init__(self, base_url="http://localhost:1995"):
        self.base_url = base_url

    def store_memory(self, message):
        """Store a single message memory."""
        url = f"{self.base_url}/api/v1/memories"
        response = requests.post(url, json=message)
        response.raise_for_status()
        return response.json()

    def search_memories(self, query, user_id=None, **kwargs):
        """Search for relevant memories."""
        url = f"{self.base_url}/api/v1/memories/search"
        params = {"query": query, **kwargs}
        if user_id:
            params["user_id"] = user_id

        response = requests.get(url, json=params)
        response.raise_for_status()
        return response.json()

# Usage
client = EverMemOSClient()

# Store memory
client.store_memory({
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+00:00",
    "sender": "user_001",
    "content": "I love playing soccer on weekends"
})

# Search memories
results = client.search_memories(
    query="What sports does the user like?",
    user_id="user_001",
    memory_types=["episodic_memory"],
    retrieve_method="rrf"
)

print(results)
```

### Custom Integration Patterns

For advanced integration scenarios:

1. **Streaming Conversations**: Integrate with chat applications to continuously store messages
2. **Custom Memory Types**: Extend the extraction pipeline for domain-specific memories
3. **Multi-tenant Systems**: Use `user_id` and `group_id` for isolation
4. **Real-time Retrieval**: Implement caching strategies for frequently accessed memories

See [API Usage Guide](../dev_docs/api_usage_guide.md) for more examples.

---

## See Also

- [Demo Guide](DEMOS.md) - Detailed demo walkthroughs
- [Batch Operations Guide](BATCH_OPERATIONS.md) - Batch processing details
- [Memory API Documentation](../api_docs/memory_api.md) - Complete API reference
- [API Usage Guide](../dev_docs/api_usage_guide.md) - Advanced API patterns
- [Evaluation Guide](../../evaluation/README.md) - Benchmarking documentation
