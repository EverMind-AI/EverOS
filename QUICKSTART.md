# Quickstart

> Five minutes from one API key to durable Markdown memory and keyword recall.

EverOS runs as a local service. The minimum production path needs only an LLM:
configure one OpenRouter key, start the server, then call the HTTP API.

## What the one-key setup includes

With only `[llm]` configured, EverOS can:

- start the server;
- extract conversations into durable Markdown;
- keep the local index in sync; and
- retrieve memories with keyword search.

Embedding, rerank, knowledge, and multimodal providers are optional upgrades.
They are not required for this walkthrough.

## Prerequisites

- Python 3.12+
- One [OpenRouter API key](https://openrouter.ai/keys)

## 1. Install

From PyPI:

```bash
pip install everos
# or: uv pip install everos
```

From source:

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
uv sync
source .venv/bin/activate
```

You can also prefix source-checkout commands with `uv run` instead of
activating the virtual environment.

## 2. Initialize EverOS

```bash
everos init
```

This creates two files under the default memory root:

```text
~/.everos/
├── everos.toml    # provider and server configuration
└── ome.toml       # memory strategy configuration
```

To use another root, run `everos init --root <path>` and pass the same
`--root <path>` to subsequent commands.

## 3. Add your OpenRouter key

Open `~/.everos/everos.toml`. The generated `[llm]` section already contains
the recommended model and base URL; replace only the empty `api_key`:

```toml
[llm]
model = "openai/gpt-4.1-mini"
api_key = "<OPENROUTER_API_KEY>"
base_url = "https://openrouter.ai/api/v1"
```

Leave `[embedding]`, `[rerank]`, and `[multimodal]` unchanged for this
walkthrough. Their empty keys make those capabilities unavailable without
preventing the core server from starting.

## 4. Start the server

```bash
everos server start
```

The server runs in the foreground on `http://127.0.0.1:8000`. Open a second
terminal and verify it:

```bash
curl http://127.0.0.1:8000/health
```

The response includes the complete capability matrix. In the one-key setup,
the important fields look like this:

```json
{
  "status": "ok",
  "capabilities": {
    "llm": true,
    "embed": false,
    "rerank": false
  },
  "disabled_features": [
    "vector_search",
    "hybrid_search",
    "agentic_search",
    "reflection",
    "skill_extraction",
    "knowledge"
  ]
}
```

The actual response also includes version, multimodal/parser capabilities, and
cascade readiness.

> [!NOTE]
> EverOS opens local index files during concurrent search and indexing. If you
> encounter file-descriptor errors, run `ulimit -n 4096` in the same shell
> before starting the server.

## 5. Add a conversation

Business endpoints live under `/api/v2`. The `/api/v1` prefix remains a legacy
compatibility alias, but new integrations should use `/api/v2`.

Timestamps are Unix epoch milliseconds in UTC:

```bash
TS=$(($(date +%s)*1000))

curl -X POST http://127.0.0.1:8000/api/v2/memory/add \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"demo-001\",
    \"app_id\": \"default\",
    \"project_id\": \"default\",
    \"messages\": [
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $TS, \"content\": \"I love climbing in Yosemite every spring.\"},
      {\"sender_id\": \"agent1\", \"role\": \"assistant\", \"timestamp\": $((TS+10000)), \"content\": \"Which routes do you enjoy most?\"},
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $((TS+20000)), \"content\": \"Mostly the cracks on El Cap.\"}
    ]
  }"
```

Messages are buffered by session until EverOS detects a boundary or the client
explicitly flushes the session.

## 6. Flush at the end of the session

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/flush \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "demo-001",
    "app_id": "default",
    "project_id": "default"
  }'
```

A successful flush returns `data.status` as `"extracted"`. The extraction is
written to Markdown, then the cascade worker projects it into the local index.

## 7. Search with the one-key method

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "app_id": "default",
    "project_id": "default",
    "query": "Where does Alice like to climb?",
    "method": "keyword",
    "top_k": 5
  }'
```

The response should contain an episode whose summary mentions Yosemite or El
Cap. If the first search is empty, wait a moment for cascade indexing and retry.

> [!IMPORTANT]
> Include `"method": "keyword"` when only the LLM is configured. The search
> API defaults to `"hybrid"`, which requires an embedding provider and returns
> HTTP 422 in the one-key tier.

Keyword retrieval returns matching episodes from the local BM25 index. Atomic
facts are created by an optional embedding-dependent strategy, so they are not
expected in the Tier 1 response.

## 8. Read the Markdown source of truth

Your extracted memory is a normal Markdown file under the memory root:

```text
~/.everos/
├── default_app/
│   └── default_project/
│       ├── users/alice/
│       │   ├── user.md
│       │   ├── episodes/
│       │   ├── .atomic_facts/
│       │   └── .foresights/
│       ├── agents/<agent_id>/
│       │   ├── agent.md
│       │   ├── .cases/
│       │   └── skills/
│       └── knowledge/
├── everos.toml
├── ome.toml
└── .index/
    ├── sqlite/system.db
    └── lancedb/
```

Markdown is canonical; SQLite and LanceDB are derived indexes. You can read,
edit, diff, and version the memory files without a database client.

## Upgrade capabilities when you need them

The generated `everos.toml` already includes commented guidance and default
models for the optional providers.

| Configuration | Available capabilities |
| --- | --- |
| `[llm]` only | Add, flush, Markdown persistence, cascade sync, keyword search |
| Add `[embedding]` | Vector/user hybrid search, reflection, skill extraction |
| Add `[rerank]` too | Agentic search, default agent hybrid search, Knowledge Wiki |
| Add `[multimodal]` and install `everos[multimodal]` | Image, PDF, audio, and office-file ingestion |

EverOS reports unavailable features through `/health`. Requests that require a
missing provider fail fast with a descriptive HTTP 422 instead of silently
degrading to a different search method.

You can replace OpenRouter with another OpenAI-compatible LLM endpoint by
changing the `[llm]` model, base URL, and key.

## Optional: run the no-key visualizer

`everos demo` is a local educational visualizer. It is hardcoded and does not
connect to the server, so it needs no API key:

```bash
everos demo
# or a copyable, non-interactive preview:
everos demo --plain
```

The live demo currently searches with the default hybrid method. Configure an
embedding provider before running `everos demo --live`.

## Stop the server

Press `Ctrl+C` in the server terminal.

## Next steps

- Integrate `/add`, `/flush`, and `/search` into your agent loop.
- Partition memory with `app_id` and `project_id`.
- Explore the full API contract in [docs/openapi.json](docs/openapi.json).
- Configure advanced retrieval in the generated `everos.toml`.
- Read [docs/architecture.md](docs/architecture.md) and
  [docs/storage_layout.md](docs/storage_layout.md).
- Set up multimodal ingestion with [docs/multimodal.md](docs/multimodal.md).
- Report problems through [CONTRIBUTING.md](CONTRIBUTING.md).
