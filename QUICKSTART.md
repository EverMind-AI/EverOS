# Quickstart

> Five minutes from one OpenRouter API key to durable Markdown memory, plus an
> embedding endpoint to search it back.

EverOS runs as a local service. Writing memory needs only an LLM: configure one
OpenRouter key, start the server, then call the HTTP API. Searching memory runs
multi-round retrieval, which also needs an embedding provider.

## What the one-key setup includes

With only `[llm]` configured, EverOS can:

- start the server;
- extract conversations into durable Markdown;
- keep the local index in sync.

Search (step 8) additionally needs `[embedding]`. Rerank, knowledge, and
multimodal providers are optional upgrades, not required for this walkthrough.

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

## 2. Try the standalone demo — no key required

Before initialization or provider setup, run:

```bash
everos demo
```

The command asks for one memory and one recall question, then opens a local
terminal visualizer. It is hardcoded and completely decoupled from the real
workflow: it needs no API key, does not start or call the EverOS server, and
does not write to your real memory root.

Press `r` to replay and `q` to quit. For a copyable non-interactive preview:

```bash
everos demo --plain
```

See [docs/everos-demo.md](docs/everos-demo.md) for the visualizer's scope.

## 3. Initialize EverOS

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

## 4. Add your OpenRouter key

Open `~/.everos/everos.toml`. The generated
`[llm]` section already contains the recommended model and base URL; replace
only the empty `api_key`:

```toml
[llm]
model = "openai/gpt-4.1-mini"
api_key = "<OPENROUTER_API_KEY>"
base_url = "https://openrouter.ai/api/v1"
```

Leave `[rerank]` and `[multimodal]` unchanged for this walkthrough. Their empty
keys do not prevent the server from starting. To run the search in step 8, also
fill `[embedding]` with an OpenAI-compatible embedding endpoint (see
[config.example.toml](config.example.toml)); without it the server starts and
writes memory, and search answers HTTP 422.

## 5. Start the server

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
    "embed": false,
    "rerank": false
  },
  "disabled_features": [
    "memory_search",
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

## 6. Add a conversation

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

## 7. Flush at the end of the session

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

## 8. Search with multi-round retrieval

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "app_id": "default",
    "project_id": "default",
    "query": "Where does Alice like to climb?",
    "method": "llm_multiround",
    "top_k": 5
  }'
```

The response should contain an episode whose summary mentions Yosemite or El
Cap. If the first search is empty, wait a moment for cascade indexing and retry.

> [!IMPORTANT]
> `llm_multiround` is the default and the only search method. It needs
> `[embedding]`: in the one-key tier the search returns HTTP 422
> `PROVIDER_NOT_CONFIGURED`.

Each round recalls the question (and later the decider's sub-queries) from the
local BM25 index and the vector index, and the decider model picks the
episodes that answer it.

## 9. Read the Markdown source of truth

Your extracted memory is a normal Markdown file under the memory root:

```text
~/.everos/
├── default_app/
│   └── default_project/
│       ├── users/alice/
│       │   ├── user.md
│       │   ├── episodes/
│       │   └── images/
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
| `[llm]` only | Add, flush, Markdown persistence, cascade sync |
| Add `[embedding]` | Multi-round search, reflection, skill extraction |
| Add `[rerank]` too | Knowledge Wiki |
| Add `[multimodal]` and install `everos[multimodal]` | Image, PDF, audio, and office-file ingestion |

EverOS reports unavailable features through `/health`. Requests that require a
missing provider fail fast with a descriptive HTTP 422 instead of silently
degrading.

You can replace OpenRouter with another OpenAI-compatible LLM endpoint by
changing the `[llm]` model, base URL, and key.

## Stop the server

Press `Ctrl+C` in the server terminal.

## Next steps

- Integrate `/add`, `/flush`, and `/search` into your agent loop.
- Partition memory with `app_id` and `project_id`.
- Explore the full API contract in [docs/openapi.json](docs/openapi.json).
- Configure advanced retrieval in the generated `everos.toml`.
- Run `everos demo --live` with your EverOS Cloud key; unlike the standalone
  demo in step 2, live mode calls the real EverOS Cloud API.
- Read [docs/architecture.md](docs/architecture.md) and
  [docs/storage_layout.md](docs/storage_layout.md).
- Set up multimodal ingestion with [docs/multimodal.md](docs/multimodal.md).
- Report problems through [CONTRIBUTING.md](CONTRIBUTING.md).
