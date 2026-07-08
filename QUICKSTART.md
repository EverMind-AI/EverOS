# Quickstart

> Five minutes from zero to "I added a conversation, queried it back, and
> can read it as plain Markdown."

EverOS runs as a **service** — start the server, then call the HTTP API.
There is no in-process library mode; an `everos` server is always in
front of your agent.

## Prerequisites

- **Python 3.12+**
- **API keys** for three capabilities: a chat LLM (memory extraction),
  an embedding model (vector retrieval), and a reranker. Any
  OpenAI-compatible endpoint works.

## 1. Install

**From PyPI** (users):

```bash
pip install everos
# or:  uv pip install everos
```

**From source** (contributors / developers):

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
uv sync          # install all deps into .venv
```

> **Note:** source install creates a `.venv` virtualenv. Subsequent
> `everos` commands need either `uv run everos ...` or activate the venv
> first (`source .venv/bin/activate`).

## 2. Configure

```bash
everos init                        # default root: ~/.everos
everos init --root /data/everos    # or specify a custom root
```

> **Root directory** — defaults to `~/.everos`. Use `--root <path>` to
> relocate; all subsequent commands (`server start`, `cascade status`,
> etc.) must use the matching `--root`. Any setting in `everos.toml` can
> also be overridden via `EVEROS_*` environment variables for containers
> and CI.

This creates `everos.toml` and `ome.toml` under the root directory.
Open `everos.toml` and fill in three sections — here's the minimum
viable config:

```toml
[llm]
model    = "gpt-4.1-mini"                        # or your preferred model
base_url = "https://openrouter.ai/api/v1"        # any OpenAI-compatible endpoint
api_key  = "sk-..."                               # your API key

[embedding]
model    = "Qwen/Qwen3-Embedding-4B"
base_url = "https://api.deepinfra.com/v1/openai"
api_key  = "..."

[rerank]
provider = "deepinfra"
model    = "Qwen/Qwen3-Reranker-4B"
base_url = "https://api.deepinfra.com/v1/inference"
api_key  = "..."
```

The generated file pre-fills recommended `model` and `base_url`
defaults — just drop in your API keys. Any OpenAI-compatible endpoint
works.

> **Multimodal** (`[multimodal]`) is optional — only needed when
> ingesting image / pdf / audio content items. See
> [docs/multimodal.md](docs/multimodal.md) for setup.


## 3. Start the server

Check your file descriptor limit — EverOS opens many LanceDB segment
files under concurrent search + indexing. Platform defaults:
**macOS 256** · **Linux 1024** · **Windows 8192**. If yours is below
4096, raise it before starting:

Run these in the **same terminal** where you will start the server —
`ulimit` is per-shell-session, not global:

```bash
ulimit -n            # check current limit
ulimit -n 4096       # raise if needed
everos server start [--root <path>]   # must be in the same session
```

> **No side effects** — `ulimit -n` only raises the per-process ceiling.
> It does not pre-allocate memory or file handles, and has zero
> performance cost. For Linux production, set `LimitNOFILE=65536` in
> your systemd unit file.

You should see:

```
starting everos on 127.0.0.1:8000
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

The server runs in the foreground. **Open a second terminal** for the
steps below.

Verify it's up:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

## 4. Add a conversation

Send messages to the server — one at a time or in batches. Each batch
belongs to a `session_id`, which represents one conversation thread.
Timestamps are Unix epoch in **milliseconds** (UTC).

First, a chat about climbing:

```bash
TS=$(($(date +%s)*1000))
curl -X POST http://127.0.0.1:8000/api/v1/memory/add \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"demo-001\",
    \"messages\": [
      {\"sender_id\": \"alice\",  \"role\": \"user\",      \"timestamp\": $TS,              \"content\": \"I just got back from a week in Yosemite. The climbing was incredible.\"},
      {\"sender_id\": \"agent1\", \"role\": \"assistant\", \"timestamp\": $((TS+10000)),  \"content\": \"That sounds amazing! Which routes did you do?\"},
      {\"sender_id\": \"alice\",  \"role\": \"user\",      \"timestamp\": $((TS+20000)),  \"content\": \"Mostly cracks on El Cap. I go every spring — it's my favorite season there.\"}
    ]
  }"
# → status: "accumulated"
```

Now the topic shifts to work:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memory/add \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"demo-001\",
    \"messages\": [
      {\"sender_id\": \"alice\",  \"role\": \"user\",      \"timestamp\": $((TS+60000)),  \"content\": \"By the way, I switched to biking to work last month. Loving it so far.\"},
      {\"sender_id\": \"agent1\", \"role\": \"assistant\", \"timestamp\": $((TS+70000)),  \"content\": \"How long is your commute?\"},
      {\"sender_id\": \"alice\",  \"role\": \"user\",      \"timestamp\": $((TS+80000)),  \"content\": \"About 25 minutes. I stop at Blue Bottle in SOMA for coffee most mornings.\"}
    ]
  }"
```

Response:

```json
{
    "data": {
        "message_count": 3,
        "status": "extracted"
    }
}
```

EverOS detected a topic shift (climbing → commute) and automatically
extracted the earlier conversation into memory.

The `status` field tells you what happened:

| Status | Meaning |
|---|---|
| `accumulated` | Messages buffered, still part of the same topic. |
| `extracted` | Topic shift detected — memory extracted from the buffer. |

> For the full API contract, see [docs/openapi.json](docs/openapi.json).

## 5. Flush (manual extraction)

If you want to extract memory without waiting for a topic shift — for
example at the end of a session — call `/flush`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memory/flush \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-001"}'
```

```json
{
    "data": {
        "status": "extracted"
    }
}
```

This forces extraction of whatever is still in the buffer.

## 6. Search

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memory/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "query": "Where do I like to climb?",
    "top_k": 5
  }'
```

Response (trimmed):

```json
{
    "data": {
        "episodes": [
            {
                "id": "alice_ep_20260528_00000002",
                "summary": "... Alice shared that she loves climbing in Yosemite every spring ...",
                "score": 0.628,
                "atomic_facts": [
                    {
                        "content": "Alice said she loves climbing in Yosemite every spring.",
                        "score": 0.628
                    }
                ]
            }
        ]
    }
}
```

Hybrid retrieval (BM25 + vector + scalar) returns the matching episode
with its atomic facts nested under it.

## 7. Your memory is just Markdown

This is what makes EverOS different — memory persists as plain Markdown:

```
<root>/                                  ← ~/.everos or your --root path
├── default_app/                        ← app_id ("default" → "default_app")
│   └── default_project/                ← project_id ("default" → "default_project")
│       ├── users/<user_id>/
│       │   ├── user.md                 ← profile
│       │   ├── episodes/               ← daily-log episodes
│       │   ├── .atomic_facts/          ← nested facts (dot-hidden)
│       │   └── .foresights/            ← predictive memory (dot-hidden)
│       ├── agents/<agent_id>/
│       │   ├── agent.md
│       │   ├── .cases/                 ← task cases
│       │   └── skills/                 ← procedural memories
│       └── knowledge/                  ← shared knowledge base
├── everos.toml                         ← provider config
├── ome.toml                            ← strategy config (hot-reloaded)
├── .index/                             ← derived indexes (rebuildable from md)
│   ├── sqlite/system.db
│   └── lancedb/
└── .tmp/
```

Every memory entry is a plain Markdown file you can directly read and
edit — no database driver needed.

## Stopping the server

`Ctrl+C` in the server terminal.

## Next steps

- **Integrate into your agent** — wrap `/add`, `/flush`, `/search` in a
  thin HTTP client and call them from your agent loop.
- **App + project scope** — pass `app_id` / `project_id` in your API
  requests to partition memory spaces inside one server (defaults to
  `"default"` when omitted).
- **Knowledge base** — upload documents via
  `/api/v1/knowledge/documents` and search with hybrid retrieval. See
  [docs/knowledge.md](docs/knowledge.md).
- **Reflection** — offline memory consolidation; enable in `ome.toml`.
  See [docs/reflection.md](docs/reflection.md).
- **Multimodal** — ingest image / pdf / audio / office documents. See
  [docs/multimodal.md](docs/multimodal.md).
- **Search modes** — four methods (`HYBRID` / `KEYWORD` / `VECTOR` /
  `AGENTIC`) with a filter DSL. See [docs/openapi.json](docs/openapi.json)
  for the full API schema.
- **Architecture** — [docs/architecture.md](docs/architecture.md) for
  DDD layering; [docs/storage_layout.md](docs/storage_layout.md) for
  on-disk layout.
- **Found a bug?** — [open an issue](CONTRIBUTING.md).
