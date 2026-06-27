# Quickstart

> Five minutes from zero to "I added a conversation, queried it back, and
> can read it as plain Markdown."

EverOS runs as a **service** — start the server, then call the HTTP API.
There is no in-process library mode; an `everos` server is always in
front of your agent.

## Prerequisites

- **Python 3.12+**
- **An LLM provider key and endpoint** — for memory extraction. OpenRouter,
  OpenAI, and other OpenAI-compatible providers work when you set
  `base_url`.
- **A multimodal provider key and endpoint** — needed only when parsing
  image / pdf / audio content items.
- **Embedding and rerank provider keys and endpoints** — for search. DeepInfra
  works for the embedding + rerank path; vLLM and DashScope are also supported
  for rerank.

Many deployments use two distinct keys by reusing one LLM key for `[llm]` and
`[multimodal]`, and one DeepInfra key for `[embedding]` and `[rerank]`. Any
setting can live in TOML or be overridden by the matching `EVEROS_*`
environment variable.

## 1. Install

```bash
pip install everos
# or:  uv pip install everos
```

## 2. Configure

Generate the starter config and fill in provider credentials:

```bash
everos init                    # writes ~/.everos/everos.toml + ome.toml (use --root to relocate)
# Edit ~/.everos/everos.toml and fill the provider fields:
#   [llm]        api_key + base_url             (chat LLM)
#   [multimodal] api_key + base_url             (optional parser LLM)
#   [embedding]  model + api_key + base_url
#   [rerank]     model + api_key + base_url     (provider can be inferred or set)
```

`everos init` generates two files: `everos.toml` (provider settings)
and `ome.toml` (offline memory engine strategy config, hot-reloaded).
Because `everos.toml` holds API keys, consider restricting access
after editing: `chmod 600 ~/.everos/everos.toml`.

The shipped template sets model defaults for `[llm]` (`gpt-4.1-mini`) and
`[multimodal]` (`google/gemini-3-flash-preview`); `[embedding]` and
`[rerank]` ship no model default — set `model` + `base_url` for those two
sections yourself (for example DeepInfra's `Qwen/Qwen3-Embedding-4B` /
`Qwen/Qwen3-Reranker-4B`). `[llm]` and `[multimodal]` ship model defaults,
but still need `api_key` and `base_url` before those capabilities are used.

> **Where config lives** — `everos init` writes into the memory root
> (`~/.everos` by default; relocate with `everos init --root <path>` and
> start the server with the matching `everos server start --root <path>`).
> `everos server start` reads `<root>/everos.toml` and exits with an error
> if it is missing. Any setting can also be overridden by an `EVEROS_*`
> environment variable (e.g. `EVEROS_LLM__API_KEY`) — handy for containers
> and CI.

## 3. Start the server

```bash
everos server start
```

You should see (port and host are configurable):

```
starting everos on 127.0.0.1:8000
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

- Default bind is `127.0.0.1` (loopback only). To expose the API
  elsewhere, put your own auth/gateway in front first
  ([see SECURITY.md](SECURITY.md)).
- The cascade index daemon runs **in the same process** as a FastAPI
  lifespan coroutine — you don't need a separate worker.
- The server runs in the foreground; **open a second terminal** for the
  steps below, and use `Ctrl+C` to stop the server when you're done.

In the second terminal, verify the server is up:

```bash
$ curl http://127.0.0.1:8000/health
{"status":"ok"}
```

## 4. Add a conversation

EverOS ingests memory at the **conversation level**, not as standalone
sentences: you POST a batch of `messages` tied to a `session_id`, and
the server accumulates them until the boundary detector trips (you can
also force a flush — see step 5).

```bash
TS=$(($(date +%s)*1000))    # Unix epoch in **milliseconds** (v1 contract)
curl -X POST http://127.0.0.1:8000/api/v1/memory/add \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"demo-001\",
    \"app_id\": \"default\",
    \"project_id\": \"default\",
    \"messages\": [
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $TS, \"content\": \"I love climbing in Yosemite every spring.\"},
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $((TS+10000)), \"content\": \"My favorite coffee shop is Blue Bottle in SOMA.\"},
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $((TS+20000)), \"content\": \"I bike to work most days.\"}
    ]
  }"
```

Response:

```json
{
    "request_id": "bf86e4e857834eba804841f8bff29106",
    "data": {
        "message_count": 3,
        "status": "accumulated"
    }
}
```

`status: "accumulated"` means the three messages are in the session
buffer, but the boundary detector hasn't decided to extract a memory
cell yet. For a quick demo we'll force it.

## 5. Force boundary extraction

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memory/flush \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-001","app_id":"default","project_id":"default"}'
```

Response (this takes a few seconds — one LLM call for extraction):

```json
{
    "request_id": "ec0e7a00c3bd4b00bb21212a411b7763",
    "data": {
        "status": "extracted"
    }
}
```

`status: "extracted"` means at least one memory cell was carved out and
written to disk + indexed.

> `/flush` is **OSS-only**. The cloud edition decides boundary timing
> server-side and does not expose this endpoint.

## 6. Search the memory you just added

```bash
curl -X POST http://127.0.0.1:8000/api/v1/memory/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "app_id": "default",
    "project_id": "default",
    "query": "Where do I like to climb?",
    "top_k": 5
  }'
```

Response (trimmed):

```json
{
    "request_id": "b53a3a94a080472d97692c503c88afdf",
    "data": {
        "episodes": [
            {
                "id": "alice_ep_20260528_00000002",
                "user_id": "alice",
                "session_id": "demo-001",
                "summary": "On May 28, 2026 ... Alice shared that she loves climbing in Yosemite every spring ...",
                "score": 0.6284722685813904,
                "atomic_facts": [
                    {
                        "id": "alice_af_20260528_00000016",
                        "content": "Alice said she loves climbing in Yosemite every spring.",
                        "score": 0.6284722685813904
                    }
                ]
            }
        ],
        "profiles": [],
        "agent_cases": [],
        "agent_skills": []
    }
}
```

The hybrid retrieval (BM25 + vector + scalar) returns the episode
that contains the climbing fact, with the matching atomic fact nested
under it. Other response arrays (`profiles` / `agent_cases` /
`agent_skills`) are always present for client-side symmetry, populated
only when the requested kind matches.

## 7. Your memory is just Markdown

This is what makes EverOS different — your memory persists as plain
Markdown files on disk:

```
$ tree ~/.everos -L 5 -a
~/.everos
├── default_app/                       ← app_id  ("default" → "default_app")
│   └── default_project/               ← project_id ("default" → "default_project")
│       ├── users/
│       │   └── alice/                  ← user_id (mirror dir: agents/<agent_id>/)
│       │       ├── episodes/
│       │       │   └── episode-2026-05-28.md
│       │       ├── .atomic_facts/      ← hidden (dot-prefix)
│       │       │   └── atomic_fact-2026-05-28.md
│       │       ├── .foresights/
│       │       │   └── foresight-2026-05-28.md
│       │       └── user.md             ← profile
│       └── knowledge/                  ← shared knowledge base (v1.1+)
│           └── .taxonomy.md
├── everos.toml                         ← provider config (API keys)
├── ome.toml                            ← strategy config (hot-reloaded)
├── .index/                             ← derived indexes (rebuildable from md)
│   ├── sqlite/system.db
│   └── lancedb/*.lance/
└── .tmp/
```

The `default` scope id materialises as `default_app` / `default_project`
on disk (with the `_app` / `_project` suffix) so the default space is
visually distinct from any user-named space. Any other id maps to itself
(e.g. `app_id: "my-app"` → `my-app/`).

Top-level `.index/` holds SQLite + LanceDB **derived** indexes — wipe it
and the cascade daemon rebuilds everything from the Markdown alone.

Read the episode we just created:

```
$ cat ~/.everos/default_app/default_project/users/alice/episodes/episode-2026-05-28.md
---
id: episode_log_alice_2026-05-28
type: episode_daily
file_type: episode_daily
schema_version: 1
user_id: alice
track: user
date: '2026-05-28'
entry_count: 1
last_appended_at: '2026-05-28T08:32:24.966944+00:00'
---
<!-- entry:ep_20260528_00000002 -->
## ep_20260528_00000002

**owner_id**: alice
**session_id**: demo-001
**timestamp**: 2026-05-28T08:32:13+00:00
**parent_type**: memcell
**parent_id**: mc_3779c20f1c53
**sender_ids**: [alice]

### Subject
Alice's Outdoor Activities and Daily Routine on May 28, 2026 Morning

### Content
On May 28, 2026 at 8:32 AM UTC, Alice shared that she loves climbing in
Yosemite every spring, highlighting a recurring seasonal outdoor activity.
She also mentioned that her favorite coffee shop is Blue Bottle located in
SOMA, indicating a preferred local spot. Additionally, Alice stated that
she bikes to work most days, revealing a habitual commuting practice.
<!-- /entry:ep_20260528_00000002 -->
```

Every memory entry is a plain Markdown file you can:

- `cat` / `grep` / `vim` directly — no driver, no service to query
- Version with Git (or rsync to backup)
- Open the `~/.everos/default_app/default_project/users/alice/` folder
  in Obsidian (the dotfile directories stay hidden by default)

## Stopping the server

`Ctrl+C` in the server terminal. Uvicorn catches `SIGINT` and shuts each
lifespan provider down in reverse order (cascade → LanceDB → SQLite →
LLM → metrics) before exiting.

## Next steps

- **Integrate into your agent** — wrap the three endpoints (`/add`,
  `/flush`, `/search`) in a thin Python client (`httpx.AsyncClient`) and
  call them from your agent loop.
- **App + project scope** — set `app_id` / `project_id` to anything
  other than `"default"` to partition memory spaces inside one server.
- **Knowledge base** — upload documents (PDF / HTML / DOCX) via
  `/api/v1/knowledge/documents` and search them with hybrid retrieval
  at `/api/v1/knowledge/search`. Ships with a 20-category default
  taxonomy. See [docs/knowledge.md](docs/knowledge.md).
- **Reflection** — offline memory self-improvement that consolidates
  related episodes. Disabled by default; enable in `ome.toml`
  (`[strategies.reflect_episodes] enabled = true`). Changes are
  hot-reloaded, no server restart needed.
- **Multi-modal messages** — `messages[].content` accepts a list of
  typed `ContentItem`s (`text` / `image` / `audio` / `doc` / `pdf` /
  `html` / `email`) for non-text input. Install the optional extra
  to enable parsing:
  `uv pip install 'everos[multimodal]'`. Office documents
  (`doc` / `docx` / `xls` / `ppt` / `…`) additionally need
  **LibreOffice** on the host (`brew install --cask libreoffice` /
  `apt-get install libreoffice`) — without it those uploads return
  HTTP 503 (`CAPABILITY_UNAVAILABLE`); PDF / image / audio / HTML
  still work.
- **Filter DSL and search modes** — `/search` supports a filter DSL
  (`AND` / `OR` / scalar predicates) and four methods (`HYBRID` /
  `KEYWORD` / `VECTOR` / `AGENTIC`). The OpenAPI docs UI is served at
  `/docs` only when the server runs with `ENV=DEV`; the default (`prod`)
  serves the API without the docs UI. The schema also lives at
  [docs/openapi.json](docs/openapi.json).
- **Architecture** — see [docs/architecture.md](docs/architecture.md)
  for the DDD layering and cascade design, and
  [docs/storage_layout.md](docs/storage_layout.md) for the on-disk
  layout.
- **Found a bug?** — open an issue (see [CONTRIBUTING.md](CONTRIBUTING.md);
  external pull requests are not merged).
