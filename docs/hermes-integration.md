# Hermes Agent Integration

Wire [Hermes Agent](https://github.com/NousResearch/hermes-agent) to an
EverOS server as its long-term, cross-session memory backend — durable
markdown memory, semantic + lexical recall, and offline consolidation,
with no `hermes-agent` PR required.

The integration ships **inside the EverOS repo** as a Hermes
`MemoryProvider` plugin bundle at
[`integrations/hermes/`](../integrations/hermes/). The
`everos integrations install hermes` command symlinks the bundle into
`~/.hermes/plugins/everos/`; Hermes discovers user-installed plugins and
loads them via the `MemoryProvider` subclass-fallback path.

## Prerequisites

- An EverOS server you can reach (local or remote). See
  [QUICKSTART](../QUICKSTART.md) for install + `everos server start`.
- Hermes Agent installed (`hermes` on your `PATH`).
- For **OSS mode** (local EverOS with self-supplied models): an
  OpenAI-protocol LLM, embedding, and (optionally) rerank endpoint —
  e.g. DeepInfra, OpenRouter, OpenAI, Together, or a local vLLM/Ollama
  exposing the OpenAI shape.

## Install

```bash
# from the EverOS checkout (editable/dev install — the common case)
everos integrations install hermes

# or, for a non-editable install, point at the bundle explicitly:
everos integrations install hermes --source /path/to/EverOS/integrations/hermes

# activate it in Hermes
hermes config set memory.provider everos
```

Verify Hermes discovered the plugin:

```bash
hermes memory status
# → Provider: everos  ·  Plugin: installed ✓  ·  Status: available ✓
#   Installed plugins: … everos (local) ← active
```

Then configure it with `hermes memory setup` (select `everos`) or by
editing `$HERMES_HOME/everos.json` directly (see [Config](#config)).

## Modes

| Mode | When | Config |
|---|---|---|
| `platform` | A vendor-hosted / remote EverOS server | `api_url` only |
| `oss` (default) | EverOS runs locally with your own LLM / embedding / rerank | `api_url` + `~/.everos/everos.toml` (the setup wizard writes it) |

`hermes memory setup` walks you through either mode. In OSS mode the
wizard writes `~/.everos/everos.toml` with `[llm]`, `[embedding]`, and
`[rerank]` blocks (chmod 600) — see
[configuration.md](configuration.md) for the full key reference. When
`agent_track_enabled` is set, it also seeds `~/.everos/ome.toml` to turn
on the agent-track OME strategies (`extract_agent_case`,
`extract_agent_skill`, `trigger_skill_clustering`).

## Config

Behavioral settings live in `$HERMES_HOME/everos.json`. Secrets belong in
`$HERMES_HOME/.env` (e.g. `EVEROS_API_KEY`), not `everos.json`.

| Key | Default | Description |
|---|---|---|
| `api_url` | `http://127.0.0.1:8000` | EverOS server URL |
| `mode` | `oss` | `platform` or `oss` |
| `user_id` | `hermes-user` | EverOS user id to index your memories under. When unset, the gateway-native id (Telegram/Discord/Slack) flows through so the same human gets one merged store |
| `agent_id` | `hermes` | EverOS agent id (agent-track attribution) |
| `app_id` | `default` | EverOS app scope |
| `project_id` | `default` | EverOS project scope |
| `agent_track_enabled` | `false` | Also write/search the agent track (cases + skills) |
| `api_key` | — | CLI-only; read by the `hermes everos` subcommands. The provider itself does not authenticate (EverOS is loopback/no-auth by default) |

`app_id` / `project_id` must match `^[a-zA-Z0-9_.@+-]+$` (the server-side
`PathSafeId` charset) and reject `.` / `..`.

## How it works

### Write path (every turn → durable memory)

After each completed Hermes turn, `sync_turn` builds two `MessageItem`s
(user + assistant, Unix-ms timestamps) and `POST /api/v1/memory/add`s them
on a daemon thread. EverOS buffers them per `session_id`; its boundary
detector decides when a segment is done, then the extractor (an LLM)
writes an **Episode** to markdown synchronously —
`~/.everos/.../users/<user_id>/episodes/episode-<date>.md`. At session end,
`on_session_end` calls `POST /memory/flush` to force extraction of any
pending buffer.

Markdown is the source of truth; SQLite (state) and LanceDB
(vectors + BM25) are derived and rebuildable. See
[how-memory-works.md](how-memory-works.md).

### Recall path (every turn → relevant context injected)

Before each turn, `prefetch` fires a background `POST /memory/search`
(`include_profile=True`) and waits up to 1.5 s. The result — top episodes
(subject + truncated narrative) + nested atomic facts + a profile
one-liner, score-sorted — is injected into the agent's context for that
turn. If the server is slow, the result surfaces on the next turn
(mem0-style two-phase).

### Long-term, consolidating

Episodes persist across Hermes restarts and chats. The Offline Memory
Engine runs in the background: `extract_atomic_facts` (single-sentence
facts), `extract_user_profile` (an evolving `user.md`), and — opt-in,
weekly cron — `reflect_episodes`, which merges fragmented episodes about
the same topic into one narrative and soft-archives the originals. See
[reflection.md](reflection.md).

## Tools exposed to the agent

Beyond passive prefetch, the agent gets four tools:

| Tool | Maps to | Purpose |
|---|---|---|
| `everos_search` | `POST /memory/search` | Semantic + lexical hybrid recall |
| `everos_list` | `POST /memory/get` | Paginated browse of episodes / profile / cases / skills |
| `everos_add` | `POST /memory/add` + `/flush` | Buffer a fact and force extraction |
| `everos_flush` | `POST /memory/flush` | Force extraction of the current session buffer |

`everos_search` / `everos_list` take an `owner` (`user` | `agent`) to
select the user vs agent track, independent of `mode`.

## Hermes-side CLI

When `memory.provider` is `everos`, Hermes exposes `hermes everos …`:

```bash
hermes everos status                 # reachability + active config/scope + breaker state
hermes everos search "QUERY" [--top-k N] [--method hybrid|vector|keyword|agentic] [--owner user|agent]
hermes everos flush [--session-id ID]
hermes everos setup --mode oss --api-url ... --user-id ...   # non-interactive everos.json writer
```

`hermes everos setup` only manages `everos.json`; for full OSS setup
(including `~/.everos/everos.toml`) use `hermes memory setup`.

## Resilience

A circuit breaker trips after 5 consecutive transient failures (server
down, LLM/embedding 503) and pauses calls for 120 s. A down EverOS server
never crashes Hermes or stalls the conversation — it degrades to built-in
`MEMORY.md`/`USER.md` until the server recovers. Client errors
(`INVALID_INPUT`, `NOT_FOUND`, …) do **not** trip the breaker.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `hermes memory status` shows `Status: not available` | `everos.json` missing or `api_url` empty — run `hermes memory setup` |
| `everos_search` returns 503 | The query-embedding step failed — check `~/.everos/everos.toml` `[embedding]` is reachable |
| Memories not appearing after `everos_add` | `everos_add` buffers + flushes; extraction runs on flush. If extraction fails, check the EverOS server logs and the `[llm]` config |
| Recall misses a just-written episode | LanceDB indexing is async (sub-second, up to ~10–15 s under load); retry, or `everos cascade sync` |
| `hermes everos …` command not found | The CLI is gated on `memory.provider == everos` — run `hermes config set memory.provider everos` |

## Uninstall

```bash
everos integrations uninstall hermes          # removes the dev symlink
# revert the provider if you no longer use it:
hermes config set memory.provider ''
```

`~/.everos` (your actual memory data) is preserved.

## See also

- [Bundle README](../integrations/hermes/README.md) — full setup/config/tool reference
- [how-memory-works.md](how-memory-works.md) — the write→index→read pipeline
- [api.md](api.md) — the EverOS HTTP API v1 contract this plugin calls
- [configuration.md](configuration.md) — `everos.toml` / env-var reference
- [cli.md](cli.md) — the `everos` CLI, including `everos integrations`
