# EverOS Memory Provider

EverOS memory provider plugin for Hermes Agent. Supports both vendor-hosted EverOS servers (platform mode) and local self-hosted instances (OSS mode) with your own LLM / embedding / rerank endpoints. Server-side md-first memory extraction, hybrid search, and per-session flushing.

## Requirements

- An EverOS server running and reachable (e.g. `everos server start`).
- `httpx` (declared in `plugin.yaml`; Hermes installs it automatically).

## Install

From the EverOS checkout, symlink this bundle into Hermes' plugin directory:

```bash
everos integrations install hermes
```

This links `integrations/hermes/` to `$HERMES_HOME/plugins/everos` (default
`~/.hermes/plugins/everos`). To point at a non-default bundle source, set
`EVEROS_HERMES_PLUGIN_SOURCE=/path/to/bundle` or pass `--source PATH`. To
replace an existing real directory at the target, pass `--force`.

> The bundle ships in-repo (`integrations/hermes/`), not in the EverOS
> wheel. Wheel-only installs (no checkout) therefore need
> `--source PATH` (or `EVEROS_HERMES_PLUGIN_SOURCE`) pointing at a
> separate checkout of the bundle.

Then activate EverOS as the memory provider:

```bash
hermes memory setup      # select "everos"
```

Or, non-interactively:

```bash
hermes everos setup --mode platform --api-url http://127.0.0.1:8000 --api-key sk-...
hermes config set memory.provider everos
```

> `hermes everos setup` only writes/merges `$HERMES_HOME/everos.json`. For a
> full OSS-mode setup (writing `~/.everos/everos.toml` + `ome.toml`), use
> `hermes memory setup` and select "everos".

To remove the bundle later:

```bash
everos integrations uninstall hermes
```

The `memory.provider` setting in Hermes config is left untouched — clear it
manually with `hermes config set memory.provider ''` if needed.

## Config

Behavioral settings live in `$HERMES_HOME/everos.json` (set them via
`hermes memory setup` or `hermes everos setup`). Only the secret
`EVEROS_API_KEY` belongs in `~/.hermes/.env`.

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `platform` | `platform` (vendor-hosted EverOS) or `oss` (self-hosted) |
| `api_url` | `http://127.0.0.1:8000` | EverOS API base URL |
| `api_key` | — | EverOS API key (CLI-only; secret — prefer `EVEROS_API_KEY` env). Read from `EVEROS_API_KEY` / `everos.json` ONLY by the `hermes everos` CLI subcommands (status/search/flush). The provider itself does not authenticate: EverOS is loopback/no-auth by default. `_config.load_config` does not load it. |
| `user_id` | `hermes-user` | User identifier (user-track scope) |
| `agent_id` | `hermes` | Agent identifier (agent-track scope; used when `--owner agent`) |
| `app_id` | `default` | EverOS app scope |
| `project_id` | `default` | EverOS project scope |
| `agent_track_enabled` | `false` | Enables the agent-track OME extractors (`extract_agent_case`, `extract_agent_skill`, `trigger_skill_clustering`) in `~/.everos/ome.toml`. OSS mode only. |
| `everos_root` | `~/.everos` | EverOS root directory (OSS mode; `null` in platform mode) |

The EverOS server itself is configured via `~/.everos/everos.toml`
(see `everos config show`). The plugin only needs to know how to reach it
and which scope to read/write.

Example `everos.json`:

```json
{
  "mode": "platform",
  "api_url": "http://127.0.0.1:8000",
  "user_id": "hermes-user",
  "agent_id": "hermes",
  "app_id": "default",
  "project_id": "default"
}
```

## CLI

When EverOS is the active memory provider, `hermes everos <subcommand>` is
available:

| Subcommand | Description |
|------------|-------------|
| `status` | Print reachability, active mode/user/scope, and circuit-breaker state |
| `search QUERY` | One-off search against the active config; prints JSON |
| `flush` | POST `/memory/flush` for the current session |
| `setup` | Non-interactive shortcut: write/merge `everos.json` |

```bash
hermes everos status
hermes everos search "the user's preferred editor" --top-k 5
hermes everos flush --session-id my-session
hermes everos setup --mode oss --api-url http://127.0.0.1:8000
```

## Tools

The provider exposes four agent-facing tools:

| Tool | Description |
|------|-------------|
| `everos_search` | Hybrid search by meaning; returns ranked episodes / profiles / agent cases / skills |
| `everos_list` | List stored memories (paginated, unranked) |
| `everos_add` | Buffer a fact (add_messages) then flush the session; extraction runs on flush |
| `everos_flush` | Flush the current session's buffered messages for extraction |

## Troubleshooting

### "EverOS temporarily unavailable"

The circuit breaker trips after 5 consecutive failures and pauses API calls
for 2 minutes. It resets automatically.

- Check the EverOS server is running: `everos server start` or `curl http://127.0.0.1:8000/health`.
- Check `api_url` in `$HERMES_HOME/everos.json` points at the right server.
- Run `hermes everos status` to see the breaker state and reachability.

### Server unreachable

```bash
curl http://127.0.0.1:8000/health
```

If this fails, start the EverOS server (`everos server start`) or fix
`api_url`. The plugin degrades gracefully — search returns no results and
writes are buffered until the breaker resets.

### Memories not appearing

- `everos_add` buffers the fact (`add_messages`) and immediately flushes the
  session — extraction runs on flush. For full LLM-backed extraction use the
  normal turn flow.
- Search uses hybrid matching — try broader queries.
- Confirm `user_id` / `agent_id` / `app_id` / `project_id` match between
  sessions (`$HERMES_HOME/everos.json`).
- Run `hermes everos flush` to force extraction of buffered messages.
