# CLI

The `everos` command-line entry point covers **setup and operations** —
generate starter config files (`init`), run the HTTP API server (`server
start`), inspect effective config (`config show`), operate the
md → LanceDB index queue (`cascade`), and change, forget and read the
versions of one memory (`memory`). Hot-path business (`/add` `/flush`
`/search` `/get`) is the **HTTP API**, not the CLI.

CLI commands run **in-process** — they call into the `service/` /
infrastructure layers directly rather than the HTTP loopback.

## Installation

The script is exposed via `pyproject.toml`:

```toml
[project.scripts]
everos = "everos.entrypoints.cli.main:app"
```

After `uv sync` (or `pip install -e .`) the `everos` command resolves
to [`src/everos/entrypoints/cli/main.py`](../src/everos/entrypoints/cli/main.py),
a [Typer](https://typer.tiangolo.com/) app.

## Subcommand layout

```
everos
├── init [--root PATH] [--force] [--print]   Generate starter config files (everos.toml + ome.toml)
├── prompts                         Print which prompt versions this EverOS sends (JSON)
├── config
│   └── show [--root PATH]          Show effective configuration
├── server
│   └── start [--host] [--port] [--root] [--reload] [--log-level]   Start the HTTP API server (uvicorn)
├── cascade [--root PATH]           Inspect / operate the md → LanceDB sync queue
│   ├── status [--duplicates]       Queue / LSN summary; opt-in duplicate-key scan
│   ├── sync [PATH]                 Drain the queue now (optional PATH force-enqueues)
│   ├── fix [--apply]               List failed rows / re-enqueue retryable ones
│   ├── rebuild [--yes]             Drop the index tables and re-index every md file
│   └── backfill [--phase vectors|all] [--yes]   Embed the rows written before an embedder was configured
└── memory                          Change, forget and read the versions of one memory
    ├── history ID                  Every version, current first (read-only)
    ├── update ID [--content ...]   A person's new version (held unless --release)
    ├── forget [TARGET] [--apply]   Forget by target; a dry run unless --apply
    └── undo FORGET_ID              Undo one forget by its id
```

Each subcommand lives in its own module under
[`entrypoints/cli/commands/`](../src/everos/entrypoints/cli/commands/) and is
registered in `cli/main.py`. The CLI is intentionally small — hot-path
business (`/add` `/flush` `/search` `/get`) is the **HTTP API**, not the
CLI; the CLI covers setup (`init`), running the server, and index ops
(`cascade`). There is no `reindex` command — for an incremental
catch-up run `everos cascade sync`; to re-reconcile every indexed
markdown file without re-embedding what is unchanged run `everos cascade
sync --force-all` (the first full pass after upgrading from before 1.4.2
does this by itself); to rebuild the whole index from markdown (recovery
from drift / corruption, and once after upgrading a Milvus root to 1.4.2)
run `everos cascade rebuild`.

## `everos prompts`

Prints the prompt manifest as JSON: every prompt EverOS registers (id,
version, fingerprint of its text, and whether it is sent while a store is
written or while it is searched) and the everalgo prompts it sends unchanged
(package, installed version, fingerprint). It needs no memory root, no
configuration and no model. Save it beside a store when the store is built
(`everos prompts > <store>/prompts.json`): two stores built by the same
prompts carry the same manifest, and a prompt change shows up as a new
fingerprint.

## `everos server start`

Wraps `uvicorn` to launch the FastAPI app from
[`entrypoints/api/app.py`](../src/everos/entrypoints/api/app.py)
in *factory* mode.

```bash
everos server start \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level info \
    --root ~/.everos
```

| Flag | Env var | Default |
|---|---|---|
| `--host` | `EVEROS_API__HOST` | `127.0.0.1` (loopback only; binding `0.0.0.0` logs a warning — EverOS ships no auth) |
| `--port` | `EVEROS_API__PORT` | `8000` |
| `--log-level` | `EVEROS_LOG_LEVEL` | `[logging] level` in `everos.toml` (`INFO`) |
| `--root` | `EVEROS_ROOT` | `~/.everos` |
| `--reload` | — | off (use in development) |

App boot builds the application and starts it (SQLite schema, index
connection, cascade, OME, the memorize queue); see
[`service/bootstrap.py`](../src/everos/service/bootstrap.py).

## `everos cascade backfill`

For a store written without an embedding provider (keyword-only recall),
once `[embedding]` is configured. It has one phase, `vectors`: every index
row without a vector is embedded, in batches, so multi-round search recalls
it by meaning too. `--phase` takes `vectors` or `all` (the default, the same
phase). It prints an input-token estimate and asks before it spends
anything (`--yes` skips the question). It needs the memory root to itself:
while a server holds it, it exits with code 3 before any embedding call.
Exit code 2 means no embedding provider is configured; 4 means the phase
ran but some rows still failed to embed (their ids are in the
`cascade_backfill_row_embed_failed` log events). Running it again picks up
only what is left.

## `everos memory`

A thin layer over the service functions behind `POST /memory/update`,
`GET /memory/history`, `POST /memory/forget` and `/memory/forget/undo`
([api.md](api.md#changing-and-forgetting-a-memory)), run in-process. Each
command names a user's memory with `--user` or a group's with `--group` and
`--asker`, takes `--root`, `--app-id` and `--project-id` (both `default`
unless given, as in the API), and prints its answer as one JSON document on
standard output -- its log lines go to standard error, a warning while the
settings load included, so `| jq` reads the answer as is.

```bash
everos memory history amy_ep_20261112_00000003 --user amy
everos memory update amy_ep_20261112_00000003 --user amy --content "Amy adopted a dog."
everos memory update pi_3f9a2c71d0be --user amy --release      # let the model change it
everos memory forget --user amy --session trip-chat             # dry run: what it would forget
everos memory forget --user amy --session trip-chat --apply     # forget; prints the forget_id
everos memory forget --user amy --types episode,scene --start 1762905600000 --apply
everos memory forget --group trip --asker u_chen --member u_dan --apply
everos memory undo fg_0123456789abcdef --user amy
```

Targets: `--record ID` (repeatable), `--session`, `--session` with
`--message ID` (repeatable: messages of the conversation), `--owner`
(everything of the user), `--types` with `--start` / `--end` (epoch ms),
`--image HASH`; a group's: `--record`, `--session` (with `--message` for
messages), `--member`, `--image`. `history` only
reads and runs beside a server; `update`, `forget --apply` and `undo` write
and need the memory root to themselves, as `cascade sync` does: while a
server holds it they exit with code 3 -- use the HTTP API then. An error the
service answers (no such memory, a version conflict, a forgotten memory) exits
with code 1 and the message.

## Configuration via env vars

Both CLI and HTTP server read configuration from `pydantic-settings`:

| Env var | Settings field |
|---|---|
| `EVEROS_ROOT` | memory-root path (default `~/.everos`) |
| `EVEROS_MEMORY__TIMEZONE` | `Settings.memory.timezone` (e.g. `Asia/Shanghai`) |
| `EVEROS_SQLITE__BUSY_TIMEOUT_MS` | `Settings.sqlite.busy_timeout_ms` |
| `EVEROS_LANCEDB__READ_CONSISTENCY_SECONDS` | `Settings.lancedb.read_consistency_seconds` |

Pattern: `EVEROS_<SECTION>__<KEY>` (double underscore = nesting). See
[`config/settings.py`](../src/everos/config/settings.py).

## Logging

The log level is configuration: `[logging] level` in `everos.toml`
(`INFO` by default). `everos.service.configure_process` applies it once per
process -- building the application calls it -- and every in-process log
(CLI command bodies + service / infra layers) flows through the same
handler. `server start --log-level` (or `EVEROS_LOG_LEVEL`) overrides the
configured level for the server; interactive commands (`cascade`, `init`,
`config`, `demo`) log `WARNING`, or `INFO` with `--verbose`.

```bash
everos server start --log-level debug   # see all sql / lance traffic
```

## API ↔ CLI division of labour

| Responsibility | API | CLI |
|---|---|---|
| Hot-path business (`/add` `/flush` `/search` `/get`) | ✅ | — (HTTP only) |
| Setup (generate config files) | — | `everos init` |
| Inspect effective config | — | `everos config show` |
| Run the server | — | `everos server start` |
| Index ops (drain / inspect / fix the cascade queue, rebuild, backfill) | — | `everos cascade {status,sync,fix,rebuild,backfill}` |
| Change / forget / version history of one memory | `/memory/update`, `/forget`, `/history` | `everos memory {update,forget,undo,history}` |
| Health probe | `GET /health` | (use HTTP) |
| Metrics scrape | `GET /metrics` | (use HTTP) |

The CLI is the **shell-friendly** surface for ops + scripting; the
HTTP API is the **process-friendly** surface for clients (web UIs,
agents, automation).

## See also

- [architecture.md](architecture.md) — DDD layering between
  entrypoints / service / memory / infra
- [`entrypoints/cli/main.py`](../src/everos/entrypoints/cli/main.py)
- [`entrypoints/cli/commands/server.py`](../src/everos/entrypoints/cli/commands/server.py)
