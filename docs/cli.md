# CLI

The `everos` command-line entry point covers **setup and operations** —
generate a starter `.env` (`init`), run the HTTP API server (`server
start`), and operate the md → LanceDB index queue (`cascade`). Hot-path
business (`/add` `/flush` `/search` `/get`) is the **HTTP API**, not the
CLI.

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
├── init                            Generate a starter .env from the packaged template
├── server
│   └── start                       Start the HTTP API server (uvicorn)
└── cascade                         Inspect / operate the md → LanceDB sync queue
    ├── status                      Queue / LSN summary
    ├── sync                        Drain the queue now (force md → LanceDB)
    └── fix                         List failed rows / re-enqueue retryable ones
```

Each subcommand lives in its own module under
[`entrypoints/cli/commands/`](../src/everos/entrypoints/cli/commands/) and is
registered in `cli/main.py`. The CLI is intentionally small — hot-path
business (`/add` `/flush` `/search` `/get`) is the **HTTP API**, not the
CLI; the CLI covers setup (`init`), running the server, and index ops
(`cascade`). There is no `reindex` command — rebuild by deleting
`<root>/.index/lancedb` and restarting, or run `everos cascade sync`.

## `everos server start`

Wraps `uvicorn` to launch the FastAPI app from
[`entrypoints/api/app.py`](../src/everos/entrypoints/api/app.py)
in *factory* mode.

```bash
everos server start \
    --host 127.0.0.1 \
    --port 8000 \
    --log-level info \
    --env-file .env
```

| Flag | Env var | Default |
|---|---|---|
| `--host` | `EVEROS_API__HOST` | `127.0.0.1` (loopback only; binding `0.0.0.0` logs a warning — EverOS ships no auth) |
| `--port` | `EVEROS_API__PORT` | `8000` |
| `--log-level` | `EVEROS_LOG_LEVEL` | `INFO` |
| `--env-file` | — | searched: `./.env` → `$XDG_CONFIG_HOME/everos/.env` → `~/.everos/.env` |
| `--reload` | — | off (use in development) |

Lifespan startup wires the storage backends (SQLite engine + LanceDB
connection) on app boot; see
[`entrypoints/api/lifespans/`](../src/everos/entrypoints/api/lifespans/).

## Configuration via env vars

Both CLI and HTTP server read configuration from `pydantic-settings`:

| Env var | Settings field |
|---|---|
| `EVEROS_MEMORY__ROOT` | `Settings.memory.root` (memory-root path) |
| `EVEROS_MEMORY__TIMEZONE` | `Settings.memory.timezone` (e.g. `Asia/Shanghai`) |
| `EVEROS_SQLITE__BUSY_TIMEOUT_MS` | `Settings.sqlite.busy_timeout_ms` |
| `EVEROS_LANCEDB__READ_CONSISTENCY_SECONDS` | `Settings.lancedb.read_consistency_seconds` |

Pattern: `EVEROS_<SECTION>__<KEY>` (double underscore = nesting). See
[`config/settings.py`](../src/everos/config/settings.py).

## Logging

`configure_logging` runs at CLI startup and configures `structlog` with
the resolved log level. All in-process logs (CLI command bodies +
service / infra layers) flow through the same handler.

```bash
everos server start --log-level debug   # see all sql / lance traffic
```

## API ↔ CLI division of labour

| Responsibility | API | CLI |
|---|---|---|
| Hot-path business (`/add` `/flush` `/search` `/get`) | ✅ | — (HTTP only) |
| Setup (generate `.env`) | — | `everos init` |
| Run the server | — | `everos server start` |
| Index ops (drain / inspect / fix the cascade queue) | — | `everos cascade {status,sync,fix}` |
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
