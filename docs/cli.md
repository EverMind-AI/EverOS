# CLI

The `everos` command-line entry point covers **setup and operations** —
generate starter config files (`init`), run the HTTP API server (`server
start`), inspect effective config (`config show`), and operate the
md → LanceDB index queue (`cascade`). Hot-path
business (`/add` `/flush` `/search` `/get`) is the **HTTP API**, not the
CLI.

CLI commands run **in-process** — they call into the `service/` /
infrastructure layers directly rather than the HTTP loopback.

## Installation

The script is exposed via `pyproject.toml`:

```toml
[project.scripts]
everos = "everos.entrypoints.cli.launcher:main"
```

After `uv sync` (or `pip install -e .`) the `everos` command resolves
to the [launcher](../src/everos/entrypoints/cli/launcher.py), which configures
the process before invoking the [Typer app](../src/everos/entrypoints/cli/main.py).
`python -m everos.entrypoints.cli.main` uses the same launcher.

### Linux allocator default

On Linux with glibc, the CLI defaults `MALLOC_ARENA_MAX` to `2` to reduce
unused memory retained by allocation arenas. It re-executes the interpreter
once so glibc reads the setting at process startup; command arguments and
Python interpreter flags are preserved. Reload workers inherit the setting.
macOS, Windows and non-glibc Linux keep their existing allocator behavior.

Where jemalloc is installed (`apt install libjemalloc2` on Debian / Ubuntu;
found the way `ctypes.util.find_library("jemalloc")` finds it) the re-exec
also preloads it: `LD_PRELOAD=libjemalloc.so.2` with
`MALLOC_CONF=oversize_threshold:0,dirty_decay_ms:1000,narenas:4`. On a 4-core,
8 GB server it doubled keyword search throughput and halved hybrid search
latency, with a long-run footprint no larger than glibc's two arenas. Child
processes the server starts (a document parser, for one) inherit the preload.

An explicitly supplied `MALLOC_ARENA_MAX` or
`GLIBC_TUNABLES=glibc.malloc.arena_max=...` takes precedence and keeps glibc;
an explicit `LD_PRELOAD` or `MALLOC_CONF` keeps the operator's allocator and
gets only the arena default. For example:

```bash
MALLOC_ARENA_MAX=2 everos server start  # glibc with two arenas, no jemalloc
MALLOC_ARENA_MAX=4 everos server start  # favor allocation concurrency
MALLOC_ARENA_MAX=0 everos server start  # retain glibc's automatic arena limit
```

This is a process-launch setting, not a TOML or `.env` application setting.
The default applies to CLI commands; importing EverOS into another Python
application does not change that application's allocation policy. For direct
Uvicorn or Python service usage, set it before starting the interpreter:

```bash
MALLOC_ARENA_MAX=2 uvicorn everos.entrypoints.api.app:create_app --factory
MALLOC_ARENA_MAX=2 python my_application.py
```

The limit controls glibc arenas, not a total memory ceiling or other native
allocators. Fewer arenas can trade memory savings for allocation contention;
see [glibc's allocator documentation](https://sourceware.org/glibc/manual/latest/html_node/Memory-Allocation-Tunables.html).

## Subcommand layout

```
everos
├── init [--root PATH] [--force] [--print]                         Generate starter config files (everos.toml + ome.toml)
├── export OUT [--root PATH] [--include-input-log] [--force]       Package Markdown, config, and a SHA-256 manifest into a .tar.gz archive
├── verify ARCHIVE                                                 Verify archive checksums, layout, and entry counts offline
├── import ARCHIVE [--root PATH] [--force] [--no-rebuild]          Verify and restore an archive into a memory root and rebuild its index
├── config
│   └── show [--root PATH]          Show effective configuration
├── server
│   └── start [--host] [--port] [--root] [--reload] [--log-level]   Start the HTTP API server (uvicorn)
└── cascade [--root PATH]           Inspect / operate the md → LanceDB sync queue
    ├── status [--duplicates]       Queue / LSN summary; opt-in duplicate-key scan
    ├── sync [PATH]                 Drain the queue now (optional PATH force-enqueues)
    └── fix [--apply]               List failed rows / re-enqueue retryable ones
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

## Portable memory export and restore (`export` / `verify` / `import`)

Because Markdown is EverOS's single source of truth, a memory root can be moved
between machines without shipping derived `.index/` files or (by default) raw
`.input_log/` conversation transcripts:

```bash
everos export backup.tar.gz --root ~/.everos
everos verify backup.tar.gz
everos import backup.tar.gz --root /data/everos-restored
```

- **`everos export <out>`** writes a `.tar.gz` archive containing Markdown
  source-of-truth files, root configuration (`everos.toml`, `ome.toml`), and a
  `manifest.json` with EverOS version, file list, per-file SHA-256 digests, and
  entry counts per kind. `.index/`, `.tmp/`, and `.lock` are always excluded;
  `.input_log/` is excluded unless `--include-input-log` is passed.
- **`everos verify <archive>`** validates the archive structure, path layout,
  every SHA-256 digest, and per-kind entry counts without touching any live
  memory root.
- **`everos import <archive>`** verifies the archive completely before writing
  anything, refuses to overwrite a non-empty root unless `--force` is passed
  (replacing rather than merging), rebuilds the derived index from restored
  Markdown (unless `--no-rebuild` is given), and verifies that before/after
  entry counts and rebuilt index row counts match the manifest.

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
| Portable archive (export / verify / restore) | — | `everos {export,verify,import}` |
| Inspect effective config | — | `everos config show` |
| Run the server | — | `everos server start` |
| Index ops (drain / inspect / fix the cascade queue) | — | `everos cascade {status,sync,fix}` |
| Health probe | `GET /health` | (use HTTP) |
| Metrics scrape | `GET /metrics` | (use HTTP) |

The CLI is the **shell-friendly** surface for ops + scripting; the
HTTP API is the **process-friendly** surface for clients (web UIs,
agents, automation).

A purge ([`POST /api/v2/memory/purge`](api.md#post-apiv2memorypurge)) is
an API-only operation and blocks no CLI command: with a purge interrupted
and the server stopped, CLI commands run as usual. When the server starts
again the purge resumes from the start, so anything written into the purged
range in the meantime is deleted too. Do not run `everos cascade backfill` while a
purge is unfinished: it re-emits work from rows of the range not deleted yet,
which can land after the purge ends; if you did, purge the range again once
the first purge has finished.

## See also

- [architecture.md](architecture.md) — DDD layering between
  entrypoints / service / memory / infra
- [`entrypoints/cli/main.py`](../src/everos/entrypoints/cli/main.py)
- [`entrypoints/cli/commands/server.py`](../src/everos/entrypoints/cli/commands/server.py)
