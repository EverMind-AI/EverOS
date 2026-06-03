# How Memory Works

How EverOS turns a stream of messages into durable, searchable memory —
the storage stack, the path layout on disk, the write→index→read
pipeline, and the consistency guarantees.

This is the narrative companion to the reference docs: see
[storage_layout.md](storage_layout.md) for the exact file encoding,
[architecture.md](architecture.md) for the layer boundaries, and
[api.md](api.md) for the HTTP contract.

## Table of contents

- [The storage stack](#the-storage-stack)
- [Storage paths](#storage-paths)
- [How a memory is born](#how-a-memory-is-born)
- [Memory types & storage strategies](#memory-types--storage-strategies)
- [The cascade daemon](#the-cascade-daemon)
- [The Offline Memory Engine (OME)](#the-offline-memory-engine-ome)
- [Consistency model](#consistency-model)
- [Zero external services](#zero-external-services)
- [Operating it](#operating-it)

## The storage stack

Three embedded pieces, each owning what it is best at. Markdown is the
**source of truth**; the other two are **derived and rebuildable**.

| Layer | Backed by | Holds | Rebuildable? |
|---|---|---|---|
| **Markdown + YAML frontmatter** | plain `.md` files | the memory content itself — the only portable, human-editable asset | — (it *is* the truth) |
| **SQLite** (`aiosqlite`) | `.index/sqlite/*.db` | system state, audit log, the cascade queue, the boundary buffer, OME engine state | ✅ from markdown |
| **LanceDB** (Arrow) | `.index/lancedb/*.lance` | vector + BM25 + scalar columns for retrieval | ✅ from markdown |

!!! note "The one rule that follows from this"
    Delete the entire `.index/` directory and **no memory is lost** — it
    rebuilds from the `.md` tree. There is no separate "export"; the
    markdown *is* the export. (How to trigger a rebuild:
    [Operating it](#operating-it).)

## Storage paths

The default memory root is **`~/.everos/`** (override with
`EVEROS_MEMORY__ROOT` or `[memory] root` in TOML). Configuration (the
`.env` file) is separate from data (the memory root): the server searches
`./.env` → `$XDG_CONFIG_HOME/everos/.env` → `~/.everos/.env`.

Memory is partitioned by **`<app_id>/<project_id>`** *before* the
user-visible directories, so different `(app, project)` spaces never share
a directory or cross in search. The reserved id `"default"` materialises as
`default_app` / `default_project` on disk (so a default space stays
visually distinct from a user-named one).

```
~/.everos/                                  ← memory root (EVEROS_MEMORY__ROOT)
├── default_app/                            ← <app_id>   ("default" → default_app)
│   └── default_project/                    ← <project_id> ("default" → default_project)
│       ├── users/                          ← user-visible (source of truth)
│       │   └── <user_id>/
│       │       ├── user.md                 single-file   (profile)
│       │       ├── episodes/
│       │       │   └── episode-<YYYY-MM-DD>.md      daily-log append
│       │       ├── .atomic_facts/                   daily-log (hidden)
│       │       │   └── atomic_fact-<YYYY-MM-DD>.md
│       │       └── .foresights/                     daily-log (hidden)
│       │           └── foresight-<YYYY-MM-DD>.md
│       ├── agents/
│       │   └── <agent_id>/
│       │       ├── .cases/                          daily-log (hidden)
│       │       │   └── agent_case-<YYYY-MM-DD>.md
│       │       └── skills/                          skill-named dir
│       │           └── skill_<name>/SKILL.md        (+ references/ scripts/)
│       └── knowledge/                      ← shared / global (reserved)
│
├── .index/                                 ← system-managed, rebuildable (gitignore)
│   ├── sqlite/
│   │   ├── system.db                       state / audit / cascade queue (md_change_state) / buffer / LSN
│   │   ├── ome.db                          Offline Memory Engine state
│   │   ├── ome.aps.db                      APScheduler jobstore (split to avoid lock contention)
│   │   └── ome.db.lock                     OME single-engine guard (portalocker)
│   └── lancedb/
│       └── <kind>.lance/                   one Arrow table per kind
│
├── ome.toml                                ← user-editable OME strategy overrides (hot-reloaded)
└── .tmp/                                   atomic-write staging
```

!!! warning "Differences from older PRD-era docs"
    The index dir is **`.index/`** (dot-prefixed), not `_index/`. The
    cascade queue and LSN/audit state live in **SQLite** (`system.db`,
    table `md_change_state`) — there is no `.cascade.log` / `.manifest.json`
    file in the current implementation. The `<app>/<project>` nesting is
    real and always present (`default_app/default_project` for the default
    scope). There is **no `everos reindex` command** (see
    [Operating it](#operating-it)).

The path manager is
[`MemoryRoot`](../src/everos/core/persistence/memory_root.py); every path
above is a property on it. `MemoryRoot.ensure()` creates the runtime dirs
(`.index/{sqlite,lancedb}/`, `.tmp/`) and copies the OME template to
`ome.toml`; user-visible dirs appear on first write.

## How a memory is born

A message does not become memory immediately — it accumulates, a boundary
is detected, an LLM extracts a cell, writers persist markdown, and the
index catches up asynchronously.

```
 POST /add  ──▶  unprocessed_buffer (SQLite)        ← messages accumulate per (session, app, project)
                      │
                      ├─ boundary detector trips  ─┐
 POST /flush ─────────┤  (or you force it)          │  one LLM call
                      │                             ▼
                      │                       extract MemCell  ──▶  memcell row (SQLite)
                      │                             │
                      │              ┌──────────────┴───────────────┐
                      │              ▼                              ▼
                      │   UserMemoryPipeline (sync)      AgentMemoryPipeline (fire-and-forget)
                      │   writes episode .md NOW          emits AgentPipelineStarted
                      ▼              │                              │
   (response returns once md is on disk)                           │
                                     ▼                              ▼
                          ┌───────────────────  Offline Memory Engine (OME)  ───────────────────┐
                          │  async strategies write derived .md:                                 │
                          │  atomic_facts · foresight · user profile · agent cases · agent skills │
                          └───────────────────────────────┬──────────────────────────────────────┘
                                                           ▼
                                            cascade daemon watches the .md tree
                                                           ▼
                                       md_change_state queue (SQLite, durable)
                                                           ▼
                                            rebuild LanceDB rows  ──▶  searchable
```

- **`/add`** appends messages to a per-`(session_id, app_id, project_id)`
  buffer and returns `accumulated` (or `extracted` if the boundary tripped
  on this call). See [api.md](api.md).
- **`/flush`** forces the boundary now (one extraction LLM call), used at
  the end of a chat/agent run.
- Episode markdown is written **synchronously** — when `/flush` returns
  `extracted`, the episode file is already on disk.
- Everything else (atomic facts, foresight, profile, agent cases/skills)
  is produced **asynchronously** by the OME — see
  [the OME section](#the-offline-memory-engine-ome).
- The **cascade daemon** turns every `.md` write into LanceDB rows so the
  content becomes searchable.

## Memory types & storage strategies

Six business memory kinds today, each user- or agent-owned, each picking
one of three on-disk patterns:

| Kind | Owner | Dir / file | Strategy | Produced by |
|---|---|---|---|---|
| **episode** | user | `episodes/episode-<date>.md` | daily-log | extraction (sync) |
| **atomic_fact** | user | `.atomic_facts/atomic_fact-<date>.md` (hidden) | daily-log | OME |
| **foresight** | user | `.foresights/foresight-<date>.md` (hidden) | daily-log | OME |
| **profile** | user | `user.md` | single-file rewrite | OME |
| **agent_case** | agent | `.cases/agent_case-<date>.md` (hidden) | daily-log | OME |
| **agent_skill** | agent | `skills/skill_<name>/SKILL.md` | skill-named dir | OME (clustering) |

The three strategies:

| Strategy | Shape | Why |
|---|---|---|
| **Daily-log append** | `<prefix>-<YYYY-MM-DD>.md`, one entry appended per memory | collapses thousands of per-entry files into one file per day |
| **Single-file rewrite** | a fixed filename overwritten in place | for a single evolving document (a user/agent profile) |
| **Skill-named dir** | one directory per skill | a skill is a richer unit (body + optional `references/` `scripts/`) |

!!! note
    The single-file writer also supports `agent.md` / `soul.md` /
    `tools.md` / `behaviors.md`, but no shipped OME strategy produces those
    yet — today only `user.md` is written. Detailed frontmatter and
    entry-id encoding live in [storage_layout.md](storage_layout.md).

## The cascade daemon

The cascade subsystem keeps LanceDB in sync with the markdown tree. It runs
**in-process** with the server (a coroutine started by the app lifespan),
not as a separate OS daemon.

1. A native filesystem watcher (`watchdog`: FSEvents on macOS, inotify on
   Linux) sees a `.md` create/modify.
2. The change is enqueued in the **`md_change_state`** table (SQLite) —
   durable, so a crash mid-sync replays on restart.
3. A worker drains the queue at **entry-level** granularity: it diffs the
   file, re-embeds only changed entries (keyed by `content_sha256`), and
   upserts the LanceDB rows.

Because markdown is the source of truth, **editing a file directly is
fully supported** — open an episode in VSCode / Obsidian / Vim, change an
entry, save, and the daemon re-indexes just that entry. Operate the queue
with `everos cascade` ([Operating it](#operating-it)); deeper runbook in
[cascade_runbook.md](cascade_runbook.md).

## The Offline Memory Engine (OME)

Most memory kinds are **not** extracted on the request path — they are
derived later by the OME, an in-process async strategy engine. When
extraction carves a MemCell, it emits an event; OME strategies pick it up
and write their markdown when ready:

- `extract_atomic_facts` — single-sentence facts from an episode
- `extract_foresight` — anticipatory notes
- `extract_user_profile` — the aggregated `user.md`
- `extract_agent_case` — a reusable agent trajectory (only when the cell is
  substantive enough; thin trajectories are skipped by design)
- `extract_agent_skill` — clusters related cases into a named skill

Strategies are configurable without a code change via **`ome.toml`** at the
memory root (hot-reloaded within ~2 s). Example — turn two off:

```toml
[strategies.extract_foresight]
enabled = false

[strategies.extract_user_profile]
enabled = false
```

OME keeps its own state in `.index/sqlite/ome.db` (run records, counters)
and its scheduler jobstore in `.index/sqlite/ome.aps.db` (split so the sync
APScheduler writer and the async OME writer never contend for one file
lock).

!!! tip "Implication for clients"
    After `/flush` returns `extracted`, the **episode** is queryable soon
    (once cascade indexes it), but **atomic facts / profile / agent cases**
    appear only after their OME strategy runs — typically seconds later.
    Poll / retry if you need them immediately.

## Consistency model

Two paths, two guarantees:

| Path | Guarantee | Detail |
|---|---|---|
| **Write** (`/add`, `/flush`) | **strong** | the episode `.md` is on disk before the call returns `extracted`; never blocks on LanceDB |
| **Read** (`/search`, `/get`) | **eventual** | reads LanceDB, which lags md by the cascade processing time — sub-second typically, up to ~10–15 s under load |

So a `/search` immediately after the `/flush` that produced a record may
miss it. The markdown is durable regardless; index lag never loses data. If
you need read-your-write, retry with backoff, or force the queue with
`everos cascade sync`.

Integrity is anchored by a few invariants (details in
[storage_layout.md](storage_layout.md)): the frontmatter `id` /
`entry_id` is the immutable join key; `content_sha256` decides whether an
entry needs re-embedding; an LSN watermark (in `system.db`) orders
rebuilds; the durable `md_change_state` queue is the replayable audit
trail.

## Zero external services

No database server, message broker, or vector service to run. Vector ANN,
full-text BM25, and scalar filtering all execute inside the **embedded
LanceDB** engine in one query; SQLite is a local file. The whole stack is a
single directory you can copy, back up, or check the user-visible parts of
into git.

!!! note
    There is no automatic "grep over markdown" search fallback today — if
    the LanceDB index is unavailable, rebuild it from markdown (it is
    derived and disposable) rather than relying on a degraded search path.

## Operating it

The CLI ([cli.md](cli.md)) is intentionally small:

| Command | What it does |
|---|---|
| `everos init` | write a starter `.env` |
| `everos server start` | run the HTTP API (cascade + OME start with it) |
| `everos cascade status` | queue / LSN summary |
| `everos cascade sync` | drain the cascade queue now (force md → LanceDB) |
| `everos cascade fix` | list failed rows / re-enqueue retryable ones |

!!! warning "There is no `everos reindex` or `everos flush`"
    - **Reindex** = the index is rebuildable: stop the server,
      `rm -rf <memory-root>/.index/lancedb`, restart — the cascade
      rebuilds from markdown. For an incremental catch-up, use
      `everos cascade sync`.
    - **Flush** is an HTTP endpoint (`POST /api/v1/memory/flush`), not a
      CLI command — it forces *extraction* of the session buffer, which is
      a different thing from forcing *index sync* (`cascade sync`).

## References

- [storage_layout.md](storage_layout.md) — exact file encoding, frontmatter
  chassis, entry-id format, atomic-write semantics
- [architecture.md](architecture.md) — DDD layers and dependency rules
- [api.md](api.md) — the HTTP contract (`/add` `/flush` `/search` `/get`)
- [cascade_runbook.md](cascade_runbook.md) — operating the sync queue
