# Storage Layout

How `everos` lays out a memory-root on disk: directory tree, file
naming, frontmatter chassis, and entry-id encoding.

Markdown is the **source of truth for extracted memory content**. LanceDB is a
rebuildable projection. SQLite also contains coordination and in-flight state,
including buffered messages, that cannot be reconstructed from markdown alone.

## 1. Memory-root tree

A memory-root is a single directory holding all persisted memory. The
default location is `~/.everos/`; override via the `EVEROS_ROOT`
env var or `--root` on the CLI.

Memory is partitioned by **`<app_id>/<project_id>`** *before* the
user-visible scope dirs. In the managed layout, without operator-created
symlink aliases, different accepted `(app, project)` spaces do not share a
directory. App and project identifiers use a portable lowercase filesystem
grammar; uppercase and reserved system names are rejected before path
construction. The reserved id `"default"` materialises as `default_app` /
`default_project` on disk. The scope is encoded **in the path**, not in the
frontmatter (see [§3](#3-frontmatter-chassis-yaml)).

```
<memory-root>/                              default ~/.everos
│
├── .projection.lock                       shared server/CLI vs exclusive rebuild lifecycle lock
├── <app_id>/                               user-visible; "default" → default_app
│   └── <project_id>/                       "default" → default_project
│       ├── users/
│       │   └── <user_id>/
│       │       ├── user.md                          single-file rewrite (profile)
│       │       ├── episodes/                         daily-log append
│       │       │   └── episode-<YYYY-MM-DD>.md
│       │       ├── .atomic_facts/                    daily-log append (hidden)
│       │       │   └── atomic_fact-<YYYY-MM-DD>.md
│       │       └── .foresights/                      daily-log append (hidden)
│       │           └── foresight-<YYYY-MM-DD>.md
│       ├── agents/
│       │   └── <agent_id>/
│       │       ├── .cases/                           daily-log append (hidden)
│       │       │   └── agent_case-<YYYY-MM-DD>.md
│       │       └── skills/                           skill-named dir
│       │           └── skill_<name>/
│       │               ├── SKILL.md
│       │               ├── references/               (optional)
│       │               └── scripts/                  (optional)
│       └── knowledge/                                user-visible (shared / global)
│
├── .index/                              system-managed runtime state (gitignore)
│   ├── .projection.bootstrap.lock        serializes first-time marker/schema/index setup
│   ├── sqlite/
│   │   ├── system.db                    state / cascade queue (md_change_state) / buffer / audit / LSN  (+ -wal / -shm)
│   │   ├── ome.db                        Offline Memory Engine state
│   │   ├── ome.aps.db                    APScheduler jobstore (split to avoid lock contention)
│   │   └── ome.db.lock                   OME single-engine guard (portalocker)
│   └── lancedb/
│       ├── .storage_identity.json         READY/REBUILDING storage-key generation gate
│       └── <kind>.lance/                one directory per LanceDB table
│
├── ome.toml                             user-editable OME strategy overrides (hot-reloaded)
└── .tmp/                                staging dir for batch / multi-step writes
```

> Cascade queue state, the LSN watermark, and the change audit all live in
> SQLite (`system.db`, table `md_change_state`) — crash-recovery replays
> from that durable queue, not a log file. (`MemoryRoot` also exposes a
> `.lock` anchor for the `memory_root_lock` primitive; there is no
> `.cascade.log` / `.manifest.json`.)

`.projection.lock` is retained in shared mode by API servers and mutating
cascade commands, and in exclusive mode by `cascade rebuild`. First-time
marker, schema, and index setup is serialized by a separate bootstrap lock.
The storage-identity marker is published as `READY(2)` after a successful
rebuild or when a marker-less root has neither source markdown nor LanceDB
artifacts. Retained SQLite state is separate and still requires the upgrade
audit described in the cascade runbook. Missing, malformed, stale, or
`REBUILDING` state blocks normal writers.

The path manager is [`MemoryRoot`](../src/everos/core/persistence/memory_root.py),
exposing every path as a property. `MemoryRoot.ensure()` creates the
runtime-required dirs (`.index/{sqlite,lancedb}/`, `.tmp/`); the
user-visible dirs are *not* pre-created — they appear on first write.
Config files (`everos.toml`, `ome.toml`) are created by `everos init`.

> The single-file writer also supports `agent.md` / `soul.md` / `tools.md`
> / `behaviors.md`, but no shipped strategy produces those today — only
> `user.md` is written. `memcell` is a SQLite-only kind (the boundary
> ledger); it has no markdown file.

## 2. Three storage strategies

Each business memory kind picks one of three on-disk patterns:

| Strategy | Filename | Mutation | Examples |
|---|---|---|---|
| **Daily-log append** | `<FILE_PREFIX>-<YYYY-MM-DD>.md` under `<DIR_NAME>/` | append entries | episode / atomic_fact / foresight / agent_case |
| **Skill-named dir** | `skills/skill_<name>/SKILL.md` (+ `references/` `scripts/`) | overwrite the file | agent skills (procedural memory) |
| **Single-file rewrite** | `user.md` (writer also supports `agent.md` / `soul.md` / `tools.md` / `behaviors.md`, not yet produced) | overwrite the file | user profile |

Markdown IO primitives live in
[`core/persistence/markdown/`](../src/everos/core/persistence/markdown/);
business-aware writers live in
[`infra/persistence/markdown/writers/`](../src/everos/infra/persistence/markdown/writers/)
and pick the right strategy via a base class.

For a step-by-step recipe to add a new memory kind, see the
[`/add-memory-kind`](../.claude/skills/add-memory-kind/SKILL.md) skill.

## 3. Frontmatter chassis (YAML)

Every markdown file carries a YAML frontmatter block at the top:

```markdown
---
id: episode_log_alice_2026-06-01
type: episode_daily
file_type: episode_daily
schema_version: 1
user_id: alice
track: user
date: '2026-06-01'
entry_count: 11
last_appended_at: '2026-06-01T09:12:13+00:00'
---
<!-- entry:ep_20260601_00000001 -->
...content...
<!-- /entry:ep_20260601_00000001 -->
```

Scope (`app_id` / `project_id`) is **not** a frontmatter field — it is
carried by the `<app>/<project>` path segments and recovered by the
cascade path parser. The frontmatter only holds the file-level owner
(`user_id` / `agent_id`) and `track`.

The chassis lives in [`core/persistence/markdown/frontmatter.py`](../src/everos/core/persistence/markdown/frontmatter.py)
(Pydantic v2):

```
BaseFrontmatter            id / type / schema_version + SCOPE_DIR ClassVar
   ├─ UserScopedFrontmatter   + user_id / track="user" + SCOPE_DIR="users"
   └─ AgentScopedFrontmatter  + agent_id / track="agent" + SCOPE_DIR="agents"
```

Concrete business schemas subclass one of the scope mixins and add
per-kind fields plus three more ClassVars that drive path resolution
+ entry-id assembly:

```python
class EpisodeDailyFrontmatter(DailyLogPathMixin, UserScopedFrontmatter):
    ENTRY_ID_PREFIX: ClassVar[str] = "ep"
    DIR_NAME: ClassVar[str] = "episodes"
    FILE_PREFIX: ClassVar[str] = "episode"
    type: Literal["episode_daily"] = "episode_daily"
    date: dt.date
    entry_count: int = 0
    last_appended_at: dt.datetime | None = None
```

## 4. Entry-id encoding

Inside daily-log files each entry is bracketed by HTML-comment markers
so the raw markdown stays clean for human readers:

```
<!-- entry:<entry_id> -->
...content...
<!-- /entry:<entry_id> -->
```

`<entry_id>` is `<prefix>_<YYYYMMDD>_<NNNNNNNN>` (8-digit sequence),
e.g. `ep_20260601_00000001`:

| Segment | Source |
|---|---|
| `prefix` | `Frontmatter.ENTRY_ID_PREFIX` (declared by the schema subclass) |
| `<YYYYMMDD>` | The daily-log file's date bucket |
| `NNNNNNNN` | Per-file sequence, 8-digit zero-padded, restarts at `00000001` each day per scope |

Implementation: [`core/persistence/markdown/entries.py`](../src/everos/core/persistence/markdown/entries.py)
(`EntryId.parse / format / next_for`).

> **File-level seq, not global**: the same `ep_20260601_00000001` may
> appear across two different `user_id`s (each user has its own daily file).
> Cross-table joins must therefore key on **`(app_id, project_id, owner_id,
> entry_id)`** rather than `entry_id` alone. LanceDB uses a separate opaque,
> injective storage key for this complete identity while public response IDs
> retain their historical shape.

## 5. SQLite runtime state + LanceDB projection

```
.index/
├── sqlite/
│   └── system.db          state / audit / cascade queue + buffer / LSN
│                           (system tables: md_change_state, memcell,
│                            unprocessed_buffer, conversation_status, cluster)
└── lancedb/
    ├── .storage_identity.json  fail-closed storage-key generation marker
    └── <kind>.lance/      one Arrow table per business kind — the per-kind
                            rows (text / vector / tokens / metadata) live here
```

- **SQLite** ([`infra/persistence/sqlite/tables/`](../src/everos/infra/persistence/sqlite/tables/))
  holds only system / coordination tables — `md_change_state` (cascade
  queue), `memcell` (boundary ledger), `unprocessed_buffer`,
  `conversation_status`, `cluster`, `knowledge`, `reflection_report` — **not**
  per-kind business rows. `reflection_report` is the audit trail for
  Reflection merges (cluster_id, mode, source_members, merged_entry_id,
  status).
- **LanceDB** ([`infra/persistence/lancedb/tables/`](../src/everos/infra/persistence/lancedb/tables/))
  holds the per-kind business rows. Owner-scoped tables use an opaque,
  injective storage key over app, project, owner, and the type-specific
  logical id. Logical references remain in their explicit columns; each
  table's `Vector(N)` dimension matches the embedding model output.

Episode and AtomicFact LanceDB tables carry a `deprecated_by: str | None`
column. When an episode is superseded by a Reflection merge,
`deprecated_by` is set to the merged episode's entry_id. Search filters
automatically exclude rows where `deprecated_by IS NOT NULL`.

LanceDB business rows are a rebuildable projection of markdown. SQLite is not
fully disposable: it also contains `unprocessed_buffer` messages that have not
yet become markdown. Do not wipe `.index/` or `.index/lancedb` manually.
Use `everos cascade rebuild`, which preserves SQLite state, resets the durable
queue, rebuilds LanceDB, and publishes the required storage-identity marker.

## 6. Atomic write semantics

`MarkdownWriter` uses a same-directory temp file
(`.<name>.tmp.<uuid>`) + `os.replace` for atomicity. Keeping the temp
file in the same directory guarantees `os.replace` is atomic on POSIX
(the rename is only atomic within a single filesystem).

`MarkdownWriter.append_entry` reads → merges frontmatter →
appends an entry block → atomic write back. The caller passes a full
`EntryId` (built via `EntryId.next_for(prefix, date, current_count)`);
this primitive is **schema-agnostic** — field-level semantics
(`entry_count` / `last_appended_at`) are a business writer's job
(see `BaseDailyWriter._frontmatter_updates` in
[`infra/persistence/markdown/writers/base.py`](../src/everos/infra/persistence/markdown/writers/base.py)).

## 7. References

- Skill: [`/add-memory-kind`](../.claude/skills/add-memory-kind/SKILL.md)
- Code:
  - [`core/persistence/memory_root.py`](../src/everos/core/persistence/memory_root.py)
  - [`core/persistence/markdown/`](../src/everos/core/persistence/markdown/)
  - [`infra/persistence/{markdown,sqlite,lancedb}/`](../src/everos/infra/persistence/)
  - [`memory/cascade/`](../src/everos/memory/cascade/) (md → LanceDB sync)
