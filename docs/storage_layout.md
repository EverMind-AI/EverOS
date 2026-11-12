# Storage Layout

How `everos` lays out a memory-root on disk: directory tree, file
naming, frontmatter chassis, and entry-id encoding.

The contents are the **source of truth**; SQLite and the configured
vector/BM25 index backend are derived indexes that can be rebuilt from
markdown alone.

## 1. Memory-root tree

A memory-root is a single directory holding all persisted memory. The
default location is `~/.everos/`; override via the `EVEROS_ROOT`
env var or `--root` on the CLI.

Memory is partitioned by **`<app_id>/<project_id>`** *before* the
user-visible scope dirs, so different `(app, project)` spaces never share
a directory. The reserved id `"default"` materialises as `default_app` /
`default_project` on disk. The scope is encoded **in the path**, not in
the frontmatter (see [§3](#3-frontmatter-chassis-yaml)).

```
<memory-root>/                              default ~/.everos
│
├── <app_id>/                               user-visible; "default" → default_app
│   └── <project_id>/                       "default" → default_project
│       ├── users/
│       │   └── <user_id>/
│       │       ├── user.md                          single-file rewrite (profile)
│       │       ├── episodes/                         daily-log append
│       │       │   └── episode-<YYYY-MM-DD>.md
│       │       ├── images/                           daily-log append
│       │       │   └── image-<YYYY-MM-DD>.md
│       │       └── scenes/                           daily-log append (sc_ ids)
│       │           └── scene-<YYYY-MM-DD>.md         one entry per scene version (scenes.md)
│       └── groups/                                   group memory (group-memory.md)
│           └── <group_id>/
│               ├── group.md                          meta + append-only roster log
│               ├── episodes/                         daily-log append (gep_ ids)
│               │   └── episode-<YYYY-MM-DD>.md
│               ├── images/                           daily-log append (gim_ ids)
│               │   └── image-<YYYY-MM-DD>.md
│               ├── scenes/                           daily-log append (gsc_ ids)
│               │   └── scene-<YYYY-MM-DD>.md         one entry per scene version
│               ├── members/                          single-file rewrite
│               │   └── <member_id>.md                member-in-group profile
│               └── profile/                          group profile
│                   ├── group_profile-<YYYY-MM-DD>.md daily-log append (gp_ ids), one entry per item version
│                   └── pending.md                    the pending batch (frontmatter only, not indexed)
│
├── figures/                             picture originals, content-addressed
│   ├── <first two hex>/<sha256>.<ext>    one file per distinct picture
│   └── .cache/<sha256>-<edge>.<ext>      derived copies (jpeg; png when greyscale), rebuildable
│
├── .index/                              system-managed, rebuildable (gitignore)
│   ├── store_version                    the store's EverOS major (`2`); see below
│   ├── sqlite/
│   │   ├── system.db                    state / cascade queue (md_change_state) / buffer / audit / LSN  (+ -wal / -shm)
│   │   ├── ome.db                        Offline Memory Engine state
│   │   ├── ome.aps.db                    APScheduler jobstore (split to avoid lock contention)
│   │   └── ome.db.lock                   OME single-engine guard (portalocker)
│   ├── lancedb/
│   │   └── <kind>.lance/                default derived index backend
│
├── ome.toml                             user-editable OME strategy overrides (hot-reloaded)
└── .tmp/                                staging dir for batch / multi-step writes
```

> Cascade queue state, the LSN watermark, and the change audit all live in
> SQLite (`system.db`, table `md_change_state`) — crash-recovery replays
> from that durable queue, not a log file. (`MemoryRoot` also exposes a
> `.lock` anchor for the `memory_root_lock` primitive; there is no
> `.cascade.log` / `.manifest.json`.)

The path manager is [`MemoryRoot`](../src/everos/infra/persistence/memory_root.py),
exposing every path as a property. `MemoryRoot.ensure()` creates the
runtime-required dirs (`.index/{sqlite,lancedb}/`, `.tmp/`); the
user-visible dirs are *not* pre-created — they appear on first write.
Config files (`everos.toml`, `ome.toml`) are created by `everos init`.

> `memcell` is a SQLite-only kind (the boundary ledger); it has no markdown
> file.

## 2. Two storage strategies

Each business memory kind picks one of two on-disk patterns:

| Strategy | Filename | Mutation | Examples |
|---|---|---|---|
| **Daily-log append** | `<FILE_PREFIX>-<YYYY-MM-DD>.md` under `<DIR_NAME>/` | append entries | episode / image_memory / scene versions (user and group) / group-profile items |
| **Single-file rewrite** | `user.md`; `groups/<g>/members/<member_id>.md` | overwrite the file | a person's profile in a space: one schema (`ProfileFrontmatter`), one store and one cascade handler for the user's own profile and a member's in a group |

Markdown IO primitives live in
[`infra/persistence/markdown/_base/`](../src/everos/infra/persistence/markdown/_base/);
business-aware writers live in
[`infra/persistence/markdown/_writers/`](../src/everos/infra/persistence/markdown/_writers/)
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
(`user_id` / `group_id`) and `track`.

The chassis lives in [`infra/persistence/markdown/_base/frontmatter.py`](../src/everos/infra/persistence/markdown/_base/frontmatter.py)
(Pydantic v2):

```
BaseFrontmatter            id / type / schema_version + SCOPE_DIR ClassVar
   ├─ UserScopedFrontmatter   + user_id / track="user" + SCOPE_DIR="users"
   └─ GroupScopedFrontmatter  + group_id / track="group" + SCOPE_DIR="groups"
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

A group's kinds use their own prefixes (`gep`, `gim`, `gsc`, `gp`), so a
group whose id equals a user's id never shares that user's row ids.

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

Implementation: [`infra/persistence/markdown/_base/entries.py`](../src/everos/infra/persistence/markdown/_base/entries.py)
(`EntryId.parse / format / next_for`).

> **File-level seq, not global**: the same `ep_20260601_00000001` may
> appear across two different `user_id`s (each user has its own daily file).
> Cross-table joins must therefore key on **`(app_id, project_id, owner_id, entry_id)`**
> rather than `entry_id` alone — see the derived index tables that follow.

## 5. SQLite + derived indexes

```
.index/
├── sqlite/
│   └── system.db          state / audit / cascade queue + buffer / LSN
│                           (system tables: md_change_state, memcell,
│                            unprocessed_buffer, conversation_status, cluster, memorize_inbox)
├── lancedb/
│   └── <kind>.lance/      default derived index backend
```

- **SQLite** ([`infra/persistence/sqlite/_tables/`](../src/everos/infra/persistence/sqlite/_tables/))
  holds only system / coordination tables — `md_change_state` (cascade
  queue), `memcell` (boundary ledger), `unprocessed_buffer`,
  `conversation_status`, `cluster` / `cluster_member` (scene clusters),
  `memorize_inbox` (the `async_mode` queue), `forget_log` /
  `message_forget`, `image_original` (picture metadata) and the group
  roster mirror (`group_meta`, `group_member`, `session_binding`) — **not**
  per-kind business rows.
- The **derived index backend** holds the per-kind business rows, keyed by
  each table's declared `PRIMARY_KEY` — `(app_id, project_id, id)` for every
  business table, where `id` is `<owner_id>_<entry_id>` for the daily-log kinds
  (so cross-table joins use `(app_id, project_id, owner_id, entry_id)`; `id`
  alone is unique only within one project).
  LanceDB is the default backend under `.index/lancedb/`; Milvus can be enabled
  as the same rebuildable index backend and lives outside the memory root in a
  configured Milvus Server or Zilliz Cloud deployment.

Episode index rows carry a `deprecated_by: str | None` column: the entry id
of the version that superseded the row, or the mark a forget set (projected
from the file's `deprecated_entries`). Search filters automatically exclude
rows where `deprecated_by IS NOT NULL`.

EverOS 2.0 keeps no atomic facts and no foresight. A store written before
they were removed may still hold `users/<id>/.atomic_facts/` and
`users/<id>/.foresights/` directories and `atomic_fact` / `foresight` index
tables: nothing watches, indexes, searches, deletes or rebuilds them any more.
A cascade change still queued for them is marked failed (`cascade status`
counts it; health does not), and a background run still pending for their
strategies is dropped with a warning. They can be deleted by hand.

EverOS 2.0 also keeps no agent memory, no knowledge base and no Reflection.
A 2.0 store written before they were removed keeps its leftovers untouched:
the `agents/` and `knowledge/` markdown matches no cascade kind and is never
scanned; the `agent_case`, `agent_skill` and `knowledge_topic` index tables
and the `knowledge_document`, `knowledge_topic` and `reflection_report`
SQLite tables are neither opened nor created; the `memcell.case_*` columns
stay unused, and `cluster` rows of kind `user_memory` or `agent_case` are
never read. An episode an earlier Reflection merged (`parent_type: cluster`,
`session_id` empty) is read, listed and searched like any other, and the
episodes it replaced still show `merged` in their history.
`everos cascade sync --force-all` re-queues every ledger row; a row of a
removed kind is then marked failed (`no handler registered`), which
`cascade status` counts and `/health`'s `healthy` ignores. A background run
still pending for a removed strategy is dropped with a warning
(`ome_run_dropped_unknown_strategy`). The leftover files and tables can be
deleted by hand.

Both layers are **fully derivable from markdown** — wipe `.index/`
and the in-process cascade subsystem re-builds everything by scanning the
user-visible tree (the durable `md_change_state` SQLite queue covers
crash-recovery replay). A root with markdown and no `.index/` is opened as a
new store (see below); `everos cascade rebuild` re-indexes without losing the
SQLite state (the buffer of unprocessed messages, the cascade ledger).

### Store version

`.index/store_version` holds the store's EverOS major, one integer. EverOS 2
opens only the stores it built: every process that opens a root -- the
server, every `everos cascade` command, backfill -- checks it before anything
in the root changes.

| The root | What happens |
|---|---|
| holds `2` | Opens. Changes inside the major keep migrating as before (LanceDB `.table_schema_version`, SQLite added columns). |
| holds another number (an EverOS 1.x root someone stamped, a newer major) or no number | Refused, unchanged. |
| has no store state -- no `system.db`, no index table | A new store: `2` is written. Markdown already there (hand-written, or left after `.index/` was wiped) is indexed by this build: markdown carries no store state. |
| has store state but no marker, and something only EverOS 2 writes (index table schema v3 or later, the `image_memory` / `group_profile_item` index tables, a 2.0 system table such as `image_original`, `figures/`) | A 2.0 store from before the marker: adopted (`store_version_adopted`, with the evidence) and stamped. |
| has store state but no marker and none of that | An EverOS 1.x store: refused, unchanged (`StoreVersionError`; the CLI exits 3). 2.0's extraction and prompts changed, so a 1.x store is rebuilt from its source conversations, not upgraded in place. |

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
[`infra/persistence/markdown/_writers/base.py`](../src/everos/infra/persistence/markdown/_writers/base.py)).

## 7. References

- Skill: [`/add-memory-kind`](../.claude/skills/add-memory-kind/SKILL.md)
- Code:
  - [`infra/persistence/memory_root.py`](../src/everos/infra/persistence/memory_root.py)
  - [`infra/persistence/markdown/_base/`](../src/everos/infra/persistence/markdown/_base/)
  - [`infra/persistence/{markdown,sqlite,lancedb,milvus,index}/`](../src/everos/infra/persistence/)
  - [`infra/cascade/`](../src/everos/infra/cascade/) (md → derived index sync)
