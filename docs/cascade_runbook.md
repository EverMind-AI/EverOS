# Cascade Runbook

The cascade daemon keeps LanceDB in sync with the markdown files under
the memory root. Service / entry points only ever write markdown; the
daemon is the **sole** writer of the LanceDB index. This runbook covers
the recurring operational questions.

## What runs where

When `everos server start` boots, the FastAPI lifespan wires four
providers in order:

1. **Metrics** — Prometheus collector.
2. **SQLite** — system DB + schema (`SQLModel.metadata.create_all`).
3. **LanceDB** — async connection + schema verification + FTS indexes.
4. **Cascade** — watcher + scanner + worker, all in-process tasks.

The cascade subsystem itself is three independent loops:

| Loop | Source signal | Effect |
|---|---|---|
| Watcher | `watchdog` filesystem events (sync thread) | `md_change_state.upsert` per registered kind |
| Scanner | Periodic walk (`scan_interval_seconds`, default 30 s) | Same — catches changes the watcher missed |
| Worker | `claim_pending_batch` polling (default 1 s when idle) | Handler dispatch → LanceDB upsert / delete |

Every loop talks to the same `md_change_state` sqlite table. The
worker's claim mode (`pending → processing → done/failed`) keeps
concurrent workers honest.

## Health: `everos cascade status`

```
queue:
  pending:                   3
  done:                      1247
  failed (retryable=TRUE):   1     (eligible for `cascade fix --apply`)
  failed (retryable=FALSE):  1     (fix md and re-save to recover)
lsn:
  max:           1252
  last_processed: 1250
  lag:            2
```

- `lag > 0` means the worker is behind. Steady state should hover near
  zero; sustained lag points at a slow handler or a stuck retry.
- `failed (retryable=FALSE)` is always user-actionable. Cascade will
  never auto-clear these — they represent malformed md the user must
  edit.

## Recovering from failures: `everos cascade fix`

`cascade fix` (no flag) lists every failed row. With `--apply`:

1. `UPDATE md_change_state SET status='pending', retry_count=0
   WHERE status='failed' AND retryable=TRUE` (the partial index
   `idx_md_change_retryable` makes this O(retryable)).
2. Drain the worker once so the retry runs synchronously.

Retryable failures cover transient embedding / HTTP errors (5xx, 429,
network resets) after the inline `MAX_RETRY=3` was exhausted. The
fix command resets the counter so a working backend gets a clean
start.

`retryable=FALSE` rows require the user to edit the md (typically a
YAML frontmatter issue) and re-save; the watcher picks the change up
naturally.

## One-shot replay: `everos cascade sync [PATH]`

Use this when the watcher missed an event (WSL mount, network share,
external editor with no inotify) or when you want a deterministic
flush before, say, a smoke test:

```bash
everos cascade sync                           # drain everything pending
everos cascade sync users/u1/episodes/X.md    # re-enqueue + drain
```

The CLI builds the same `CascadeOrchestrator` as the daemon but only
calls `sync_once` / `drain_once` — no watcher / scanner background
task. So it's safe to run in parallel with a live `everos server`.

## Recovery paths

### LanceDB schema drift on startup

`LanceDBLifespanProvider.startup` calls `verify_business_schemas`. If
an on-disk table has columns the current Pydantic schema does not
declare (or vice versa), the boot fails with:

```
LanceDB table 'episode' schema drift: missing=[...], extra=[...].
The index is rebuildable from md — recover with
`rm -rf ~/.everos/.index/lancedb` and restart.
```

This is the documented recovery: delete the index, restart the
server, the scanner will pick up every md file on its first sweep and
the worker repopulates LanceDB. Markdown is the source of truth, so
no data is lost.

### inotify watch-limit exhaustion (Linux)

Default kernel limit is 8 192 watches per user. On a sizeable memory
root the watcher may silently miss events. Symptoms:

- Scanner catches the file changes but the watcher never logs an
  event for the same path.
- `cat /proc/sys/fs/inotify/max_user_watches` is at the limit.

Fix by bumping the kernel parameter:

```bash
echo fs.inotify.max_user_watches=524288 | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### WSL2 / network mounts

Filesystem events do not propagate from the Windows host into WSL2
(or across most SMB / NFS shares). The watcher will start without
error and silently see nothing.

Workarounds:

- Rely on the scanner — at default 30 s interval, throughput is
  bounded but eventually-consistent.
- Drop the scan interval to ~5 s if the memory root is small.
- Run `everos cascade sync` explicitly after batch edits.

### Daemon process crash mid-batch

`claim_pending_batch` flips rows to `processing` *atomically*. If the
process dies before `mark_done` / `mark_failed`, those rows stay in
`processing` until the next boot. **The orchestrator auto-recovers**
on startup: `CascadeOrchestrator.start` calls
`md_change_state_repo.recover_orphan_processing()` before launching
the watcher / scanner / worker, which resets every `processing` row
back to `pending`. Single-process cascade means no race — at boot
time no other worker could legitimately own a `processing` row.

No operator action required; the structured log line
`cascade_recovered_orphan_processing` reports the count when it
fires.

### FD exhaustion (`os error 24` / EMFILE)

Symptoms (any of these on a long-running daemon):

- LanceDB query / index build fails with `lance error: ... Too many
  open files (os error 24)`.
- `lsof -p <pid> | wc -l` grows monotonically over hours / days.
- Health log lines like `cascade_lancedb_optimize_failed` /
  `cascade_lancedb_rebuild_failed` carrying `OSError: [Errno 24]`.

Cause (verified against `lance crate 4.0`): the LanceDB *index* cache
(`GlobalIndexCache`) holds one reader object per opened FTS / vector
/ scalar index, and each reader pins the file descriptors of its
`_indices/<uuid>/...` files. With a long-running daemon and steady-
state cascade ingest, every `optimize()` call adds new readers; with
LanceDB's own default (`index_cache_size_bytes=None`, unbounded), they
**are never evicted** and the FDs leak monotonically.

`drop_index` does **not** help — it is a manifest-only operation and
leaves the on-disk UUID directories untouched. Even an explicit
`optimize(cleanup_older_than=0)` `unlink()`-ing the files does not
release FDs: POSIX keeps the inode alive as long as a process holds
an open FD on it (the entries show as `(deleted)` in `lsof`). Only an
LRU eviction inside the cache (or a connection close) actually closes
the FDs.

Fix (already wired in `LanceDBSettings.index_cache_size_bytes` —
default 16 MB, ~290 FD ceiling): see
[Tuning knobs § LanceDB index cache](#lancedb-index-cache-index_cache_size_bytes)
for the sizing table and the env-var override path.

If you have already hit EMFILE in a running process, the cleanest
recovery is a daemon restart — the open connection closes, every FD
is released, and the next start comes up with the capped Session in
place.

## Tuning knobs

### Cascade scheduler knobs

All defaults live in `everos.memory.cascade.orchestrator.CascadeConfig`
and `everos.memory.cascade.worker.CascadeWorker`:

| Knob | Default | Effect |
|---|---|---|
| `scan_interval_seconds` | 30 | Scanner sweep cadence |
| `worker_batch_size` | 50 | Rows claimed per worker cycle |
| `worker_max_retry` | 3 | Inline retries before `mark_failed(retryable=TRUE)` |
| `worker_poll_interval_seconds` | 1 | Idle wait between empty drain attempts |
| `worker_retry_backoff_seconds` | 2 | Linear backoff seed; doubles per attempt |

Tuning surface is intentionally not in `Settings` yet — once we have
wall-clock numbers from real workloads, the values that need
operator override will surface there.

### LanceDB index cache (`index_cache_size_bytes`)

Lives in `LanceDBSettings`; overridable via the
`EVEROS_LANCEDB__INDEX_CACHE_SIZE_BYTES` environment variable. This
is the only knob that bounds the steady-state file-descriptor count
of a long-running EverOS daemon — see
[Recovery paths § FD exhaustion](#fd-exhaustion-os-error-24--emfile)
for why nothing else (prune, rebuild, `drop_index`) helps.

Measured cap → FD ceiling (30 add+optimize cycles + 100-query stress
on the real `Episode` schema):

| Cap | FD ceiling | Query latency (p50) | Safe under `ulimit -n` |
|---|---|---|---|
| `2 MB` | ~45 | ~5 ms | macOS default 256 (5× headroom) |
| `4 MB` | ~52 | ~3 ms | macOS default 256 |
| `8 MB` | ~140 | ~2.4 ms | macOS default 256 (1.8× headroom) |
| **`16 MB`** (default) | **~290** | **~2.3 ms** | **Linux default 1024 (3.5× headroom); macOS needs `ulimit -n 1024`** |
| `32 MB` | ~630 | ~1.4 ms | Linux default 1024 (1.6× headroom) |
| `unbounded` | grows forever | ~1.3 ms | NEVER use in a daemon |

EverOS's measured steady-state working set after a `rebuild_indexes`
cycle is roughly **50-100 readers / 3-6 MB resident** (5 tables × ~7
BM25 columns × ~10 `part_N` reader entries each), so the 16 MB default
provides ~3× headroom for burst traffic and stale-but-not-yet-evicted
readers.

When to override:

- **Tight `ulimit -n` environments** (containers; macOS dev boxes
  that haven't bumped the default 256) → drop to `4 MB` or `8 MB`.
  Query latency increases by ~1-3 ms but correctness is unaffected.
- **Larger working sets** (many more tables or much wider FTS
  indexes than the default schema set) → bump to `32-64 MB`. Verify
  your platform's `ulimit -n` covers the corresponding FD ceiling
  with at least 2× headroom.
- **Diagnostic-only**: set to a tiny value (e.g. `1 MB`) to
  *force* LRU thrashing and reproduce cache-miss latency in tests.

Do **not** set `metadata_cache_size_bytes` — it is intentionally left
at LanceDB's default (unbounded) because the metadata cache holds
parsed manifests / fragment stats and has zero effect on FD count;
capping it just thrashes parsing work without solving anything.

## Concurrency

The worker is async, not multi-process. Inside one drain cycle,
`asyncio.gather(*[_process_one(row) for row in batch])` runs every
claimed row concurrently — cascade is IO-bound (embedding HTTP calls
dominate wall time) so single-process coroutine concurrency saturates
the bottleneck. The `worker_batch_size` knob (default 50) caps
in-flight rows.

Multi-process workers are a scaling axis we'd reach for only if a
single process becomes CPU-bound, which the current design does not
anticipate. `claim_pending_batch` is already race-safe (the
``WHERE status='pending'`` filter ensures each row lands in exactly
one batch even if multiple workers raced), so adding processes later
is a deployment-side change with no schema work.

## What cascade does NOT do (yet)

- **Schema migration**: LanceDB column changes require `rm -rf`.
- **Parent-id back-link**: Episode rows currently carry
  `parent_id=None`; the writer doesn't preserve the source memcell id
  in the entry inline. Tracked separately.
- **Reference-file change detection (agent_skill)**: edits to
  `references/*.md` siblings won't trigger a re-index — only changes
  to `SKILL.md` itself fire the watcher. Workaround: run
  `everos cascade sync agents/<a>/skills/skill_<n>/SKILL.md` after
  editing references.
