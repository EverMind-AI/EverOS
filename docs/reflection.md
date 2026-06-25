# Reflection

Reflection periodically consolidates the memory fragments scattered across
many conversations into a single, chronologically-organized narrative. It
runs offline in the background: it merges the multiple Episodes inside one
similarity cluster into one, resolves stale information by keeping the
latest state, and soft-archives the originals it replaces — so memory gets
more accurate and more compact with use, instead of piling up into noise.

## Prerequisites

Reflection's merge step calls an LLM, and re-clustering the merged narrative
calls an embedding model. Configure both as you would for the rest of EverOS
— an OpenAI-compatible `[llm]` and `[embedding]` block in `<root>/everos.toml`,
or the matching `EVEROS_LLM__*` / `EVEROS_EMBEDDING__*` environment variables.
If a provider is unavailable, the affected clusters are skipped and logged
rather than failing the run.

## Quick start

> The examples below assume EverOS is running on the default port 8000.
> `<root>` is the EverOS memory root (see [QUICKSTART](../QUICKSTART.md)).

Reflection is off by default. Turn it on in `<root>/ome.toml` — **one line**:

```toml
[strategies.reflect_episodes]
enabled = true
```

Once enabled, it **runs automatically every Monday at 02:00**. This is how
Reflection is meant to work — nothing else to configure. To change the run
time, add a `cron` line (a schedule expression; optional):

```toml
[strategies.reflect_episodes]
enabled = true
cron = "0 3 * * 0"        # optional: change the run time (here: Sundays at 03:00)
```

> **Don't run it too often.** Once a week at most is recommended. Each run
> is a lossy LLM merge; repeatedly re-consolidating the same memories can
> make the narrative *worse*, not better — which is why the default is
> weekly.

Config changes hot-reload (no restart needed). From then on, at each
scheduled time, Reflection consolidates each user's memory once.

**What does it produce, and where do I see it?** Each run **appends one
merged narrative** to the relevant user's Episode log, and marks the older
fragments it replaces as archived (removed from default search). Markdown is
the source of truth — just open the user's Episode log file:

```
<root>/default_app/default_project/users/<user_id>/episodes/episode-<date>.md
```

The new entry carries `parent_type: cluster` (Episodes produced by ordinary
conversation are `parent_type: memcell`). It looks roughly like this:

```markdown
---
owner_id: u_andrew
timestamp: 2026-10-11T02:00:00+00:00
parent_type: cluster          # <- marks it as a Reflection merge product
parent_id: cl_a1b2c3d4e5f6
---
## Subject
Andrew's pet adoption journey

## Content
Andrew initially had no pets. He later adopted a dog named Toby, and then
adopted another dog named Buddy. He currently has two dogs.
```

A search on the topic afterwards returns this single complete narrative
rather than the scattered old fragments.

> To inspect which clusters were consolidated, how many entries were
> archived, etc., see [Auditing & troubleshooting](#auditing--troubleshooting)
> (advanced; not needed for everyday use).
> For debugging without waiting for the schedule, you can trigger a run by
> hand — see [Triggering a run](#triggering-a-run).

## How it works

Reflection runs *offline*, separate from the live conversation path. The
online path keeps extracting Episodes and clustering them; Reflection later
consumes those clusters — it never sits between a user and a response:

```
Online (never blocked)                Offline (scheduled)
──────────────────────                ───────────────────
conversation → Episode → Cluster ───► Reflection consolidates the clusters
```

A scheduled run processes every user across all app/project tenants that have
clusters.

After each conversation, EverOS extracts an **Episode** (a summary of a
conversation segment), and geometric clustering groups semantically similar,
time-adjacent Episodes into a **Cluster**. The same topic thus ends up
scattered as several point-in-time snapshots within one cluster:

```
Cluster cl_xxx
├── ep_0001  "Andrew has no pets yet"       (August)
├── ep_0002  "Andrew adopted Toby"          (September)
└── ep_0003  "Andrew also adopted Buddy"    (October)
```

Reflection consolidates one cluster at a time, in four steps:

```
Select ─→ Merge ─→ Re-extract ─→ Deprecate
```

1. **Select** — pick clusters worth consolidating: not yet consolidated and
   holding ≥ 2 members, or already consolidated and since joined by new
   members. At most 10 clusters per run, largest first.
2. **Merge** — hand the cluster's Episodes to the LLM in chronological order
   and merge them into one narrative: preserve facts, resolve contradictions
   by keeping the latest state, restore the timeline, drop duplicates, and
   end with the current state. A previously consolidated cluster is updated
   incrementally — only the new fragments are folded into the existing
   narrative.
3. **Re-extract** — the merged narrative is written to Markdown and triggers
   re-extraction of atomic facts, keeping derived data consistent with it.
4. **Deprecate** — the replaced original Episodes and their atomic facts get
   `deprecated_by` pointing at the new narrative; cluster membership is
   updated; an audit record is written.

The result:

```
Cluster cl_xxx
└── ep_0042  "Andrew initially had no pets. He later adopted a dog named
              Toby, then another named Buddy. He currently has two dogs."
              (originals ep_0001 / ep_0002 / ep_0003 → deprecated_by = ep_0042)
```

The merged narrative is, structurally, just an ordinary Episode
(`parent_type="cluster"`) — transparent to retrieval, no search-pipeline
changes required. Default search excludes any memory carrying `deprecated_by`,
so a query like "how many pets does Andrew have" only hits the one complete
narrative.

## Storage layout

Memory uses Markdown as the single source of truth; SQLite and LanceDB are
derived indexes built automatically by the cascade daemon.

| Store | What it holds | Role |
|---|---|---|
| Markdown | Episode bodies, merged narratives, archive markers | Single source of truth; human-readable and editable |
| SQLite | Clusters and members, consolidation audit records | Structured state and queries |
| LanceDB | Vectors + BM25 index for Episodes / atomic facts | Search (rebuildable from Markdown) |

The **merged narrative** is written to the Episode daily-log Markdown; its
frontmatter marks that it came from a cluster:

```yaml
---
owner_id: u_andrew
timestamp: 2026-10-10T12:00:00+00:00
parent_type: cluster
parent_id: cl_a1b2c3d4e5f6
---
## Subject
Andrew's pet adoption journey

## Content
Andrew initially had no pets. He later adopted a dog named Toby, and then
adopted another dog named Buddy. He currently has two dogs.
```

The **replaced originals** are not deleted. Their file's frontmatter records
the archive mapping, and the index layer writes `deprecated_by`:

```yaml
---
# added to the original Episode file's frontmatter:
deprecated_entries:
  ep_20260810_0001: ep_20261010_0042
  ep_20260910_0002: ep_20261010_0042
---
```

> Soft-archive, not delete: even if SQLite / LanceDB are corrupted, as long
> as the Markdown is intact the indexes can be fully rebuilt — and every
> consolidation remains traceable back to its original content.

## Configuration

| Setting | Location | Default | Description |
|---|---|---|---|
| `reflect_episodes.enabled` | `<root>/ome.toml` | `false` | Set to `true` to enable (the only setting needed) |
| `reflect_episodes.cron` | `<root>/ome.toml` | `0 2 * * 1` | Run time, as a standard cron expression (`0 2 * * 1` = Mondays at 02:00); **optional**, omit to use the built-in default. Running more than weekly is not recommended |
| `clustering.threshold` | `<root>/everos.toml` | `0.65` | Clustering similarity threshold |
| `clustering.time_window_days` | `<root>/everos.toml` | `7.0` | Clustering time window (days) |

Two files, two scopes: `ome.toml` holds OME-strategy config (Reflection's
on/off switch and schedule); `everos.toml` holds general settings (clustering
and the like). Both live under the memory root, and you only write the keys
you want to override — everything else falls back to the shipped defaults in
`config/default.toml`, which you never edit by hand.

Changes to `ome.toml` hot-reload (~1–2s); no server restart needed.

> Setting `enabled` back to `false` stops the *next* run from starting; a run
> already in progress finishes normally.

## API reference

Reflection's normal mode of operation is the scheduled automatic run (see
[Quick start](#quick-start)). The endpoint below triggers a run **on
demand** — for testing, debugging, or when you want to consolidate
immediately. It is an auxiliary path, not the normal mode (there is no
dedicated CLI command).

### Triggering a run

```
POST /api/v1/ome/trigger
Content-Type: application/json
```

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | string | — | Strategy name; use `reflect_episodes` |
| `timeout` | float | 120.0 | Max seconds to wait for the run to finish |
| `force` | bool | false | When `true`, runs even if `enabled=false` |

**Response**:

```json
{ "status": "ok", "name": "reflect_episodes" }
```

`status` is `"ok"` (finished) or `"timeout"` (did not finish in time); an
unknown strategy name returns 404.

**Example — Python**:

```python
import httpx


async def trigger_reflection() -> str:
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        resp = await client.post(
            "/api/v1/ome/trigger",
            json={"name": "reflect_episodes", "timeout": 120, "force": True},
        )
        resp.raise_for_status()
        return resp.json()["status"]
```

**Example — curl**:

```bash
curl -X POST http://localhost:8000/api/v1/ome/trigger \
  -H "Content-Type: application/json" \
  -d '{"name": "reflect_episodes", "timeout": 120, "force": true}'
```

## Auditing & troubleshooting

> This section is **advanced**. For everyday use you don't need it — just
> read the Markdown (see [Quick start](#quick-start)). It's here for
> inspecting consolidation details or diagnosing problems.

Each run writes one `reflection_report` audit record, useful for reviewing
consolidation history:

| Field | Description |
|---|---|
| `cluster_id` | The cluster that was consolidated |
| `mode` | `init` (first merge) or `update` (incremental update) |
| `source_count` | Number of fragments merged |
| `merged_entry_id` | The merged-narrative Episode produced |
| `deprecated_fact_count` | Number of atomic facts archived alongside |
| `created_at` | Consolidation time |

```bash
sqlite3 <root>/.index/sqlite/system.db \
  "SELECT cluster_id, mode, source_count, merged_entry_id
   FROM reflection_report ORDER BY created_at DESC LIMIT 10;"
```

| Symptom | Likely cause |
|---|---|
| No consolidation records after triggering | No eligible clusters (a cluster needs ≥ 2 members) |
| Response `status: "timeout"` | Downstream re-extraction is slow; raise `timeout` and retry |
| Old fragments still appear in search | Index syncs asynchronously, usually 1–3s; wait and retry |
| 404 returned | Strategy name must be `reflect_episodes` |

## Design notes

Why Reflection is shaped the way it is:

- **Offline and scheduled.** Merging is a heavy, lossy LLM operation, so it
  runs off the request path — conversations stay fast — and a weekly cadence
  lets enough new fragments accumulate to be worth re-merging.
- **Soft-archive, never delete.** Originals stay in Markdown, so every
  consolidation is traceable and the indexes can always be rebuilt from the
  Markdown source of truth.
- **A merged narrative is just an Episode.** Reusing the Episode type means
  search and every downstream consumer keep working unchanged — Reflection
  introduces no new retrieval path.

## Limitations

- **Merging is lossy** — LLM consolidation may drop individual details. The
  original fragments are retained in storage and remain traceable, but are
  not in default search results.
- **Clustering is by similarity** — Reflection consolidates the output of
  similarity clustering; a single cluster is not guaranteed to be strictly
  one topic.
- **No one-click rollback yet** — originals are fully retained, but there is
  currently no endpoint to undo a specific consolidation.

## End-to-end walkthrough

The walkthrough triggers a run by hand to demonstrate the full flow; in a
real deployment, once enabled it runs automatically on schedule, so this step
isn't needed.

```bash
BASE=http://localhost:8000/api/v1

# 1. With Reflection enabled (set enabled = true in <root>/ome.toml),
#    trigger a run by hand here (for the demo; in production it runs on schedule)
curl -s -X POST "$BASE/ome/trigger" \
  -H "Content-Type: application/json" \
  -d '{"name": "reflect_episodes", "timeout": 120, "force": true}' \
  | jq .
# → { "status": "ok", "name": "reflect_episodes" }

# 2. Review the consolidation audit
sqlite3 <root>/.index/sqlite/system.db \
  "SELECT mode, source_count, merged_entry_id
   FROM reflection_report ORDER BY created_at DESC LIMIT 1;"
# → init|3|ep_20261010_0042

# 3. Verify via search: the hit is the merged narrative, not the old fragments
curl -s -X POST "$BASE/memory/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "how many pets does Andrew have", "top_k": 5}' \
  | jq '.data.episodes[0] | {subject, episode, session_id}'
# → session_id is null on a merged narrative (the aggregation-product marker);
#   episode holds the full narrative text
```

## See also

- [how-memory-works.md](how-memory-works.md) — Episodes and the memory extraction pipeline
- [storage_layout.md](storage_layout.md) — Markdown + SQLite + LanceDB stack
- [api.md](api.md) — full HTTP API reference
