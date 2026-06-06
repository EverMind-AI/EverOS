# Architecture

> Companion: [.claude/rules/architecture.md](../.claude/rules/architecture.md) (auto-loaded coding rules)

## DDD layered architecture

```
┌──────────────────────────────────────────────────────┐
│  entrypoints/  (Presentation)                         │
│    cli + api                                          │
├──────────────────────────────────────────────────────┤
│  service/      (Application — Use Case orchestration) │
│    memorize / retrieve / evolve / manage              │
├──────────────────────────────────────────────────────┤
│  memory/       (Domain — Business core)               │
│    models + extract + search + cascade + prompt_slots │
├──────────────────────────────────────────────────────┤
│  infra/persistence  (Storage adapters; infra/ may host other adapter types)    │
│    markdown + sqlite + lancedb                        │
└──────────────────────────────────────────────────────┘

Cross-cutting (used by all layers, depends on none):
  component/  ← Injectable providers (LLM / Embedding / config / utils)
  core/       ← Runtime base (observability / lifespan / context)
  config/     ← Configuration data (Settings schema + default.toml)
```

## Dependency direction (single-direction, enforced)

```
entrypoints → service → memory → infra
```

| from → to | Allowed? |
|---|---|
| entrypoints → service | ✅ |
| entrypoints → memory / infra | ❌ (must go through service) |
| service → memory | ✅ |
| memory → infra | ✅ |
| memory → service | ❌ |
| infra → memory | ❌ |
| infra cross-subpackage (e.g. lancedb → markdown within persistence/) | ❌ (use service to orchestrate) |
| any → component / core / config | ✅ (cross-cutting) |

Enforced via `import-linter` in CI:

```toml
[tool.importlinter]
root_packages = ["everos"]

[[tool.importlinter.contracts]]
name = "Layered architecture"
type = "layers"
layers = [
    "everos.entrypoints",
    "everos.service",
    "everos.memory",
    "everos.infra",
]
```

## Storage three-piece set

```
┌────────────────────────────────────────────────────────────────┐
│             md-first storage stack                              │
└────────────────────────────────────────────────────────────────┘

   ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐
   │   Markdown   │   │   SQLite     │   │    LanceDB      │
   │  (truth)     │   │  (state)     │   │  (index)        │
   ├──────────────┤   ├──────────────┤   ├─────────────────┤
   │ entries +    │   │ change queue │   │ vector ANN      │
   │ frontmatter  │   │ + state/LSN  │   │ BM25 (Tantivy)  │
   │ Git friendly │   │ buffer /     │   │ scalar filter   │
   │ Obsidian OK  │   │   audit      │   │ multi-modal     │
   └──────────────┘   └──────────────┘   └─────────────────┘
          │                  │                    │
          ▼                  ▼                    ▼
    memory-root/         .index/sqlite/      .index/lancedb/
   (truth source)       (system data)       (rebuildable)
```

## Write path

```
External message
       │
       ▼
1. service.memorize           (entrypoint of write path)
       │
       ▼
2. memory.extract.pipeline    (calls everalgo)
       │
       ▼
3. infra.persistence.markdown.write       (atomic: tmp + fsync + rename)
       │  ✅ md write success → return immediately
       │
   ┌───┴────┐
   │        │
   ▼        ▼
4a. SQLite   4b. memory.cascade  (async daemon)
    audit        watches md → diff entries → LanceDB sync
```

**Key guarantee**: md write is strongly consistent (fsync). LanceDB is eventually consistent. LanceDB unavailability does not block response — changes buffer in the SQLite `md_change_state` queue, replayed on recovery.

## Read path

```
User query
   │
   ▼
1. service.retrieve
   │
   ▼
2. memory.search.hybrid       single LanceDB query =
                                BM25 + vector ANN + scalar filter
   │
   ▼
3. (optional) read md         original markdown for context
   │
   ▼
   Return
```

## Key components

### `memory/extract/`

```
extract/
├── ingest/      Standardized message intake + multi-modal parser dispatch
├── pipeline/    Main extraction pipeline (calls everalgo + dual-track split + writes store)
└── evolution/   Async memory evolution (event/counter/cron triggers)
```

### `memory/cascade/`

Daemon that watches markdown changes and syncs to LanceDB:

- inotify / FSEvents file watcher (cross-platform via `watchdog`)
- 500ms debounce
- Entry-level diff (added / changed / removed)
- LanceDB single-transaction update (text + vector columns atomic)
- LSN-based crash recovery via the SQLite `md_change_state` queue

### `memory/prompt_slots/`

Three-layer prompt overlay:

```
config/prompt_slots/*.yaml          (Layer 1: defaults, ships with package)
       ↓
~/.everos/prompt_slots/*.yaml       (Layer 2: app-level override)
       ↓
runtime override                    (Layer 3: per-call override)
```

everalgo receives PromptSlot as parameter — no hardcoded prompts in algorithm code.

### `core/observability/`

Three-piece observability:

- `metrics/` — Prometheus counter / gauge / histogram + global registry
- `logging/` — structlog with context processor (trace_id propagation)
- `tracing/` — OpenTelemetry tracer + span helpers

## Markdown layout

```
~/.everos/                                  # memory root (default; EVEROS_MEMORY__ROOT)
└── <app_id>/<project_id>/                  # scope ("default" → default_app/default_project)
    ├── users/<user_id>/
    │   ├── user.md                                     # profile (single-file rewrite)
    │   ├── episodes/episode-<YYYY-MM-DD>.md            # daily-log append
    │   ├── .atomic_facts/atomic_fact-<YYYY-MM-DD>.md   # hidden, framework-derived
    │   └── .foresights/foresight-<YYYY-MM-DD>.md       # hidden, framework-derived
    ├── agents/<agent_id>/
    │   ├── .cases/agent_case-<YYYY-MM-DD>.md           # hidden, framework-derived
    │   └── skills/skill_<name>/SKILL.md                # named-dir
    └── knowledge/                                      # global shared knowledge
```

System-managed entries (`.index/`, `.tmp/`) and `ome.toml` live directly
under the memory root.
Full tree + frontmatter chassis: [storage_layout.md](storage_layout.md) and
[how-memory-works.md](how-memory-works.md). Frontmatter has 4-tier field
protection (L1 read-only / L2 system / L3 business / L4 user).

## everalgo boundary

[`everalgo`](https://github.com/EverMind-AI/EverAlgo) is a separate Python library (published as the `everalgo-*` PyPI packages) holding **only memory extraction algorithms**:

- `everalgo.parser` — multi-modal parsing
- `everalgo.user_memory` — ConvMemCell / Episode / Foresight / AtomicFact / Profile extractors
- `everalgo.agent_memory` — AgentMemCell / Case / Skill extractors
- `everalgo.knowledge` — file-to-knowledge

everalgo is:

- **Stateless** — pure functions, no class hierarchy
- **No I/O** — does not touch md files / LanceDB / SQLite
- **No prompts inline** — receives `PromptSlot` parameter, project supplies defaults

This boundary lets everalgo be reused across product forms (this open-source build, EverMind Cloud, OpenClaw plugins, etc.).

## Further reading

- [docs/overview.md](overview.md) — vision and scope
- [docs/engineering.md](engineering.md) — engineering tooling and CI / CD
- [.claude/rules/architecture.md](../.claude/rules/architecture.md) — short-form rules for Claude Code
