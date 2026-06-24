# Architecture

> Companion: [.claude/rules/architecture.md](../.claude/rules/architecture.md) (auto-loaded coding rules)

## DDD layered architecture

```
┌──────────────────────────────────────────────────────┐
│  entrypoints/  (Presentation)                         │
│    cli + api                                          │
├──────────────────────────────────────────────────────┤
│  service/      (Application — Use Case orchestration) │
│    memorize / search / get / knowledge                │
├──────────────────────────────────────────────────────┤
│  memory/       (Domain — Business core)               │
│    models + extract + search + cascade + prompt_slots │
├──────────────────────────────────────────────────────┤
│  infra/persistence  (Storage adapters; infra/ may host other adapter types)    │
│    markdown + sqlite + lancedb                        │
└──────────────────────────────────────────────────────┘

Cross-cutting (used by all layers, depends on none):
  component/  ← Injectable providers (LLM / Embedding / parser / config / utils)
  core/       ← Runtime base (observability / lifespan / context / errors)
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
1. service.search
   │
   ▼
2. memory.search (hybrid)     single LanceDB query =
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
~/.everos/                                  # memory root (default; EVEROS_ROOT)
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

`everalgo` is a set of PyPI-published packages (`everalgo-core`,
`everalgo-boundary`, `everalgo-user-memory`, `everalgo-agent-memory`,
`everalgo-rank`, plus the optional `everalgo-parser` extra), imported under
the `everalgo` namespace, holding **only memory extraction algorithms**:

- `everalgo.parser` — multi-modal parsing (optional `[multimodal]` extra)
- `everalgo.user_memory` — ConvMemCell / Episode / Foresight / AtomicFact / Profile extractors
- `everalgo.agent_memory` — AgentMemCell / Case / Skill extractors
- `everalgo.boundary` / `everalgo.rank` — boundary detection / fusion + rerank

everalgo is:

- **Stateless** — pure functions, no class hierarchy
- **No I/O** — does not touch md files / LanceDB / SQLite
- **No prompts inline** — receives `PromptSlot` parameter, project supplies defaults

This boundary lets everalgo be reused across product forms (this open-source build, EverOS Cloud, OpenClaw plugins, etc.).

## Error handling architecture

### Exception hierarchy

All application exceptions derive from `AppError` (`core/errors.py`),
split into four branches by nature:

```
AppError
├── DomainError                    (client-side / business-rule violations)
│   ├── NotFoundError              → 404 NOT_FOUND
│   │   ├── DocumentNotFoundError
│   │   └── TopicNotFoundError
│   ├── ConflictError              → 409 CONFLICT
│   │   └── DuplicateDocumentError
│   ├── InvalidInputError          → 422 INVALID_INPUT
│   │   ├── ExtractionEmptyError   → 422 EXTRACTION_EMPTY
│   │   └── FilterError
│   ├── PathTraversalError         → 400 BAD_REQUEST
│   └── UnsupportedModalityError   → 415 UNSUPPORTED_FORMAT
├── InfrastructureError            (transient, retryable)     → 503
│   ├── StorageError
│   ├── VectorStoreError
│   └── ExternalServiceError
│       ├── LLMServiceError
│       ├── EmbeddingServiceError
│       └── RerankServiceError
├── CapabilityError                (permanent, not retryable) → 503
│   └── MultimodalNotEnabledError
└── ConfigurationError             (misconfiguration)         → 500
```

### Error propagation strategy

Exceptions are raised at the layer where the error is detected and
propagate naturally — **service and route layers do not catch-and-wrap**.
The entrypoints layer registers per-type exception handlers
(`entrypoints/api/exception_handlers.py`) via Starlette's MRO dispatch.
Each handler converts the exception into a canonical error envelope with
an `ErrorCode` enum value and the appropriate HTTP status code.

### Boundary translation

Third-party exception types are translated at the component boundary to
prevent external types from leaking into upper layers:

- `everalgo.llm.LLMError` → `LLMServiceError` at `component/parser/_core.py`
- Embedding / rerank provider errors → `EmbeddingServiceError` / `RerankServiceError` at their respective protocol modules

## Further reading

- [docs/overview.md](overview.md) — vision and scope
- [docs/engineering.md](engineering.md) — engineering tooling and CI / CD
- [.claude/rules/architecture.md](../.claude/rules/architecture.md) — short-form rules for Claude Code
