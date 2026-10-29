# Architecture

> Companion: [.claude/rules/architecture.md](../.claude/rules/architecture.md) (auto-loaded coding rules)

## DDD layered architecture

- `entrypoints/` (Presentation) — cli + api + tui; translates protocols to
  service calls and results / errors back, nothing else.
- `service/` (Application) — `bootstrap` (assembly) + the `MemoryService` /
  `KnowledgeService` / `OpsService` facades.
- `memory/` (Domain) — models + ports + extract + search + custom_instructions
  + reflection + strategies + get + events.
- `infra/` (Adapters) — `persistence/` (markdown + sqlite + derived index)
  implementing `memory.ports`, and the cascade and OME engines.

Cross-cutting (used by all layers, depends on none):

- `component/` — injectable providers (LLM / embedding / rerank / parser /
  tokenizer / config loader).
- `core/` — runtime base (errors / observability / middleware / context /
  `utils.datetime`).
- `config/` — configuration data (Settings schema + default.toml +
  custom-instruction YAML).

## Dependency direction (single-direction, enforced)

```
entrypoints → service → memory ← infra
                         (ports)
```

The domain owns its storage contracts. `memory/ports` declares what the
domain needs from storage; `infra` implements those ports and depends on the
domain, never the other way round (dependency inversion).

| from → to | Allowed? |
|---|---|
| entrypoints → service | ✅ |
| entrypoints → memory / infra / component / everalgo | ❌ (must go through service; DTOs and errors come from `everos.service` / `everos.core`) |
| service → memory | ✅ |
| memory → infra | ❌ (go through `memory/ports`) |
| memory → service | ❌ |
| infra → memory.ports / models / events / mappers | ✅ (adapters implement the ports in domain types) |
| infra → any other memory module, service, entrypoints | ❌ |
| infra.ome → memory.ports | ✅ (`infra.ome` imports only `everos.memory.ports` — the types the domain declares for it); any other memory module ❌ |
| service.bootstrap → infra / component implementations | ✅ — the one module that builds adapters and providers and injects them (`service-usecases-forbidden`) |
| service use cases (`memory` / `knowledge` / `ops` / …) → infra / component implementations | ❌ — they receive what they need; protocols are imported for annotations only (backfill included: its collaborators arrive as `BackfillDeps`) |
| infra cross-subpackage (e.g. lancedb → markdown within persistence/) | ❌ (use service to orchestrate); one carve-out: every index backend reads the shared record declarations in `persistence/records/` (`lancedb → records` is whitelisted in import-linter) |
| any → component / core / config | ✅ (cross-cutting) |
| `infra/persistence/index → component.embedding` | ✅ — the only network-capable provider `infra` reaches for. Startup measures the embedding model's real output width before the index is opened for use; nothing else can answer that question, and asking it from a higher layer would mean every index user had to remember to ask. Elsewhere `infra` imports only `component` protocols (`EmbeddingProvider`, `Tokenizer`) for the collaborators it is handed. |

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
]

# plus forbidden contracts: entrypoints-forbidden, infra-inverted,
# memory-forbidden, core-forbidden, component-forbidden, mappers-pure,
# ome-ports-only, service-usecases-forbidden, no-bypass-facade
# (see pyproject.toml)
```

`make lint` also runs `scripts/check_singletons.py`: no module-level lazy
singleton slot (`_x: X | None = None`), no no-argument `get_*()` accessor and
no import-time `*_repo` instance in `src/`. Objects are built by the
application bootstrap and injected.

### Storage ports

`memory/ports` is the domain's declaration of what it needs from the outside
world, in domain types only — storage, and the event / strategy vocabulary
the scheduler runs:

| Module | Contents |
|---|---|
| `ports/query.py` | `Filter` (backend-neutral predicate tree) and the value objects `Scope`, `SessionKey`, `Page`, `PageResult`, `MemoryRef`, `Stored`, `SearchHit`, `ScoredKnowledgeTopic` |
| `ports/stores.py` | write side, the truth and system state: `EpisodeStore`, `AtomicFactStore`, `ForesightStore`, `ProfileStore`, `AgentCaseStore`, `AgentSkillStore`, `SessionBufferStore`, `MemcellStore`, `ClusterStore`, `MemorizeInboxStore`, `ReflectionReportStore`, and for the project-level knowledge base `KnowledgeStore` (markdown documents + taxonomy) and `KnowledgeCatalog` (the cascade's document / topic records) |
| `ports/indexes.py` | read side, the rebuildable derived indexes: `EpisodeIndex`, `AtomicFactIndex`, `AgentCaseIndex`, `AgentSkillIndex`, `ProfileIndex`, `KnowledgeTopicIndex` |
| `ports/events.py` | `Event` (frozen, JSON round-trippable, identified by `topic()`) and the scheduler ticks `CronTick` / `IdleTick` / `ManualTick` |
| `ports/strategy.py` | the triggers `Immediate` / `Cron` / `Idle`, the `Counter` gate, and `StrategySpec` — one strategy's name, function, trigger, `emits`, `applies_to`, gate, retries and enabled flag |
| `ports/event_bus.py` | `EventBus`, the port through which the service publishes events |

`memory/deps.py` bundles what a strategy may use into `MemoryDeps`: the LLM
and embedding providers (`None` when not configured) and the `Stores` /
`Indexes` containers of ports.

Adapters: `Markdown*Store` (`infra/persistence/markdown/stores.py`),
`Sqlite*Store` (`infra/persistence/sqlite/stores.py`), and one
backend-neutral `*IndexAdapter` per kind
(`infra/persistence/index/adapters.py`). Index writes and housekeeping are not
domain concerns: they go through `IndexMaintenance`
(`infra/persistence/index/maintenance.py`), used only by the cascade engine
and backfill. The domain never writes an index; it writes markdown and the
cascade projects it.

### Storage objects

Every storage handle is an object the bootstrap builds over one memory root
and passes down; nothing is reached through a module-level instance.

| Object | Owns | Opens |
|---|---|---|
| `SqliteDatabase` (`sqlite/sqlite_manager.py`) | the `system.db` engine and session factory | the file on first connection |
| `SqliteRepos` (`sqlite/repositories.py`) | one repository per system table, built by `SqliteRepos.over(database)` | — |
| `LanceDbConnection` / `MilvusConnection` | the engine connection and, for LanceDB, the table-handle cache | on first use |
| `LanceIndexBackend` / `MilvusIndexBackend` | the engine's `IndexRepositories` and its migrations / schema checks | — |
| `DerivedIndex` (`index/derived_index.py`) | the selected backend plus the engine-neutral checks: record width, primary keys, configuration verification | — |

`infra/persistence/factories.py` builds the engine-dependent pieces:
`build_derived_index(settings, root)` picks the backend once,
`build_stores(root, sqlite)` and `build_indexes(index)` wrap the repositories
in the domain ports. Why objects rather than module instances: a test or a
second root gets its own storage by building another set, a repository's
database is visible at its construction, and closing is the owner's call
(`Application.aclose`, `OpsService.index_session`), not a process-wide reset.

## Storage three-piece set

```
┌────────────────────────────────────────────────────────────────┐
│             md-first storage stack                              │
└────────────────────────────────────────────────────────────────┘

   ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐
   │   Markdown   │   │   SQLite     │   │ Derived index   │
   │  (truth)     │   │  (state)     │   │ LanceDB/Milvus  │
   ├──────────────┤   ├──────────────┤   ├─────────────────┤
   │ entries +    │   │ change queue │   │ vector ANN      │
   │ frontmatter  │   │ + state/LSN  │   │ BM25 (Tantivy)  │
   │ Git friendly │   │ buffer /     │   │ scalar filter   │
   │ Obsidian OK  │   │   audit      │   │ multi-modal     │
   └──────────────┘   └──────────────┘   └─────────────────┘
          │                  │                    │
          ▼                  ▼                    ▼
    memory-root/         .index/sqlite/      .index/<backend>/
   (truth source)       (system data)       (rebuildable)
```

## Write path

```
External message
       │
       ▼
1. MemoryService.memorize     (entrypoint of write path)
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
4a. SQLite   4b. infra.cascade  (async daemon)
    audit        watches md → diff entries → index sync
```

**Key guarantee**: md write is strongly consistent (fsync). The derived
index is eventually consistent. Index backend unavailability does not block
response — changes buffer in the SQLite `md_change_state` queue, replayed on
recovery.

## Read path

```
User query
   │
   ▼
1. MemoryService.search
   │
   ▼
2. memory.search (hybrid)     BM25 + vector ANN + scalar filter
                              through the configured index backend
   │
   ▼
3. (optional) read md         original markdown for context
   │
   ▼
   Return
```

## Key components

### `service/`

The application layer has two jobs, kept in separate modules:

- **Assembly** — `service/bootstrap.py`. `build_application(settings)` builds
  every provider (`build_providers`), storage adapter, the OME engine with all
  strategies registered, the event bus, the cascade orchestrator and the domain
  pipelines and managers, and returns an `Application`: three facades plus a
  `Capabilities` snapshot of which optional providers were configured. Building
  opens nothing; `Application.start()` brings up the system DB, the index, the
  cascade, OME and the `async_mode` queue in that order (the queue publishes
  into OME, so it starts last and stops first), and `aclose()` unwinds what was
  started, then closes the storage connections whether or not `start()` ran.
  `aclose()` is idempotent and `async with build_application(settings) as
  app:` closes on exit. Tracing and metrics are configured once here
  (`configure_tracing` / `configure_metrics`).
- **Use cases** — one facade per kind of caller, one method per use case:
  `MemoryService` (memorize / flush / search / get, plus the `async_mode`
  queue), `KnowledgeService` (document CRUD and knowledge search) and
  `OpsService` (cascade status / sync / fix / rebuild / quiesce / backfill, OME
  trigger, health signals, task readers). They receive their collaborators;
  none builds or looks one up. The rules an entry point would otherwise carry
  live here too: which task statuses exist, when an `/add` is queued instead
  of run inline, how an upload is size-checked and parsed, what a failed
  health probe reads as, when a backfill run stops. A facade that outgrows
  one module becomes a package with the facade class in `__init__` and the
  use-case steps in private modules (`service/memory/`, `service/ops/` —
  backfill's three phases are `service/ops/_backfill_*`).

Entry points hold the one `Application`: the HTTP lifespan builds and starts it
(`app.state.application`), routes call its facades, and a CLI command builds it
without starting it (`async with`) and opens storage per command through
`OpsService.index_session`. Why one assembly point: every object's
construction is visible in one file, a test swaps any provider by passing
`Providers`, and no module can reach a half-configured singleton.

### `component/`

Each injectable capability exposes three things: a protocol, its
implementations, and a factory — no process-wide accessor.

| Package | Protocol | Implementations | Factory |
|---|---|---|---|
| `llm` | `everalgo.llm.LLMClient` | `OpenAIProvider` | `build_llm_provider(resolve_*_config(settings), observer=observer_for(settings))` |
| `embedding` | `EmbeddingProvider` | `OpenAIEmbeddingProvider` | `build_embedding_provider(settings.embedding)` |
| `rerank` | `RerankProvider` | DeepInfra / vLLM / DashScope providers | `build_rerank_provider(settings.rerank)` |
| `parser` | everalgo parser entry | — | `aparse_file(raw, llm=...)` takes the multimodal LLM from its caller |

A factory raises when its section is missing or wrong
(`LLMNotConfiguredError` / `ValueError`); the bootstrap's `build_providers`
turns that into `None` with a warning, and the `None` is the whole
"not configured" signal: search refuses with a 422, cascade writes rows
without vectors, knowledge uploads refuse, and the server's `start()` refuses
to run without an extraction LLM. Why the bootstrap decides and the factory
does not: "absent" is an application decision, while "misconfigured" must
stay visible to whoever builds the provider.

### `memory/extract/`

```
extract/
├── ingest/       Standardized message intake + multi-modal parser dispatch
├── parser/       Input parsing (format normalization, message preprocessing)
├── boundary/     BoundaryDetection: session buffer -> memcells (+ ledger, quarantine)
├── pipeline/     Per-cell extraction (Episode md + the events that announce it)
├── llm_retry.py  The one host-side retry of everalgo's LLMOutputError
└── evolution/    Async memory evolution (event/counter/cron triggers)
```

`BoundaryDetection` is built by the bootstrap and injected into
`MemoryService`; it owns the session buffer, the `memcell` ledger and the
quarantine of a session the model keeps failing to segment. Why in the
domain: deciding where an episode ends is a memory rule, and keeping it out
of the service lets it be tested against in-memory stores and a scripted
model. Why one retry module: everalgo operators never retry (an unusable
reply raises `LLMOutputError`), so the host owns the policy, and one place
keeps the attempt count from multiplying across layers.

### `infra/cascade/`

Daemon that watches markdown changes and syncs to the configured derived
index backend:

- inotify / FSEvents file watcher (cross-platform via `watchdog`)
- 500ms debounce
- Entry-level diff (added / changed / removed)
- Per-entry index upsert / delete (text + vector columns update together)
- LSN-based crash recovery via the SQLite `md_change_state` queue
- Handlers for all eight business kinds: episode, atomic_fact, foresight,
  user_profile, agent_case, agent_skill, knowledge_document, knowledge_topic

### `memory/custom_instructions/`

Per-operator requirements appended to everalgo's bundled rules
(`config/custom_instructions/<operator>.yaml`, all disabled by default).
`CustomInstructions.load(operator)` returns the text or `None`; the operator
receives it as `custom_instructions=` and can neither lose a rule nor change
its output shape. Operators: boundary detection, episode extraction, profile
extraction, reflection. See [custom_instructions.md](custom_instructions.md).

### `memory/reflection/`

Offline memory self-improvement (`orchestrator.py`): Select → Merge →
Re-extract → Deprecate, merging fragmented episodes within a cluster into a
single coherent narrative. `ReflectionOrchestrator` selects and merges, driven
by the `reflect_episodes` OME strategy (cron, disabled by default), and returns
the merged episode with a `ReflectionContext`; once `extract_atomic_facts` has
re-extracted its facts, `finish_reflection` hands that context to
`ReflectionFinisher`, which deprecates the replaced members.

### `memory/strategies/`

OME strategy implementations — one file per strategy. A strategy is an async
function `fn(event, deps: MemoryDeps) -> list[Event]`: it works only through
`deps` and returns its follow-up events instead of publishing them. Each file
declares its `StrategySpec` next to the function; `STRATEGY_SPECS` collects
them. Strategy names are stable — run records and `ome.toml` overrides key on
them.

- `extract_atomic_facts` / `extract_foresight` / `extract_user_profile` — user pipeline
- `extract_agent_case` / `extract_agent_skill` — agent pipeline
- `reflect_episodes` / `finish_reflection` — offline episode consolidation
  (cron merge, then deprecation once the merged episode's facts exist)
- `trigger_profile_clustering` / `trigger_skill_clustering` — clustering triggers

Who publishes: the domain never does. The memorize service publishes each
pipeline cell's events on the `EventBus` (`OmeEventBus` over the engine), and
`infra/ome/adapter.py` (`to_ome_strategy`) binds the deps into a spec, checks
every returned event against its `emits` and publishes them for the run. Why
this shape: a strategy is testable as a plain function with in-memory doubles
(`tests/fakes`), and the emit contract is enforced in one place instead of in
every strategy body.

### `core/observability/`

Three-piece observability:

- `metrics/` — Prometheus counter / gauge / histogram over the registry
  `configure_metrics` settles
- `logging/` — structlog with context processor (trace_id propagation);
  `bound_log_context` adds fields to every event in a block
- `tracing/` — OpenTelemetry tracer + span helpers; the recall-score sink is
  installed by `init_score_sink` and is a no-op until then

These are the facades: business layers (`entrypoints` / `service` / `memory` /
`infra`) never import `structlog`, `logging` or `opentelemetry` directly
(`no-bypass-facade`), so a backend is swapped where it is configured — once,
in the bootstrap — not where it is called.

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

`everalgo` is a set of PyPI-published packages (`everalgo-user-memory`,
`everalgo-agent-memory`, `everalgo-rank`, `everalgo-knowledge`, plus the
optional `everalgo-parser` extra), imported under the `everalgo` namespace,
holding **only memory extraction algorithms**:

- `everalgo.parser` — multi-modal parsing (optional `[multimodal]` extra)
- `everalgo.user_memory` — ConvMemCell / Episode / Foresight / AtomicFact / Profile extractors
- `everalgo.agent_memory` — AgentMemCell / Case / Skill extractors
- `everalgo.boundary` — boundary detection (conversation -> MemCells)
- `everalgo.rank` — fusion + rerank
- `everalgo.knowledge` — KnowledgeExtractor (document parse + topic extraction)

everalgo is:

- **Stateless** — pure functions, no class hierarchy
- **No I/O** — does not touch md files, derived indexes, or SQLite
- **Owns its prompts** — every operator ships its rules and its strict reply schema together; EverOS can only append requirements (`custom_instructions=`), never replace a prompt
- **Never retries** — an unusable reply raises `LLMOutputError`; the host decides (`memory/extract/llm_retry.py`)

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
