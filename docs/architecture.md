# Architecture

> Companion: [.claude/rules/architecture.md](../.claude/rules/architecture.md) (auto-loaded coding rules)

This document defines responsibility boundaries and describes the current
mechanisms that implement them. Directory and type examples are not mandatory
layouts. Passing import checks does not establish delivery, recovery,
idempotency, or upgrade guarantees; those require separate evidence.

## DDD layered architecture

- `entrypoints/` (Presentation) — cli + api + tui; translates protocols to
  service calls and results / errors back, nothing else.
- `service/` (Application) — `bootstrap` (assembly) + the `MemoryService` /
  `KnowledgeService` / `OpsService` facades.
- `memory/` (Domain) — models + ports + extract + search + custom_instructions
  + reflection + strategies + get + events.
- `infra/` (Adapters) — `persistence/` (markdown + sqlite + derived index)
  implementing `memory.ports`, and the cascade and OME engines.

Cross-cutting capabilities have no dependency on business layers. Access
depends on the capability: shared data and utilities differ from external
services or storage access.

- `component/` — injectable providers (LLM / embedding / rerank / parser /
  tokenizer / config loader).
- `core/` — runtime base (errors / observability / context / `utils.datetime`);
  it knows no web framework.
- `config/` — configuration data (Settings schema + default.toml +
  custom-instruction YAML).

## Dependency direction and responsibility

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
| entrypoints → business operations | Through service; direct imports of memory / infra / component / everalgo, including data types, are prohibited at entrypoints. Use service request/response contracts and permitted core/config entry APIs. Current import checks cover this package boundary. |
| service → memory | ✅ |
| memory → infra | ❌ (go through `memory/ports`) |
| memory → service | ❌ |
| infra → memory.ports / models / events / pure conversions / dependency declarations | ✅ when implementing a declared contract; a conversion must not invoke business workflows or perform hidden I/O. Shared data is not a dependency on a business service. |
| infra → any other memory module, service, entrypoints | ❌ |
| infra.ome → scheduling/event contracts | ✅ — the engine consumes declared contracts without interpreting memory business rules. The current `ome-ports-only` guard restricts direct domain imports to `memory.ports`. |
| service.bootstrap → infra / component implementations | ✅ — selects application-level implementations, wires collaborators, and assigns lifecycle ownership. |
| service use cases → external capabilities | Receive collaborators instead of locating or choosing application-level implementations. Operational protocols and result data are reviewed by responsibility, not rejected solely because their package is infra. |
| business logic → concrete storage rows / layout | ❌ — adapters and projections own row conversion, file grouping, and backend-specific updates. |
| infra cross-subpackage | Internal storage collaboration and shared record declarations are allowed where needed; infra must not take over application use-case orchestration. |
| layers → core / config helpers and shared data | ✅ without introducing reverse business dependencies. Configuration that determines business behavior should be explicit; a formatting helper is not a service locator. |
| memory / infra → component capabilities | Distinguish pure helpers, fixed algorithm integration, and replaceable I/O capabilities. The latter need explicit configuration and ownership; a direct import alone does not establish which category applies. |

The embedding provenance check currently constructs a short-lived provider
with a separate probe budget and untruncated output width. That is an internal
infrastructure collaborator, not a global service lookup. If application-level
provider substitution must also replace this probe, expose the probe capability
at assembly; do not require every internal helper to become an injected service.

`import-linter` enforces a subset of these boundaries in CI:

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

The current configuration excludes `TYPE_CHECKING` imports and uses explicit
forbidden-module lists. An annotation import may describe shared data, a
protocol, or a concrete implementation; its presence is not itself a defect.
Conversely, a green check does not prove that a mapper is free of I/O or that
runtime dependencies are explicit. Changes to guards need both rejected and
allowed examples that reflect the responsibility being protected.

`make lint` also runs `scripts/check_singletons.py`, which detects selected lazy
singleton and import-time repository patterns. Review must still reject hidden
business service lookup. Configured process-wide logging, metrics, tracing, and
request context are permitted runtime facilities, not business service locators.

### Storage ports

`memory/ports` is the domain's declaration of what it needs from the outside
world, in domain types only — storage, and the event / strategy vocabulary
the scheduler runs. A port never exposes a backend row class, a markdown
parse structure or a backend handle, and never asks business logic to
interpret storage layout.

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
| `DerivedIndex` (`index/derived_index.py`) | the selected backend plus the engine-neutral checks: primary keys, configuration verification | — |

`infra/persistence/factories.py` builds the engine-dependent pieces:
`build_derived_index(settings, root)` picks the backend once,
`build_stores(root, sqlite)` and `build_indexes(index)` wrap the repositories
in the domain ports. Why objects rather than module instances: a test or a
second root gets its own storage by building another set, a repository's
database is visible at its construction, and closing is the owner's call
(`Application.aclose`, `OpsService.index_session`), not a process-wide reset.

The table declarations every index backend reads (`TABLE_NAME`,
`BM25_FIELDS`, `PRIMARY_KEY`, the columns, which columns are embeddings) live
in `infra/persistence/records/` as plain pydantic models (`IndexRecord`). An
embedding column is marked `DenseVector` and carries no width. The vector
width is deployment configuration: `build_derived_index` hands
`settings.embedding.dim` to the backend, and `index.schema_for(model, dim=...)`
turns a record into the neutral `IndexSchema` at that width. Each backend
derives its physical schema from `IndexSchema` alone -- LanceDB its Arrow
schema (`lancedb/_base/arrow.py`), Milvus its collection layout
(`milvus/_layout.py`) -- and builds its own indexes in its own package
(LanceDB's FTS and IVF upkeep is `lancedb/_base/indexes.py`).

Why: with the width baked into the record classes, they had to read the
configuration while being imported, a second configuration reader that could
disagree with `Settings` and needed a startup check to catch it. With the
records neutral there is one reader and one path for the width
(`Settings` -> bootstrap -> backend -> repositories), a third backend needs
only `IndexSchema`, and the `records-neutral` import contract keeps
`lancedb`, `pymilvus` and `pyarrow` out of `records/`.

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

The following is a responsibility sketch, not a transaction protocol:

```
External message
       │
       ▼
1. MemoryService.add          (entrypoint of write path)
       │
       ▼
2. memory.extract.pipeline    (calls everalgo)
       │
       ▼
3. infra.persistence.markdown.write       (file persistence)
       │  service coordinates results and follow-up events
       │
   ┌───┴────┐
   │        │
   ▼        ▼
4a. SQLite   4b. infra.cascade  (async daemon)
    audit        watches md → diff entries → index sync
```

Markdown is the source for derived index content. Cascade uses a persisted
change queue and reconciliation to synchronize the index asynchronously.
A durable file write does not atomically commit event publication, SQLite
state, or every downstream result. Neither this flow nor eventual consistency
alone proves that every interrupted operation will recover.

### Completion, recovery, and retry responsibilities

| Owner | Required responsibility |
|---|---|
| service | Distinguish request acceptance, business persistence, and derived completion; coordinate the handoff between them through declared contracts. |
| memory | Define business identity, permissible duplicates, completion with no output, and the invariants that partial execution must preserve. |
| infra | Implement persistence, scheduling, recovery, and deduplication mechanisms; cascade synchronizes and repairs derived indexes. Engines do not invent business duplicate semantics. |
| component | Expose classified failures and bounded call behavior that can be accounted for in the caller's budget. |
| entrypoints | Report the service's actual completion and failure semantics without upgrading their guarantees. |

These are acceptance obligations, not a claim that every current path has
passed them. In the `0d3b502` implementation, write-to-publication handoff,
interrupted OME recovery, repeated side effects, and combined retry policy
remain areas requiring retained, reproducible fault tests and explicit
contracts. See [API completion behavior](api.md) for the public interface.

Retries must be evaluated along the actual call path: SDK transport attempts,
provider reply validation, host operator retries, OME execution, and queue
replay do not apply to every failure in the same way. OME already has backoff
and a per-attempt timeout; neither is an end-to-end deadline. Define the owner
of the combined call/time budget, cancellation behavior, and the budget after
recovery. A terminal result for one provider call is neither a blanket ban on
future execution under changed conditions nor permission for unconditional
outer retries. Delivery and duplicate guarantees must be selected explicitly;
this architecture does not prescribe exactly-once delivery, outbox, or UoW.

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

- **Assembly** — `service/bootstrap.py`. `build_application(settings)` selects
  application providers (`build_providers`), storage adapters, the OME engine with
  strategies registered, the event bus, the cascade orchestrator and the domain
  pipelines and managers, and returns an `Application`: three facades plus a
  `Capabilities` snapshot of which optional providers were configured.
  `Application.start()` brings up the system DB, the index, the
  cascade, OME and the `async_mode` queue in that order (the queue publishes
  into OME, so it starts last and stops first), and `aclose()` unwinds what was
  started, then closes the storage connections whether or not `start()` ran.
  `aclose()` is idempotent and `async with build_application(settings) as
  app:` closes on exit. Process observability is configured through
  `bootstrap.configure_process`; internal helpers can be constructed by their
  owning component without becoming separate application dependencies.
- **Use cases** — one facade per kind of caller, one method per use case:
  `MemoryService` (memorize / flush / search / get, plus the `async_mode`
  queue), `KnowledgeService` (document CRUD and knowledge search) and
  `OpsService` (cascade status / sync / fix / rebuild / quiesce / backfill, OME
  trigger, health signals, task readers). They receive external collaborators;
  they do not select application providers or look up hidden business services.
  The rules an entry point would otherwise carry
  live here too: which task statuses exist, when an `/add` is queued instead
  of run inline, how an upload is size-checked and parsed, what a failed
  health probe reads as, when a backfill run stops. A facade that outgrows
  one module becomes a package with the facade class in `__init__` and the
  use-case steps in private modules (`service/memory/`, `service/ops/`,
  `service/knowledge/` — backfill's three phases are `service/ops/backfill/`).

Entry points hold the one `Application`: the HTTP lifespan builds and starts it
(`app.state.application`), routes call its facades, and a CLI command builds it
without starting it (`async with`) and opens storage per command through
`OpsService.index_session`. Application-level selection and lifecycle ownership
are visible at assembly; tests can replace those providers through `Providers`.
This does not imply that every internal helper or independently constructed
probe is replaced by that argument.

Operational use cases may consume management protocols and result data. They
should coordinate confirmation, phases, progress, and outcomes, while storage
adapters/projections own concrete row conversion and updates. In particular,
normal projection and backfill should share embedding-source and column rules;
duplicating those rules in both paths is a maintenance risk, independent of
where a protocol's Python type is declared.

`infra.persistence.index.VectorProjection` and `VECTOR_PROJECTIONS` provide
the shared embedding declarations for normal cascade projection and vector
backfill. The row-level backfill -- scanning rows with a missing vector side,
embedding them in batches with row-by-row fallback, writing them back and
compacting each table -- lives next to those declarations in
`infra.persistence.index.vector_backfill` (`ProjectionVectorBackfill`). The
bootstrap builds it and hands it to the vectors phase through `BackfillDeps`;
the phase keeps preflight, estimate, confirmation and the summary, and is typed
against the `VectorBackfill` protocol under `TYPE_CHECKING` only. No
import-linter exception is needed, and a guard test keeps it that way.

Reflection selects ordered retirement references and retains business report
counts. Markdown stores group those references by file and apply replacement
markers in file order. This boundary does not add cross-file atomicity or
change partial-write failure behavior.

### `entrypoints/`

`entrypoints/api/` splits protocol translation into six parts:

- `app.py` — assembles FastAPI: the lifespan builds the `Application` through
  `bootstrap`, then handlers, middleware and routers are registered (the
  business routers under both `/api/v1` and `/api/v2`).
- `routes/` — one module per resource; a route validates the DTO, calls a
  facade and wraps the envelope. No business judgement.
- `schemas/` — request / response models (pydantic), the public wire
  contract; `docs/openapi.json` is generated from them (`make openapi`).
- `dependencies.py` — `ApplicationDep` / `RequestIdDep` (`Depends` aliases);
  tests swap the application through `app.dependency_overrides`.
- `exception_handlers.py` — one table from exception class to status code /
  `ErrorCode`.
- `middleware/` — the HTTP middleware (CORS defaults, request id, Prometheus
  metrics, profiling); it reaches logging and metrics through
  `core.observability`.

`entrypoints/cli/`: `main.py` registers the command groups, `commands/` holds
one module per group, and `_process.py` sets the log level a command runs at.

### `component/`

Replaceable external capabilities expose a contract and implementations, with
application selection at assembly. A factory is useful where construction
varies; pure helpers and fixed integrations do not each need a new protocol.
Do not add process-wide accessors for business services or providers.

| Package | Protocol | Implementations | Factory |
|---|---|---|---|
| `llm` | `everalgo.llm.LLMClient` | `OpenAIProvider` | `build_llm_provider(resolve_*_config(settings), observer=observer_for(settings))` |
| `embedding` | `EmbeddingProvider` | `OpenAIEmbeddingProvider` | `build_embedding_provider(settings.embedding)` |
| `rerank` | `RerankProvider` | DeepInfra / vLLM / DashScope providers | `build_rerank_provider(settings.rerank)` |
| `parser` | everalgo parser entry | — | `aparse_file(raw, llm=...)` takes the multimodal LLM from its caller |

Calling EverAlgo's parser from this component is normal integration. Reusing
`RawFile` or `ParsedContent` does not introduce storage dependencies. Separate
file acquisition and access policy from content conversion where independent
testing or substitution is needed; inject a callable when sufficient instead
of requiring a class for every function. Stable behavior settings should be
passed explicitly; deliberate live configuration needs an explicit source and
compatibility rules for existing environment overrides.

A factory raises when its section is missing or wrong
(`LLMNotConfiguredError` / `ValueError`). The extraction LLM is required:
`build_providers` lets its `LLMNotConfiguredError` propagate, so neither the
server nor a CLI command that builds the application gets past assembly
without `[llm]`, and nothing downstream of the bootstrap types the LLM (or
the decider, which falls back to `[llm]`) as optional. For embedding, rerank
and multimodal the bootstrap turns the error into `None` with a warning, and
the `None` is the whole "not configured" signal: search refuses with a 422,
cascade writes rows without vectors, knowledge uploads refuse, and parsing a
non-text upload refuses without a multimodal LLM. Why the bootstrap decides
and the factory does not: "absent" is an application decision, while
"misconfigured" must stay visible to whoever builds the provider.

### `memory/extract/`

```
extract/
├── ingest/       Standardized message intake + multi-modal parser dispatch
├── parser/       Input parsing (format normalization, message preprocessing)
├── boundary/     BoundaryDetection: session buffer -> memcells (+ ledger, quarantine)
├── pipeline/     Per-cell extraction (Episode md + the events that announce it)
└── llm_retry.py  The one host-side retry of everalgo's LLMOutputError
```

`BoundaryDetection` is built by the bootstrap and injected into
`MemoryService`; it owns the session buffer, the `memcell` ledger and the
quarantine of a session the model keeps failing to segment. Why in the
domain: deciding where an episode ends is a memory rule, and keeping it out
of the service lets it be tested against in-memory stores and a scripted
model. `llm_retry.py` centralizes the host's retry of `LLMOutputError` for
boundary detection and user episode extraction. It does not eliminate SDK,
provider, or scheduler retries; their combined behavior needs the separate
budget and failure-classification checks described above.

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
function `fn(event, deps: MemoryDeps) -> list[Event]`: it receives collaborators
through `deps` and returns follow-up events instead of publishing them. It may
perform I/O and is not a pure function. Behavior-affecting configuration must
also be explicit; current deep configuration reads are not evidence of complete
dependency isolation. Each file declares its `StrategySpec` next to the
function; `STRATEGY_SPECS` collects
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
(`no-bypass-facade`). Backend configuration belongs at the process assembly
entry point. These controlled runtime facilities may hold process state, but
must not own business truth or provide access to Store, Service, or provider
instances. The current logging settings expose the level with console output;
additional formats or sinks are capability choices, not prerequisites for a
sound dependency structure.

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
providing memory algorithms, parsing, and their data contracts:

- `everalgo.parser` — multi-modal parsing (optional `[multimodal]` extra)
- `everalgo.user_memory` — ConvMemCell / Episode / Foresight / AtomicFact / Profile extractors
- `everalgo.agent_memory` — AgentMemCell / Case / Skill extractors
- `everalgo.boundary` — boundary detection (conversation -> MemCells)
- `everalgo.rank` — fusion + rerank
- `everalgo.knowledge` — KnowledgeExtractor (document parse + topic extraction)

The integration separates responsibilities:

- **Algorithm contracts** belong to EverAlgo. Host orchestration and callbacks
  may use those contracts directly; sharing a candidate type is not the same
  as duplicating ranking rules in the host.
- **Host persistence** belongs to EverOS. Algorithm calls may invoke an injected
  LLM or parsing capability; that does not make them pure functions or grant
  them responsibility for EverOS markdown, index rows, or SQLite transactions.
- **Operator prompts** belong to EverAlgo. Custom instructions append host
  requirements through the operator's supported contract.
- **Retry policy** belongs to the layers responsible for the operation being
  retried. An operator output error is distinct from provider transport or
  validation failure; a retry must preserve the applicable side-effect rules.

This boundary lets everalgo be reused across product forms (this open-source build, EverOS Cloud, OpenClaw plugins, etc.).

Import the LLM contract types (`LLMClient`, `LLMMessage`, `LLMOutputError`,
`StrictOutputModel`) from `everalgo.llm`; EverOS does not re-export them.

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
│   ├── UnsupportedModalityError   → 415 UNSUPPORTED_FORMAT
│   ├── PayloadTooLargeError       → 413 PAYLOAD_TOO_LARGE
│   └── QueueSaturatedError        → 429 QUEUE_SATURATED
├── InfrastructureError            (infrastructure failure)  → 503 EXTERNAL_SERVICE_UNAVAILABLE
│   ├── StorageError
│   ├── VectorStoreError
│   └── ExternalServiceError
│       ├── LLMServiceError
│       ├── EmbeddingServiceError
│       └── RerankServiceError
├── CapabilityError                (capability unavailable)  → 503 CAPABILITY_UNAVAILABLE
│   └── MultimodalNotEnabledError
└── ConfigurationError             (misconfiguration)         → 500 CONFIGURATION_ERROR
    └── ProviderNotConfiguredError → 422 PROVIDER_NOT_CONFIGURED
```

### Error propagation strategy

Exceptions are raised at the layer where the error is detected and
normally propagate to the boundary that can interpret them. Catching an error
to coordinate cleanup, partial results, or a documented fallback is legitimate;
avoid indiscriminate wrapping that loses its classification.
The entrypoints layer maps exceptions in
`entrypoints/api/exception_handlers.py`: one table from `AppError` subclass
to HTTP status, `ErrorCode`, optional fixed message and optional log line,
served by one generic handler and resolved through Starlette's MRO dispatch
(the most specific registered class wins). FastAPI's `HTTPException`,
`RequestValidationError` and the catch-all keep dedicated handlers. Every
response is the canonical error envelope (`schemas.ErrorResponse`); an
unmapped exception becomes a 500 `INTERNAL_ERROR` without its message.

### Boundary translation

Translate provider-specific failures at their integration boundary. Shared
EverAlgo contract errors may remain visible to callers that own their policy:

- `LLMCallError` → `LLMServiceError` at `component/parser/_core.py`
- Embedding / rerank provider errors → `EmbeddingServiceError` / `RerankServiceError` at their respective protocol modules

An exception family or HTTP status alone does not decide whether repeating a
partially completed operation is safe. Classify the cause, request conditions,
and side effects, then apply the combined budget rather than treating every
infrastructure error as transient.

## Further reading

- [docs/overview.md](overview.md) — vision and scope
- [docs/engineering.md](engineering.md) — engineering tooling and CI / CD
- [.claude/rules/architecture.md](../.claude/rules/architecture.md) — short-form rules for Claude Code
