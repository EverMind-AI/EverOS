# Architecture rule (always loaded)

EverOS is a DDD-layered framework with the storage ports owned by the domain
(dependency inversion):

```
entrypoints  →  service  →  memory  ←  infra
cross-cutting, usable by every layer: core / component / config
```

- `entrypoints/` — CLI + HTTP API + TUI: protocol translation only (request →
  facade call → response, exception → status / exit code). No business
  judgement, no construction. The HTTP middleware lives in
  `entrypoints/api/middleware/`.
- `service/` — `bootstrap.py` (the composition root: the one place that
  selects and builds adapters, providers and engines and owns their lifecycle)
  and three facades: `MemoryService` (memorize / flush / search / get + the
  async queue), `KnowledgeService` (documents, uploads, search), `OpsService`
  (cascade, backfill, OME trigger, health, tasks). A facade module over
  ~300 lines triggers a split review: split by responsibility into a subpackage
  (`service/memory/`, `service/ops/`, `service/knowledge/`) with the facade
  class in `__init__` and the use-case steps in private modules. The number is
  the trigger, not the criterion — never contort a signature to stay under it.
- `memory/` — domain: models, ports (`*Store` / `*Index` / `EventBus` /
  `StrategySpec`), extract / search / strategies / reflection / mappers /
  custom_instructions.
- `infra/` — adapters implementing `memory.ports` (markdown + sqlite + lancedb
  + milvus) and the two domain-agnostic engines (cascade, OME).
- `component/` — injectable providers (llm / embedding / rerank / parser /
  tokenizer / config loader).
- `core/` — runtime base: errors, observability, request context, stateless
  utils (`core.utils.datetime`); no web framework.
- `config/` — Settings, default TOML, custom-instruction YAML.

## Hard constraints (enforced by `import-linter`, run in `make lint`)

1. **Layering**: `entrypoints → service → memory`; `memory` never imports
   `infra`. Infra reaches the domain only through `memory.ports`, `models`,
   `events` and the pure `mappers`; it never invokes a domain workflow
   (extract / search / strategies / reflection).
2. **Entry points only see the service**: `entrypoints` must not import
   `memory`, `infra`, `component` or `everalgo` — not even their data types.
   DTOs and error types come through the `everos.service` facade or
   `everos.core`.
3. **One assembler**: inside `service`, only `bootstrap` imports `infra` and
   provider implementations; use-case modules receive what they need. A
   component may build its own cohesive internal helpers (a short-lived probe
   provider, a parser backend) — bootstrap is the factory for application-level
   choices, not for every object. Dependencies that change behaviour
   (configuration, external IO) are explicit arguments, never looked up.
4. **A type import is not a dependency on an implementation**: importing a
   protocol or a result type for annotations (`TYPE_CHECKING` imports are
   excluded from the contracts) does not breach a layer; constructing or
   calling the concrete backend does.
5. **OME isolation**: `infra.ome` imports only `everos.memory.ports` — never
   other `memory` modules, `persistence`, `service`, or `entrypoints`.
6. **Cross-cutting stays below**: `core` / `config` import no business layer;
   `component` imports only `core` and `config` (explicit settings types and
   pure config helpers) and never hides behaviour behind a global
   configuration lookup.

If a change needs to cross a boundary the wrong way, the design is wrong —
refactor, don't add an exception. `make lint` enforces only what the
contracts can see: a green run is not evidence that a responsibility sits in
the right layer. A boundary exception, when one is unavoidable, needs a
rationale, an impact analysis and a guard test, recorded next to the contract
in `pyproject.toml`; never relax a rule silently to fit existing code.

Full rationale: [docs/architecture.md](../../docs/architecture.md).
