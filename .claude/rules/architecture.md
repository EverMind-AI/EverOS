# Architecture rule (always loaded)

EverOS is a DDD-layered framework with the storage ports owned by the domain
(dependency inversion):

```
entrypoints  →  service  →  memory  ←  infra
cross-cutting, usable by every layer: core / component / config
```

- `entrypoints/` — CLI + HTTP API + TUI: protocol translation only (request →
  facade call → response, exception → status / exit code). No business
  judgement, no construction.
- `service/` — `bootstrap.py` (the one place that builds adapters, providers
  and engines) and three facades: `MemoryService` (memorize / flush / search /
  get + the async queue), `KnowledgeService` (documents, uploads, search),
  `OpsService` (cascade, backfill, OME trigger, health, tasks). A facade module
  over ~300 lines becomes a subpackage (`service/memory/`, `service/ops/`)
  with the facade class in `__init__` and the use-case steps in private
  modules.
- `memory/` — domain: models, ports (`*Store` / `*Index` / `EventBus` /
  `StrategySpec`), extract / search / strategies / reflection / mappers /
  custom_instructions.
- `infra/` — adapters implementing `memory.ports` (markdown + sqlite + lancedb
  + milvus) and the two domain-agnostic engines (cascade, OME).
- `component/` — injectable providers (llm / embedding / rerank / parser /
  tokenizer / config loader).
- `core/` — runtime base: errors, observability, middleware, request context,
  stateless utils (`core.utils.datetime`).
- `config/` — Settings, default TOML, custom-instruction YAML.

## Hard constraints (enforced by `import-linter`, run in `make lint`)

1. **Layering**: `entrypoints → service → memory`; `memory` never imports
   `infra`, and `infra` reaches the domain only through `memory.ports` /
   `models` / `events`.
2. **Entry points only see the service**: `entrypoints` must not import
   `memory`, `infra`, `component` or `everalgo` — DTOs and error types come
   through the `everos.service` facade or `everos.core`.
3. **One assembler**: inside `service`, only `bootstrap` imports `infra` and
   provider implementations; use-case modules receive what they need.
4. **Private internals**: `service`, `memory`, and `entrypoints` must not import
   `infra.persistence.{markdown,lancedb,sqlite,...}.**` internals — go through
   the package facade.
5. **OME isolation**: `infra.ome` imports only `everos.memory.ports` — never
   other `memory` modules, `persistence`, `service`, or `entrypoints`.
6. **Cross-cutting stays below**: `core` / `config` import no business layer;
   `component` imports only `core`.

If a change needs to cross a boundary the wrong way, the design is wrong — refactor,
don't add an exception.

Full rationale: [docs/architecture.md](../../docs/architecture.md).
