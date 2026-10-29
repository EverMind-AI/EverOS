# Architecture rule (always loaded)

EverOS is a DDD-layered framework. The dependency direction is **single, downward only**:

```
entrypoints  →  service  →  memory  →  infra
                              ↓
                        component / core / config
```

- `entrypoints/` — CLI + HTTP API (presentation). No business logic.
- `service/` — use-case orchestration (memorize / retrieve / evolve / manage).
- `memory/` — domain (extract / search / cascade / prompt_slots / models).
- `infra/` — storage adapters (markdown + sqlite + lancedb) and the OME subsystem.
- `component/` — injectable providers (llm / embedding / config / utils).
- `core/` — runtime base (observability / lifespan / context / persistence primitives).
- `config/` — configuration data (Settings + default TOML).

## Hard constraints (enforced by `import-linter`, run in `make lint`)

1. **Layering**: an outer layer may import an inner layer, never the reverse.
   `entrypoints → service → memory → infra`.
2. **Private internals**: `service`, `memory`, and `entrypoints` must not import
   `infra.persistence.{markdown,lancedb,sqlite}.**` internals — go through the
   package facade (`from everos.infra.persistence.markdown import ...`).
3. **OME isolation**: `infra.ome` imports only `everos.memory.ports` (the types
   the domain declares for it) — never other `memory` modules, `persistence`,
   `service`, or `entrypoints`. It is a scheduler with no domain logic.

If a change needs to cross a boundary the wrong way, the design is wrong — refactor,
don't add an exception.

Full rationale: [docs/architecture.md](../../docs/architecture.md).
