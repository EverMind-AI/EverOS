# `everos` package

Source layout for the `everos` Python package. This README is a quick
orientation; full architectural detail lives elsewhere.

## Layout

```
everos/
├── entrypoints/    Presentation: cli + api
├── service/        Application: use case orchestration
├── algo/           Every memory algorithm (types / llm / boundary / extractors / rank), stateless
├── memory/         Domain: extract + search + cascade + custom_instructions + models
├── infra/          Infrastructure: persistence/{markdown, sqlite, lancedb, milvus, index}
├── component/      Cross-cutting providers: llm / embedding / rerank / parser (incl. backend) / tokenizer
├── core/           Runtime base: observability / lifespan / context
└── config/         Data: Settings + default.toml + custom_instructions files
```

Each subpackage has a top-level `__init__.py` describing its responsibility
and public API.

## Dependency rule

```
entrypoints → service → memory ← infra
                         ↓
                        algo
```

Single-direction; enforced by `import-linter` in CI.

`algo/` imports no other EverOS layer. Callers supply data, configuration, and
external capabilities and own all persistent state. See [algo/README.md](algo/README.md).

## Further reading

- Architecture: [../../docs/architecture.md](../../docs/architecture.md)
- Coding rules (auto-loaded by Claude Code): [../../.claude/rules/](../../.claude/rules/)
