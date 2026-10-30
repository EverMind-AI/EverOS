# EverOS — md-first Memory Extraction Framework

This is a Python framework for md-first memory extraction (lightweight; single-user or small-team).

## Quick commands

```bash
uv sync                    # install deps
make lint                  # ruff (check + format-check) + import-linter
make format                # auto-fix formatting
make test                  # pytest tests/unit
make integration           # pytest tests/integration
make ci                    # full CI: lint + test + integration
```

## Architecture

DDD layers with domain-owned ports, plus cross-cutting:

```
entrypoints  →  service  →  memory  ←  infra
cross-cutting, usable by every layer: core / component / config
```

- `entrypoints/` — cli + api (incl. HTTP middleware) + tui (protocol translation only)
- `service/` — `bootstrap.py` (the one assembler) + `MemoryService` / `KnowledgeService` / `OpsService` facades
- `memory/` — domain (models + ports + extract + search + strategies + reflection + custom_instructions)
- `infra/` — adapters implementing `memory.ports` (markdown + sqlite + lancedb + milvus) + the cascade and OME engines
- `component/` — injectable providers (llm / embedding / rerank / parser / tokenizer)
- `core/` — runtime base (errors / observability / context / utils.datetime)
- `config/` — configuration data (Settings + default.toml + custom_instructions)

**Dependency rule**: `entrypoints → service → memory ← infra`; entry points import only `service` (plus `core` / `config`). Enforced by `import-linter`.

Detailed: [docs/architecture.md](docs/architecture.md).

## Engineering practices

- **Coding rules** auto-loaded from [.claude/rules/](.claude/rules/) (10 rules; the three always-loaded ones cover architecture / code-style / language-policy, the rest are path-scoped and load when Claude Code opens a matching file)
- **Workflows** as slash commands in [.claude/skills/](.claude/skills/) — `/commit`, `/new-branch`, `/pr`, `/release`
- **Project-level decisions** in [docs/](docs/) (low-frequency, human-judgment-required)
- **Language policy**: the project targets a global audience — docs and code are English; CJK appears only in test fixtures and locale-suffixed mirrors. Scanned by `make check-cjk`.
- **Datetime discipline**: never call `datetime.now()` / `time.time()` directly — use `everos.core.utils.datetime`. Enforced by `make check-datetime`.

Contributor engineering reference — build, test, CI gates, branch & commit conventions: [docs/engineering.md](docs/engineering.md).

## Branch strategy

`main` is the default and protected branch. Create scoped branches from
`main` (`feat/*`, `fix/*`, `docs/*`, `ci/*`, `chore/*`, `refactor/*`) and open
pull requests back to `main` after the required checks pass.

See [.claude/skills/new-branch/SKILL.md](.claude/skills/new-branch/SKILL.md)
for the full branch workflow.

## GitHub sync guard

When refreshing this repository from an internal source archive, preserve
GitHub-only contributor and automation files. Do not overwrite `CLAUDE.md`,
`.claude/skills/*`, `CONTRIBUTING.md`, or `.github/*` workflow/template files
without checking [docs/github-sync.md](docs/github-sync.md).

## Storage three-piece set

```
Markdown (truth)  +  SQLite (state)  +  LanceDB (vector + BM25 + scalar)
```

- Memory root: `~/.everos/{agents,users,knowledge}/` (md files = single source of truth)
- System DB: `~/.everos/.index/sqlite/system.db` (state + audit + queue + metadata)
- Index: `~/.everos/.index/lancedb/` (rebuildable from md)

Selection rationale: [docs/architecture.md](docs/architecture.md).

## Source layout

**src layout** (`src/everos/<...>`): standard PyPA project structure — code lives under `src/` so the working tree is not on the import path until installed, preventing accidental imports of in-development modules.

Algorithm assets (prompts, extractors) live in the separate [`everalgo`](https://github.com/EverMind-AI/EverAlgo) library, consumed here as the `everalgo-*` PyPI packages.

## Where things go

| Want to... | Look at |
|---|---|
| Understand architecture | [docs/architecture.md](docs/architecture.md) |
| Understand storage choice | [docs/architecture.md](docs/architecture.md) (storage section) |
| Build, test, CI & conventions | [docs/engineering.md](docs/engineering.md) |
| Add a new module | [.claude/rules/init-py-and-reexport.md](.claude/rules/init-py-and-reexport.md) |
| Make a commit | use `/commit` |
| Open a branch / PR | use `/new-branch` / `/pr` |
| Run checks before pushing | `make ci` |
