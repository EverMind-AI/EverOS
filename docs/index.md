# EverOS Documentation

Documentation for [EverOS](../README.md) — md-first memory extraction
framework. Organised by [Diátaxis](https://diataxis.fr/) — what kind of
question you have determines which section to read.

## Tutorials

Learning-oriented entry points — start here to get a feel for the system
before wiring it into a real workflow.

| Doc | Purpose |
|---|---|
| [everos-demo.md](everos-demo.md) | `everos demo` — local educational TUI to feel the memory lifecycle before configuring keys |
| [use-cases.md](use-cases.md) | Worked examples and integrations showing what persistent memory enables, to study and adapt |

## Reference

Technical reference: contracts, commands, schemas — read these when you
already know what you want to do and need to know exactly how.

| Doc | Purpose |
|---|---|
| [api.md](api.md) | HTTP API v1 reference — endpoints, request / response, error contracts |
| [knowledge.md](knowledge.md) | Knowledge base module — upload, search, taxonomy, storage layout |
| [reflection.md](reflection.md) | Reflection — offline memory consolidation: enable, schedule, storage, triggering |
| [cli.md](cli.md) | `everos` CLI subcommands + env var conventions |
| [storage_layout.md](storage_layout.md) | Memory-root tree + frontmatter chassis + EntryId encoding |
| [prompt_slots.md](prompt_slots.md) | PromptSlot loader — bundled default prompts (Layer 1 live; app/runtime overlays planned) |
| [configuration.md](configuration.md) | TOML / env-var configuration reference |
| [multimodal.md](multimodal.md) | Multimodal content items — image / PDF / audio / doc parsing |

## Explanation

Design decisions and architectural concepts — read these to understand
why the system is shaped the way it is.

| Doc | Purpose |
|---|---|
| [overview.md](overview.md) | Project vision, scope, design philosophy |
| [how-memory-works.md](how-memory-works.md) | Storage stack + on-disk paths + write→index→read pipeline + consistency |
| [architecture.md](architecture.md) | DDD layered architecture + dependency rules |
| [datetime.md](datetime.md) | Two-zone discipline — UTC at storage, display tz at boundaries |

## How-to

Task-driven operational guides — read these when you need to do a
specific thing (drain a queue, recover from a stuck row, etc.).

| Doc | Purpose |
|---|---|
| [cascade_runbook.md](cascade_runbook.md) | Cascade subsystem ops — drain queue, recover stuck rows |
| [github-sync.md](github-sync.md) | Guardrails for refreshing GitHub from internal exports without overwriting GitHub-only workflow files |
| [locomo_benchmark.md](locomo_benchmark.md) | LoCoMo benchmark — run and evaluate |

## Engineering

For contributors working on the framework itself — how to build, test,
and ship a change.

| Doc | Purpose |
|---|---|
| [engineering.md](engineering.md) | Contributor engineering reference — build, test, CI gates, branch & commit conventions |

## See also

Top-level project files live next to the repo root:

- [README.md](../README.md) — quick start & feature overview
- [QUICKSTART.md](../QUICKSTART.md) — 5-minute walkthrough (install → service → search)
- [CONTRIBUTING.md](../CONTRIBUTING.md) — how to contribute (issue-only model)
- [CHANGELOG.md](../CHANGELOG.md) — release notes
- [release-notes-1.1.0.md](release-notes-1.1.0.md) — EverOS 1.1.0 highlights (Knowledge, Reflection, OME)
- [migration-to-1.0.0.md](migration-to-1.0.0.md) — migrate off pre-1.0.0 APIs / infrastructure
- [SECURITY.md](../SECURITY.md) — security policy & private vulnerability reporting
- [CITATION.md](../CITATION.md) — academic citation info
- [ACKNOWLEDGMENTS.md](../ACKNOWLEDGMENTS.md) — third-party acknowledgments

Coding conventions and slash command workflows are auto-loaded by
Claude Code from [.claude/rules/](../.claude/rules/) and
[.claude/skills/](../.claude/skills/).
