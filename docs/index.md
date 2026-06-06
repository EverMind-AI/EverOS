# EverOS Documentation

Documentation for [EverOS](../README.md) — md-first memory extraction
framework. Organised by [Diátaxis](https://diataxis.fr/) — what kind of
question you have determines which section to read.

## Reference

Technical reference: contracts, commands, schemas — read these when you
already know what you want to do and need to know exactly how.

| Doc | Purpose |
|---|---|
| [api.md](api.md) | HTTP API v1 reference — endpoints, request / response, error contracts |
| [cli.md](cli.md) | `everos` CLI subcommands + env var conventions |
| [storage_layout.md](storage_layout.md) | Memory-root tree + frontmatter chassis + EntryId encoding |
| [prompt_slots.md](prompt_slots.md) | YamlConfigLoader + three-layer prompt override |
| [migration-to-1.0.0.md](migration-to-1.0.0.md) | Legacy API and infrastructure migration notes for EverOS 1.0.0 |

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

## Engineering / Internal

For maintainers and contributors working on the framework itself,
not for using it.

| Doc | Purpose |
|---|---|
| [engineering.md](engineering.md) | Engineering & dev-efficiency infrastructure (CI / tooling / Claude Code) |

## See also

Top-level project files live next to the repo root:

- [README.md](../README.md) — quick start & feature overview
- [QUICKSTART.md](../QUICKSTART.md) — 5-minute walkthrough (install → service → search)
- [use-cases.md](use-cases.md) — full use-case gallery and integration examples
- [CONTRIBUTING.md](../CONTRIBUTING.md) — how to contribute (issue-only model)
- [CHANGELOG.md](../CHANGELOG.md) — release notes
- [SECURITY.md](../SECURITY.md) — security policy & private vulnerability reporting
- [CITATION.md](../CITATION.md) — academic citation info
- [ACKNOWLEDGMENTS.md](../ACKNOWLEDGMENTS.md) — third-party acknowledgments

Coding conventions and slash command workflows are auto-loaded by
Claude Code from [.claude/rules/](../.claude/rules/) and
[.claude/skills/](../.claude/skills/).
