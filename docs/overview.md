# EverOS — Project Overview

## Vision

Build an open-source Python memory framework where **AI agents' long-term memory is plain Markdown files on the user's disk**, not opaque rows in a hosted database.

## Scope

**In scope (v1)**:

- Local deployment for personal agents or small teams
- Conversation, workflow, agent-trace, file-knowledge → structured memory
- Hybrid retrieval (BM25 + vector + scalar filter)
- Cascade index sync (md edit → LanceDB sub-second)
- Dual-track memory (user-track / agent-track)
- Offline memory evolution (Foresight / AtomicFact / Profile / Skill)
- CLI + HTTP API

**Out of scope (v1, future v2)**:

- Multi-tenant / group / community deployment (10K+ users)
- End-to-cloud sync (planned for v2)
- Distributed deployment / sharding

## Design philosophy

### 1. Markdown as Source of Truth

```
delete all LanceDB / SQLite files → can rebuild from md
delete any md file               → memory is gone
```

User trust comes from physical visibility — the user can `cat` / `vim` / `grep` their own memory at any time.

### 2. Three-piece storage with clear job boundaries

| Component | Role | Does NOT do |
|---|---|---|
| Markdown files | Truth source — entries, frontmatter | Search (grep is degraded fallback only) |
| SQLite | Queue, cascade audit log, sensitive data isolation | Vector / full-text |
| LanceDB | Vector ANN + BM25 + scalar filter, single-query hybrid | Be the source of truth (loss = rebuild from md) |

### 3. Algorithm-orchestration separation

`everalgo` (a set of separate PyPI packages — `everalgo-core` / `-boundary` / `-user-memory` / `-agent-memory` / `-rank`, plus the optional `-parser` extra) holds the extraction algorithms (MemCell extraction, Episode generation, Profile evolution). EverOS calls everalgo via the PromptSlot interface; everalgo knows nothing about storage.

This boundary lets the same algorithm power both this open-source lightweight version and other product forms.

### 4. DDD layered architecture

```
entrypoints  →  service  →  memory  →  infra
                              ↓
                        component / core / config
```

Strict single-direction dependency, enforced by `import-linter` in CI.

## Why src layout (`src/everos/`)

- Standard PyPA project structure used when shipping to PyPI
- Avoid namespace collision with system packages named `memory`, `infra`, etc.
- Avoid accidental import of working-tree code in dev (PyPA recommendation)

## Comparable projects (where EverOS differs)

| Project | Position | Difference |
|---|---|---|
| [mem0](https://github.com/mem0ai/mem0) | API-first memory service | mem0 stores in vector DB; we store in md files |
| [Letta](https://github.com/letta-ai/letta) | Agent OS w/ Core/Recall/Archival | Letta uses Postgres; we use markdown filesystem |
| [MemOS](https://github.com/MemTensor/MemOS) | Multi-classification memory | MemOS targets enterprise; we target lightweight (single-user / small team) |
| [memsearch](https://github.com/zilliztech/memsearch) | md-first search engine | Closest to us; we add memory extraction (not just search) |

## Roadmap

- **v0.1 (MVP)** — Phase 1 core loop: markdown + lancedb + cascade + episode extraction
- **v0.2** — Full extraction pipeline (workspace / agent / knowledge), evolution framework
- **v0.3** — Production hardening, full CLI, HTTP API, Obsidian demo
- **v1.0** — Stable API, PyPI release, comprehensive docs
- **v2** (future) — Edge-to-cloud sync via EverMe (separate project)

## Status

**Stable (v1.0.1)** — Released on PyPI; the v1 API is stable. Development
continues on `dev` toward v1.1.
