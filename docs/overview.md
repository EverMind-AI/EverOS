# EverOS — Project Overview

## Vision

Build an open-source Python memory framework where **AI agents' long-term memory is plain Markdown files on the user's disk**, not opaque rows in a hosted database.

## Scope

**In scope (2.0)**:

- Local deployment for personal agents or small teams
- Chat memory: one-to-one and group chats, text and pictures (other content
  items are parsed into text) → episodes, picture memories, profiles,
  scenes and a group's profile
- Change, forget and history of a memory: every change is a new version,
  every forget a reversible mark
- Multi-round retrieval (BM25 + vector + scalar filter per sub-query, RRF,
  an LLM decider that selects the core memories)
- Cascade index sync (md edit → derived index)
- CLI + HTTP API

**Out of scope (2.0)**:

- Agent memory (trajectories, cases, skills) and a knowledge base of
  uploaded documents
- Multi-tenant / community deployment (10K+ users)
- End-to-cloud sync
- Distributed deployment / sharding

## Design philosophy

### 1. Markdown as Source of Truth

```
delete all derived index / SQLite files → can rebuild from md
delete any md file               → memory is gone
```

User trust comes from physical visibility — the user can `cat` / `vim` / `grep` their own memory at any time.

### 2. Three-piece storage with clear job boundaries

| Component | Role | Does NOT do |
|---|---|---|
| Markdown files | Truth source — entries, frontmatter | Search (grep is degraded fallback only) |
| SQLite | Queue, cascade audit log, sensitive data isolation | Vector / full-text |
| Derived index | Vector ANN + BM25 + scalar filter | Be the source of truth (loss = rebuild from md) |

### 3. Algorithm-orchestration separation

`everalgo` (a set of separate PyPI packages — `everalgo-user-memory` / `-boundary` / `-clustering` / `-rank` on `-core`, plus the optional `-parser` extra) holds the extraction algorithms (memory-cell extraction, episode generation, profile evolution). EverOS calls everalgo's extractor functions directly — passing storage-free data in, getting structured results out; for boundary detection, episode, profile and the scene narrative it can append requirements to the operator's bundled rules via custom instructions. everalgo knows nothing about storage.

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
- **v1.1** — Knowledge base + Reflection (offline memory consolidation)
- **v2.0** — Chat memory only: group memory, scenes, versioned and
  reversible memory, multi-round retrieval as the one search method; agent
  memory, the knowledge base and Reflection removed
- **Later** — Edge-to-cloud sync via EverMe (separate project)

## Status

**Latest stable release: v1.1.4** (PyPI) — the v1 API is stable.
