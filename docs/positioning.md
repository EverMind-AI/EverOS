# Positioning

This page keeps EverOS public language consistent across the README, docs,
release notes, banners, and demos.

## Category

EverOS is a **local-first memory runtime for AI agents**.

Use this category when the audience cares about agent infrastructure,
developer tools, and local data ownership. Avoid positioning EverOS as only a
vector database, RAG library, chat-history SDK, or dashboard.

## One-Liner

EverOS gives agents long-term memory stored as readable Markdown, with local
SQLite and LanceDB indexes rebuilt from that source of truth.

## Primary Proof Points

- Markdown source of truth: readable, editable, diffable, and Git-versioned.
- Local runtime stack: Markdown + SQLite + LanceDB, no required hosted database.
- User + agent memory tracks: user episodes/profile and agent cases/skills are
  separate first-class surfaces.
- Knowledge + Reflection: source-backed knowledge pages and offline memory
  consolidation extend beyond retrieval-only memory.
- Rebuildable indexes: derived SQLite and LanceDB state can be regenerated from
  the Markdown files.

## Claims To Avoid Without Fresh Evidence

Do not use benchmark or research claims in public hero copy unless they link to
an owned benchmark or citation page in this repository.

Avoid unsupported claims such as:

- Exact LoCoMo accuracy numbers.
- Exact p95 latency numbers.
- Exact token-savings percentages.
- Paper counts or publication claims.
- "Best", "fastest", or "most accurate" market-wide comparisons.

Use capability language instead:

- "Local-first memory for AI agents, stored as Markdown."
- "Memory agents can read, edit, search, and evolve."
- "A Markdown-native memory runtime for AI agents."

## Comparison Notes

These notes are for maintainers writing public copy. Keep comparisons factual,
linked, and framed around fit rather than winner-takes-all claims.

| Project | Public positioning | EverOS distinction |
|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | Universal memory layer with SDKs, CLI, integrations, self-hosted server, and managed platform paths. | Position EverOS around local-first ownership, Markdown source files, direct file editing, and rebuildable local indexes. |
| [MemOS](https://github.com/MemTensor/MemOS) | Memory OS / memory-augmented generation system for persistent memory, hybrid retrieval, and cross-task skill reuse. | Position EverOS as the practical local runtime with a concrete Markdown + SQLite + LanceDB storage model rather than a broad memory-OS taxonomy. |
| [OpenViking](https://github.com/volcengine/OpenViking) | Context database for AI agents using a filesystem paradigm for context, memory, resources, and skills. | Position EverOS around memory extraction, Markdown source of truth, Knowledge, Reflection, and derived indexes, not just context storage. |
| [MemPalace](https://github.com/MemPalace/mempalace) | Personal memory OS for agents and developers, with local-first and integration-oriented workflows. | Position EverOS around typed storage boundaries, source-backed Markdown, tested cascade/index rebuilds, and first-class user + agent memory tracks. |

## Banner Direction

Prefer a capability-led banner over metric-led copy:

```text
EverOS
Local-first memory for AI agents, stored as Markdown.
Markdown source of truth · SQLite + LanceDB · User + agent memory · Knowledge + Reflection
```

When metrics are available, link the banner or README badge to the benchmark
page that explains model stack, dataset, hardware, sample size, and command to
reproduce the result.
