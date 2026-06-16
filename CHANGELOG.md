# Changelog

All notable changes to **EverOS** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Unreleased changes on `dev` will be listed here._

## [1.0.1] - 2026-06-16

### Security

- **Path-traversal hardening for caller-supplied identifiers.** `sender_id`
  now carries the same path-safety guard as `app_id` / `project_id`: a
  character whitelist plus rejection of the `.` / `..` tokens. The whitelist
  admits `@` and `+` so email-style ids and plus-addressing still pass.
- **Defense-in-depth write containment.** `MarkdownWriter` now rejects any
  write target that resolves outside the configured memory root before reading,
  creating parent directories, or writing files. The API layer maps this
  backstop error to HTTP 400.

### Documentation

- Add a multimodal usage guide and correct the multimodal error semantics
  after end-to-end verification.
- Document the upcoming Knowledge Wiki and idle/offline Reflection/Dreaming
  roadmap in the README and documentation set.
- Rename outdated algorithm-library references to `everalgo` across docs and
  code comments; no code identifiers changed.
- Fix accuracy drift found in a documentation audit; reflect the `everalgo`
  packages being published and the v1.0.0 stable status.

## [1.0.0] - 2026-06-03

First public release of EverOS — a Markdown-first memory extraction framework
for AI agents.

### Added

- **Markdown as source of truth** — all memory persists as plain `.md` files you
  can open, edit, grep, and version with Git.
- **Lightweight three-piece storage** — Markdown (truth) + SQLite (state / queue
  / audit) + LanceDB (vector + BM25 + scalar index). No external services
  required.
- **Hybrid retrieval** — BM25, vector, and scalar filtering in a single LanceDB
  query.
- **Cascade index sync** — editing a `.md` file triggers a file watcher →
  entry-level diff → sub-second LanceDB sync.
- **Dual-track memory** — user-track (Episodes / Profiles) and agent-track
  (Cases / Skills).
- **Multi-source extraction** — conversations, workflows, agent traces, and file
  knowledge.
- **CLI + HTTP API** — the `everos` command-line tool and a FastAPI server,
  async-first throughout.
- **Pluggable providers** — LLM / embedding / rerank via the OpenAI-compatible
  protocol (works with OpenAI, OpenRouter, vLLM, Ollama, …).
- **Decoupled algorithms** — memory extraction algorithms live in the standalone
  `everalgo-*` libraries published on PyPI.

[Unreleased]: https://github.com/EverMind-AI/EverOS/compare/v1.0.1...HEAD
[1.0.1]: https://github.com/EverMind-AI/EverOS/releases/tag/v1.0.1
[1.0.0]: https://github.com/EverMind-AI/EverOS/releases/tag/v1.0.0
