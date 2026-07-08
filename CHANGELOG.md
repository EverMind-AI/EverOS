# Changelog

All notable changes to **EverOS** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.2] - 2026-07-07

### Fixed

- **Agent-track search broken by `deprecated_by IS NULL` filter** —
  `compile_filters()` unconditionally appended a `deprecated_by IS NULL`
  clause to every LanceDB query, but only `episode` and `atomic_fact`
  tables have this column. Agent-track search (`agent_case`,
  `agent_skill`) failed on any method. The clause is now conditional on
  `owner_type == "user"`.

## [1.1.1] - 2026-07-06

### Added

- **DashScope rerank provider** — Aliyun Bailian `gte-rerank-v2` adapter;
  configure with `rerank.provider = "dashscope"` in `everos.toml`.
- **`everos demo` TUI command** — Textual-based interactive CLI demo for
  showcasing EverOS core features.
- **Benchmark runner** — full LoCoMo benchmark suite: `benchmarks/run.py`
  with TOML configuration, automated ingestion, search evaluation, and
  scoring.
- **Hybrid search: heap-expand algorithm** — rewrote `hierarchy.py` to
  heap-driven lazy expansion with global top-N competition, replacing the
  serial four-layer pipeline.

### Fixed

- **Knowledge: atomic upsert prevents StaleDataError** — cascade handler
  switched from get→update to `INSERT ... ON CONFLICT DO UPDATE`, fixing
  concurrent cascade race conditions.
- **API: OpenAPI version read from `__version__`** — no longer hardcoded to
  `0.1.0`; version now stays in sync with `pyproject.toml`.
- **Profile middleware no longer swallows exceptions** — inner handler
  errors now re-raise correctly instead of silently returning HTTP 200.

### Performance

- **Cascade optimize throttle 1s → 10s** — reduced unnecessary LanceDB
  `optimize()` I/O by raising the minimum interval between calls.

### CI / Build

- **CI Python version matrix** — test and integration jobs now run on both
  Python 3.12 and 3.13.
- **pyproject.toml improvements** — added `project.urls`, `Typing :: Typed`
  classifier, relaxed `jieba` version constraint, removed unused
  `python-dotenv` dependency, cleaned up sdist include list, added `RUF`
  lint rules and coverage configuration.
- **`make ci` includes coverage** — `ci` target now runs
  `lint + test + integration + cov`.

### Documentation

- Fixed stale references across 13 files (v1.1.0 freshness sweep).
- Added GitHub sync guide (`docs/github-sync.md`).
- Added v1.1.0 release notes and v1.0.0 migration guide as standalone docs.
- Added `README.zh-CN.md` (Chinese README).
- Expanded `QUICKSTART.md` with source install instructions and `uv run`
  usage notes.
- Clarified cascade `optimize()` semantics in docstrings and runbook.

## [1.1.0] - 2026-06-24

### Added

- **Knowledge base subsystem** — full-stack document management exposed via
  `/api/v1/knowledge/*`. Upload documents (PDF / HTML / DOCX via multimodal
  parser), CRUD operations, and hybrid search (BM25 + vector + rerank +
  category boost). Ships with a 20-category default taxonomy
  (`.taxonomy.md`, auto-generated on first use). Original uploaded files are
  preserved alongside extracted Markdown. New settings group:
  `knowledge.*` (search tuning, `max_upload_bytes`, etc.).
- **Reflection V1** — offline memory self-improvement engine.
  Select → Merge → Re-extract → Deprecate: clusters related episodes within
  existing 7-day windows, merges them via LLM, re-extracts consolidated
  episodes, and deprecates the originals. Runs as an OME strategy
  (`reflect_episodes`); configure via `ome.toml`
  (`[strategies.reflect_episodes]`, cron `0 2 * * 1`), changes are
  hot-reloaded within ~2 s, no restart needed; **disabled by default**.
  Requires `everalgo-user-memory>=0.3.1`.
- **Standardized error response contract.** All API errors now return a
  canonical envelope with a semantic `ErrorCode` (10 codes: `NOT_FOUND`,
  `CONFLICT`, `INVALID_INPUT`, `EXTRACTION_EMPTY`, `UNSUPPORTED_FORMAT`,
  `EXTERNAL_SERVICE_UNAVAILABLE`, `CAPABILITY_UNAVAILABLE`,
  `CONFIGURATION_ERROR`, `INTERNAL_ERROR`, `BAD_REQUEST`), per-type
  exception handlers with MRO dispatch, and an `ErrorResponse` Pydantic
  model visible in OpenAPI docs. Replaces the v1.0 two-code scheme
  (`HTTP_ERROR` / `SYSTEM_ERROR`).
- **Search: hierarchical fact eviction** (Layer-4) with `min_score` floor —
  low-confidence atomic facts are evicted before fusion, improving
  precision.
- **Knowledge search degradation guidance** — when the embedding or rerank
  provider fails at call time, the knowledge search route enriches the
  error message with actionable guidance (e.g. retry with `method=keyword`,
  which needs no embedding) before returning `503`.
- **Knowledge topic recaller** — dual-column BM25 recall for knowledge
  topics, integrated into the search manager alongside existing recall
  types.

### Changed

- **`everos init` now generates `gpt-4.1-mini`** as the default LLM model
  (was `gpt-4o-mini`). Existing user configurations are not affected.
- **API error `code` values have changed.** v1.0 returned only `HTTP_ERROR`
  (all 4xx) and `SYSTEM_ERROR` (all 5xx). v1.1 returns fine-grained
  semantic codes (see Added above). Clients that match on `error.code`
  string values need to update. The envelope structure
  (`request_id` + `error.{code, message, timestamp, path}`) is unchanged.
- **DDD-aligned exception hierarchy** — domain errors reorganized:
  `ValidationError` → `InvalidInputError`;
  `DocumentAlreadyExistsError` → `DuplicateDocumentError`;
  `EmbeddingError` → `EmbeddingServiceError`;
  `RerankError` → `RerankServiceError`;
  `LLMError` → `LLMServiceError` (at the boundary);
  `MultimodalError` split into `UnsupportedModalityError` (domain) +
  `MultimodalNotEnabledError` (infrastructure).
  New base classes: `CapabilityError`, `ConfigurationError`.
- **`infra/` restructured** — storage adapters moved under
  `infra/persistence/{markdown,sqlite,lancedb}`; each sub-package's
  `__init__.py` is the sole public API (enforced by import-linter).
- **Parser capability extracted** to `component/parser` (shared by memorize
  and knowledge upload paths).

### Fixed

- **Knowledge search no longer returns a bare `500 INTERNAL_ERROR` when the
  embedding or rerank provider is unconfigured.** `_require_search_providers`
  now raises `ConfigurationError` → `500 CONFIGURATION_ERROR`. A provider
  that is configured but fails at call time still surfaces as
  `503 EXTERNAL_SERVICE_UNAVAILABLE`.
- **Knowledge document uploads are capped** at `knowledge.max_upload_bytes`
  (default 50 MiB); oversized uploads are rejected with `422` before parsing.
- **Knowledge search `query` is bounded** to 2000 chars.
- **`GET /knowledge/documents?sort_by=updated_at`** is now accepted.
- **`POST /knowledge/documents` returns `original_file_path`** so callers no
  longer need a follow-up `GET` to locate the preserved upload.
- **Rerank providers no longer echo the upstream HTTP response body** into the
  client-facing `503` message (vLLM / DeepInfra); the body is logged instead.
- **Knowledge FK cascade race** — removed the foreign key on
  `knowledge_topics.doc_id` that caused delete-order race conditions;
  cascade cleanup handled at application level.
- **Knowledge `replace_document`** — atomic PUT: backup old Markdown before
  re-extraction; removed explicit SQLite delete for atomicity.
- **Knowledge duplicate `doc_id`** rejected on create; title collision
  resolved by appending `doc_id` to directory name.
- **Knowledge `md_path` resolution** fixed in `delete_document` (was not
  resolved against `memory_root`).
- **OME file-handle leak** — portalocker file handle is now closed on lock
  contention instead of being left open.
- **jieba / Python 3.12 compatibility** — deferred jieba import to avoid
  `SyntaxError` from invalid escape sequences; suppressed
  `DeprecationWarning` in tests.
- **Test isolation** — tests no longer leak `.env` state or depend on module
  import ordering.

### Documentation

- Added knowledge base technical documentation.
- Corrected the onboarding flow: `everos init` writes `everos.toml` +
  `ome.toml` (TOML), not a `.env` file; removed the nonexistent
  `--xdg` / `--env-file` options and the false `0600`-permissions claim
  from `README.md` / `QUICKSTART.md`; fixed the stable-version line
  (`v1.0.1`) and completed the `docs/cli.md` command tree.
- Updated error handling docs to match the new DDD exception hierarchy.

## [1.0.1] - 2026-06-16

### Security

- **Path-traversal hardening for caller-supplied identifiers.** `sender_id`
  (which flows through to `owner_id` and becomes a directory segment on the
  episode write path) now carries the same path-safety guard as `app_id` /
  `project_id`: a character whitelist plus rejection of the `.` / `..` tokens.
  The whitelist admits `@` and `+` so real-world ids (email-style,
  plus-addressing) still pass.
- **Defense-in-depth write containment.** `MarkdownWriter` now rejects any
  write target that resolves outside the configured memory root, before any
  filesystem touch (both the write `mkdir` and the append read-modify-write
  read). This backstop holds even if an identifier reaches the writer
  unsanitised (e.g. an `owner_id` set in the extract pipeline rather than from
  the DTO). The API layer maps the resulting error to HTTP 400.

### Documentation

- Add a multimodal usage guide and correct the multimodal error semantics
  after end-to-end verification.
- Rename the algorithm library to `everalgo` across docs and
  code comments (no code identifiers changed).
- Fix accuracy drift found in an adversarial doc audit; reflect the
  `everalgo` packages being published and the v1.0.0 stable status.

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

[Unreleased]: https://github.com/EverMind-AI/everos/compare/v1.1.2...HEAD
[1.1.2]: https://github.com/EverMind-AI/everos/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/EverMind-AI/everos/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/EverMind-AI/everos/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/EverMind-AI/everos/releases/tag/v1.0.1
[1.0.0]: https://github.com/EverMind-AI/everos/releases/tag/v1.0.0
