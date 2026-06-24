# EverOS 1.1.0 Release Notes

EverOS 1.1.0 expands the memory system beyond user episodes with first-class
knowledge management, reflection, and stronger operational guarantees around
search, persistence, and API contracts.

## Highlights

- Added Knowledge APIs and storage for document creation, listing, patching,
  deletion, taxonomy handling, and knowledge-topic search.
- Added Reflection orchestration for periodically merging and refining episode
  clusters.
- Expanded OME runtime support with event IDs, run-record storage migration,
  configuration reload behavior, and testing harness improvements.
- Improved search and get behavior for agent-owned memory, including agent
  cases, agent skills, and owner-type isolation.
- Reworked API error handling around typed exception handlers and consistent
  error envelopes.
- Updated docs, OpenAPI schema, configuration examples, and test coverage for
  the 1.1.0 surface area.

## Compatibility Notes

- Existing local TUI demo registration remains available in this PR.
- Existing DashScope rerank support is preserved; the DashScope provider file
  is not replaced by the 1.1.0 archive.
- The update intentionally leaves directories outside the 1.1.0 archive scope,
  including existing use-case and iOS demo material, untouched.
- Knowledge search requires configured embedding and rerank providers. Missing
  providers now fail explicitly with configuration errors rather than silently
  returning degraded results.

## Upgrade Notes

- Regenerate or review `docs/openapi.json` after route or DTO changes.
- Run `uv sync --frozen` against the updated `uv.lock`.
- Review `config.example.toml`, `src/everos/config/default.toml`, and
  `src/everos/config/default_ome.toml` for new Knowledge and OME settings.
- If running e2e tests without live provider credentials, use dummy provider
  environment variables for startup-only checks; live vector and rerank paths
  remain behind slow/live markers.

## Verification

This PR was checked with:

- `uv run ruff check .`
- `uv run pytest tests/unit`
- `uv run pytest tests/integration`
- `uv run lint-imports`
- `uv run pytest tests/e2e` with dummy LLM and embedding environment variables
- Targeted search/get regression tests for agent and deprecated-filter handling
