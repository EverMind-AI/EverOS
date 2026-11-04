# Code style rule (always loaded)

- **Formatter & linter**: `ruff` is the single tool (replaces black / isort / flake8).
  Line length 88, target `py312`. Run `make format` to auto-fix; `make lint` checks.
  Exception: the code migrated from EverAlgo (`src/everos/algo/**`,
  `src/everos/component/parser/backend/**`, `tests/fakes/llm_client.py`,
  `tests/fakes/llm_conformance.py`) was written at 120 columns and is exempt from
  `E501`; reflow a docstring when you edit it, never a prompt.
- **Active ruff rule sets**: `E F I N UP B SIM ASYNC`. Don't disable a rule inline
  unless there's a genuine reason — prefer fixing the code.
- **Type hints**: annotate every public function signature (params + return). The
  codebase is ~100% typed; keep it that way.
- **`from __future__ import annotations`** at the top of every module — annotations
  are strings, so forward refs and `X | None` unions are free.
- **Prefer `collections.abc`** (`Sequence`, `Mapping`) over concrete `list`/`dict`
  in signatures; use `Protocol` for structural interfaces.
- **No dead code**: no commented-out blocks, no unused imports, no speculative
  abstractions. Delete rather than comment out.
- **Naming**: `*Manager` (orchestrators), `*Provider` (injectable services),
  `*Reader`/`*Writer` (persistence), `*Recaller` (search routes). Follow the
  established suffix when adding a sibling.
