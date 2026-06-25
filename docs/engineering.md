# Engineering Reference

> Companion docs: system design lives in [architecture.md](architecture.md);
> coding rules live in [../.claude/rules/](../.claude/rules/). This document
> is the contributor-facing reference for **building, testing, and shipping**
> a change — the toolchain, the CI gates, and the branch / commit conventions
> your pull request must satisfy.

## Toolchain

| Tool | Role |
|---|---|
| [uv](https://docs.astral.sh/uv/) | sole package manager (`uv sync`; do not use `pip install`) |
| [hatchling](https://hatch.pypa.io/) | wheel build backend (src layout under `src/everos`) |
| [ruff](https://docs.astral.sh/ruff/) | lint + format (replaces black / isort / flake8) |
| [import-linter](https://import-linter.readthedocs.io/) | enforces the DDD layer dependency direction |
| [pytest](https://docs.pytest.org/) | unit + integration tests |
| [pre-commit](https://pre-commit.com/) | local gate run before each commit |
| `Makefile` | single entry point for every command — CI invokes the same targets |

All tool configuration lives in a single `pyproject.toml` (ruff, pytest,
coverage, and the import-linter layer contracts) — there are no separate
`pylintrc` / `pytest.ini` / `.isort.cfg` files.

## Local development

```bash
make install      # uv sync --frozen
make format       # ruff fix + format
make lint         # ruff check + format-check + import-linter + datetime/asset/name guards
make test         # pytest tests/unit
make integration  # pytest tests/integration
make cov          # unit + integration with coverage (gate: 80%)
make ci           # lint + test + integration  — run this before pushing
make help         # list every target
```

CI runs the **same** `make` targets, so a green `make ci` locally predicts a
green pipeline.

### Configuration

Settings load in ascending priority:

1. `src/everos/config/default.toml` — shipped with the package (lowest)
2. `<memory-root>/everos.toml` — user config (optional)
3. `EVEROS_*` environment variables (highest)

Run `everos init` to generate starter config and `everos config show` to
inspect the effective result. Full reference: [configuration.md](configuration.md).

## Quality gates

Each stage can independently fail a change; there is no `--no-verify` bypass.

```
1. Editor      ruff (lint + format) on save
2. pre-commit  ruff, trailing-whitespace / EOF, yaml & toml checks,
               large-file & private-key guards, merge-conflict check,
               and gitlint (commit-msg stage) — see "Commits" below
3. make ci     lint + unit + integration — run before pushing
4. GitHub CI   re-runs the same make targets on every pull request
5. Review      1 approval + all conversations resolved + all checks green
```

## Continuous integration

CI runs on GitHub Actions ([.github/workflows/](../.github/workflows/)). Every
pull request into `main` must pass:

| Check | Command | Guards |
|---|---|---|
| lint | `make lint` | ruff style, DDD layer direction (import-linter), datetime discipline, asset & deprecated-name guards |
| unit tests | `make test` | `tests/unit` |
| integration tests | `make integration` | `tests/integration` |
| package build | `make package` | the wheel builds and imports cleanly |
| docs | `make docs-check` | Markdown and internal-link validity |
| commit messages / PR title | `make check-commits` / `make check-pr-title` | Conventional Commits format |

`main` is a protected branch: no direct pushes; changes land through a
reviewed pull request with all checks green.

## Contributing workflow

- **Branch** off `main`, then open a pull request back into `main`.
- **Commits and the PR title** follow
  [Conventional Commits](https://www.conventionalcommits.org/): `type(scope): subject`,
  with the subject ≤ 72 characters and no leading emoji. Allowed types:
  `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`,
  `build`, `ci`, `revert`. This is enforced both locally (gitlint, commit-msg
  stage) and in CI.
- **Pull requests** use [.github/PULL_REQUEST_TEMPLATE.md](../.github/PULL_REQUEST_TEMPLATE.md)
  (changes / scope / API impact / tests / checklist).
- **Issues** use the templates under [.github/ISSUE_TEMPLATE/](../.github/ISSUE_TEMPLATE/).

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full onboarding walkthrough.

> This repository also ships Claude Code configuration — coding rules under
> `.claude/rules/` and slash-command workflows under `.claude/skills/` — that
> encode the conventions above. It is optional convenience tooling: using
> Claude Code is not required to contribute, and the CI gates remain the
> source of truth.

## References

- Architecture: [architecture.md](architecture.md)
- Coding rules: [../.claude/rules/](../.claude/rules/)
- Contributor onboarding: [../CONTRIBUTING.md](../CONTRIBUTING.md)
- [uv](https://docs.astral.sh/uv/) ·
  [ruff](https://docs.astral.sh/ruff/) ·
  [import-linter](https://import-linter.readthedocs.io/) ·
  [pre-commit](https://pre-commit.com/) ·
  [Conventional Commits](https://www.conventionalcommits.org/)
