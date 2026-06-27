# Contributing to EverOS

Thanks for your interest in EverOS! This page explains how contribution works
on this project.

## How EverOS accepts contributions

EverOS accepts **curated pull request contributions**. The EverMind core team
reviews every change for product fit, architecture consistency, licensing, and
long-term maintainability before it is merged.

What we actively welcome from the community:

| Type | Where |
|---|---|
| 🐛 Bug reports | [Open a bug issue](https://github.com/EverMind-AI/EverOS/issues/new?template=bug_report.yml) |
| 💡 Feature ideas / use cases | [Open a feature issue](https://github.com/EverMind-AI/EverOS/issues/new?template=feature_request.yml) |
| 🔧 Focused fixes | A pull request linked to a bug, design note, or clear reproduction |
| ❓ Questions & discussion | [GitHub Discussions](https://github.com/EverMind-AI/EverOS/discussions) / [Discord](https://discord.gg/gYep5nQRZJ) |

Large architectural changes should start as an issue or discussion before code.
Small documentation, test, CI, and bug-fix PRs can be opened directly.

## Reporting a bug

Use the [bug report template](https://github.com/EverMind-AI/EverOS/issues/new?template=bug_report.yml). Include:

- Clear reproduction steps
- Expected vs. actual behavior
- Environment (OS, Python version, everos version)
- Relevant logs (**with secrets redacted**)

## Suggesting a feature

Use the [feature request template](https://github.com/EverMind-AI/EverOS/issues/new?template=feature_request.yml). Provide:

- The use case / problem being solved
- Proposed API or behavior
- Backward-compatibility considerations

## Sending a pull request

1. Create a branch from `main`.
2. Keep the change scoped to one purpose.
3. Do not commit images, videos, or asset/media directories. Use external
   hosting, release artifacts, or approved storage and link from docs.
4. Run `make ci` locally before requesting review.
5. Use a Conventional Commit title, such as `fix(search): guard empty profile`.
6. Open a pull request to `main` and fill out the PR template.

By submitting a pull request, you agree that your contribution is licensed under
the project's [Apache-2.0](LICENSE) license.

## Reporting security issues

**Do not** open a public issue for security vulnerabilities. Follow the private
process in [SECURITY.md](SECURITY.md).

## Code of Conduct

This project and everyone participating in it is governed by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you
are expected to uphold it. Report unacceptable behavior to evermind@shanda.com.

## Questions

- [GitHub Discussions](https://github.com/EverMind-AI/EverOS/discussions) — general Q&A
- [Discord](https://discord.gg/gYep5nQRZJ) — community chat
- Email: evermind@shanda.com

---

## For maintainers (core team)

The workflow below is for core-team members with write access. **You do not need
any of this to file an issue** — it documents how the team develops EverOS
internally.

### Prerequisites

- **Python 3.12+**
- [`uv`](https://docs.astral.sh/uv/) package manager
- Git

> No Docker / database services required — EverOS is lightweight (Markdown +
> SQLite + LanceDB embedded).

### Setup

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
make install             # deps + pre-commit hooks (one-stop dev setup)
everos init              # write ~/.everos/everos.toml + ome.toml
make ci                  # verify: lint + unit + integration + package
```

### Code style

Conventions are auto-loaded by Claude Code from [.claude/rules/](.claude/rules/).
Highlights:

- **Python 3.12+**, Ruff formatting (88-char line)
- **Absolute imports** only
- **English only** in code / comments / docstrings (no CJK — see
  [.claude/rules/language-policy.md](.claude/rules/language-policy.md))
- **Type hints** required on signatures; Pydantic v2 for data models
- **`__init__.py`** in every package; subpackages re-export public API via
  `from .x import Y as Y` + `__all__`
- **DDD layered**: `entrypoints → service → memory → infra`, single direction,
  enforced by `import-linter`

```bash
make format    # ruff fix + format
make lint      # ruff check + import-linter + hard repo hygiene gates
```

### Branch strategy

| Branch | Role |
|---|---|
| `main` | Default and protected branch |
| `feat/<scope>-<desc>` | Feature work |
| `fix/<scope>-<desc>` | Bug fixes |
| `docs/<scope>-<desc>` | Documentation-only changes |
| `ci/<scope>-<desc>` | CI, build, and developer-experience changes |

Do not push directly to `main`. Do not force-push shared branches. Merge through
PRs after required checks pass.

### Commit messages

**[Conventional Commits](https://www.conventionalcommits.org)**: `<type>[(scope)]: <description>`
(e.g. `feat: add agentic rerank`, `fix(search): guard empty profile`). Enforced
locally by `gitlint` in the `commit-msg` hook and in GitHub Actions by the
`Commit lint` workflow. Use `/commit` for guided generation; full type list:
[.claude/skills/commit/SKILL.md](.claude/skills/commit/SKILL.md).

### Testing

```bash
make test          # tests/unit
make integration   # tests/integration
make cov           # coverage report
```

- Add unit tests for new functions (`tests/unit/test_<module>/test_<action>_<expected>.py`)
- Add integration or e2e coverage for behavior changes (`tests/integration/`, `tests/e2e/`)

Full conventions: [.claude/rules/testing.md](.claude/rules/testing.md).

### Slash commands (Claude Code)

- `/new-branch` — create branch with proper naming
- `/commit` — generate a Conventional Commits message
- `/pr` — open a GitHub pull request with the correct target branch

---

Thank you for helping make EverOS better! 🎉
