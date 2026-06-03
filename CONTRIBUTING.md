# Contributing to EverOS

Thanks for your interest in EverOS! This page explains how contribution works
on this project.

## How EverOS accepts contributions

EverOS follows an **"open source, not open contribution"** model (similar to
SQLite). The codebase is developed and maintained by the EverMind core team, and
we **do not merge external pull requests**. This keeps copyright provenance
clean and the architecture coherent.

What we actively welcome from the community:

| Type | Where |
|---|---|
| 🐛 Bug reports | [Open a bug issue](https://github.com/EverMind-AI/everos/issues/new?template=bug_report.md) |
| 💡 Feature ideas / use cases | [Open a feature issue](https://github.com/EverMind-AI/everos/issues/new?template=feature_request.md) |
| 🔧 Suggested fixes | An issue with a code snippet / patch attached (see below) |
| ❓ Questions & discussion | [GitHub Discussions](https://github.com/EverMind-AI/everos/discussions) / [Discord](https://discord.gg/pfwwskxp) |

> **Pull requests opened against this repository will be closed** with a pointer
> to this policy. Please open an issue instead — it is the fastest path to
> getting a change in.

## Reporting a bug

Use the [bug report template](https://github.com/EverMind-AI/everos/issues/new?template=bug_report.md). Include:

- Clear reproduction steps
- Expected vs. actual behavior
- Environment (OS, Python version, everos version)
- Relevant logs (**with secrets redacted**)

## Suggesting a feature

Use the [feature request template](https://github.com/EverMind-AI/everos/issues/new?template=feature_request.md). Provide:

- The use case / problem being solved
- Proposed API or behavior
- Backward-compatibility considerations

## Suggesting a fix (code welcome)

Found the bug *and* the fix? Great — paste a minimal patch or code snippet
**in the issue**. Treat it as a proposal: the core team will review it, adapt it
to the project's conventions, and land the actual commit (crediting you in the
commit message / changelog).

> By posting a code suggestion in an issue, you agree it may be incorporated into
> EverOS under the project's [Apache-2.0](LICENSE) license.

## Reporting security issues

**Do not** open a public issue for security vulnerabilities. Follow the private
process in [SECURITY.md](SECURITY.md).

## Code of Conduct

This project and everyone participating in it is governed by the
[Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you
are expected to uphold it. Report unacceptable behavior to evermind@shanda.com.

## Questions

- [GitHub Discussions](https://github.com/EverMind-AI/everos/discussions) — general Q&A
- [Discord](https://discord.gg/pfwwskxp) — community chat
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
git clone <repo-url>
cd everos
make install             # deps + pre-commit hooks (one-stop dev setup)
everos init      # set EVEROS_LLM__API_KEY (OpenAI-protocol)
make ci                   # verify
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
make lint      # ruff check + import-linter
```

### Branch strategy (GitFlow Lite)

| Branch | Role |
|---|---|
| `master` | Released stable |
| `dev` | Default integration branch |
| `feat/<scope>-<desc>` | New features (from dev → dev) |
| `fix/<scope>-<desc>` | Bug fixes (from dev → dev) |
| `hotfix/<scope>-<desc>` | Emergency fixes (from master → master + dev) |

Full rationale: [.claude/skills/new-branch/SKILL.md](.claude/skills/new-branch/SKILL.md).

### Commit messages

**Gitmoji** format: `<emoji> <type>: <description>`. Use `/commit` for guided
generation. Full table: [.claude/skills/commit/SKILL.md](.claude/skills/commit/SKILL.md).

### Testing

```bash
make test          # tests/unit
make integration   # tests/integration
make cov           # coverage report
```

- Add unit tests for new functions (`tests/unit/test_<module>/test_<action>_<expected>.py`)
- Add golden fixtures for behavior changes (`tests/golden/`)

Full conventions: [.claude/rules/testing.md](.claude/rules/testing.md).

### Slash commands (Claude Code)

- `/new-branch` — create branch with proper naming
- `/commit` — generate Gitmoji commit message
- `/pr` — internal merge request with correct target branch

---

Thank you for helping make EverOS better! 🎉
