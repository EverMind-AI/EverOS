# Engineering & Dev-Efficiency Infrastructure

> Companions: business architecture lives in [architecture.md](architecture.md);
> hard coding constraints live in [../.claude/rules/](../.claude/rules/).
> This document covers the surrounding tooling, configuration, and processes
> — what we adopted, what role each piece plays, and how they fit together.
> CI runs on GitHub Actions; all checks are invoked through the `Makefile`.

---

## 1. Scope

Engineering / dev-efficiency infrastructure does not solve business problems —
it solves **team + code + time** problems:

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   Business architecture (docs/architecture.md)           │
│      — answers "how to build the system"                 │
│                                                          │
│   Engineering rules (.claude/rules/)                     │
│      — answers "how to write the code"                   │
│                                                          │
│   Engineering / dev-efficiency infrastructure (this doc) │
│      — answers "how the team collaborates,               │
│         how code is auto-checked,                        │
│         how releases are automated,                      │
│         how tools land in the project"                   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Reasons this is documented separately:

- **Cross-project reusable** — `CLAUDE.md` / rules / `pyproject.toml` are
  patterns, not content. The next project can adopt them as-is.
- **Decoupled from business** — business architecture changes do not affect
  these; upgrading these does not affect business.
- **Onboarding-oriented** — new contributors read this first to understand
  what the tooling looks like.

---

## 2. Infrastructure overview

```
┌─────────────────────────────────────────────────────────────────────┐
│            Team collaboration / Code quality / CI/CD                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─ Claude Code engineering layer ────────────────────────────┐    │
│   │                                                            │    │
│   │   CLAUDE.md  ←  team-shared context (auto loaded into     │    │
│   │                 system prompt)                             │    │
│   │   .claude/                                                 │    │
│   │   ├── CLAUDE.md          subdir context (optional)        │    │
│   │   ├── rules/  (10)       path-scoped hard coding rules    │    │
│   │   ├── skills/ (3)        slash command workflows          │    │
│   │   └── settings.json      permissions allowlist            │    │
│   │                                                            │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   ┌─ Code quality gates ───────────────────────────────────────┐    │
│   │                                                            │    │
│   │   pre-commit          runs locally before commit           │    │
│   │     ├ ruff (lint+fmt)                                      │    │
│   │     ├ trailing-whitespace / end-of-file-fixer              │    │
│   │     ├ check-yaml / check-toml                              │    │
│   │     ├ check-added-large-files (≥1MB warn)                  │    │
│   │     ├ detect-private-key                                   │    │
│   │     ├ no committed images/videos/assets                    │    │
│   │     └ gitlint (commit-msg stage)                           │    │
│   │                                                            │    │
│   │   ruff                lint + format                        │    │
│   │                       (replaces black / isort / flake8)    │    │
│   │   import-linter       DDD layer-direction enforcement      │    │
│   │   repo asset gate     blocks images/videos/assets in git   │    │
│   │   pytest              unit / integration                   │    │
│   │                                                            │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   ┌─ Dependencies & build ─────────────────────────────────────┐    │
│   │                                                            │    │
│   │   uv                  sole package manager                 │    │
│   │                       (no `pip install`)                   │    │
│   │   pyproject.toml      src layout + extras + groups         │    │
│   │   uv.lock             checked in; CI uses --frozen         │    │
│   │   hatchling           wheel build backend                  │    │
│   │   Makefile            unified entry; CI calls it           │    │
│   │   src/everos/templates/env.template                       │    │
│   │                       environment variable template        │    │
│   │                                                            │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   ┌─ CI/CD (GitHub Actions) ───────────────────────────────────┐    │
│   │                                                            │    │
│   │   CI:    .github/workflows/ci.yml    lint / test / integ   │    │
│   │                                      / package build        │    │
│   │   Docs:  .github/workflows/docs.yml  Markdown + YAML check │    │
│   │   Gates invoke Makefile targets; the Makefile is the       │    │
│   │   single source of truth for commands.                     │    │
│   │                                                            │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   ┌─ Collaboration workflow ───────────────────────────────────┐    │
│   │                                                            │    │
│   │   Branch model: protected main + short-lived PR branches   │    │
│   │   PR template: .github/PULL_REQUEST_TEMPLATE.md            │    │
│   │   ISSUE_TEMPLATE: bug / feature / use-case / docs / config │    │
│   │   CONTRIBUTING.md: contributor onboarding                  │    │
│   │                                                            │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Claude Code engineering layer

### 3.1 Loading mechanism

Claude Code automatically loads the following into the system prompt at
session start (no manual import):

```
┌────────────────────────┬──────────────────────────────────────────┐
│  File                   │  Purpose                                 │
├────────────────────────┼──────────────────────────────────────────┤
│  CLAUDE.md (repo root)  │  Team-shared context: architecture       │
│                         │  overview, commands, convention index    │
│  .claude/rules/*.md     │  Hard coding constraints                 │
│                         │  (path-scoped on-demand load)            │
│  .claude/settings.json  │  Permissions allowlist (not in prompt)   │
│  ~/.claude/CLAUDE.md    │  User-level (personal preferences)       │
│  CLAUDE.local.md        │  Project-local personal (gitignored)     │
└────────────────────────┴──────────────────────────────────────────┘
```

### 3.2 Rules (10 files, path-scoped)

| File | Paths (auto-load condition) |
|---|---|
| architecture.md | always loaded (no paths) |
| code-style.md | always loaded (no paths) |
| language-policy.md | always loaded (no paths) |
| imports.md | `src/**/*.py`, `tests/**/*.py` |
| init-py-and-reexport.md | `src/**/__init__.py`, `src/**/*.py` |
| module-docstring.md | `src/{infra,memory,service,component,core}/**/*.py` |
| async-programming.md | `src/**/*.py`, `tests/**/*.py` |
| datetime-handling.md | `src/**/*.py`, `tests/**/*.py` |
| logging-observability.md | `src/**/*.py` |
| testing.md | `tests/**/*.py` |

**Why path-scoped**: avoid loading 1000+ lines of rules every session
(~5–8K tokens). At startup only architecture + code-style + language-policy
load (~1.5–2K tokens); the rest load on demand when Claude Code reads a
matching `.py` file.

### 3.3 Skills (3 slash commands)

| Command | Purpose | When to use |
|---|---|---|
| `/commit` | Generate a Conventional Commits message | After a focused change, ready to commit |
| `/new-branch` | Create branch from protected main | Starting a new feat / fix / ci branch |
| `/pr` | Open a GitHub PR with the repo template | Ready to merge |

Skills and rules use **independent loading mechanisms**: rules auto-load
into the system prompt, skills only trigger when the user types `/<name>`.

### 3.4 settings.json

```json
{
  "permissions": {
    "allow": ["Bash(uv sync*)", "Bash(make*)", "Bash(uv run pytest*)", ...]
  }
}
```

**Purpose**: reduce permission prompts. Team-shared config goes into
`settings.json` (in git); personal preferences go into `settings.local.json`
(gitignored).

---

## 4. Code quality gates

```
        ┌──────────────────────────────────────────────────────┐
        │     Each stage can independently fail the change      │
        └──────────────────────────────────────────────────────┘

[Local editor]
     │
     ▼
Stage 1: editor real-time feedback
     ├ ruff (lint + format) on save
     └ path-relevant .claude/rules guide Claude Code

     │
     ▼
Stage 2: pre-commit (triggered by `git commit`)
     ├ ruff fix + format
     ├ trailing-whitespace, end-of-file-fixer
     ├ check-yaml, check-toml
     ├ check-added-large-files (≥1MB)
     ├ detect-private-key
     ├ no-repo-assets (rejects images/videos/assets in git)
     └ gitlint  (commit-msg stage; rejects malformed messages)

     │
     ▼
Stage 3: local `make ci` (manual, before push)
     ├ make lint        (ruff + import-linter + repo hygiene gates)
     ├ make test        (pytest tests/unit)
     ├ make integration (pytest tests/integration)
     └ make package     (sdist/wheel build + import smoke test)

     │
     ▼
Stage 4: CI (GitHub Actions, push + PR triggered)
     └ re-runs the same `make lint / test / integration / package` targets

     │
     ▼
Stage 5: PR review
     ├ ≥ 1 approval
     └ all threads resolved + all CI green
```

**Key design**: when any stage fails, **never merge** — there is no
`--no-verify` / `--allow-failure` escape hatch.

---

## 5. Dependencies & build

### 5.1 pyproject.toml overview

```toml
[project]
name = "everos"
requires-python = ">=3.12"
dependencies = [...]               # runtime deps (minimal set)

[project.optional-dependencies]
multimodal = [...]                 # extras (install on demand)

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/everos"]          # src layout

[project.scripts]
everos = "everos.entrypoints.cli.main:app"  # exposes CLI command

[tool.ruff]                        # code style
[tool.pytest.ini_options]          # tests
[tool.coverage.run]                # coverage config (gate lives in `make cov`)
[tool.importlinter]                # dependency direction

[dependency-groups]
dev = ["ruff", "pytest", "pytest-asyncio", "pytest-cov",
       "import-linter", "pre-commit", "ipdb"]
```

**Single-file principle**: configuration that used to live in `pylintrc`,
`pytest.ini`, `.isort.cfg` is **all consolidated into `pyproject.toml`**.

### 5.2 Makefile commands

```
make help          list all targets
make install       uv sync --frozen
make format        ruff fix + format
make lint          ruff + import-linter + repo asset/media + datetime discipline + openapi drift
make test          pytest tests/unit
make integration   pytest tests/integration
make package       build sdist/wheel + smoke-test wheel import
make cov           pytest unit + integration, coverage gate (fail under 80%)
make ci            lint + test + integration + package
make clean         clear caches
```

**Single source of truth**: CI only invokes `make <target>`, so local and CI
run identical commands and cannot drift.

### 5.3 env.template (slimmed down)

The template lives at `src/everos/templates/env.template` (bundled
inside the wheel as package data, copied to `./.env` via `everos init`).
It groups settings by provider, each block sharing the OpenAI-protocol
`MODEL` / `API_KEY` / `BASE_URL` triple:

```
EVEROS_LLM__*           # text model (model / api_key / base_url)
EVEROS_MULTIMODAL__*    # vision model for image/office inputs
EVEROS_EMBEDDING__*     # embedding model (vector index)
EVEROS_RERANK__*        # cross-encoder reranker
EVEROS_MEMORY__ROOT     # memory-root (md files + .index/{sqlite,lancedb}/)
EVEROS_LOG_LEVEL        # DEBUG | INFO | WARNING | ERROR
EVEROS_LOG_FORMAT       # json | text
TZ                      # display timezone (storage is always UTC)
```

Every key has a sensible default except the `API_KEY` fields, which you fill in.

---

## 6. CI/CD (GitHub Actions)

### 6.1 Strategy

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   GitHub Actions   (.github/workflows/)                  │
│     ci.yml    push (main) + PR                           │
│       ├ lint              make lint                      │
│       ├ unit tests        make test                      │
│       ├ integration tests make integration               │
│       └ package build     make package                   │
│     docs.yml  Markdown link check + issue-template YAML  │
│       └ make docs-check                                  │
│     commits.yml Conventional Commit subject check        │
│       └ make check-commits                               │
│                                                          │
│   Consistency:                                           │
│     ├ astral-sh/setup-uv (cache keyed by uv.lock)        │
│     ├ Makefile is the single source of CI commands       │
│     └ pre-commit runs locally first to reduce CI churn   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 6.2 CI checklist

| Check | Tool | Failure condition |
|---|---|---|
| Lint | `make lint` (ruff check + ruff format --check) | any error |
| Layer direction | `make lint` (lint-imports inside) | layer violation |
| Repository media | `make lint` (check_repo_assets.py) | images/videos/assets committed |
| Datetime discipline | `make lint` (check_datetime_discipline.py) | bypasses helper module |
| OpenAPI drift | `make lint` (dump_openapi.py --check) | schema ≠ committed openapi.json |
| Unit | `make test` (pytest tests/unit) | any failure |
| Integration | `make integration` (pytest tests/integration) | any failure |
| Package build | `make package` (sdist/wheel + import smoke test) | build or import failure |
| Commit message | `Commit lint` workflow | non-Conventional Commit subject |

Integration tests run with a `FakeLLMClient` — no live credentials are needed in CI.
Commit message format is enforced locally via `gitlint` in the `commit-msg`
pre-commit stage and remotely via the `Commit lint` workflow.

### 6.3 Branch protection

| Branch | Rule |
|---|---|
| **main** | branch protection: PR + two reviews + green required checks; no direct push |
| feat / fix / docs / ci | contributor branches; merge through PR |

---

## 7. Collaboration workflow

### 7.1 Branch model

EverOS uses a simple protected-main model after the 1.0 history reset:

```
main  ●────●────●────●────► protected, releasable
       ▲    ▲    ▲
       │    │    └─ PR from ci/*
       │    └────── PR from fix/*
       └─────────── PR from feat/*
```

All work starts from `main`, lands through a pull request, and requires green
checks. Force-pushing `main` is reserved only for repository recovery work.

### 7.2 PR template

A single PR template at [`.github/PULL_REQUEST_TEMPLATE.md`](../.github/PULL_REQUEST_TEMPLATE.md)
with five sections: **Summary / Area / Verification / Checklist / Notes for
Reviewers**. The `/pr` skill fills it in (see
[../.claude/skills/pr/SKILL.md](../.claude/skills/pr/SKILL.md)).

### 7.3 Commit convention (Conventional Commits)

Format: `<type>[(scope)][!]: <description>` per
[Conventional Commits](https://www.conventionalcommits.org).

```
feat:     new feature
fix:      bug fix
refactor: restructuring (no behavior change)
test:     add / update tests
docs:     documentation
style:    formatting
perf:     performance optimization
chore:    configuration / build / tooling
build:    build system or dependencies
ci:       CI configuration
revert:   revert a previous commit
```

`gitlint` enforces the format locally via its `contrib-title-conventional-commits`
rule in the commit-msg pre-commit stage. GitHub Actions runs the same policy on
pushes to `main` and pull requests. See
[../.claude/skills/commit/SKILL.md](../.claude/skills/commit/SKILL.md).

---

## 8. Issue templates / user support

```
.github/ISSUE_TEMPLATE/
├── bug_report.yml           structured bug report (form)
├── feature_request.yml      feature proposal (form)
├── use_case.yml             share a use case / integration
├── docs.yml                 documentation issue
└── config.yml               disable blank issues + community links

CONTRIBUTING.md              contributor onboarding: setup / code style /
                             branch / commit / PR / testing
```

---

## 9. Infrastructure summary table

```
┌─────────────────────┬──────────────────────────────────────┬─────────────┐
│  Facility            │  Location / file                      │  Failure    │
│                      │                                       │  impact     │
├─────────────────────┼──────────────────────────────────────┼─────────────┤
│  CLAUDE.md           │  /CLAUDE.md                          │  cc loses   │
│                      │                                      │  context    │
│  Team rules          │  /.claude/rules/ (10)                │  cc unaware │
│                      │                                      │  of conv.   │
│  Team skills         │  /.claude/skills/ (3)                │  no slash   │
│                      │                                      │  workflows  │
│  Permissions         │  /.claude/settings.json              │  cc prompts │
│                      │                                      │  on each op │
├─────────────────────┼──────────────────────────────────────┼─────────────┤
│  pyproject           │  /pyproject.toml                     │  build fail │
│  Lock file           │  /uv.lock                            │  dep drift  │
│  Makefile            │  /Makefile                           │  no unified │
│                      │                                      │  entry      │
│  pre-commit          │  /.pre-commit-config.yaml            │  no local   │
│                      │                                      │  gate       │
│  env template        │  /src/everos/templates/env.template │  newcomers  │
│                      │                                      │  lost on env│
├─────────────────────┼──────────────────────────────────────┼─────────────┤
│  CI                  │  /.github/workflows/ci.yml           │  PR cannot  │
│                      │                                      │  merge      │
│  Docs CI             │  /.github/workflows/docs.yml         │  broken     │
│                      │                                      │  doc links  │
│  PR template         │  /.github/PULL_REQUEST_TEMPLATE.md   │  no PR temp │
│  Issue templates     │  /.github/ISSUE_TEMPLATE/ (5)        │  scattered  │
│  CONTRIBUTING        │  /CONTRIBUTING.md                    │  contrib.   │
│                      │                                      │  confused   │
└─────────────────────┴──────────────────────────────────────┴─────────────┘
```

---

## 10. Future extensions

```
Near-term
  □ /new-module    skill: scaffold a subpackage that complies with rules
  □ ruff rule sets: add D (docstring), ANN (annotations)
  □ Static type checking (pyright or mypy) once hot paths stabilize

Mid-term
  □ release-please / Conventional Commits → automated changelog
  □ Automated PyPI wheel upload on tag
  □ Multi-Python version matrix (3.12 / 3.13)
  □ Performance benchmark CI with historical comparison

Long-term
  □ Mutation testing (mutmut)
  □ Coverage ratchet (raise the 80% gate as the suite matures)
```

---

## 11. On investing in engineering infrastructure

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   Plain business code ≠ an engineering project            │
│                                                          │
│   Engineering project = business code +                   │
│                         coding rules +                    │
│                         quality gates (pre-commit + CI) + │
│                         automation (Makefile + skills) +  │
│                         collaboration (branch + PR) +     │
│                         knowledge base (CLAUDE.md +       │
│                                         rules + docs)     │
│                                                          │
│   The earlier this infrastructure lands, the faster and   │
│   farther the team can run.                               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Old project vs. new project after this rewrite:

| Dimension | Old project | New project |
|---|---|---|
| Lint tools | black + isort + pylint | ruff (single tool) |
| Config files | pyproject + pylintrc + pyrightconfig + pytest.ini | unified pyproject.toml |
| pre-commit | basic | adds gitlint commit-msg + import / yaml / private-key checks |
| Layer direction | not enforced | import-linter enforced in CI |
| Commit format | freeform | gitlint pre-commit hook (Conventional Commits) |
| Claude Code integration | partial rules | rules + skills + settings (full) |
| CI platform | ad hoc | GitHub Actions calling Makefile targets |
| Tests | basic | unit + integration + e2e + coverage report |

These are not perfectionism — they are baseline requirements for
**multi-person collaboration, long-term maintenance, and sustainable
evolution**.

---

## 12. References

- Hard coding rules: [../.claude/rules/](../.claude/rules/) (auto-loaded by Claude Code)
- Slash command workflows: [../.claude/skills/](../.claude/skills/)
- Contributor onboarding: [../CONTRIBUTING.md](../CONTRIBUTING.md)
- Architecture: [architecture.md](architecture.md)
- Claude Code memory mechanism: [code.claude.com/docs/en/memory.md](https://code.claude.com/docs/en/memory.md)
- Claude Code skills: [code.claude.com/docs/en/skills.md](https://code.claude.com/docs/en/skills.md)
- ruff: [docs.astral.sh/ruff](https://docs.astral.sh/ruff/)
- import-linter: [import-linter.readthedocs.io](https://import-linter.readthedocs.io/)
- gitlint: [jorisroovers.com/gitlint](https://jorisroovers.com/gitlint/)
- uv: [docs.astral.sh/uv](https://docs.astral.sh/uv/)
- pre-commit: [pre-commit.com](https://pre-commit.com/)
- Conventional Commits: [conventionalcommits.org](https://www.conventionalcommits.org/)
- GitHub Actions: [docs.github.com/en/actions](https://docs.github.com/en/actions)
