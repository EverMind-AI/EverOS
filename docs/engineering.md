# Engineering & Dev-Efficiency Infrastructure

> Companions: business architecture lives in [architecture.md](architecture.md);
> hard coding constraints live in [../.claude/rules/](../.claude/rules/).
> This document covers the surrounding tooling, configuration, and processes
> — what we adopted, what role each piece plays, and how they fit together.

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
│   │   ├── skills/ (5)        slash command workflows          │    │
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
│   │     └ gitlint (commit-msg stage)                           │    │
│   │                                                            │    │
│   │   ruff                lint + format                        │    │
│   │                       (replaces black / isort / flake8)    │    │
│   │   import-linter       DDD layer-direction enforcement      │    │
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
│   │   config/default.toml  default settings (shipped)           │    │
│   │                                                            │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   ┌─ Dual-platform CI/CD ──────────────────────────────────────┐    │
│   │                                                            │    │
│   │   Primary:  GitLab CI       .gitlab-ci.yml                 │    │
│   │   Mirror:   GitHub Actions  .github/workflows/ci.yml       │    │
│   │   Both invoke Makefile targets; the Makefile is the        │    │
│   │   single source of truth for commands.                     │    │
│   │                                                            │    │
│   └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│   ┌─ Collaboration workflow ───────────────────────────────────┐    │
│   │                                                            │    │
│   │   Branch model: dev / master (GitFlow Lite)                │    │
│   │   PR / MR templates: same template across platforms        │    │
│   │   CODEOWNERS: by DDD layer ownership                       │    │
│   │   ISSUE_TEMPLATE: bug / feature / config                   │    │
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

### 3.3 Skills (5 slash commands)

| Command | Purpose | When to use |
|---|---|---|
| `/commit` | Generate Gitmoji-format commit message | After a focused change, ready to commit |
| `/new-branch` | Create branch under dev/master strategy | Starting a new feat / fix / hotfix |
| `/pr` | Create GitLab MR or GitHub PR with template | Ready to merge |
| `/add-memory-kind` | Scaffold a new business memory kind end-to-end | Adding a new memory type (md + sqlite + lancedb) |
| `/release` | Cut a PyPI release (rc / stable) | Shipping a version |

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
     ├ check-merge-conflict
     ├ detect-private-key
     └ gitlint  (commit-msg stage; rejects malformed messages)

     │
     ▼
Stage 3: local `make ci` (manual, before push)
     ├ make lint        (ruff + format-check + import-linter + datetime + openapi-drift)
     ├ make test        (pytest tests/unit)
     └ make integration (pytest tests/integration)

     │
     ▼
Stage 4: CI (PR triggered, GitLab + GitHub)
     └ re-runs the same `make lint / test / integration` targets

     │
     ▼
Stage 5: PR / MR review
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
[tool.coverage.run]                # coverage (unit+integration; make cov gates at 80%)
[tool.importlinter]                # dependency direction

[dependency-groups]
dev = ["ruff", "pytest", "pytest-asyncio", "pytest-cov",
       "pytest-rerunfailures", "import-linter", "pre-commit",
       "pyinstrument", "ipdb"]
```

**Single-file principle**: configuration that used to live in `pylintrc`,
`pytest.ini`, `.isort.cfg` is **all consolidated into `pyproject.toml`**.

### 5.2 Makefile commands

```
make help          list all targets
make install       uv sync --frozen
make format        ruff fix + format
make lint          ruff + format --check + import-linter + datetime + openapi-drift
make test          pytest tests/unit
make integration   pytest tests/integration
make cov           pytest unit+integration with coverage, gate at 80%
make ci            lint + test + integration   ← CI invokes these targets
make clean         clear caches
```

> Plus `install-deps` (CI's `uv sync --frozen`), `openapi` (regenerate
> `docs/openapi.json`), `check-openapi` / `check-datetime` (the lint
> sub-gates), and `check-cjk` (advisory) — see the `Makefile` for the
> full list.

**Single source of truth**: CI configuration only invokes `make <target>`,
preventing drift between GitHub and GitLab. Local and CI run identical
commands.

### 5.3 Configuration model

Settings are loaded in ascending priority:

1. `config/default.toml` (shipped with the package; lowest priority)
2. `<memory-root>/everos.toml` (user config; optional)
3. `EVEROS_*` environment variables (highest priority)

`everos init [--root PATH]` generates starter config files
(`everos.toml` + `ome.toml`) in the memory root. `everos.toml` holds
all config (API keys, model, storage); override individual fields with
`EVEROS_*` env vars. Inspect the effective config with
`everos config show [--root PATH]`.

Key `EVEROS_*` env vars:

```
EVEROS_LLM__MODEL          # model name (provider-agnostic)
EVEROS_LLM__API_KEY        # any OpenAI-protocol API key
EVEROS_LLM__BASE_URL       # optional: custom endpoint (Ollama bridge etc.)
EVEROS_ROOT                # memory-root (default ~/.everos)
EVEROS_LOG_LEVEL
TZ
```

---

## 6. Dual-platform CI/CD

### 6.1 Dual-platform strategy

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   Primary: GitLab CI         (.gitlab-ci.yml)            │
│     ├ internal team dev          stages: lint / test     │
│     ├ MR triggered                                       │
│     └ uv cache (keyed by uv.lock)                        │
│                                                          │
│   Mirror: GitHub Actions     (.github/workflows/ci.yml)  │
│     ├ public OSS mirror          same make targets       │
│     ├ push + PR triggered                                │
│     └ astral-sh/setup-uv@v3                              │
│                                                          │
│   Consistency:                                           │
│     ├ Makefile is the single source of CI commands       │
│     └ pre-commit runs locally first to reduce CI churn   │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 6.2 CI checklist

| Check | Tool | Platform | Failure condition |
|---|---|---|---|
| Lint | `make lint` (ruff + format-check + import-linter + datetime + openapi-drift) | both | any error |
| Layer direction | `make lint` (lint-imports inside) | both | layer violation |
| Unit | `make test` (pytest tests/unit) | both | any failure |
| Integration | `make integration` (pytest tests/integration) | both | any failure (PR + master/dev push only) |

Commit message format is enforced **locally** via `gitlint` in the
`commit-msg` pre-commit stage; it does not run in CI.

### 6.3 Branch protection

| Branch | GitLab rule | GitHub rule |
|---|---|---|
| **master** | no direct push; MR + 1 approval + green pipeline | branch protection + 1 review + status checks |
| **dev** | same as above | same as above |
| feat / fix / hotfix | free push; rebase parent before merge | same |

---

## 7. Collaboration workflow

### 7.1 Branch model (GitFlow Lite)

```
                              v0.1                              v0.2                                v1.0
                                ▲                                 ▲                                   ▲
                                │ release PR                      │ release PR                        │ release PR
                                │ (dev→master+tag)                │ (dev→master+tag)                  │ (dev→master+tag)
master   ●──────────────────────●─────────────●──────────────────●──────────────────────────────────●────►  stable / released
                                │             ▲                  │                                  │
                                │             │ merge hotfix     │                                  │
                                │             │                  │                                  │
                                │       ●──●──┘                  │                                  │
                                │       │ hotfix branch          │                                  │
                                │       │ (cut from master)      │                                  │
                                │       │                        │                                  │
                                │       ▼ sync to dev            │                                  │
                                │       │                        │                                  │
dev   ●──●──●──●──●──●──●──●──●─●──●──●─●──●──●──●──●──●──●──●──●─●──●──●──●──●──●──●──●──●──●──●──●─────►  integration
            ▲                   ↑                                ↑                                  ↑
            │             release point                   release point                       release point
       feat/A             (dev HEAD →                     (dev HEAD →                         (dev HEAD →
       ●──●──●             master + v0.1)                  master + v0.2)                      master + v1.0)


  feat/*   : cut from dev → PR → merge into dev
  hotfix/* : cut from master → merge into master + sync into dev (double merge)
  release  : dev → master + tag on master (no separate release branch)

  Vertical │ in the diagram = "dev HEAD merged into master via release PR + v0.x tag"
```

Details in [../.claude/skills/new-branch/SKILL.md](../.claude/skills/new-branch/SKILL.md).

### 7.2 PR / MR template (shared across platforms)

Six sections: changes / target branch / scope / API impact / tests /
checklist.

File locations:

- GitLab: `.gitlab/merge_request_templates/default.md`
- GitHub: `.github/PULL_REQUEST_TEMPLATE.md`

### 7.3 CODEOWNERS (by DDD layer)

```
/src/everos/memory/         @chandler.zhang @libin.zhang001
/src/everos/infra/          @chandler.zhang @yeanhua
/src/everos/component/      @chandler.zhang
/src/everos/core/           @chandler.zhang
/src/everos/service/        @chandler.zhang @libin.zhang001
/src/everos/entrypoints/    @chandler.zhang
/.claude/                   @chandler.zhang
/.gitlab-ci.yml             @chandler.zhang @jianhua.yao
```

At least one owner per directory; two owners for critical modules. Edits
auto-mention the corresponding owners.

### 7.4 Commit convention (Gitmoji)

```
✨ feat: new feature
🐛 fix: bug fix
♻️ refactor: refactoring (no behavior change)
✅ test: add / update tests
📝 docs: documentation
🎨 style: formatting
⚡️ perf: performance optimization
🔧 chore: configuration / build
🚧 wip: work in progress (must not land on master)
```

`gitlint` enforces format **locally** (commit-msg pre-commit stage). See
[../.claude/skills/commit/SKILL.md](../.claude/skills/commit/SKILL.md).

---

## 8. Issue templates / user support

```
.github/ISSUE_TEMPLATE/
├── bug_report.md            software deps: lancedb / sqlite / ruff
├── feature_request.md       generic template
└── config.yml               disable blank issue + Discord / Discussions links

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
│  Team skills         │  /.claude/skills/ (5)                │  no slash   │
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
│  default config      │  /src/everos/config/default.toml    │  newcomers  │
│                      │                                      │  lost on cfg│
├─────────────────────┼──────────────────────────────────────┼─────────────┤
│  GitLab CI           │  /.gitlab-ci.yml                     │  MR cannot  │
│                      │                                      │  merge      │
│  GitHub Actions      │  /.github/workflows/ci.yml           │  PR cannot  │
│                      │                                      │  merge      │
│  CODEOWNERS          │  /.gitlab/CODEOWNERS                 │  no auto    │
│                      │                                      │  reviewer   │
│  GitLab MR template  │  /.gitlab/merge_request_templates/   │  no MR temp │
│  GitHub PR template  │  /.github/PULL_REQUEST_TEMPLATE.md   │  no PR temp │
│  Issue templates     │  /.github/ISSUE_TEMPLATE/ (3)        │  scattered  │
│  CONTRIBUTING        │  /CONTRIBUTING.md                    │  contrib.   │
│                      │                                      │  confused   │
└─────────────────────┴──────────────────────────────────────┴─────────────┘
```

---

## 10. Future extensions

```
Near-term
  ☑ Coverage threshold — `make cov` now gates at 80% (unit + integration)
  □ /new-module    skill: scaffold a subpackage that complies with rules
  □ /run-eval      skill: run behavior-consistency eval
  □ ruff rule sets: add D (docstring), ANN (annotations)

Mid-term (before v0.5)
  □ Type checking re-introduction (pyright or mypy) once hot paths stabilize
  □ release-please / Conventional Commits → automated changelog
  □ pre-commit autoupdate cadence
  □ Performance benchmark CI with historical comparison

Long-term (after v1.0)
  □ /security-review  skill: automated security review
  □ Mutation testing (mutmut)
  □ Multi-Python version matrix (3.12 / 3.13)
  □ Automated PyPI wheel upload
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
│                         collaboration (branch + PR +      │
│                                        CODEOWNERS) +      │
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
| Commit format | freeform | gitlint pre-commit hook (Gitmoji) |
| Claude Code integration | partial rules | rules + skills + settings (full) |
| CI platform | GitLab only | GitLab + GitHub mirror, both calling Makefile |
| Tests | basic | unit + integration + golden + coverage report |

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
- Gitmoji: [gitmoji.dev](https://gitmoji.dev/)
- GitLab CI: [docs.gitlab.com/ee/ci](https://docs.gitlab.com/ee/ci/)
- GitHub Actions: [docs.github.com/en/actions](https://docs.github.com/en/actions)
- CODEOWNERS: [docs.gitlab.com/ee/user/project/codeowners](https://docs.gitlab.com/ee/user/project/codeowners/)
