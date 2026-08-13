# GitHub Sync Guard

This page records the rules for refreshing the public GitHub repository
(EverMind-AI/EverOS) from the internal GitLab source.

## Core Principle

GitLab `dev` is the source of truth. GitHub `main` is the public release
mirror. The two repos have **no shared commit history** — GitHub receives
packaged exports, not git pushes.

---

## 1. Must Delete on GitHub

These files exist on GitHub but are obsolete — delete them during sync:

| File | Reason |
|---|---|
| `docs/locomo_benchmark.md` | Replaced by `benchmarks/README.md` |
| `tests/run_locomo_10x3.sh` | Replaced by `benchmarks/run.py` |
| `tests/run_locomo_batch.sh` | Replaced by `benchmarks/run.py` |
| `tests/run_locomo_full.sh` | Replaced by `benchmarks/run.py` |
| `tests/test_locomo.py` | Replaced by `benchmarks/run.py` |
| `src/everos/memory/strategies/_partition_locks.py` | Wrong path; correct location is `src/everos/memory/_partition_locks.py` |

## 2. Must Preserve on GitHub (Do NOT Overwrite)

These files have GitHub-specific content (branch model, PR workflow,
community-facing text). Keep the GitHub versions:

### Workflow & CI

- `CLAUDE.md` — says `main` branch, not `dev`/`master`
- `CONTRIBUTING.md` — says "curated PR contributions", not "no external PRs"
- `.github/PULL_REQUEST_TEMPLATE.md` — GitHub PR template
- `.github/BRANCH_PROTECTION.md` — GitHub branch protection rules
- `.github/workflows/ci.yml` — targets `main`, uses `actions/checkout@v6`
- `.github/workflows/docs.yml` — GitHub-only doc checks
- `.github/workflows/commits.yml` — GitHub-only commit lint
- `.github/ISSUE_TEMPLATE/**` — `.yml` format (GitLab uses `.md`)
- `.github/dependabot.yml` — GitHub-only dependency scanning

### Claude Code rules & skills

- `.claude/rules/*.md` — GitHub versions are simplified for contributors
- `.claude/skills/commit/SKILL.md` — Conventional Commits (vs Gitmoji)
- `.claude/skills/new-branch/SKILL.md` — branches from `main` (vs `dev`)
- `.claude/skills/pr/SKILL.md` — GitHub PR (vs GitLab MR)
- `.claude/settings.json` — GitHub permission set (includes `gh pr`)

### GitHub-only scripts & tests

- `scripts/check_commit_messages.py`
- `scripts/check_deprecated_names.py`
- `scripts/check_docs.py`
- `scripts/check_github_contributor_docs.py`
- `scripts/check_pr_title.py`
- `scripts/check_repo_assets.py`
- `tests/unit/test_scripts/test_check_*.py`

### Root docs (GitHub version has different content)

- `README.md` — GitHub version has banner, badges, use-case gallery
- `Makefile` — GitHub version has extra targets (docs-check, package, etc.)

## 3. Overwrite from GitLab

Everything not in sections 1 or 2 should be overwritten from GitLab,
including but not limited to:

- All `src/everos/**` source code
- All `tests/unit/**` and `tests/integration/**` and `tests/e2e/**`
- `pyproject.toml`, `uv.lock`
- `QUICKSTART.md`, `SECURITY.md`
- `CHANGELOG.md`, `CITATION.md`, `ACKNOWLEDGMENTS.md`
- `docs/*.md` (except those listed in Must Delete)
- `docs/openapi.json`
- `benchmarks/**`
- `.gitignore`
- `.gitlab-ci.yml` (harmless on GitHub, good to keep in sync)

## 4. Do NOT Sync to GitHub

These files are GitLab-internal and should not appear on GitHub:

| File | Reason |
|---|---|
| `.gitlab/merge_request_templates/` | GitLab-only |
| `.vscode/` | IDE preference |
| `.claude/skills/release/SKILL.md` | Internal release workflow |
| `local/` | Temporary design docs |
| `evaluation/` | Empty module |

## 5. Do NOT Sync to GitLab

These files are GitHub-only and should stay only on GitHub:

| Category | Files |
|---|---|
| use-cases/ | 90 files (claude-code-plugin, game-of-throne-demo, openher) |
| GitHub CI scripts | `scripts/check_*.py` + tests |
| GitHub CI workflows | `commits.yml`, `docs.yml` |
| Config | `.gitlint`, `.env.example` (root level) |
| Docs | `docs/use-cases.md` |

## Safe Sync Pattern

```bash
# 1. Clone both repos
git clone <gitlab> gitlab-export
git clone <github> github-checkout

# 2. Sync with excludes
rsync -a --delete \
  --exclude '.git' \
  --exclude 'CLAUDE.md' \
  --exclude 'CONTRIBUTING.md' \
  --exclude 'README.md' \
  --exclude 'Makefile' \
  --exclude '.github/' \
  --exclude '.claude/rules/' \
  --exclude '.claude/skills/' \
  --exclude '.claude/settings.json' \
  --exclude 'scripts/check_*.py' \
  --exclude 'tests/unit/test_scripts/' \
  --exclude 'use-cases/' \
  --exclude 'local/' \
  --exclude 'evaluation/' \
  --exclude '.vscode/' \
  --exclude '.gitlab/' \
  --exclude '.gitlab-ci.yml' \
  gitlab-export/ github-checkout/

# 3. Delete obsolete files
cd github-checkout
rm -f docs/locomo_benchmark.md
rm -f tests/run_locomo_10x3.sh tests/run_locomo_batch.sh tests/run_locomo_full.sh
rm -f tests/test_locomo.py
rm -f src/everos/memory/strategies/_partition_locks.py

# 4. Verify
make lint
make test
```

## Review Checklist

Before opening the sync PR on GitHub:

- [ ] `CLAUDE.md` says branches are created from `main`
- [ ] `.claude/skills/pr/SKILL.md` creates PRs with `--base main`
- [ ] `CONTRIBUTING.md` says "GitHub pull request", not internal terminology
- [ ] Obsolete files from section 1 are deleted
- [ ] `make lint` and `make test` pass
- [ ] No `.gitlab-ci.yml`, `.vscode/`, `local/`, `evaluation/` leaked in
