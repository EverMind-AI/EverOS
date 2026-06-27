# GitHub Sync Guard

This page records the GitHub-only files that must be preserved when refreshing
this public repository from the internal GitLab source/export.

## Rule

When a GitLab export, archive, rsync, or scripted copy is used to refresh the
GitHub repository, do **not** blindly overwrite GitHub-only contributor,
automation, and public-facing workflow files.

These files describe the public GitHub workflow: protected `main`, scoped
branches, GitHub pull requests, GitHub Actions, and community contribution
rules. Internal workflow text can mislead contributors and AI coding tools if
it is copied into GitHub.

## Preserve These GitHub Files

Keep the GitHub versions of these files unless a maintainer deliberately updates
them for the public GitHub workflow:

- `CLAUDE.md`
- `CONTRIBUTING.md`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/BRANCH_PROTECTION.md`
- `.github/workflows/ci.yml`
- `.github/workflows/docs.yml`
- `.github/workflows/commits.yml`
- `.github/ISSUE_TEMPLATE/**`
- `.claude/skills/commit/SKILL.md`
- `.claude/skills/new-branch/SKILL.md`
- `.claude/skills/pr/SKILL.md`
- `scripts/check_github_contributor_docs.py`
- `tests/unit/test_scripts/test_check_github_contributor_docs.py`

## Safe Sync Pattern

Prefer an explicit exclude list in the sync job rather than manual cleanup after
the overwrite. Example shape:

```bash
rsync -a --delete \
  --exclude 'CLAUDE.md' \
  --exclude 'CONTRIBUTING.md' \
  --exclude '.github/PULL_REQUEST_TEMPLATE.md' \
  --exclude '.github/BRANCH_PROTECTION.md' \
  --exclude '.github/workflows/ci.yml' \
  --exclude '.github/workflows/docs.yml' \
  --exclude '.github/workflows/commits.yml' \
  --exclude '.github/ISSUE_TEMPLATE/' \
  --exclude '.claude/skills/commit/SKILL.md' \
  --exclude '.claude/skills/new-branch/SKILL.md' \
  --exclude '.claude/skills/pr/SKILL.md' \
  --exclude 'scripts/check_github_contributor_docs.py' \
  --exclude 'tests/unit/test_scripts/test_check_github_contributor_docs.py' \
  <gitlab-export>/ <github-checkout>/
```

After any sync, run:

```bash
make docs-check
make lint
```

The `check_github_contributor_docs.py` gate catches the known failure mode:
public contributor docs drifting back to an internal branch/review model.

## Review Checklist

Before opening a sync PR:

- Confirm `CLAUDE.md` says branches are created from `main`.
- Confirm `.claude/skills/pr/SKILL.md` creates PRs with `--base main`.
- Confirm `CONTRIBUTING.md` says GitHub pull request, not internal review
  terminology.
- Confirm `make docs-check` and `make lint` both pass.
