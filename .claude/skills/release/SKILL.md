---
name: release
description: Cut a versioned release and publish everos to PyPI via the tag-triggered workflow
---

# /release

Publish a new version of `everos` to PyPI. Publishing is automated: pushing a
`vX.Y.Z` tag triggers [.github/workflows/release.yml](../../../.github/workflows/release.yml),
which builds, smoke-tests, and uploads via PyPI **Trusted Publishing** (OIDC —
no stored token) behind the `release` environment's manual-approval gate.

## Preconditions

- On `main`, up to date, with green CI (the tag builds from `main`'s tree).
- Decide the version per SemVer: patch = fixes, minor = back-compatible
  features, major = breaking changes.

## Steps

```
1. Bump the version    → pyproject.toml [project] version = "X.Y.Z"
   (single source; everos.__version__ reads installed package metadata)
2. Update CHANGELOG.md → move the Unreleased entries under a new
   ## [X.Y.Z] - <date> heading
3. Commit              → git commit -m "chore(release): vX.Y.Z"
4. Open a PR, merge to main after green CI
5. Tag main + push     → git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
6. Approve             → the release.yml run pauses on the `release`
   environment; a reviewer approves in the Actions run
7. Verify              → https://pypi.org/project/everos/X.Y.Z/
```

The tag must equal the `pyproject.toml` version — the workflow refuses to
publish on a mismatch.

## Pre-releases

PEP 440 pre-release tags publish too (PyPI accepts them; `pip install everos`
ignores them unless `--pre`): `vX.Y.ZrcN`, `vX.Y.ZaN`, `vX.Y.ZbN`. Set the same
suffix in `pyproject.toml` version before tagging.

## One-time setup (project owner, not doable from CI)

1. **PyPI trusted publisher** — PyPI → project `everos` → Settings →
   Publishing → add: owner `EverMind-AI`, repo `EverOS`, workflow
   `release.yml`, environment `release`.
2. **GitHub environment** — repo Settings → Environments → create `release`
   with required reviewers, so every publish needs a manual approval.

No PyPI API token is ever stored; the workflow mints a short-lived OIDC token
at publish time.
