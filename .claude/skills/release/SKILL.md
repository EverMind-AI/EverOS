---
name: release
description: Cut a versioned release and publish everos to PyPI via the tag-triggered workflow
---

# /release

Publish a new version of `everos` to PyPI. Publishing is automated: pushing a
`vX.Y.Z` tag triggers [.github/workflows/release.yml](../../../.github/workflows/release.yml),
which builds, smoke-tests, and uploads via PyPI **Trusted Publishing** (OIDC —
no stored token) behind the `release` environment's manual-approval gate, then
drafts the GitHub Release page from the CHANGELOG.

A release is not finished when PyPI accepts the upload: the GitHub Release page
is drafted, never auto-published, and someone has to write its lead summary and
click **Publish**.

## Preconditions

- On `main`, up to date, with green CI (the tag builds from `main`'s tree).
- Decide the version per SemVer: patch = fixes, minor = back-compatible
  features, major = breaking changes.

## Steps

```
1. Bump the version    → pyproject.toml [project] version = "X.Y.Z"
   (single source; everos.__version__ reads installed package metadata)
2. Update CHANGELOG.md → move the Unreleased entries under a new
   ## [X.Y.Z] - <date> heading, and write the release page's prose here
   (lead paragraph + `### Upgrade` group — see "The release page")
3. Commit              → git commit -m "chore(release): vX.Y.Z"
4. Open a PR, merge to main after green CI
5. Tag main + push     → git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z
6. Approve             → the release.yml run pauses on the `release`
   environment; a reviewer approves in the Actions run
7. Verify              → https://pypi.org/project/everos/X.Y.Z/
8. Publish the page    → the run leaves a DRAFT GitHub Release, already
   complete if step 2 was done properly. Read it once, click Publish.
```

The tag must equal the `pyproject.toml` version — the workflow refuses to
publish on a mismatch. A stable tag with no matching `## [X.Y.Z]` CHANGELOG
section fails the release job for the same reason.

## The release page

**The whole page is written in CHANGELOG.md, during the release PR.** Nothing
is meant to be composed at publish time — by then the changes are weeks old and
the text gets no review. Write the version section like this:

```markdown
## [X.Y.Z] - 2026-09-01

**What this release is for.** One paragraph, prose, no bullets — it becomes the
lead of the release page. Say what changed for a user, not what was refactored.

### Added
### Changed
### Fixed

### Upgrade

What a reader must know before upgrading: what happens on first startup, which
command recovers a bad state, which pins moved. Omit the group entirely when a
plain `pip install --upgrade` is all there is — 1.1.4 and 1.2.0 have nothing
here. Do not write filler.
```

CI turns that into the page: everything above `### Upgrade` is lifted verbatim
with the group headings demoted to `##`, and the Upgrade prose is wrapped in the
boilerplate — pip line above it, compare link below — which is the shape every
release since 1.1.3 has. See
[1.2.1](https://github.com/EverMind-AI/EverOS/releases/tag/v1.2.1).

So publishing is a read-through and a click. If the draft looks wrong, the fix
belongs in CHANGELOG.md on `main`, not only in the draft — otherwise the two
drift apart and the next release inherits the habit.

> While it is a draft, GitHub serves the release at
> `releases/tag/untagged-<hash>`, and that URL keeps serving a stale page after
> publication with no redirect. Never share it — a reader who opens it later
> concludes the release never went out. Link `releases/tag/vX.Y.Z` instead; the
> job prints both URLs in its step summary.

Publish with the **Publish release** button in the web UI. Publishing through
the API by flipping `draft` alone drops the tag: GitHub rebinds the release to
the `untagged-<hash>` placeholder and creates a git tag by that name against the
default branch (observed on 1.2.2). Pass `tag_name` if you must do it from the
CLI:

```bash
gh api -X PATCH "repos/EverMind-AI/EverOS/releases/<id>" \
  -F draft=false -f tag_name=vX.Y.Z -f make_latest=true
```

Re-running the release job replaces its own draft and leaves an already-published
release untouched, so a re-run is always safe.

## Pre-releases

PEP 440 pre-release tags publish too (PyPI accepts them; `pip install everos`
ignores them unless `--pre`): `vX.Y.ZrcN`, `vX.Y.ZaN`, `vX.Y.ZbN`. Set the same
suffix in `pyproject.toml` version before tagging.

Their release page is drafted as a **pre-release** and never becomes
`/releases/latest`. A pre-release does not need its own CHANGELOG section — the
draft falls back to a one-line placeholder body when there is none.

## One-time setup (project owner, not doable from CI)

1. **PyPI trusted publisher** — PyPI → project `everos` → Settings →
   Publishing → add: owner `EverMind-AI`, repo `EverOS`, workflow
   `release.yml`, environment `release`.
2. **GitHub environment** — repo Settings → Environments → create `release`
   with required reviewers, so every publish needs a manual approval.

No PyPI API token is ever stored; the workflow mints a short-lived OIDC token
at publish time.
