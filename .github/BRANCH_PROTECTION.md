# Branch Protection Baseline

Use this as the admin checklist for `main` after the EverOS 1.0 history reset.

## Required Repository Rule

- Require pull requests before merging.
- Require two approving reviews for normal work.
- Require conversation resolution before merge.
- Block force pushes.
- Block branch deletion.
- Do not grant routine admin bypasses.

## Required Status Checks

Mark these checks as required before merge:

- `CI / lint`
- `CI / unit tests`
- `CI / integration tests`
- `CI / package build`
- `Docs / links`
- `Commit lint / commit messages`

## Optional Repository Checks

Do not require checks that are not emitted for every pull request. Treat these
as advisory unless GitHub shows they run on all normal PRs:

- `.github/dependabot.yml`

## Merge Policy

- Work on feature branches.
- Push branches normally; do not force-push shared branches.
- Merge through PRs after checks are green.
- Delete merged branches.

Temporary admin bypass should be reserved for repository recovery work only.
