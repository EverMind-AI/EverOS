---
name: pr
description: Open a GitHub PR targeting the correct branch with the project template
---

# /pr

Open a pull request on GitHub using the `gh` CLI and the repo's PR template.

## Steps

1. Confirm the branch and target:
   - Base branch is **`main`**.
   - Head branch should be a scoped branch such as `feat/*`, `fix/*`,
     `docs/*`, `ci/*`, `chore/*`, or `refactor/*`.
2. Ensure local checks pass first:
   ```bash
   make ci
   ```
   Do not open a PR with failing lint/tests.
3. Push the branch:
   ```bash
   git push -u origin HEAD
   ```
4. Create the PR, filling the template
   ([.github/PULL_REQUEST_TEMPLATE.md](../../../.github/PULL_REQUEST_TEMPLATE.md)):
   ```bash
   gh pr create --base main --fill-first
   ```
   Then edit the body to complete each section:
   - **Summary** — what changed and why.
   - **Area** — tick the relevant box (architecture / benchmark / use case /
     docs / DX / CI-build-release).
   - **Verification** — paste the commands you ran (`make ci`, manual checks).
   - **Checklist** — tick honestly; don't tick boxes you didn't satisfy.
   - **Notes for Reviewers** — anything subtle.

## Notes

- Keep the PR scoped to one area. Split unrelated changes.
- If `make ci` was not fully run, say so in Verification rather than implying it passed.
- Never retarget a community PR away from `main` unless a maintainer explicitly
  asks for it.
