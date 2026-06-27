---
name: commit
description: Stage and create a Conventional Commits message for the current change
---

# /commit

Create a well-formed commit following the [Conventional Commits](https://www.conventionalcommits.org)
standard. The format is enforced by `gitlint` in the `commit-msg` hook.

## Steps

1. Run `git status` and `git diff` (and `git diff --staged`) to see what changed.
2. Review recent history for style: `git log --oneline -10`.
3. Group the change into a single focused commit. If the working tree mixes
   unrelated changes, stage selectively (`git add -p` / specific paths) rather
   than committing everything at once.
4. Write the message in **Conventional Commits** form:

   ```
   <type>[(scope)][!]: <imperative summary, ≤72 chars>

   <optional body: what & why, wrapped at 72>

   <optional footer: BREAKING CHANGE: …, Refs: #123>
   ```

5. Never use `--no-verify`. If pre-commit hooks fail, fix the cause and re-commit.
6. Do not commit secrets, generated artifacts, or work-in-progress directly to
   the protected `main` branch.

## Types

| Type | Use for |
|---|---|
| `feat` | new feature |
| `fix` | bug fix |
| `refactor` | behavior-preserving restructure |
| `test` | add / update tests |
| `docs` | documentation |
| `style` | formatting only |
| `perf` | performance |
| `chore` | config / build / tooling |
| `build` | build system or dependencies |
| `ci` | CI configuration |
| `revert` | revert a previous commit |

## Notes

- No emoji — the title must start with the `type`.
- One logical change per commit; keep the history bisectable.
- The summary is imperative mood: "add", not "added" / "adds".
- `scope` is optional: `fix(search): …`. A `!` before the colon (or a
  `BREAKING CHANGE:` footer) marks a breaking change.
