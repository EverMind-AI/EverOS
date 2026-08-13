---
name: new-branch
description: Create a GitHub branch from main with the project naming convention
---

# /new-branch

Cut a new branch for the GitHub repository.

## Branch model

```
main       = default and protected branch
feat/*     = feature work
fix/*      = bug fixes
docs/*     = documentation-only changes
ci/*       = CI, build, and developer-experience changes
chore/*    = repository maintenance
refactor/* = behavior-preserving code structure changes
```

## Steps

1. Ask (or infer) the change type: `feat`, `fix`, `docs`, `ci`, `chore`, or
   `refactor`.
2. Update `main` first:
   ```bash
   git checkout main
   git pull --ff-only
   ```
3. Create the branch with a kebab-case slug:
   ```bash
   git checkout -b <type>/<short-slug>
   ```
4. Keep the branch scoped to one purpose and open a pull request back to
   `main`.

## Naming

- `feat/add-agentic-rerank`, `fix/empty-profile-crash`,
  `docs/quickstart-config`, `ci/check-github-docs`.
- Lowercase, hyphen-separated, no spaces, concise.

Never commit directly to `main` — always use a branch and pull request.
