# Contributing to EverMemOS

Thanks for contributing. This guide is intentionally short so you can start fast.

## Fast Path

1. Check existing [issues](https://github.com/EverMind-AI/EverMemOS/issues) and [discussions](https://github.com/EverMind-AI/EverMemOS/discussions).
2. Fork the repo and create a branch.
3. Run the local setup.
4. Make a focused change.
5. Run checks and open a PR.

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/EverMemOS.git
cd EverMemOS

# install dependencies
uv sync --group dev

# configure environment
cp env.template .env

# start infrastructure (optional for docs-only changes)
docker compose up -d
```

## Before Opening a PR

Run these checks from repo root:

```bash
PYTHONPATH=src uv run pytest tests/
uv run black --check src tests demo
uv run isort --check-only src tests demo
```

Optional helper targets:

```bash
make lint
make test
```

## Coding Conventions

- Python style follows PEP 8.
- Use type hints for new/changed code.
- Use `async`/`await` for I/O paths.
- Formatting is Black with line length `88`.
- Use absolute imports from project modules.
- Avoid wildcard imports (`from x import *`).
- For time conversion logic, prefer `common_utils.datetime_utils`.

## Branch and Commit Conventions

Suggested branch names:

- `feature/<short-name>`
- `fix/<short-name>`
- `docs/<short-name>`
- `refactor/<short-name>`

Commit style:

- Commit messages are validated by a `commit-msg` hook using Conventional Commits.
- Required format: `<type>(<scope>)?: <description>` or `<type>!: <description>`.
- Allowed types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.

Examples:

- `feat(api): add memory search pagination`
- `fix(core): handle empty tenant id`
- `refactor(di)!: simplify lifecycle bootstrap`

## Pull Request Expectations

Please include:

- What changed and why.
- Linked issue (if available).
- How you tested it.
- Screenshots for UI changes (if applicable).

Keep PRs focused. Smaller PRs review faster.

## Reporting Bugs and Features

- Bug report: <https://github.com/EverMind-AI/EverMemOS/issues/new?template=bug_report.md>
- Feature request: <https://github.com/EverMind-AI/EverMemOS/issues/new?template=feature_request.md>

## Security Reports

Do not report security vulnerabilities in public issues.

- See: [Security Policy](SECURITY.md)
- Report via email: `evermind@shanda.com`

## License

By contributing, you agree your contributions are licensed under the Apache License 2.0 in this repository.
