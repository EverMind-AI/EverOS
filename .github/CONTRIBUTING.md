# Contributing to EverOS

Thanks for helping improve EverOS. This repository brings together architecture
methods, benchmarks, and use cases for long-term memory in self-evolving agents,
so there are several useful ways to contribute.

## Ways to Contribute

- Improve or extend an architecture method in `methods/`.
- Add benchmark tasks, adapters, or reproducibility notes in `benchmarks/`.
- Add a memory-enabled app, demo, or integration in `use-cases/`.
- Fix documentation, examples, setup steps, or broken links.
- Report bugs with clear reproduction steps and environment details.

## Development Setup

Most core development happens in EverCore:

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS/methods/EverCore

docker compose up -d
uv sync
cp env.template .env
uv run python src/run.py
```

Verify the server:

```bash
curl http://localhost:1995/health
```

## Common Commands

```bash
cd methods/EverCore
make test                     # Run tests
make lint                     # Run formatting and i18n checks
uv sync --group evaluation    # Install evaluation dependencies
```

## Pull Request Checklist

Before opening a PR, please check:

- The change is scoped to the relevant area: `methods/`, `benchmarks/`, or
  `use-cases/`.
- Setup or behavior changes are documented.
- Tests or manual verification are included when relevant.
- No secrets, `.env` files, generated build output, or dependency folders are
  committed.
- Active relative links in Markdown files resolve.

## Use-Case Contributions

Use cases should be easy for a new developer to inspect and run. Each use case
should include:

- A README with what it does, how to run it, and what memory feature it shows.
- A small `.env.example` when configuration is required.
- No committed images, build output, dependency folders, or secrets.

Images should be hosted with GitHub user attachments or another external asset
URL instead of committed to the repository.

## Style Notes

- Follow existing patterns before adding new abstractions.
- EverCore I/O is async; use `await`.
- EverCore is multi-tenant; keep data tenant-scoped.
- Keep prompt changes aligned across
  `methods/EverCore/src/memory_layer/prompts/en/` and
  `methods/EverCore/src/memory_layer/prompts/zh/` when applicable.

## Community

Please keep discussions respectful, constructive, and welcoming. See
`CODE_OF_CONDUCT.md` for expectations.

By contributing, you agree that your contributions are licensed under the
Apache License 2.0.
