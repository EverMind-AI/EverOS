# EverCore

EverCore is the long-term memory operating system at the center of EverOS. It extracts, structures, and retrieves durable knowledge from conversations so agents can remember across sessions and adapt over time.

## Start Here

| Goal | Link |
| :--- | :--- |
| Install and run EverCore locally | [Setup Guide](docs/installation/SETUP.md) |
| Browse the documentation index | [Documentation](docs/) |
| Try usage examples and demos | [Usage Examples](docs/usage/USAGE_EXAMPLES.md) |
| Review the architecture | [Architecture](docs/ARCHITECTURE.md) |
| Run evaluations | [Evaluation Guide](evaluation/) |

## Quick Start

```bash
docker compose up -d
uv sync
uv run python src/run.py
```

The server runs at `http://localhost:1995` by default. See the [full setup guide](docs/installation/SETUP.md) for environment variables, service configuration, and troubleshooting.

## Folder Guide

- [src/](src/) - application, memory, infrastructure, and API layers.
- [docs/](docs/) - setup, usage, architecture, API, and development documentation.
- [demo/](demo/) - interactive examples and memory extraction demos.
- [evaluation/](evaluation/) - benchmark runners and reports.
- [tests/](tests/) - unit and integration tests.
