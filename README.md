# EverOS

> Local-first markdown memory framework for AI agents and user chats — lightweight, dev-friendly, small-team.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)

---

## What is EverOS

EverOS is an open-source Python framework that turns conversations, agent trajectories, and files into **structured, retrievable, evolving long-term memory** for AI agents and user chats. Designed for **lightweight local deployments** (small teams, individual developers), with three core principles:

1. **Markdown as Source of Truth** — All memory persists as plain `.md` files. Open, edit, grep, version with Git, view in Obsidian. No black-box database lock-in.
2. **Lightweight three-piece storage** — `Markdown` files (truth) + `SQLite` (state/queue) + `LanceDB` (vector + BM25 + scalar). No MongoDB / Elasticsearch / Milvus / Redis / Kafka required.
3. **EverAlgo as pure algorithm library** — Memory extraction algorithms are decoupled into a separate library; this project orchestrates and persists.

## Architecture at a glance

```
┌───────────────────────────────────────────────┐
│  entrypoints/  (CLI + HTTP API)                │  presentation
├───────────────────────────────────────────────┤
│  service/      (use cases: memorize/retrieve)  │  application
├───────────────────────────────────────────────┤
│  memory/       (extract + search + cascade)    │  domain
├───────────────────────────────────────────────┤
│  infra/        (markdown / sqlite / lancedb)   │  infrastructure
└───────────────────────────────────────────────┘
        ↑                    ↑
   component/            core/
   (LLM/Embedding)       (observability/lifespan)
```

DDD 5 layers, single-direction dependency. See [docs/architecture.md](docs/architecture.md).

## Quick start

### Install as a package

```bash
uv pip install everos               # or: pip install everos

# Generate starter config (model defaults bundled inside the wheel)
everos init                          # writes ~/.everos/everos.toml + ome.toml (use --root to relocate)
# Edit ~/.everos/everos.toml and fill the api_key fields (see comments inside).

everos --help
everos server start
```

`everos init` writes two TOML files into the memory root (`~/.everos` by
default; relocate with `--root`): `everos.toml` (app settings + provider
credentials) and `ome.toml` (offline-engine schedules). `everos server start`
reads `<root>/everos.toml` and exits with an error if it is missing. Any
setting can also be overridden by an `EVEROS_*` environment variable
(e.g. `EVEROS_LLM__API_KEY`). The endpoint stack is OpenAI-protocol
compatible (OpenAI / OpenRouter / vLLM / Ollama / DeepInfra …) — set the
`base_url` field in each provider section of `everos.toml` to point at any
of them.

#### Multi-modal (optional)

To ingest non-text content (image / pdf / audio / office documents)
through `/api/v1/memory/add` `content` items, install the optional
extra:

```bash
uv pip install 'everos[multimodal]'   # or: pip install 'everos[multimodal]'
```

This pulls in `everalgo-parser` (with the `[svg]` bundle for SVG
support via cairosvg) and wires up the multimodal LLM client
(the `[multimodal]` section in `everos.toml`, defaults to
`google/gemini-3-flash-preview`).

**Office document support requires LibreOffice as a system dependency.**
The parser shells out to `soffice` (LibreOffice's headless renderer) to
convert `.doc` / `.docx` / `.ppt` / `.pptx` / `.xls` / `.xlsx` to PDF
before feeding the result into the multimodal LLM. Without LibreOffice,
office uploads return HTTP 503 (`CAPABILITY_UNAVAILABLE`) with a clear
error message; PDF / image
/ audio / HTML / email parsing is unaffected.

Install on the host before serving office documents:

```bash
brew install --cask libreoffice              # macOS
sudo apt-get install -y libreoffice          # Debian / Ubuntu
```

For the full multimodal contract (supported modalities, `uri` vs
`base64`, config, error semantics, end-to-end curl examples), see
[docs/multimodal.md](docs/multimodal.md).

For a step-by-step walkthrough (add a conversation → flush → search →
read the markdown), see [QUICKSTART.md](QUICKSTART.md).

### Develop locally

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
uv sync                              # creates ./.venv and installs deps
source .venv/bin/activate            # — or skip activation and prefix every command with `uv run`
everos init                         # fill in the [llm] api_key in the generated everos.toml

everos --help
make test
```

## Storage layout

```
~/.everos/
├── default_app/                  # app_id  ("default" → "default_app" on disk)
│   └── default_project/          # project_id ("default" → "default_project")
│       ├── users/<user_id>/
│       │   ├── user.md           # profile
│       │   ├── episodes/         # daily-log episodes (visible)
│       │   ├── .atomic_facts/    # nested facts (dotfile-hidden)
│       │   └── .foresights/      # predictive memory (dotfile-hidden)
│       └── agents/<agent_id>/
│           ├── agent.md
│           ├── .cases/           # one task case per entry
│           └── skills/           # named procedural memories
├── .index/                       # derived indexes (rebuildable from md)
│   ├── sqlite/system.db          # state + queue + audit
│   └── lancedb/*.lance/          # vector + BM25 + scalar
└── .tmp/                         # transient working files
```

Open any `<app>/<project>/users/<user_id>/` folder in Obsidian — your
agent's brain is just files. The dotfile directories (`.atomic_facts/`,
`.foresights/`, `.cases/`) stay hidden by default so the visible folder
is the user-facing memory surface, while extracted derivatives sit
quietly alongside.

## Features

- **Hybrid retrieval**: BM25 + vector (HNSW/IVF-PQ) + scalar filter, single-query in LanceDB
- **Cascade index sync**: edit a `.md` → file watcher → entry-level diff → LanceDB sync, sub-second
- **Multi-source extraction**: conversations / agent trajectories / file knowledge
- **Dual-track memory**: user-track (Episodes / Profiles) + agent-track (Cases / Skills)
- **Async-first**: full asyncio, single event loop
- **Multi-modal**: text + small image / audio inline; large media via S3/OSS reference

## Project structure

```
everos/                        # repo root
├── src/everos/                # main package (src layout)
│   ├── entrypoints/           # cli + api
│   ├── service/               # use case orchestration
│   ├── memory/                # domain: extract + search + cascade + prompt_slots
│   ├── infra/                 # storage: markdown + lancedb + sqlite
│   ├── component/             # cross-cutting: llm / embedding / config / utils
│   ├── core/                  # runtime: observability / lifespan / context
│   └── config/                # configuration data + Settings schema
├── tests/                     # unit / integration / golden / fixtures
├── docs/                      # design docs
└── .claude/                   # team-shared rules + skills (auto-loaded by Claude Code)
```

## Documentation

- [docs/overview.md](docs/overview.md) — Project overview & vision
- [docs/architecture.md](docs/architecture.md) — DDD layered architecture & dependency rules
- [docs/engineering.md](docs/engineering.md) — Engineering & dev-efficiency infrastructure (CI / tooling / Claude Code)
- [docs/multimodal.md](docs/multimodal.md) — Multimodal memory: ingest image / pdf / audio / office docs via the HTTP API
- [CHANGELOG.md](CHANGELOG.md) — Release notes
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute
- [.claude/rules/](.claude/rules/) — Detailed coding conventions (auto-loaded by Claude Code)

## Use Cases

See [use-cases/README.md](use-cases/README.md) for the full gallery.

<table>
</table>

## Status

**Stable (v1.1.0)** — Released on PyPI; the v1 API is stable.

## License

[Apache License 2.0](LICENSE) — see [NOTICE](NOTICE) for third-party attributions.

## Citation

If you use EverOS in research, see [CITATION.md](CITATION.md).

---

**Acknowledgments**: This project builds on prior research and tooling — see [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md).
