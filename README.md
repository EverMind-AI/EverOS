<div align="center" id="readme-top">

![banner-gif](https://github.com/user-attachments/assets/0bf97efd-580f-4a53-a2a2-58d6daea7290)

<p align="center">
  <a href="https://x.com/evermind"><img src="https://img.shields.io/badge/EverMind-000000?labelColor=gray&style=for-the-badge&logo=x&logoColor=white" alt="X"></a>
  <a href="https://huggingface.co/EverMind-AI"><img src="https://img.shields.io/badge/🤗_HuggingFace-EverMind-F5C842?labelColor=gray&style=for-the-badge" alt="HuggingFace"></a>
  <a href="https://discord.gg/gYep5nQRZJ"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FgYep5nQRZJ%3Fwith_counts%3Dtrue&query=%24.approximate_presence_count&suffix=%20online&label=Discord&color=404EED&labelColor=gray&style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/EverMind-AI/EverOS/discussions/67"><img src="https://img.shields.io/badge/WeCom-EverMind_社区-07C160?labelColor=gray&style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

[Website](https://evermind.ai) · [Documentation](https://docs.evermind.ai) · [Blog](https://evermind.ai/blogs)

</div>


<br>

<details open>
  <summary><kbd>Table of Contents</kbd></summary>

<br>

- [EverOS 1.0.0 in 30 seconds](#everos-100-in-30-seconds)
- [What is EverOS](#what-is-everos)
- [Quick start](#quick-start)
- [Architecture at a glance](#architecture-at-a-glance)
- [Storage layout](#storage-layout)
- [Features](#features)
- [Project structure](#project-structure)
- [Documentation](#documentation)
- [Use Cases](#use-cases)
- [Stay Tuned](#stay-tuned)
- [EverMind ecosystem](#evermind-ecosystem)
- [Contributing](#contributing)

<br>

</details>


## EverOS 1.0.0 in 30 seconds

EverOS gives AI agents a local memory system that developers can inspect,
edit, rebuild, and extend.

<table>
<tr>
<td width="50%" valign="top">
<strong>Memory you can read</strong><br>
Plain Markdown is the source of truth. SQLite and LanceDB are local,
rebuildable indexes.
</td>
<td width="50%" valign="top">
<strong>Easy to run locally</strong><br>
Install with Python. No Docker, MongoDB, Elasticsearch, Redis, Kafka, or
hosted database is required.
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>User and agent memory</strong><br>
Store profiles, episodes, task cases, and skills with scoped search by
user, agent, app, project, and session.
</td>
<td width="50%" valign="top">
<strong>Evolving, multimodal memory</strong><br>
Ingest text, image, audio, docs, PDF, HTML, and email. Online extraction
and offline evolution are configurable; Reflection and LLM Wiki are next.
</td>
</tr>
</table>

<br>


## What is EverOS

EverOS is an open-source Python framework for long-term memory in AI
agents and user chats. It turns conversations, agent trajectories, and
files into structured memory, then keeps local retrieval indexes in sync
so the memory can be searched and reused.

The system is built around three boundaries:

1. **Memory content stays readable** - Markdown is the durable source of truth.
2. **Runtime state stays local** - SQLite tracks state and LanceDB handles vector, BM25, and scalar-filter search.
3. **Algorithms stay modular** - [EverAlgo](https://github.com/EverMind-AI/EverAlgo) owns memory algorithms; EverOS owns runtime, persistence, online flows, and offline evolution.

<br>

## Quick start

### 1. Install EverOS

```bash
uv pip install everos
# or: pip install everos
```

### 2. Initialize configuration

Generate a starter `.env` file, then fill the API key fields shown in
the generated comments.

```bash
everos init
```

`everos init` writes `./.env` by default. Use `everos init --xdg` to
write `${XDG_CONFIG_HOME:-~/.config}/everos/.env` instead.

### 3. Start the server

```bash
everos --help
everos server start
```

`everos server start` searches for `.env` in this order: `--env-file <path>` →
`./.env` (cwd) → `${XDG_CONFIG_HOME:-~/.config}/everos/.env` → `~/.everos/.env`.
The endpoint stack is OpenAI-protocol compatible (OpenAI / OpenRouter / vLLM /
Ollama / DeepInfra) - override `*__BASE_URL` in the generated `.env` to point
at any of them.

For a step-by-step walkthrough (add a conversation, flush, search, then
read the markdown), see [QUICKSTART.md](QUICKSTART.md).

### Optional: ingest multimodal files

To ingest non-text content (image / pdf / audio / office documents)
through `/api/v1/memory/add` `content` items, install the optional
extra:

```bash
uv pip install 'everos[multimodal]'   # or: pip install 'everos[multimodal]'
```

This pulls in `everalgo-parser` (with the `[svg]` bundle for SVG
support via cairosvg) and wires up the multimodal LLM client
(`EVEROS_MULTIMODAL__*` fields in `.env`, defaults to
`google/gemini-3-flash-preview` via OpenRouter).

**Office document support requires LibreOffice as a system dependency.**
The parser shells out to `soffice` (LibreOffice's headless renderer) to
convert `.doc` / `.docx` / `.ppt` / `.pptx` / `.xls` / `.xlsx` to PDF
before feeding the result into the multimodal LLM. Without LibreOffice,
office uploads return HTTP 415 with a clear error message; PDF / image
/ audio / HTML / email parsing is unaffected.

Install on the host before serving office documents:

```bash
brew install --cask libreoffice              # macOS
sudo apt-get install -y libreoffice          # Debian / Ubuntu
```

### For contributors

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
uv sync                              # creates ./.venv and installs deps
source .venv/bin/activate            # or prefix commands with `uv run`
everos init                          # fill the four API key slots in .env (two distinct keys)

everos --help
make test
```

<br>

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

<br>

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

<br>

## Features

- **Hybrid retrieval**: BM25 + vector (HNSW/IVF-PQ) + scalar filter, single-query in LanceDB
- **Cascade index sync**: edit a `.md` → file watcher → entry-level diff → LanceDB sync, sub-second
- **Multi-source extraction**: conversations / agent trajectories / file knowledge
- **Dual-track memory**: user-track (Episodes / Profiles) + agent-track (Cases / Skills)
- **Async-first**: full asyncio, single event loop
- **Multi-modal**: text + small image / audio inline; large media via S3/OSS reference

<br>

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
<br>

## Documentation

- [docs/overview.md](docs/overview.md) — Project overview & vision
- [docs/architecture.md](docs/architecture.md) — DDD layered architecture & dependency rules
- [docs/engineering.md](docs/engineering.md) — Engineering & dev-efficiency infrastructure (CI / tooling / Claude Code)
- [docs/use-cases.md](docs/use-cases.md) — Full use-case gallery and integration examples
- [docs/migration-to-1.0.0.md](docs/migration-to-1.0.0.md) — Legacy API and infrastructure migration notes
- [CHANGELOG.md](CHANGELOG.md) — Release notes
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute
- [.claude/rules/](.claude/rules/) — Detailed coding conventions (auto-loaded by Claude Code)

<br>

## Use Cases

EverOS is already being used across AI coding, personal memory, creative
assistants, games, browser agents, and research workflows. The README keeps
the path short; the full showcase lives in [docs/use-cases.md](docs/use-cases.md).

<table>
<tr>
<td width="33%" valign="top">
<strong>AI coding memory</strong><br>
Persistent context for coding assistants and multi-agent engineering workflows.
<br><br>
<a href="https://github.com/tt-a1i/evermemos-mcp">Coding assistant memory</a> ·
<a href="https://github.com/tt-a1i/hive">Hive</a>
</td>
<td width="33%" valign="top">
<strong>Personal memory products</strong><br>
Memory for companions, browser agents, wearables, education, and daily life.
<br><br>
<a href="https://evermind.ai/usecase_reunite">Reunite</a> ·
<a href="https://github.com/TonyLiangDesign/MemoCare">MemoCare</a>
</td>
<td width="33%" valign="top">
<strong>Research and demos</strong><br>
Memory-aware data analysis, graph visualization, games, and product prototypes.
<br><br>
<a href="https://github.com/yuansui123/AI-Data-Technician-EverMemOS">AI Data Technician</a> ·
<a href="https://main.d2j21qxnymu6wl.amplifyapp.com/graph.html">Memory graph</a>
</td>
</tr>
</table>

[Explore the full use-case gallery](docs/use-cases.md)

<br>

## Stay Tuned

Star the repo or join the community links above to follow new architecture methods, benchmark releases, and memory-enabled use cases.

![star us gif](https://github.com/user-attachments/assets/0c512570-945a-483a-9f47-8e067bd34484)

<br>

## EverMind ecosystem

EverMind is an open-source ecosystem for long-term memory, self-evolving agents, and memory evaluation. EverOS is the core runtime architecture; EverMemOS is the paper and research line carrying our strongest memory-system benchmark runs; EverAlgo supplies the next-generation algorithms that make the system modular and reusable.

<table>
<tr>
<th colspan="2">EverMind Open-Source Ecosystem</th>
</tr>
<tr>
<td><strong>Core memory architecture</strong></td>
<td><a href="https://github.com/EverMind-AI/EverOS">EverOS</a> / EverMemOS - the local memory operating system and research-backed runtime for agent and user memory.</td>
</tr>
<tr>
<td><strong>Algorithm engine</strong></td>
<td><a href="https://github.com/EverMind-AI/EverAlgo">EverAlgo</a> - stateless extraction, ranking, parsing, and memory operators that power EverOS.</td>
</tr>
<tr>
<td><strong>Alternative architecture</strong></td>
<td><a href="https://github.com/EverMind-AI/HyperMem">HyperMem</a> - hypergraph memory for long-term conversations, with its own benchmark-backed topic -> episode -> fact retrieval method.</td>
</tr>
<tr>
<td><strong>Benchmarks</strong></td>
<td><a href="https://github.com/EverMind-AI/EverMemBench">EverMemBench</a> · <a href="https://github.com/EverMind-AI/EvoAgentBench">EvoAgentBench</a> - evaluation suites for conversational memory and agent self-evolution.</td>
</tr>
<tr>
<td><strong>Long-context research</strong></td>
<td><a href="https://github.com/EverMind-AI/MSA">MSA</a> - Memory Sparse Attention for scalable latent memory and 100M-token contexts.</td>
</tr>
<tr>
<td><strong>Personal memory layer</strong></td>
<td><a href="https://github.com/EverMind-AI/EverMe">EverMe</a> - CLI and agent plugin suite for cross-device, cross-agent personal memory.</td>
</tr>
<tr>
<td><strong>Developer integrations</strong></td>
<td><a href="https://github.com/EverMind-AI/evermem-claude-code">evermem-claude-code</a> · <a href="https://github.com/EverMind-AI/everos-plugins">everos-plugins</a> - plugins, skills, and migration tooling for AI coding agents.</td>
</tr>
</table>

Together, these repositories form EverMind's research-to-runtime stack: new memory methods, reusable algorithms, benchmark evidence, and practical agent integrations.

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Contributing

Contributions are welcome across the whole repository: architecture methods, benchmark coverage, use-case examples, documentation, and bug fixes. Browse [Issues](https://github.com/EverMind-AI/EverOS/issues) to find a good entry point, then open a PR when you are ready.

<br>

> [!TIP]
>
> **Welcome all kinds of contributions** 🎉
>
> Help make EverOS better. Code, documentation, benchmark reports, use-case write-ups, and integration examples are all valuable. Share your projects on social media to inspire others.
>
> Connect with one of the EverOS maintainers [@elliotchen200](https://x.com/elliotchen200) on 𝕏 or [@cyfyifanchen](https://github.com/cyfyifanchen) on GitHub for project updates, discussions, and collaboration opportunities.

![divider](https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only)
![divider](https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only)

### Code Contributors

[![EverOS Contributors](https://contrib.rocks/image?repo=EverMind-AI/EverOS)](https://github.com/EverMind-AI/EverOS/graphs/contributors)

![divider](https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only)
![divider](https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only)

### License

[Apache License 2.0](LICENSE) — see [NOTICE](NOTICE) for third-party attributions.

### Citation

If you use EverOS in research, see [CITATION.md](CITATION.md).

<br>

<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>
