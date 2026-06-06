<div align="center" id="readme-top">

![EverOS banner](https://github.com/EverMind-AI/EverOS/releases/download/v1.0.0/everos-readme-banner.jpg)

<p align="center">
  <a href="https://x.com/evermind"><img src="https://img.shields.io/badge/EverMind-000000?labelColor=gray&style=for-the-badge&logo=x&logoColor=white" alt="X"></a>
  <a href="https://huggingface.co/EverMind-AI"><img src="https://img.shields.io/badge/🤗_HuggingFace-EverMind-F5C842?labelColor=gray&style=for-the-badge" alt="HuggingFace"></a>
  <a href="https://discord.gg/gYep5nQRZJ"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FgYep5nQRZJ%3Fwith_counts%3Dtrue&query=%24.approximate_presence_count&suffix=%20online&label=Discord&color=404EED&labelColor=gray&style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/EverMind-AI/EverOS/discussions/67"><img src="https://img.shields.io/badge/WeCom-EverMind_社区-07C160?labelColor=gray&style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

[Website](https://evermind.ai) · [Documentation](https://docs.evermind.ai) · [Blog](https://evermind.ai/blogs) · [中文](README.zh-CN.md)

</div>


<br>

<details>
  <summary><kbd>Table of Contents</kbd></summary>

<br>

- [EverOS 1.0.0 Highlights](#everos-100-highlights)
- [What Is EverOS](#what-is-everos)
- [Quick Start](#quick-start)
- [Architecture At A Glance](#architecture-at-a-glance)
- [Storage Layout](#storage-layout)
- [Features](#features)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Use Cases](#use-cases)
- [Stay Tuned](#stay-tuned)
- [EverMind Ecosystems](#evermind-ecosystems)
- [Contributing](#contributing)

<br>

</details>


## EverOS 1.0.0 Highlights

> [!IMPORTANT]
>
> **EverOS 1.0.0 is a major release for self-evolving memory.** It brings a
> local-first runtime, Markdown as the source of truth, hybrid retrieval,
> multimodal ingestion, user and agent memory scopes, and modular algorithms
> through [EverAlgo](https://github.com/EverMind-AI/EverAlgo).
>
> **Watch this repository** for the next wave of memory-system work, including
> Wiki-style knowledge layers and Dreaming for deeper offline evolution.

<table>
<tr>
<td width="33%" valign="top">
<strong>Markdown-First Memory</strong><br>
Memory is persisted as plain Markdown: visible, auditable, hand-editable,
Git-friendly, and owned by the user.
</td>
<td width="33%" valign="top">
<strong>Lightweight Local Stack</strong><br>
Install with Python. SQLite tracks runtime state; LanceDB powers vector,
BM25, and scalar-filter retrieval locally.
</td>
<td width="33%" valign="top">
<strong>Layered Memory Model</strong><br>
User memory and agent memory are first-class today. Wiki-style knowledge
is the next layer in the roadmap.
</td>
</tr>
<tr>
<td width="33%" valign="top">
<strong>Self-Evolving Agents</strong><br>
Agent memory can extract reusable cases and skills from repeated
experience, so workflows become smarter over time.
</td>
<td width="33%" valign="top">
<strong>Multimodal Ingestion</strong><br>
Text, image, audio, documents, PDF, HTML, and email can be parsed into
memory through the optional multimodal pipeline.
</td>
<td width="33%" valign="top">
<strong>Online And Offline Strategy Control</strong><br>
Online extraction and offline evolution stay separate, with configurable
prompts and models at each step. Dreaming is coming next.
</td>
</tr>
<tr>
<td width="33%" valign="top">
<strong>Orthogonal Memory Scope</strong><br>
Owner, memory type, and scope are independent: search by user, agent,
app, project, session, and structured filters.
</td>
<td width="33%" valign="top">
<strong>Progressive Disclosure</strong><br>
Readable memory surfaces stay simple while deeper facts, cases, and
skills remain available.
</td>
<td width="33%" valign="top">
<strong>Modular By Design</strong><br>
EverAlgo owns algorithms; EverOS owns runtime, persistence, online flows,
and offline evolution.
</td>
</tr>
</table>

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>


## What Is EverOS

EverOS is an open-source Python framework for self-evolving long-term
memory across agents and platforms. It gives makers one portable memory
layer for every agent they use - Claude Code, Codex, OpenClaw, Hermes,
and more - so context, decisions, files, and trajectories can follow the
work instead of staying trapped in one tool.

EverOS stores conversations, agent trajectories, and files as readable
Markdown, then syncs local SQLite and LanceDB indexes for fast retrieval.
Agents can reuse past cases and skills, improve from repeated workflows,
and become more proactive over time.

The system is built around three boundaries:

1. **Memory content stays readable** - Markdown is the durable source of truth.
2. **Runtime state stays local** - SQLite tracks state and LanceDB handles vector, BM25, and scalar-filter search.
3. **Algorithms stay modular** - [EverAlgo](https://github.com/EverMind-AI/EverAlgo) owns memory algorithms; EverOS owns runtime, persistence, online flows, and offline evolution.

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>


## Quick Start

### 1. Install EverOS

```bash
uv pip install everos
# or: pip install everos
```

### 2. Initialize Configuration

Generate a starter `.env` file, then fill the API key fields shown in
the generated comments.

```bash
everos init
```

`everos init` writes `./.env` by default. Use `everos init --xdg` to
write `${XDG_CONFIG_HOME:-~/.config}/everos/.env` instead.

### 3. Start The Server

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

### Optional: Ingest Multimodal Files

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

### For Contributors

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
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Architecture At A Glance

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
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Storage Layout

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
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Features

- **Hybrid retrieval**: BM25 + vector (HNSW/IVF-PQ) + scalar filter, single-query in LanceDB
- **Cascade index sync**: edit a `.md` → file watcher → entry-level diff → LanceDB sync, sub-second
- **Multi-source extraction**: conversations / agent trajectories / file knowledge
- **Dual-track memory**: user-track (Episodes / Profiles) + agent-track (Cases / Skills)
- **Async-first**: full asyncio, single event loop
- **Multi-modal**: text + small image / audio inline; large media via S3/OSS reference

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Project Structure

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
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

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
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>


## Use Cases

Use cases show what persistent memory makes possible in real products and
workflows. Some examples are packaged in this repository; others point to
external demos or integrations you can study and adapt.

<table>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/840470d7-a838-4c05-8685-dd797d4e9cdf)](https://evermind.ai/usecase_reunite)

#### Reunite - Find With EverOS

Parents describe what they remember. Children describe what they recall. Reunite uses semantic memory to surface the connections.

[Learn more](https://evermind.ai/usecase_reunite)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/7282b38b-56bf-4356-aa7b-06a845e7683d)](https://github.com/tt-a1i/hive)

#### Hive Orchestrator

Browser-native hive-mind for CLI coding agents - Claude Code, Codex, Gemini, and OpenCode collaborate as real PTY processes via a team protocol.

[Code](https://github.com/tt-a1i/hive)

</td>
</tr>

<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/867d9329-ce9a-496f-ab1e-15c77974e5fa)](https://github.com/tt-a1i/evermemos-mcp)

#### AI Coding Assistants With EverOS

Universal long-term memory layer for AI coding assistants, powered by EverOS.

[Code](https://github.com/tt-a1i/evermemos-mcp)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/a4f0fd86-1c81-4445-bebc-e51eb5e33b30)](https://github.com/yuansui123/AI-Data-Technician-EverMemOS)

#### AI Data Technician

An agentic AI system that learns from scientist interaction to inspect, analyze, and classify high-dimensional time series data - with persistent memory that improves across sessions.

[Code](https://github.com/yuansui123/AI-Data-Technician-EverMemOS)

</td>
</tr>

<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/650b901b-c9ba-4001-bac7-626b009df830)

#### Rokid AI Assistant With EverOS

Connect to EverOS within Rokid Glasses enabling long-term memory for all of your smart activities.

Coming soon

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/85b338b2-e48e-4a65-9f30-0bc6998df872)

#### Creative Assistant With Memory

Creative assistant with long-term memory, so your creative context stays available across sessions.

Coming soon

</td>
</tr>

<tr>
<td colspan="2" align="right">
<a href="#readme-top"><img src="https://img.shields.io/badge/-Back_to_top-gray?style=flat-square" alt="Back to top"></a>
</td>
</tr>

<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/f30617a1-adc0-4271-bc0e-c3a0b28cb903)](https://github.com/xunyud/Earth-Online)

#### Earth Online Memory Game

Earth Online is a memory-aware productivity game that turns everyday planning into a living quest log.

[Code](https://github.com/xunyud/Earth-Online)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/57d8cda7-35a5-4561-b794-5520dffc917b)](https://github.com/golutra/golutra)

#### Multi-Agent Orchestration Platform

Golutra presents a multi-agent workforce for engineering teams, extending the IDE model from a single assistant to coordinated agents.

[Code](https://github.com/golutra/golutra)

</td>
</tr>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/75f19db5-30f6-4eed-9b1e-c9c6a0e6b7de)](https://github.com/Yangtze-Seventh/taste-verse)

#### Your Personal Tasting Universe

Record, visualize, and explore your tasting journey through an immersive 3D star map.

[Code](https://github.com/Yangtze-Seventh/taste-verse)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/93ac2a68-4f18-4fcb-8d87-80aeb00a9d7c)](https://github.com/kellyvv/OpenHer)

#### EverOS Open Her

Build AI that feels. Open-source persona engine - personality emerges from neural drives, not prompts. Inspired by Her.

[Code](https://github.com/kellyvv/OpenHer)

</td>
</tr>

<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/550071c1-dc39-4964-9f67-ffdfad792345)](https://chromewebstore.google.com/detail/ruminer-browser-agent/lbccjohfpdpimbhpckljimgolndfmfif)

#### Browser Agent For Personal Memory

Ruminer brings persistent memory to a browser agent so it can carry personal context across web tasks.

[Plugin](https://chromewebstore.google.com/detail/ruminer-browser-agent/lbccjohfpdpimbhpckljimgolndfmfif)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/c258a6c4-fe70-497a-98d1-3dade4a932f6)](https://github.com/nanxingw/EverMem)

#### EverMem Sync With EverOS

One command to connect any AI coding CLI to EverMemOS long-term memory.

[Code](https://github.com/nanxingw/EverMem)

</td>
</tr>

<tr>
<td colspan="2" align="right">
<a href="#readme-top"><img src="https://img.shields.io/badge/-Back_to_top-gray?style=flat-square" alt="Back to top"></a>
</td>
</tr>

<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/39274473-ceb3-48fb-a031-e22230decbe2)](https://github.com/mco-org/mco)

#### MCO - Orchestrate AI Coding Agents

MCO equips your primary agent with an agent team that can work together to solve complex tasks.

[Code](https://github.com/mco-org/mco)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/314c9126-8e08-4688-bbbb-8555ad58cf67)](https://github.com/onenewborn/StudyBuddy-public)

#### Study Buddy With Self-Evolving Memory

Study proactively with an agent that has self-evolving memory.

[Code](https://github.com/onenewborn/StudyBuddy-public)

</td>
</tr>

<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/21da76aa-9a8b-48e0-9134-42429d7390e7)](https://github.com/TonyLiangDesign/MemoCare)

#### Alzheimer's Memory Assistant

Empowering individuals with advanced memory support and daily assistance.

[Code](https://github.com/TonyLiangDesign/MemoCare)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/e2428df3-ea11-4e88-8f9c-dad437dd8998)](https://github.com/AlexL1024/NeuralConnect)

#### Memory-Driven Multi-Agent NPC Experience

An iOS sci-fi mystery game where players explore and uncover the truth.

[Code](https://github.com/AlexL1024/NeuralConnect)

</td>
</tr>

<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/e6eaf308-a874-483f-8874-6934bf95a78f)](https://github.com/elontusk5219-prog/Mobi)

#### Mobi Companion

An iOS app where users create, nurture, and live with a personalized AI companion called Mobi.

[Code](https://github.com/elontusk5219-prog/Mobi)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/9aabcaa9-f97a-49d2-9109-0b5bb696ed41)](https://github.com/JaMesLiMers/EvermemCompetition-Spiro)

#### AI Wearable With Memory

A context-native AI wearable that listens to everyday life and converts conversations into memory.

[Code](https://github.com/JaMesLiMers/EvermemCompetition-Spiro)

</td>
</tr>

<tr>
<td colspan="2" align="right">
<a href="#readme-top"><img src="https://img.shields.io/badge/-Back_to_top-gray?style=flat-square" alt="Back to top"></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/df9677ec-386f-4c56-a428-08bca25c54dc)](docs/migration-to-1.0.0.md)

#### Legacy OpenClaw Agent Memory

Archived pre-1.0.0 plugin reference. New integrations should use the EverOS 1.0.0 API.

[Learn more](docs/migration-to-1.0.0.md)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/3a2357a1-c0c3-464a-8979-0d1cdfc9b0d4)](https://github.com/TEN-framework/ten-framework/tree/04cb80601374fa9e35b4e544b2dbd23286ca7763/ai_agents/agents/examples/voice-assistant-with-EverMemOS)

#### Live2D Character With Memory

Add long-term memory to a real-time Live2D character, powered by [TEN Framework](https://github.com/TEN-framework/ten-framework).

[Code](https://github.com/TEN-framework/ten-framework/tree/04cb80601374fa9e35b4e544b2dbd23286ca7763/ai_agents/agents/examples/voice-assistant-with-EverMemOS)

</td>
</tr>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/c36bdc04-97d3-4fe9-97d9-4b93b475595a)](https://screenshot-analysis-vercel.vercel.app/)

#### Computer-Use With Memory

Run screenshot-based analysis with computer-use and store the results in memory.

[Live Demo](https://screenshot-analysis-vercel.vercel.app/)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/54a7cf8f-62c4-4fbc-9d50-b214d034e051)](use-cases/game-of-throne-demo)

#### Game Of Thrones Memories

A demonstration of AI memory infrastructure through an interactive Q&A experience with *A Game of Thrones*.

[Code](use-cases/game-of-throne-demo)

</td>
</tr>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/af37c1f6-7ba5-430c-b99d-2a7e7eac618f)](use-cases/claude-code-plugin)

#### Claude Code Plugin

Persistent memory for Claude Code. Automatically saves and recalls context from past coding sessions.

[Code](use-cases/claude-code-plugin)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/d521d28c-0ccd-44ff-aecc-828245e2f973)](https://main.d2j21qxnymu6wl.amplifyapp.com/graph.html)

#### Memory Graph Visualization

Explore stored entities and relationships in a graph interface. Frontend demo; backend integration is in progress.

[Live Demo](https://main.d2j21qxnymu6wl.amplifyapp.com/graph.html)

</td>
</tr>
</table>

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Stay Tuned

Star the repo or join the community links above to follow new architecture methods, benchmark releases, memory-enabled use cases, Wiki-style memory, and Dreaming updates.

![star us gif](https://github.com/user-attachments/assets/0c512570-945a-483a-9f47-8e067bd34484)

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=EverMind-AI/EverOS&type=Date)](https://www.star-history.com/#EverMind-AI/EverOS&Date)

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## EverMind Ecosystems

EverMind is an open-source ecosystem for long-term memory, self-evolving agents, and memory evaluation. EverOS is the core runtime architecture; EverMemOS is the paper and research line carrying our strongest memory-system benchmark runs; EverAlgo supplies the next-generation algorithms that make the system modular and reusable.

<table>
<tr>
<th colspan="2">EverMind Open-Source Ecosystem</th>
</tr>
<tr>
<td><strong>Core Memory Architecture</strong></td>
<td><a href="https://github.com/EverMind-AI/EverOS">EverOS</a> / EverMemOS - the local memory operating system and research-backed runtime for agent and user memory.</td>
</tr>
<tr>
<td><strong>Algorithm Engine</strong></td>
<td><a href="https://github.com/EverMind-AI/EverAlgo">EverAlgo</a> - stateless extraction, ranking, parsing, and memory operators that power EverOS.</td>
</tr>
<tr>
<td><strong>Alternative Architecture</strong></td>
<td><a href="https://github.com/EverMind-AI/HyperMem">HyperMem</a> - hypergraph memory for long-term conversations, with its own benchmark-backed topic -> episode -> fact retrieval method.</td>
</tr>
<tr>
<td><strong>Benchmarks</strong></td>
<td><a href="https://github.com/EverMind-AI/EverMemBench">EverMemBench</a> · <a href="https://github.com/EverMind-AI/EvoAgentBench">EvoAgentBench</a> - evaluation suites for conversational memory and agent self-evolution.</td>
</tr>
<tr>
<td><strong>Long-Context Research</strong></td>
<td><a href="https://github.com/EverMind-AI/MSA">MSA</a> - Memory Sparse Attention for scalable latent memory and 100M-token contexts.</td>
</tr>
<tr>
<td><strong>Personal Memory Layer</strong></td>
<td><a href="https://github.com/EverMind-AI/EverMe">EverMe</a> - CLI and agent plugin suite for cross-device, cross-agent personal memory.</td>
</tr>
<tr>
<td><strong>Developer Integrations</strong></td>
<td><a href="https://github.com/EverMind-AI/evermem-claude-code">evermem-claude-code</a> · <a href="https://github.com/EverMind-AI/everos-plugins">everos-plugins</a> - plugins, skills, and migration tooling for AI coding agents.</td>
</tr>
</table>

Together, these repositories form EverMind's research-to-runtime stack: new memory methods, reusable algorithms, benchmark evidence, and practical agent integrations.

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

<br>

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
