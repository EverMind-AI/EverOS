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

- [What is EverOS](#what-is-everos)
- [Architecture at a glance](#architecture-at-a-glance)
- [Quick start](#quick-start)
- [Storage layout](#storage-layout)
- [Features](#features)
- [Project structure](#project-structure)
- [Documentation](#documentation)
- [Use Cases](#use-cases)
- [Stay Tuned](#stay-tuned)
- [Contributing](#contributing)

<br>

</details>


## What is EverOS

EverOS is an open-source Python framework that turns conversations, agent trajectories, and files into **structured, retrievable, evolving long-term memory** for AI agents and user chats. Designed for **lightweight local deployments** (small teams, individual developers), with three core principles:

1. **Markdown as Source of Truth** — All memory persists as plain `.md` files. Open, edit, grep, version with Git, view in Obsidian. No black-box database lock-in.
2. **Lightweight three-piece storage** — `Markdown` files (truth) + `SQLite` (state/queue) + `LanceDB` (vector + BM25 + scalar). No MongoDB / Elasticsearch / Milvus / Redis / Kafka required.
3. **[EverAlgo](https://github.com/EverMind-AI/EverAlgo) as pure algorithm library** — Memory extraction algorithms are decoupled into a separate library; this project orchestrates and persists.

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

## Quick start

### Install as a package

```bash
uv pip install everos               # or: pip install everos

# Generate a starter .env (OpenRouter + DeepInfra defaults; bundled inside the wheel)
everos init                          # writes ./.env (use --xdg for ~/.config/everos/.env)
# Edit .env and fill the API key fields (see comments inside).

everos --help
everos server start
```

`everos server start` searches for `.env` in this order: `--env-file <path>` →
`./.env` (cwd) → `${XDG_CONFIG_HOME:-~/.config}/everos/.env` → `~/.everos/.env`.
The endpoint stack is OpenAI-protocol compatible (OpenAI / OpenRouter / vLLM /
Ollama / DeepInfra …) — override `*__BASE_URL` in the generated `.env` to point
at any of them.

#### Multi-modal (optional)

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

For a step-by-step walkthrough (add a conversation → flush → search →
read the markdown), see [QUICKSTART.md](QUICKSTART.md).


### Develop locally

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
uv sync                              # creates ./.venv and installs deps
source .venv/bin/activate            # — or skip activation and prefix every command with `uv run`
everos init                          # fill the four API key slots in .env (two distinct keys)

everos --help
make test
```

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
- [CHANGELOG.md](CHANGELOG.md) — Release notes
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute
- [.claude/rules/](.claude/rules/) — Detailed coding conventions (auto-loaded by Claude Code)

<br>

## Use Cases

Use cases show what persistent memory makes possible in real products and workflows. Some examples are packaged in this repository; others point to external demos or integrations you can study and adapt.

<table>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/840470d7-a838-4c05-8685-dd797d4e9cdf)](https://evermind.ai/usecase_reunite)

#### Reunite - Find with EverOS

Parents describe what they remember. Children describe what they recall. Reunite uses semantic memory to surface the connections.

[Learn more](https://evermind.ai/usecase_reunite)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/7282b38b-56bf-4356-aa7b-06a845e7683d)](https://github.com/tt-a1i/hive)

#### Hive Orchestrator

Browser-native hive-mind for CLI coding agents — Claude Code, Codex, Gemini, and OpenCode collaborate as real PTY processes via a team protocol.

[Code](https://github.com/tt-a1i/hive)

</td>
</tr>

<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/867d9329-ce9a-496f-ab1e-15c77974e5fa)](https://github.com/tt-a1i/evermemos-mcp)

#### AI Coding Assistants with EverOS

Universal long-term memory layer for AI coding assistants, powered by EverOS.

[Code](https://github.com/tt-a1i/evermemos-mcp)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/a4f0fd86-1c81-4445-bebc-e51eb5e33b30)](https://github.com/yuansui123/AI-Data-Technician-EverMemOS)

#### AI Data Techician

An agentic AI system that learns from scientist interaction to inspect, analyze, and classify high-dimensional time series data — with persistent memory that improves across sessions.

[Code](https://github.com/yuansui123/AI-Data-Technician-EverMemOS)

</td>
</tr>

<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/650b901b-c9ba-4001-bac7-626b009df830)

#### Rokid AI Assistant with EverOS

Connect to EverOS within Rokid Glasses enabling long-term memory for all of your smart activities.

Coming soon

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/85b338b2-e48e-4a65-9f30-0bc6998df872)

#### Creative Assistant with Memory

Creative assistant with long-term memory, never forget your crativites anymore.

Coming soon

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

Build AI that feels. Open-source persona engine — personality emerges from neural drives, not prompts. Inspired by Her.

[Code](https://github.com/kellyvv/OpenHer)

</td>
</tr>

<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/550071c1-dc39-4964-9f67-ffdfad792345)](https://chromewebstore.google.com/detail/ruminer-browser-agent/lbccjohfpdpimbhpckljimgolndfmfif)

#### Browser Agent for Personal Memory

Ruminer brings persistent memory to a browser agent so it can carry personal context across web tasks.

[Plugin](https://chromewebstore.google.com/detail/ruminer-browser-agent/lbccjohfpdpimbhpckljimgolndfmfif)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/c258a6c4-fe70-497a-98d1-3dade4a932f6)](https://github.com/nanxingw/EverMem)

#### EverMem Sync with EverOS

One command to connect any AI coding CLI to EverMemOS long-term memory.

[Code](https://github.com/nanxingw/EverMem)

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

#### Study Buddy with Self-Evolving Memory

Study proactively with an agent that has self-evolving memory.

[Code](https://github.com/onenewborn/StudyBuddy-public)

</td>
</tr>

<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/21da76aa-9a8b-48e0-9134-42429d7390e7)](https://github.com/TonyLiangDesign/MemoCare)

#### Alzheimer’s Memory Assistant

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

#### AI Wearable with Memory

A context-native AI wearable that listens to everyday life and converts conversations into memory.

[Code](https://github.com/JaMesLiMers/EvermemCompetition-Spiro)

</td>
</tr>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/df9677ec-386f-4c56-a428-08bca25c54dc)](https://github.com/EverMind-AI/EverOS/tree/0f49826ba0f9a94e1974c97614a46a68e0a08b52/evermemos-openclaw-plugin)

#### OpenClaw Agent Memory

A 24/7 agent workflow with continuous learning memory across sessions.

[Plugin](https://github.com/EverMind-AI/EverOS/tree/0f49826ba0f9a94e1974c97614a46a68e0a08b52/evermemos-openclaw-plugin)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/3a2357a1-c0c3-464a-8979-0d1cdfc9b0d4)](https://github.com/TEN-framework/ten-framework/tree/04cb80601374fa9e35b4e544b2dbd23286ca7763/ai_agents/agents/examples/voice-assistant-with-EverMemOS)

#### Live2D Character with Memory

Add long-term memory to a real-time Live2D character, powered by [TEN Framework](https://github.com/TEN-framework/ten-framework).

[Code](https://github.com/TEN-framework/ten-framework/tree/04cb80601374fa9e35b4e544b2dbd23286ca7763/ai_agents/agents/examples/voice-assistant-with-EverMemOS)

</td>
</tr>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/c36bdc04-97d3-4fe9-97d9-4b93b475595a)](https://screenshot-analysis-vercel.vercel.app/)

#### Computer-Use with Memory

Run screenshot-based analysis with computer-use and store the results in memory.

[Live Demo](https://screenshot-analysis-vercel.vercel.app/)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/54a7cf8f-62c4-4fbc-9d50-b214d034e051)](use-cases/game-of-throne-demo)

#### Game of Thrones Memories

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

Star the repo or join the community links above to follow new architecture methods, benchmark releases, and memory-enabled use cases.

![star us gif](https://github.com/user-attachments/assets/0c512570-945a-483a-9f47-8e067bd34484)

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
