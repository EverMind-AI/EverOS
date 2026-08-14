<div align="center" id="readme-top">

![EverOS banner](https://github.com/user-attachments/assets/806e9d7f-c861-4b89-9141-11e38f8753e3)

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

- [Why Ever OS](#why-ever-os)
- [Quick Start](#quick-start)
- [Use Cases](#use-cases)
- [Documentation](#documentation)
- [EverMind Ecosystems](#evermind-ecosystems)
- [Contributing](#contributing)

<br>

</details>


## Why Ever OS

EverOS is a Python library and local-first memory runtime for agents and
makers. It gives one portable memory layer across coding assistants, apps,
devices, and workflows from day one. It stores conversations, files, and agent
trajectories as readable Markdown, then syncs local SQLite and LanceDB indexes
for fast retrieval and self-evolving reuse.

<table>
<tr>
<th width="28%">Title</th>
<th width="36%">EverOS</th>
<th width="36%">Other Agent Memory Libraries</th>
</tr>
<tr>
<td><strong>Markdown source of truth</strong></td>
<td>✅ Canonical <code>.md</code> files that are readable, editable, diffable, and Git-versioned</td>
<td>❌ Usually API, vector, graph, dashboard, or database state</td>
</tr>
<tr>
<td><strong>Direct file editing</strong></td>
<td>✅ Edit <code>.md</code> files; cascade watcher syncs</td>
<td>❌ Usually SDK, API, dashboard, or backend update paths</td>
</tr>
<tr>
<td><strong>Local three-part stack</strong></td>
<td>✅ Markdown + SQLite + LanceDB; no MongoDB, Elasticsearch, or Redis required</td>
<td>❌ Often depends on managed services, vector DBs, graph DBs, or server stacks</td>
</tr>
<tr>
<td><strong>User + agent tracks</strong></td>
<td>✅ User <code>episodes/profile</code> and agent <code>cases/skills</code> are separate first-class surfaces</td>
<td>❌ Usually centered on chat history, profiles, entities, facts, or retrieval records</td>
</tr>
<tr>
<td><strong>Orthogonal retrieval</strong></td>
<td>✅ Search by <code>user_id</code>, <code>agent_id</code>, <code>app_id</code>, <code>project_id</code>, and <code>session_id</code></td>
<td>❌ Usually app, namespace, tenant, thread, or graph scoped</td>
</tr>
<tr>
<td><strong>Knowledge Wiki</strong></td>
<td>✅ Editable, source-backed Markdown knowledge pages with taxonomy, CRUD APIs, and topic search</td>
<td>❌ Usually separate from memory, trapped in a dashboard, or not tied back to source files</td>
</tr>
<tr>
<td><strong>Reflection</strong></td>
<td>✅ Offline memory evolution that merges episode clusters and refines profiles and skills between sessions</td>
<td>❌ Usually retrieval-only memory with little background consolidation or long-horizon improvement</td>
</tr>
</table>

<br>

## Quick Start

> One OpenRouter API key is enough to start EverOS, write durable memories,
> and retrieve them with keyword search.

### Prerequisites

- Python 3.12+
- One [OpenRouter API key](https://openrouter.ai/keys)

### 1. Install

```bash
uv pip install everos
# or: pip install everos
```

### 2. Try the standalone demo — no key required

No API key or server setup required—run one command to quickly experience how
EverOS stores and recalls memory:

```bash
everos demo
```

Enter something EverOS should remember, then ask a related question to watch
the memory move through ingest -> extract -> index -> recall.

<https://github.com/user-attachments/assets/98cb8e1e-2ca8-4504-b0a6-0b9a040a0a5c>

### 3. Initialize and add your OpenRouter key

```bash
everos init
```

This creates `~/.everos/everos.toml` and `~/.everos/ome.toml`. Open
`~/.everos/everos.toml`; the generated model and OpenRouter URL are already
correct, so replace only the empty `api_key`:

```toml
[llm]
model = "openai/gpt-4.1-mini"
api_key = "<OPENROUTER_API_KEY>"
base_url = "https://openrouter.ai/api/v1"
```

This is the smallest Tier 1 setup: memory add, flush, Markdown persistence,
cascade indexing, and keyword search.

Use `everos init --root <path>` if you want a different memory root. Pass the
same `--root <path>` to subsequent commands.

### 4. Start EverOS

```bash
everos server start
```

Keep the server running, then open a second terminal and check it:

```bash
curl http://127.0.0.1:8000/health
```

Look for `"status":"ok"`. With this one-key setup, `capabilities.llm` is
`true`; embedding and rerank remain `false` until you configure them.

### 5. Add and retrieve your first memory

> [!NOTE]
> Business endpoints live under `/api/v2`. The older `/api/v1` prefix still
> resolves to the same handlers so existing integrations keep working, but it
> is a legacy alias that may be removed in a future major release — write new
> code against `/api/v2`.

Add a tiny conversation:

```bash
TS=$(($(date +%s)*1000))

curl -X POST http://127.0.0.1:8000/api/v2/memory/add \
  -H 'Content-Type: application/json' \
  -d "{
    \"session_id\": \"demo-001\",
    \"app_id\": \"default\",
    \"project_id\": \"default\",
    \"messages\": [
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $TS, \"content\": \"I love climbing in Yosemite every spring.\"},
      {\"sender_id\": \"alice\", \"role\": \"user\", \"timestamp\": $((TS+10000)), \"content\": \"My favorite coffee shop is Blue Bottle in SOMA.\"}
    ]
  }"
```

Flush the memory at the end of the session:

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/flush \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-001","app_id":"default","project_id":"default"}'
```

Search it back:

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "app_id": "default",
    "project_id": "default",
    "query": "Where do I like to climb?",
    "method": "keyword",
    "top_k": 5
  }'
```

You should see the Yosemite memory in the response. Keep
`"method": "keyword"` in this one-key setup because the API defaults to hybrid
search, which requires an embedding provider.

> [!TIP]
> **First memory unlocked.**
> You just gave EverOS a fact, flushed it into durable Markdown-backed memory,
> and searched it back through the local index. That is the core loop.
> Want to see the source of truth? Open `~/.everos` and inspect the generated
> Markdown files.

For annotated responses and the Markdown files EverOS creates, see
[QUICKSTART.md](QUICKSTART.md).

### What works with one key?

The OpenRouter one-key setup is EverOS Tier 1. It supports server startup,
memory add and flush, durable Markdown storage, cascade indexing, and keyword
search. Add optional providers only when you need the features below:

| Configuration | Adds |
| --- | --- |
| `[llm]` only | Core memory flow and keyword search |
| Add `[embedding]` | Vector/user hybrid search, reflection, and skill extraction |
| Add `[rerank]` too | Agentic search, default agent hybrid search, and Knowledge Wiki |
| Add `[multimodal]` and parser extra | Image, PDF, audio, and office-file ingestion |

Missing optional capabilities are reported by `/health` and return a clear
HTTP 422 if you request a feature that needs them.

> [!NOTE]
> `everos demo --live` is different from the standalone demo in step 2: it
> connects to a running server and uses the real add/flush/search flow. It uses
> hybrid search, so add an embedding provider before you run it.

### Optional: Ingest Multimodal Files

To ingest non-text content (image / pdf / audio / office documents)
through `/api/v2/memory/add` `content` items, install the optional
extra:

```bash
uv pip install 'everos[multimodal]'   # or: pip install 'everos[multimodal]'
```

This pulls in `everalgo-parser` (with the `[svg]` bundle for SVG support via
cairosvg). Configure the `[multimodal]` section in `everos.toml`; its default
model is `google/gemini-3-flash-preview` via OpenRouter.

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
everos demo --plain                  # try the local educational demo; no API keys needed
everos init                          # add one OpenRouter key to ~/.everos/everos.toml

everos --help
make test
```

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Use Cases

Now that you have had your first successful EverOS moment, explore what people
are building with persistent memory across agents, apps, and community
integrations.

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

Archived pre-1.0.0 plugin reference. New integrations should use the current EverOS API.

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

## Documentation

- [docs/everos-demo.md](docs/everos-demo.md) — Demo scope and TUI source layout
- [docs/how-memory-works.md](docs/how-memory-works.md) — Markdown, SQLite, LanceDB, and recall flow
- [docs/use-cases.md](docs/use-cases.md) — Full use-case gallery and integration examples
- [docs/engineering.md](docs/engineering.md) — Contributor engineering reference: build, test, CI, conventions
- [docs/migration-to-1.0.0.md](docs/migration-to-1.0.0.md) — Legacy API migration notes
- [CHANGELOG.md](CHANGELOG.md) — Release notes
- [CONTRIBUTING.md](CONTRIBUTING.md) — How to contribute

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## EverMind Ecosystems

EverMind is an open-source ecosystem for long-term memory, self-evolving
agents, AI-native interfaces, and memory evaluation.

<table>
<tr>
<th colspan="2">EverMind Open-Source Ecosystem</th>
</tr>
<tr>
<td><strong>Memory Runtime</strong></td>
<td><a href="https://github.com/EverMind-AI/EverOS">EverOS</a> - the local memory operating system and research-backed runtime for agent and user memory.</td>
</tr>
<tr>
<td><strong>Self-Improving Agent Harness</strong></td>
<td><a href="https://github.com/EverMind-AI/Raven">Raven</a> - the self-improving agent harness that brings memory, proactivity, context control, and skill evolution into terminal-native agents.</td>
</tr>
<tr>
<td><strong>Algorithm Engine</strong></td>
<td><a href="https://github.com/EverMind-AI/EverAlgo">EverAlgo</a> - stateless extraction, ranking, parsing, and memory operators that power EverOS.</td>
</tr>
<tr>
<td><strong>Hypergraph Memory</strong></td>
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

Contributions are welcome across the whole repository: memory methods, benchmark coverage, use-case examples, documentation, and bug fixes. Browse [Issues](https://github.com/EverMind-AI/EverOS/issues) to find a good entry point, then open a PR when you are ready.

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
