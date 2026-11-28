<div align="center" id="readme-top">

<h1>EverOS</h1>

**One memory. Every agent.**

A local-first, Markdown-native memory layer for AI agents.

<p align="center">
<a href="https://x.com/evermind"><img src="https://img.shields.io/badge/EverMind-000000?labelColor=gray&style=for-the-badge&logo=x&logoColor=white" alt="X"></a>
  <a href="https://huggingface.co/EverMind-AI"><img src="https://img.shields.io/badge/🤗_HuggingFace-EverMind-F5C842?labelColor=gray&style=for-the-badge" alt="HuggingFace"></a>
  <a href="https://discord.gg/QHQ2YYp7cH"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FQHQ2YYp7cH%3Fwith_counts%3Dtrue&query=%24.approximate_presence_count&suffix=%20online&label=Discord&color=404EED&labelColor=gray&style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/EverMind-AI/EverOS/discussions/67"><img src="https://img.shields.io/badge/WeCom-EverMind_社区-07C160?labelColor=gray&style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a> <!-- cjk-allow: community name -->
</p>

[Website](https://evermind.ai) · [Documentation](https://docs.evermind.ai) · [Blog](https://evermind.ai/blogs) · [中文](README.zh-CN.md) <!-- cjk-allow: language switcher -->

[![LoCoMo: 94.61% re-judged](https://img.shields.io/badge/LoCoMo-94.61%25*-F5C842?style=flat-square&labelColor=282722)](#benchmarks)
[![PyPI](https://img.shields.io/pypi/v/everos?style=flat-square&labelColor=282722&color=F5C842)](https://pypi.org/project/everos/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-F5C842?style=flat-square&labelColor=282722)](LICENSE)

</div>

<p align="center">
<a href="#quick-start">Quick Start</a> · <a href="#why-everos">Why EverOS</a> · <a href="#benchmarks">Benchmarks</a> · <a href="#ecosystem">Ecosystem</a> · <a href="#use-cases">Use Cases</a> · <a href="#documentation">Docs</a>
</p>

EverOS stores conversations, files, and agent trajectories as readable Markdown.
SQLite tracks state; LanceDB indexes the files for retrieval. Read, edit, diff,
and version your memory with the tools you already use.

## Quick Start

Requires **Python 3.12+** and one [OpenRouter API key](https://openrouter.ai/keys).
This setup runs locally with an LLM provider and keyword search.

### 1. Install and initialize

```bash
pip install everos
# or: uv pip install everos
everos init
```

### 2. Add your key

Open `~/.everos/everos.toml` and fill in `[llm]`:

```toml
[llm]
model = "openai/gpt-4.1-mini"
api_key = "<OPENROUTER_API_KEY>"
base_url = "https://openrouter.ai/api/v1"
```

### 3. Start the server

```bash
everos server start
```

Keep it running. In a second terminal, check `curl http://127.0.0.1:8000/health`
for `"status":"ok"`.

### 4. Write and recall a memory

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

New memories may take a moment to reach the search index. If the first search
is empty, wait briefly and retry. Open `~/.everos` to inspect the Markdown files.

[Full walkthrough](QUICKSTART.md) · [DashScope setup](QUICKSTART.zh-CN.md) · [Windows / WSL2](docs/windows.md)

## Why EverOS

| Capability | What you get |
|---|---|
| **Memory you own** | Canonical Markdown files on your machine; rebuildable local indexes. |
| **Shared across agents** | User profiles and episodes alongside agent cases and skills, scoped by user, agent, app, project, and session. |
| **Several ways to recall** | Keyword, vector, hybrid, and multi-round agentic retrieval. Start with keyword; add providers as needed. |
| **Knowledge Wiki** | Editable, source-backed Markdown knowledge pages with taxonomy, CRUD APIs, and topic search. |
| **Reflection** | Background consolidation that merges episode clusters and refines profiles and skills. |

The local stack is **Markdown + SQLite + LanceDB**. No MongoDB, Elasticsearch,
or Redis is required. LLM, embedding, and reranking calls use the providers you
configure. See [how memory works](docs/how-memory-works.md) and
[provider requirements](QUICKSTART.md#upgrade-capabilities-when-you-need-them).

## Benchmarks

| Benchmark | Questions | Accuracy | Answer model |
|---|---|---|---|
| [LoCoMo](benchmarks/locomo.md) | 1,540 | **94.61%**\* | `openai/gpt-4.1-mini` |
| LongMemEval | 500 | **94.00%** | `google/gemini-3.6-flash` |
| EverMemBench | 2,400 | **66.67%** | `google/gemini-3-flash-preview` |
| SubtleMemory | 1,522 | **71.75%** | `openai/gpt-5.4` |

\* LoCoMo: published `gpt-4.1-mini` configuration, best of three `agentic`
runs. The 99 disputed gold answers are graded against the audit's corrected
answers. [Full results and scoring protocol](benchmarks/locomo.md).

**Search latency:** `hybrid` search has a **0.27 s median** in the LoCoMo
latency measurements and calls no LLM. This is a separate retrieval path from
the `agentic` run behind the 94.61% score; it is not a p95 or end-to-end answer
latency claim.

**Under load:** the pre-release 1.8.0 build sustained **14–16 hybrid searches/s**
and **120–140 keyword searches/s** on a 4-core, 8 GB server with 168,821 memories.
Model providers were stubbed to measure EverOS itself.
[Throughput setup and limits](benchmarks/throughput.md).

[Reproduce the benchmarks →](benchmarks/README.md)

## Ecosystem

Connect your agent to EverOS through a host plugin, MCP, or the HTTP API.
[Raven](https://github.com/EverMind-AI/Raven) includes EverOS memory.

| Host / workflow | Setup | Release status |
|---|---|---|
| Claude Code · Codex | [Claude Code plugin](https://github.com/EverMind-AI/plugins/tree/main/claude-code) · [Codex plugin](https://github.com/EverMind-AI/plugins/tree/main/codex) | Pre-release verification |
| DeepSeek Harness · Hermes | [DSH plugin](https://github.com/EverMind-AI/plugins/tree/main/dsh) · [Hermes plugin](https://github.com/EverMind-AI/plugins/tree/main/hermes) | Pre-release verification |
| OpenClaw | [Plugin and setup guide](https://github.com/EverMind-AI/plugins/tree/main/openclaw) | Package scope migration; see guide |
| Dify | [Self-hosted](https://github.com/EverMind-AI/plugins/tree/main/dify) · [Cloud](https://github.com/EverMind-AI/plugins/tree/main/dify_cloud) | Marketplace submission pending |
| Claude Desktop · Cursor · other MCP hosts | [MCP setup guide](docs/mcp.md) | Separate `everos-mcp` server |
| Your own app or agent | [HTTP API](docs/api.md) · [Python library](src/everos/README.md) | Included in EverOS |

Host plugins automate recall and capture; Dify exposes tools you place in your
workflow. Check the [official integration registry](https://github.com/EverMind-AI/plugins#plugins)
for current release status.

Building with EverOS? [Share an integration or use case](https://github.com/EverMind-AI/EverOS/issues/new/choose).

## Use Cases

A few projects from the community. Browse the [full gallery](docs/use-cases.md) for more examples.

<table>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/7282b38b-56bf-4356-aa7b-06a845e7683d)](https://github.com/tt-a1i/hive)

#### Hive Orchestrator

Browser-native hive-mind for CLI coding agents - Claude Code, Codex, Gemini, and OpenCode collaborate as real PTY processes via a team protocol.

[Code](https://github.com/tt-a1i/hive)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/550071c1-dc39-4964-9f67-ffdfad792345)](https://chromewebstore.google.com/detail/ruminer-browser-agent/lbccjohfpdpimbhpckljimgolndfmfif)

#### Browser Agent for Personal Memory

Ruminer brings persistent memory to a browser agent so it can carry personal context across web tasks.

[Plugin](https://chromewebstore.google.com/detail/ruminer-browser-agent/lbccjohfpdpimbhpckljimgolndfmfif)

</td>
</tr>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/314c9126-8e08-4688-bbbb-8555ad58cf67)](https://github.com/onenewborn/StudyBuddy-public)

#### Study Buddy with Self-Evolving Memory

Study proactively with an agent that has self-evolving memory.

[Code](https://github.com/onenewborn/StudyBuddy-public)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/3a2357a1-c0c3-464a-8979-0d1cdfc9b0d4)](https://github.com/TEN-framework/ten-framework/tree/04cb80601374fa9e35b4e544b2dbd23286ca7763/ai_agents/agents/examples/voice-assistant-with-EverMemOS)

#### Live2D Character with Memory

Add long-term memory to a real-time Live2D character, powered by [TEN Framework](https://github.com/TEN-framework/ten-framework).

[Code](https://github.com/TEN-framework/ten-framework/tree/04cb80601374fa9e35b4e544b2dbd23286ca7763/ai_agents/agents/examples/voice-assistant-with-EverMemOS)

</td>
</tr>
</table>

## Documentation

| I want to… | Start here |
|---|---|
| Run the complete local walkthrough | [Quick Start](QUICKSTART.md) |
| Connect a coding assistant | [MCP](docs/mcp.md) · [Host plugins](https://github.com/EverMind-AI/plugins) |
| Configure models and capabilities | [Configuration](docs/configuration.md) · [Recommended models](QUICKSTART.md#recommended-models) |
| Build against the API | [HTTP API](docs/api.md) · [OpenAPI schema](docs/openapi.json) |
| Work with knowledge, reflection, or files | [Knowledge](docs/knowledge.md) · [Reflection](docs/reflection.md) · [Multimodal](docs/multimodal.md) |
| Understand or contribute to the runtime | [Architecture](docs/architecture.md) · [Engineering](docs/engineering.md) |

[All documentation](docs/index.md) · [Changelog](CHANGELOG.md) · [Migration guide](docs/migration-to-1.0.0.md)

## More from EverMind

[Raven](https://github.com/EverMind-AI/Raven) is an agent harness with EverOS
memory. [EverMe](https://github.com/EverMind-AI/EverMe) brings personal memory
across devices and agents. [EverAlgo](https://github.com/EverMind-AI/EverAlgo)
provides the memory operators used by EverOS.

Research and evaluation: [HyperMem](https://github.com/EverMind-AI/HyperMem) ·
[MSA](https://github.com/EverMind-AI/MSA) ·
[EverMemBench](https://github.com/EverMind-AI/EverMemBench) ·
[EvoAgentBench](https://github.com/EverMind-AI/EvoAgentBench) ·
[SkillCorpus](https://github.com/EverMind-AI/SkillCorpus)

## Contributing

Code, documentation, benchmark reports, and integrations are welcome.
Read [CONTRIBUTING.md](CONTRIBUTING.md), browse
[Issues](https://github.com/EverMind-AI/EverOS/issues), or join
[Discord](https://discord.gg/QHQ2YYp7cH).

[![EverOS contributors](https://contrib.rocks/image?repo=EverMind-AI/EverOS)](https://github.com/EverMind-AI/EverOS/graphs/contributors)

[Apache License 2.0](LICENSE) · [Third-party notices](NOTICE) · [Citation](CITATION.md)

[Back to top ↑](#readme-top)
