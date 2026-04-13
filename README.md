<div align="center" id="readme-top">

![banner-gif][banner-gif]

<p align="center">
  <a href="https://arxiv.org/abs/2601.02163"><img src="https://img.shields.io/badge/arXiv-EverOS-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: EverOS"></a>
  <a href="https://arxiv.org/abs/2604.08256"><img src="https://img.shields.io/badge/arXiv-HyperMem-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: HyperMem"></a>
  <a href="https://arxiv.org/abs/2602.01313"><img src="https://img.shields.io/badge/arXiv-EverMemBench-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: EverMemBench"></a>
  <a href="https://github.com/EverMind-AI/MSA"><img src="https://img.shields.io/badge/arXiv-Memory%20Sparse%20Attention-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: Memory Sparse Attention"></a>
  <a href="https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic"><img src="https://img.shields.io/badge/🤗_HuggingFace-EverMemBench--Dynamic-F5C842?labelColor=gray&style=flat-square" alt="HuggingFace: EverMemBench-Dynamic"></a>
  <a href="https://discord.gg/gYep5nQRZJ"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FgYep5nQRZJ%3Fwith_counts%3Dtrue&query=%24.approximate_presence_count&suffix=%20online&label=Discord&color=404EED&labelColor=gray&style=flat-square&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/EverMind-AI/EverOS/discussions/67"><img src="https://img.shields.io/badge/WeChat-EverMind-07C160?labelColor=gray&style=flat-square&logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

[Website][website] · [Documentation][docs] · [Blog][blog]

</div>

<br>

<details open>
<summary><kbd>Table of Contents</kbd></summary>

<br>

- [Why EverOS](#why-everos)
- [Project Structure](#project-structure)
- [Key Results](#key-results)
- [Use Cases](#use-cases)
- [Quick Start](#quick-start)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Documentation](#documentation)
- [GitHub Codespaces](#github-codespaces)
- [Community](#community)
- [Citation](#citation)
- [Contributing](#contributing)

<br>

</details>

## Project Structure

EverOS is built around two pillars — **Methods** give agents memory and evolution capabilities, and **Benchmarks** objectively measure whether those capabilities actually work.

```
EverOS/
└── benchmarks/
│   ├── evermembench/        # Memory quality evaluation
│   └── evoagentbench/       # Agent self-evolution evaluation
├── methods/
│   ├── everos/             # Long-term memory operating system
│   └── hypermem/           # Hypergraph memory architecture
```

<br>

## EverMind Ecosystem

Large language models are evolving from single-turn chatbots to long-lived interactive agents. But when an agent needs to stay coherent across weeks of conversation, it hits a hard ceiling: finite context windows and fragmented memory.

Expanding the context window is not the answer — ultra-long contexts are expensive and degrade with distance. The future of long-term agents depends on **structured memory organization**.

EverOS is a memory operating system that transforms unlimited interaction streams into a structured "digital brain" through a three-stage memory lifecycle: **episodic trace formation**, **semantic consolidation**, and **reconstructive recall** — enabling agents to continuously convert raw interactions into structured, evolving knowledge.

<div align="center">

Benchmark Stats

<table>
  <tr>
    <td align="center"><strong>93.05%</strong><br><sub>LoCoMo</sub></td>
    <td align="center"><strong>83.00%</strong><br><sub>LongMemEval-S</sub></td>
    <td align="center"><strong>+40%</strong><br><sub>EvoAgent GDPVAL</sub></td>
  </tr>
</table>

</div>

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>



### Methods

<table>
<tr>
<td width="50%">

![banner-gif][banner-gif]

#### EverMemos

A self-organizing memory operating system inspired by biological imprinting. Extracts, structures, and retrieves long-term knowledge from conversations — enabling agents to remember, understand, and continuously evolve.

[Paper][arxiv-evermemos-link] · [Docs](methods/evermemos/) · [Quick Start](#quick-start)

</td>
<td width="50%">

![banner-gif][banner-gif]

#### HyperMem

A hypergraph-based hierarchical memory architecture that captures high-order associations through hyperedges. Organizes memory into topic, event, and fact layers for coarse-to-fine long-term conversation retrieval. LoCoMo 92.73%.

[Paper][arxiv-hypermem-link] · [Docs](methods/hypermem/)

</td>
</tr>
</table>

### Benchmarks

<table>
<tr>
<td width="50%">

#### EverMemBench

Three-layer memory quality evaluation: factual recall, applied reasoning, and personalized generalization. Evaluates memory systems and LLMs under a unified standard.

[Paper][arxiv-evermembench-link] · [Dataset][hf-link] · [Docs](benchmarks/evermembench/)

</td>
<td width="50%">

#### EvoAgentBench

Agent self-evolution evaluation — not static snapshots, but longitudinal growth curves. Measures transfer efficiency, error avoidance, and skill-hit quality through controlled experiments with and without evolution.

[Docs](benchmarks/evoagentbench/)

</td>
</tr>
</table>

> All benchmarks are designed as **open public standards**. Any memory architecture or agent framework can be evaluated under the same ruler.

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Key Results

### Memory Performance

<div align="center">

| System | LoCoMo | LongMemEval-S |
| :--- | :----: | :----: |
| **EverOS** | **93.05%** | **83.00%** |
| **HyperMem** | **92.73%** | — |
| Mem0 | 78.4% | — |
| MemOS | 74.2% | — |
| Zep | 71.6% | — |

</div>

### Self-Evolution Gains

<div align="center">

| Task Type | Agent + LLM | Baseline | + EverOS Skills | Delta |
| :--- | :--- | :----: | :----: | :----: |
| Code (Django) | OpenClaw + Qwen3.5-397B | 37% | 58% | **+21%** |
| Code (Django) | Nanobot + Qwen3.5-397B | 21% | 47% | **+26%** |
| General (GDPVAL) | OpenClaw + Qwen3.5-397B | 29% | 69% | **+40%** |
| General (GDPVAL) | OpenClaw + Qwen3.5-27B | 41% | 61% | **+20%** |

</div>

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Use Cases

[![EverMind + OpenClaw Agent Memory and Plugin][usecase-openclaw-image]][usecase-openclaw-link]

**EverMind + OpenClaw Agent Memory and Plugin**

Imagine a 24/7 agent with continuous learning memory that you can carry with you wherever you go. Check out the [agent_memory][usecase-openclaw-link] branch and the [plugin][usecase-openclaw-plugin-link] for more details.

![divider][divider-light]
![divider][divider-dark]

<br>

[![Live2D Character with Memory][usecase-live2d-image]][usecase-live2d-link]

**Live2D Character with Memory**

Add long-term memory to your anime character that can talk to you in real-time powered by [TEN Framework][ten-framework-link].
See the [Live2D Character with Memory Example][usecase-live2d-link] for more details.

![divider][divider-light]
![divider][divider-dark]

<br>

[![Computer-Use with Memory][usecase-computer-image]][usecase-computer-link]

**Computer-Use with Memory**

Use computer-use to launch screenshot-based analysis, all stored in your memory.
See the [live demo][usecase-computer-link] for more details.

![divider][divider-light]
![divider][divider-dark]

<br>

[![Game of Thrones Memories][usecase-got-image]][usecase-got-link]

**Game of Thrones Memories**

A demonstration of AI memory infrastructure through an interactive Q&A experience with "A Game of Thrones".
See the [code][usecase-got-link] for more details.

![divider][divider-light]
![divider][divider-dark]

<br>

[![EverOS Claude Code Plugin][usecase-claude-image]][usecase-claude-link]

**EverOS Claude Code Plugin**

Persistent memory for Claude Code. Automatically saves and recalls context from past coding sessions.
See the [code][usecase-claude-link] for more details.

![divider][divider-light]
![divider][divider-dark]

<br>

[![Visualize Memories with Graphs][usecase-graph-image]][usecase-graph-link]

**Visualize Memories with Graphs**

Memory Graph view that visualizes your stored entities and how they relate. This is a pure frontend demo which has not been plugged into the backend yet — we are working on it.
See the [live demo][usecase-graph-link].

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Quick Start

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
```

Then navigate to the component you need:

| | Use Case | Entry Point |
| :-- | :--- | :--- |
| **EverMemos** | Build agents with long-term memory | [methods/evermemos/](methods/evermemos/) |
| **HyperMem** | Use the hypergraph memory architecture | [methods/hypermem/](methods/hypermem/) |
| **EverMemBench** | Evaluate memory system quality | [benchmarks/evermembench/](benchmarks/evermembench/) |
| **EvoAgentBench** | Measure agent self-evolution | [benchmarks/evoagentbench/](benchmarks/evoagentbench/) |

> Each component has its own installation guide, dependency configuration, and usage examples.

### EverMemos Quick Start

```bash
cd methods/evermemos

# Start Docker services
docker compose up -d

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Configure API keys
cp env.template .env
# Edit .env and set:
#   - LLM_API_KEY (for memory extraction)
#   - VECTORIZE_API_KEY (for embedding/rerank)

# Start server
uv run python src/run.py

# Verify installation
curl http://localhost:1995/health
# Expected response: {"status": "healthy", ...}
```

Server runs at `http://localhost:1995` · [Full Setup Guide][setup-guide]

### Basic Usage

Store and retrieve memories with simple Python code:

```python
import requests

API_BASE = "http://localhost:1995/api/v1"

# 1. Store a conversation memory
requests.post(f"{API_BASE}/memories", json={
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+00:00",
    "sender": "user_001",
    "content": "I love playing soccer on weekends"
})

# 2. Search for relevant memories
response = requests.get(f"{API_BASE}/memories/search", json={
    "query": "What sports does the user like?",
    "user_id": "user_001",
    "memory_types": ["episodic_memory"],
    "retrieve_method": "hybrid"
})

result = response.json().get("result", {})
for memory_group in result.get("memories", []):
    print(f"Memory: {memory_group}")
```

[More Examples][usage-examples] · [API Reference][api-docs] · [Interactive Demos][interactive-demos]

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Demo

### Run the Demo

```bash
# Terminal 1: Start the API server
uv run python src/run.py

# Terminal 2: Run the simple demo
uv run python src/bootstrap.py demo/simple_demo.py
```

**Try it now**: Follow the [Demo Guide][interactive-demos] for step-by-step instructions.

### Full Demo Experience

```bash
# Extract memories from sample data
uv run python src/bootstrap.py demo/extract_memory.py

# Start interactive chat with memory
uv run python src/bootstrap.py demo/chat_with_memory.py
```

See the [Demo Guide][interactive-demos] for details.

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Evaluation & Benchmarking

EverOS achieves **93% overall accuracy** on the LoCoMo benchmark, outperforming comparable memory systems.

### Benchmark Results

![EverOS Benchmark Results][benchmark-image]

### Supported Benchmarks

- **[LoCoMo][locomo-link]** — Long-context memory benchmark with single/multi-hop reasoning
- **[LongMemEval][longmemeval-link]** — Multi-session conversation evaluation
- **[PersonaMem][personamem-link]** — Persona-based memory evaluation

### Run Evaluations

```bash
# Install evaluation dependencies
uv sync --group evaluation

# Run smoke test (quick verification)
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# Run full evaluation
uv run python -m evaluation.cli --dataset locomo --system evermemos

# View results
cat evaluation/results/locomo-evermemos/report.txt
```

[Full Evaluation Guide][evaluation-guide] · [Complete Results][evaluation-results-link]

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Documentation

| Guide | Description |
| ----- | ----------- |
| [Quick Start][getting-started] | Installation and configuration |
| [Configuration Guide][config-guide] | Environment variables and services |
| [API Usage Guide][api-usage-guide] | Endpoints and data formats |
| [Development Guide][dev-guide] | Architecture and best practices |
| [Memory API][memory-api-doc] | Complete API reference |
| [Demo Guide][demo-guide] | Interactive examples |
| [Evaluation Guide][evaluation-guide] | Benchmark testing |

### Advanced Techniques

- **[Group Chat Conversations][group-chat-guide]** — Combine messages from multiple speakers
- **[Conversation Metadata Control][metadata-control-guide]** — Fine-grained control over conversation context
- **[Memory Retrieval Strategies][retrieval-strategies-guide]** — Lightweight vs Agentic retrieval modes
- **[Batch Operations][batch-operations-guide]** — Process multiple messages efficiently

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## GitHub Codespaces

EverOS supports [GitHub Codespaces][codespaces-link] for cloud-based development — no Docker setup or local environment configuration needed.

[![Open in GitHub Codespaces][codespaces-badge]][codespaces-project-link]

| Machine Type | Status | Notes |
| ------------ | ------ | ----- |
| 2-core (Free tier) | Not supported | Insufficient resources for infrastructure services |
| 4-core | Minimum | Works but may be slow under load |
| 8-core | Recommended | Good performance with all services |
| 16-core+ | Optimal | Best for heavy development workloads |

> **Note:** If your company provides GitHub Codespaces, hardware limitations typically will not be an issue since enterprise plans often include access to larger machine types.

### Getting Started with Codespaces

1. Click the "Open in GitHub Codespaces" button above
2. Select a **4-core or larger** machine when prompted
3. Wait for the container to build and services to start
4. Update API keys in `.env` (`LLM_API_KEY`, `VECTORIZE_API_KEY`, etc.)
5. Run `make run` to start the server

All infrastructure services (MongoDB, Elasticsearch, Milvus, Redis) start automatically and are pre-configured to work together.

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>


## Citation

If EverOS helps your research, please cite:

```bibtex
@article{evermemos2025,
  title   = {EverMemOS: A Self-Organizing Memory Operating System for AI Agents},
  author  = {EverMind Team},
  journal = {arXiv preprint arXiv:2601.02163},
  year    = {2025}
}

@article{hypermem2026,
  title   = {HyperMem: Hypergraph Memory for Long-Term Conversations},
  author  = {Yue, Hu, Sheng, Zhou, Zhang, Liu, Guo, Deng},
  journal = {arXiv preprint arXiv:2604.08256},
  year    = {2026}
}

@article{evermembench2025,
  title   = {EverMemBench: A Comprehensive Benchmark for Long-Term Memory
             in Conversational AI},
  author  = {EverMind Team},
  journal = {arXiv preprint arXiv:2602.01313},
  year    = {2025}
}

@article{evoagentbench2025,
  title   = {EvoAgentBench: The First Objective Benchmark for Agent Self-Evolution},
  author  = {EverMind Team},
  year    = {2025}
}
```

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Contributing

Browse [Issues][issues-link] to find your entry point, join [Discussions][discussions] to share ideas, or connect with maintainers — [@elliotchen200][elliot-x-link] on X and [@cyfyifanchen][cyfyifanchen-link] on GitHub.

![divider][divider-light]
![divider][divider-dark]

### Code Contributors

[![EverOS Contributors][contributors-image]][contributors]

### Contribution Guidelines

Read our [Contribution Guidelines][contributing-doc] for code standards and Git workflow.

### License

[Apache 2.0][license]

---

<div align="center">

<a href="https://evermind.ai"><strong>EverMind</strong></a> · Keep in Mind. Evolve over Time.

</div>

<!-- Navigation -->
[readme-top]: #readme-top

<!-- Dividers -->
[divider-light]: https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only
[divider-dark]: https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only

<!-- Images -->
[banner-gif]: https://github.com/user-attachments/assets/646e813a-a7a7-4ba2-bda8-d8bdf884a890
[usecase-openclaw-image]: https://github.com/user-attachments/assets/0e06da2b-0236-430f-89b4-980b8b6a855f
[usecase-live2d-image]: https://github.com/user-attachments/assets/a80bdab3-e5d0-43b9-9e8d-0a9605012a26
[usecase-computer-image]: https://github.com/user-attachments/assets/0d306b4c-bcd7-4e9e-a244-22fa3cb7b727
[usecase-got-image]: https://github.com/user-attachments/assets/d1efe507-4eb7-4867-8996-457497333449
[usecase-claude-image]: https://github.com/user-attachments/assets/b40b2241-b0e6-4fc9-9a35-92139f3a2d81
[usecase-graph-image]: https://github.com/user-attachments/assets/6586e647-dd5f-4f9f-9b26-66f930e8241c
[benchmark-image]: figs/benchmark_2.png

<!-- Badges -->
[license-badge]: https://img.shields.io/badge/License-Apache%202.0-blue?labelColor=gray&style=flat-square
[arxiv-evermemos-badge]: https://img.shields.io/badge/arXiv-EverOS-b31b1b?style=flat-square&logo=arxiv&logoColor=white
[arxiv-hypermem-badge]: https://img.shields.io/badge/arXiv-HyperMem-b31b1b?style=flat-square&logo=arxiv&logoColor=white
[arxiv-evermembench-badge]: https://img.shields.io/badge/arXiv-EverMemBench-b31b1b?style=flat-square&logo=arxiv&logoColor=white
[hf-badge]: https://img.shields.io/badge/🤗_EverMemBench--Dynamic-F5C842?style=flat-square
[deepwiki-badge]: https://deepwiki.com/badge.svg
[back-to-top]: https://img.shields.io/badge/-Back_to_top-gray?style=flat-square
[codespaces-badge]: https://github.com/codespaces/badge.svg
[discord-members-badge]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FgYep5nQRZJ%3Fwith_counts%3Dtrue&query=%24.approximate_presence_count&suffix=%20online&label=Discord&color=404EED&style=flat-square&logo=discord&logoColor=white
[wechat-badge]: https://img.shields.io/badge/WeChat-EverMind-07C160?style=flat-square&logo=wechat&logoColor=white

<!-- Primary Links -->
[license]: https://github.com/EverMind-AI/EverOS/blob/main/LICENSE
[website]: https://evermind.ai
[blog]: https://evermind.ai/blogs
[docs]: https://docs.evermind.ai
[api-docs]: https://docs.evermind.ai/api-reference/introduction
[discussions]: https://github.com/EverMind-AI/EverOS/discussions
[discord]: https://discord.gg/gYep5nQRZJ
[wechat]: https://github.com/EverMind-AI/EverOS/discussions/67
[deepwiki]: https://deepwiki.com/EverMind-AI/EverOS

<!-- arXiv Links -->
[arxiv-evermemos-link]: https://arxiv.org/abs/2601.02163
[arxiv-hypermem-link]: https://arxiv.org/abs/2604.08256
[arxiv-evermembench-link]: https://arxiv.org/abs/2602.01313
[hf-link]: https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic

<!-- Use Case Links -->
[usecase-openclaw-link]: https://github.com/EverMind-AI/EverMemOS/tree/agent_memory
[usecase-openclaw-plugin-link]: https://github.com/EverMind-AI/EverMemOS/tree/agent_memory/evermemos-openclaw-plugin
[ten-framework-link]: https://github.com/TEN-framework/ten-framework
[usecase-live2d-link]: https://github.com/TEN-framework/ten-framework/tree/main/ai_agents/agents/examples/voice-assistant-with-EverMemOS
[usecase-computer-link]: https://screenshot-analysis-vercel.vercel.app/
[usecase-got-link]: https://github.com/EverMind-AI/evermem_got_demo
[usecase-claude-link]: https://github.com/EverMind-AI/evermem-claude-code
[usecase-graph-link]: https://main.d2j21qxnymu6wl.amplifyapp.com/graph.html

<!-- Documentation Links -->
[setup-guide]: docs/installation/SETUP.md
[usage-examples]: docs/usage/USAGE_EXAMPLES.md
[interactive-demos]: docs/usage/DEMOS.md
[group-chat-guide]: docs/advanced/GROUP_CHAT_GUIDE.md
[metadata-control-guide]: docs/advanced/METADATA_CONTROL.md
[retrieval-strategies-guide]: docs/advanced/RETRIEVAL_STRATEGIES.md
[batch-operations-guide]: docs/usage/BATCH_OPERATIONS.md
[getting-started]: docs/dev_docs/getting_started.md
[config-guide]: docs/usage/CONFIGURATION_GUIDE.md
[api-usage-guide]: docs/dev_docs/api_usage_guide.md
[dev-guide]: docs/dev_docs/development_guide.md
[memory-api-doc]: docs/api_docs/memory_api.md
[demo-guide]: demo/README.md
[evaluation-guide]: evaluation/README.md

<!-- Evaluation Links -->
[locomo-link]: https://github.com/snap-research/locomo
[longmemeval-link]: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
[personamem-link]: https://huggingface.co/datasets/bowen-upenn/PersonaMem
[evaluation-results-link]: https://huggingface.co/datasets/EverMind-AI/EverMemOS_Eval_Results

<!-- Infrastructure Links -->
[codespaces-link]: https://github.com/features/codespaces
[codespaces-project-link]: https://codespaces.new/EverMind-AI/EverOS

<!-- Community Links -->
[issues-link]: https://github.com/EverMind-AI/EverOS/issues
[elliot-x-link]: https://x.com/elliotchen200
[cyfyifanchen-link]: https://github.com/cyfyifanchen
[contributors-image]: https://contrib.rocks/image?repo=EverMind-AI/EverOS
[contributors]: https://github.com/EverMind-AI/EverOS/graphs/contributors
[contributing-doc]: CONTRIBUTING.md
