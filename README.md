<div align="center" id="readme-top">

![banner-gif](https://github.com/user-attachments/assets/0bf97efd-580f-4a53-a2a2-58d6daea7290)

<p align="center">
  <!-- <a href="https://arxiv.org/abs/2601.02163"><img src="https://img.shields.io/badge/arXiv-EverOS-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: EverOS"></a> -->
  <!-- <a href="https://arxiv.org/abs/2604.08256"><img src="https://img.shields.io/badge/arXiv-HyperMem-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: HyperMem"></a> -->
  <!-- <a href="https://arxiv.org/abs/2602.01313"><img src="https://img.shields.io/badge/arXiv-EverMemBench-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: EverMemBench"></a> -->
  <!-- <a href="https://github.com/EverMind-AI/MSA"><img src="https://img.shields.io/badge/arXiv-Memory%20Sparse%20Attention-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white" alt="arXiv: Memory Sparse Attention"></a> -->
  <!-- <a href="https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic"><img src="https://img.shields.io/badge/🤗_HuggingFace-EverMemBench--Dynamic-F5C842?labelColor=gray&style=flat-square" alt="HuggingFace: EverMemBench-Dynamic"></a> -->
  <a href="https://x.com/evermind"><img src="https://img.shields.io/badge/EverMind-000000?labelColor=gray&style=for-the-badge&logo=x&logoColor=white" alt="X"></a>
  <a href="https://discord.gg/gYep5nQRZJ"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FgYep5nQRZJ%3Fwith_counts%3Dtrue&query=%24.approximate_presence_count&suffix=%20online&label=Discord&color=404EED&labelColor=gray&style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/EverMind-AI/EverOS/discussions/67"><img src="https://img.shields.io/badge/WeChat-EverMind-07C160?labelColor=gray&style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>
  <a href="https://github.com/EverMind-AI/EverOS/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-2196F3?labelColor=gray&style=for-the-badge" alt="License"></a>
</p>

[Website](https://evermind.ai) · [Documentation](https://docs.evermind.ai) · [Blog](https://evermind.ai/blogs)

</div>

<br>

> [!IMPORTANT]
>
> ### Project Structure Update
>
> We've unified [EverCore](methods/evermemos/), [HyperMem](methods/HyperMem/), [EverMemBench](benchmarks/EverMemBench/), and [EvoAgentBench](benchmarks/EvoAgentBench/) with [ use-cases ](use-cases/) into a single repository.
>
> EverOS gives developers one place to build, evaluate, and integrate long-term memory into their self-evolving agents. 🎉

<br>

## Project Overview

**EverOS** is a collection of long-term memory **methods**, **benchmarks**, and **usecases** for building self-evolving agents.

```
EverOS/
└── methods/
    ├── EverCore/            # Long-term memory operating system
    └── HyperMem/            # Hypergraph memory architecture
├── benchmarks/
│   ├── EverMemBench/        # Memory quality evaluation
│   └── EvoAgentBench/       # Agent self-evolution evaluation
└── use-cases/               # Example use cases
    ├── claude-code-plugin/        
    └── game-of-throne-demo/            
```

<br>

## Methods

Methods are production-ready memory architectures that give agents persistent, structured long-term memory. Each can be used standalone or composed together depending on your use case.

<table>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/1bbead72-7a6b-4b19-88f2-5bfc8433e3aa)


#### EverCore

A self-organizing memory operating system inspired by biological imprinting. Extracts, structures, and retrieves long-term knowledge from conversations — enabling agents to remember, understand, and continuously evolve.

[Paper](https://arxiv.org/abs/2601.02163) · [Docs](methods/evermemos/)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/b63d8735-ea94-4ed6-9c0c-a11b55b1a2a4)

#### HyperMem

A hypergraph-based hierarchical memory architecture that captures high-order associations through hyperedges. Organizes memory into topic, event, and fact layers for coarse-to-fine long-term conversation retrieval. LoCoMo 92.73%.

[Paper](https://arxiv.org/abs/2604.08256) · [Docs](methods/HyperMem/)

</td>
</tr>
</table>

<br>

## Benchmarks

Benchmarks are designed as **open public standards**. Any memory architecture or agent framework can be evaluated under the same ruler.

<table>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/f6f11c3c-7977-4c3b-8c2b-f7cf13e8f93a)

#### EverMemBench

Three-layer memory quality evaluation: factual recall, applied reasoning, and personalized generalization. Evaluates memory systems and LLMs under a unified standard.

[Paper](https://arxiv.org/abs/2602.01313) · [Dataset](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) · [Docs](benchmarks/EverMemBench/)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/79fd03fe-cd6d-4b92-88d7-d66886d31799)

#### EvoAgentBench

Agent self-evolution evaluation — not static snapshots, but longitudinal growth curves. Measures transfer efficiency, error avoidance, and skill-hit quality through controlled experiments with and without evolution.

[Docs](benchmarks/EvoAgentBench/)

</td>
</tr>
</table>



<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

<!-- ## Key Results

### Memory Performance

| System | LoCoMo | LongMemEval-S |
| :--- | :----: | :----: |
| **EverOS** | **93.05%** | **83.00%** |
| **HyperMem** | **92.73%** | — |
| Mem0 | 78.4% | — |
| MemOS | 74.2% | — |
| Zep | 71.6% | — |


### Self-Evolution Gains

| Task Type | Agent + LLM | Baseline | + EverOS Skills | Delta |
| :--- | :--- | :----: | :----: | :----: |
| Code (Django) | OpenClaw + Qwen3.5-397B | 37% | 58% | **+21%** |
| Code (Django) | Nanobot + Qwen3.5-397B | 21% | 47% | **+26%** |
| General (GDPVAL) | OpenClaw + Qwen3.5-397B | 29% | 69% | **+40%** |
| General (GDPVAL) | OpenClaw + Qwen3.5-27B | 41% | 61% | **+20%** |


<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div> -->

## Use Cases

<table>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/9dcb3dd4-4402-45fa-ae13-e6782f42c7ea)

#### Earth Online Memory Game

Earth Online is a memory-aware productivity game that turns everyday planning into a living quest log. 

<!-- [Agent Memory](https://github.com/EverMind-AI/everos/tree/agent_memory) · [Plugin](https://github.com/EverMind-AI/everos/tree/agent_memory/everos-openclaw-plugin) -->

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/57d8cda7-35a5-4561-b794-5520dffc917b)

#### Multi‑Agent Orchestration Platform

Golutra is pitched as “beyond the IDE,” a multi-agent workforce rather than a single assistant for engineering teams.

</td>
</tr>
<tr>
<tr>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/e6eaf308-a874-483f-8874-6934bf95a78f)

#### Mobi Is a Companion

An iOS app that lets users create, nurture, and live with a personalized AI “lifeform” companion called Mobi.

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/9aabcaa9-f97a-49d2-9109-0b5bb696ed41)

#### AI Wearable with Memory

A context-native empathic AI wearable that listens to everyday life
and converts conversations into memory.

</td>
</tr>
<tr>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/df9677ec-386f-4c56-a428-08bca25c54dc)

#### OpenClaw Agent Memory

A 24/7 agent with continuous learning memory that you can carry with you wherever you go.

[Agent Memory](https://github.com/EverMind-AI/everos/tree/agent_memory) · [Plugin](https://github.com/EverMind-AI/everos/tree/agent_memory/everos-openclaw-plugin)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/3a2357a1-c0c3-464a-8979-0d1cdfc9b0d4)

#### Live2D Character with Memory

Add long-term memory to your anime character that can talk to you in real-time, powered by [TEN Framework](https://github.com/TEN-framework/ten-framework).

[Code](https://github.com/TEN-framework/ten-framework/tree/main/ai_agents/agents/examples/voice-assistant-with-everos)

</td>
</tr>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/c36bdc04-97d3-4fe9-97d9-4b93b475595a)

#### Computer-Use with Memory

Use computer-use to launch screenshot-based analysis, all stored in your memory.

[Live Demo](https://screenshot-analysis-vercel.vercel.app/)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/54a7cf8f-62c4-4fbc-9d50-b214d034e051)](use-cases/game-of-throne-demo)

#### Game of Thrones Memories

A demonstration of AI memory infrastructure through an interactive Q&A experience with "A Game of Thrones".

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

![banner-gif](https://github.com/user-attachments/assets/d521d28c-0ccd-44ff-aecc-828245e2f973)

#### Memory Graph Visualization

Visualize your stored entities and how they relate. Pure frontend demo — backend integration in progress.

[Live Demo](https://main.d2j21qxnymu6wl.amplifyapp.com/graph.html)

</td>
</tr>
</table>

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Quick Start

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
```

Then navigate to the component you need:

| | Component | Entry Point |
| :-- | :--- | :--- |
| **EverCore** | Build agents with long-term memory | [methods/everos/](methods/everos/) |
| **HyperMem** | Use the hypergraph memory architecture | [methods/HyperMem/](methods/HyperMem/) |
| **EverMemBench** | Evaluate memory system quality | [benchmarks/EverMemBench/](benchmarks/EverMemBench/) |
| **EvoAgentBench** | Measure agent self-evolution | [benchmarks/EvoAgentBench/](benchmarks/EvoAgentBench/) |

> Each component has its own installation guide, dependency configuration, and usage examples.

### EverCore Quick Start

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

Server runs at `http://localhost:1995` · [Full Setup Guide](docs/installation/SETUP.md)

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

[More Examples](docs/usage/USAGE_EXAMPLES.md) · [API Reference](https://docs.evermind.ai/api-reference/introduction) · [Interactive Demos](docs/usage/DEMOS.md)

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

<!-- ## Demo

### Run the Demo

```bash
# Terminal 1: Start the API server
uv run python src/run.py

# Terminal 2: Run the simple demo
uv run python src/bootstrap.py demo/simple_demo.py
```

**Try it now**: Follow the [Demo Guide](docs/usage/DEMOS.md) for step-by-step instructions.

### Full Demo Experience

```bash
# Extract memories from sample data
uv run python src/bootstrap.py demo/extract_memory.py

# Start interactive chat with memory
uv run python src/bootstrap.py demo/chat_with_memory.py
```

See the [Demo Guide](docs/usage/DEMOS.md) for details.

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div> -->

## Evaluation & Benchmarking

EverCore achieves **93% overall accuracy** on the LoCoMo benchmark, outperforming comparable memory systems.

### Benchmark Results

![EverOS Benchmark Results](https://github.com/user-attachments/assets/824ac1a6-4bf2-4019-9316-ea7ba7d03142)

### Supported Benchmarks

- **[LoCoMo](https://github.com/snap-research/locomo)** — Long-context memory benchmark with single/multi-hop reasoning
- **[LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)** — Multi-session conversation evaluation
- **[PersonaMem](https://huggingface.co/datasets/bowen-upenn/PersonaMem)** — Persona-based memory evaluation

### Run Evaluations

```bash
# Install evaluation dependencies
uv sync --group evaluation

# Run smoke test (quick verification)
uv run python -m evaluation.cli --dataset locomo --system everos --smoke

# Run full evaluation
uv run python -m evaluation.cli --dataset locomo --system everos

# View results
cat evaluation/results/locomo-everos/report.txt
```

[Full Evaluation Guide](evaluation/README.md) · [Complete Results](https://huggingface.co/datasets/EverMind-AI/everos_Eval_Results)

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

<!-- ## Documentation

| Guide | Description |
| ----- | ----------- |
| [Quick Start](docs/dev_docs/getting_started.md) | Installation and configuration |
| [Configuration Guide](docs/usage/CONFIGURATION_GUIDE.md) | Environment variables and services |
| [API Usage Guide](docs/dev_docs/api_usage_guide.md) | Endpoints and data formats |
| [Development Guide](docs/dev_docs/development_guide.md) | Architecture and best practices |
| [Memory API](docs/api_docs/memory_api.md) | Complete API reference |
| [Demo Guide](demo/README.md) | Interactive examples |
| [Evaluation Guide](evaluation/README.md) | Benchmark testing |

### Advanced Techniques

- **[Group Chat Conversations](docs/advanced/GROUP_CHAT_GUIDE.md)** — Combine messages from multiple speakers
- **[Conversation Metadata Control](docs/advanced/METADATA_CONTROL.md)** — Fine-grained control over conversation context
- **[Memory Retrieval Strategies](docs/advanced/RETRIEVAL_STRATEGIES.md)** — Lightweight vs Agentic retrieval modes
- **[Batch Operations](docs/usage/BATCH_OPERATIONS.md)** — Process multiple messages efficiently

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div> -->

<!-- ## GitHub Codespaces

EverOS supports [GitHub Codespaces](https://github.com/features/codespaces) for cloud-based development — no Docker setup or local environment configuration needed.

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/EverMind-AI/EverOS)

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

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div> -->


## Citation

If EverOS helps your research, please cite:

```bibtex
@article{hu2026evermemos,
  title   = {EverMemOS: A Self-Organizing Memory Operating System for Structured Long-Horizon Reasoning},
  author  = {Chuanrui Hu and Xingze Gao and Zuyi Zhou and Dannong Xu and Yi Bai and Xintong Li and Hui Zhang and Tong Li and Chong Zhang and Lidong Bing and Yafeng Deng},
  journal = {arXiv preprint arXiv:2601.02163},
  year    = {2026}
}

@article{yue2026hypermem,
  title   = {HyperMem: Hypergraph Memory for Long-Term Conversations},
  author  = {Juwei Yue and Chuanrui Hu and Jiawei Sheng and Zuyi Zhou and Wenyuan Zhang and Tingwen Liu and Li Guo and Yafeng Deng},
  journal = {arXiv preprint arXiv:2604.08256},
  year    = {2026}
}

@article{hu2026evaluating,
  title   = {Evaluating Long-Horizon Memory for Multi-Party Collaborative Dialogues},
  author  = {Chuanrui Hu and Tong Li and Xingze Gao and Hongda Chen and Yi Bai and Dannong Xu and Tianwei Lin and Xiaohong Li and Yunyun Han and Jian Pei and Yafeng Deng},
  journal = {arXiv preprint arXiv:2602.01313},
  year    = {2026}
}
```

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## 🌟 Stay Tuned

![star us gif](https://github.com/user-attachments/assets/0c512570-945a-483a-9f47-8e067bd34484)

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Contributing

We love open-source energy! Whether you are squashing bugs, shipping features, sharpening docs, or just tossing in wild ideas, every PR moves EverOS forward. Browse [Issues](https://github.com/EverMind-AI/EverOS/issues) to find your perfect entry point, then show us what you have got. Let us build the future of memory together.

<br>

> [!TIP]
>
> **Welcome all kinds of contributions** 🎉
>
> Join us in building EverOS better! Every contribution makes a difference, from code to documentation. Share your projects on social media to inspire others!
>
> Connect with one of the EverOS maintainers [@elliotchen200](https://x.com/elliotchen200) on 𝕏 or [@cyfyifanchen](https://github.com/cyfyifanchen) on GitHub for project updates, discussions, and collaboration opportunities.

![divider](https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only)
![divider](https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only)

### Code Contributors

[![EverOS Contributors](https://contrib.rocks/image?repo=EverMind-AI/EverOS)](https://github.com/EverMind-AI/EverOS/graphs/contributors)

![divider](https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only)
![divider](https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only)

### Contribution Guidelines

Read our [Contribution Guidelines](methods/evermemos/CONTRIBUTING.md) for code standards and Git workflow.

![divider](https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only)
![divider](https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only)

### License & Citation & Acknowledgments

[Apache 2.0](https://github.com/EverMind-AI/EverOS/blob/main/LICENSE) • [Acknowledgments](methods/evermemos/docs/ACKNOWLEDGMENTS.md)

<br>

<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>
