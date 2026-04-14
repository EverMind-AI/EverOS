<div align="center" id="readme-top">

![banner-gif](https://github.com/user-attachments/assets/c2cef808-3e93-4f81-a194-dffe02ddd984)

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
> We've unified [EverCore](methods/evermemos/), [HyperMem](methods/HyperMem/), [EverMemBench](benchmarks/EverMemBench/), and [EvoAgentBench](benchmarks/EvoAgentBench/) with usecases into a single repository.
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
└── usecases/                # Example applications
```

<br>

## Methods

Our methods are production-ready memory architectures that give agents persistent, structured long-term memory. Each can be used standalone or composed together depending on your use case.

<table>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/2d45227d-6817-48f5-83eb-8422d7cb989d)


#### EverCore

A self-organizing memory operating system inspired by biological imprinting. Extracts, structures, and retrieves long-term knowledge from conversations — enabling agents to remember, understand, and continuously evolve.

[Paper](https://arxiv.org/abs/2601.02163) · [Docs](methods/evermemos/)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/b68d354a-3de6-4dea-9656-6113a0a12786)

#### HyperMem

A hypergraph-based hierarchical memory architecture that captures high-order associations through hyperedges. Organizes memory into topic, event, and fact layers for coarse-to-fine long-term conversation retrieval. LoCoMo 92.73%.

[Paper](https://arxiv.org/abs/2604.08256) · [Docs](methods/HyperMem/)

</td>
</tr>
</table>

<br>

## Benchmarks

Our benchmarks are designed as **open public standards**. Any memory architecture or agent framework can be evaluated under the same ruler.

<table>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/06b4f598-73e6-44d8-b9cc-8b5483cc363e)

#### EverMemBench

Three-layer memory quality evaluation: factual recall, applied reasoning, and personalized generalization. Evaluates memory systems and LLMs under a unified standard.

[Paper](https://arxiv.org/abs/2602.01313) · [Dataset](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) · [Docs](benchmarks/EverMemBench/)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/3573198d-b4ac-4fd2-b101-d14018c75e39)

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

![banner-gif](https://github.com/user-attachments/assets/0e06da2b-0236-430f-89b4-980b8b6a855f)

#### OpenClaw Agent Memory

A 24/7 agent with continuous learning memory that you can carry with you wherever you go.

[Agent Memory](https://github.com/EverMind-AI/everos/tree/agent_memory) · [Plugin](https://github.com/EverMind-AI/everos/tree/agent_memory/everos-openclaw-plugin)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/a80bdab3-e5d0-43b9-9e8d-0a9605012a26)

#### Live2D Character with Memory

Add long-term memory to your anime character that can talk to you in real-time, powered by [TEN Framework](https://github.com/TEN-framework/ten-framework).

[Code](https://github.com/TEN-framework/ten-framework/tree/main/ai_agents/agents/examples/voice-assistant-with-everos)

</td>
</tr>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/0d306b4c-bcd7-4e9e-a244-22fa3cb7b727)

#### Computer-Use with Memory

Use computer-use to launch screenshot-based analysis, all stored in your memory.

[Live Demo](https://screenshot-analysis-vercel.vercel.app/)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/d1efe507-4eb7-4867-8996-457497333449)

#### Game of Thrones Memories

A demonstration of AI memory infrastructure through an interactive Q&A experience with "A Game of Thrones".

[Code](https://github.com/EverMind-AI/evermem_got_demo)

</td>
</tr>
<tr>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/b40b2241-b0e6-4fc9-9a35-92139f3a2d81)

#### Claude Code Plugin

Persistent memory for Claude Code. Automatically saves and recalls context from past coding sessions.

[Code](https://github.com/EverMind-AI/evermem-claude-code)

</td>
<td width="50%" valign="top">

![banner-gif](https://github.com/user-attachments/assets/6586e647-dd5f-4f9f-9b26-66f930e8241c)

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
cd methods/everos

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

![EverOS Benchmark Results](methods/evermemos/figs/benchmark_2.png)

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
