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

- [Project Overview](#project-overview)
- [Use Cases](#use-cases)
- [Quick Start](#quick-start)
- [Architecture Methods](#architecture-methods)
- [Benchmarks](#benchmarks)
- [Evaluation](#evaluation)
- [Citations](#citations)
- [Stay Tuned](#stay-tuned)
- [Contributing](#contributing)

<br>

</details>



## Project Overview

**EverOS** is a unified home for applying, building, and evaluating long-term memory in self-evolving agents. The repository is organized around three essential parts:

| Part | What it gives you | Start here |
| :--- | :--- | :--- |
| **Use cases** | Apps, demos, and integrations showing how memory changes real agent workflows. | [use-cases/](use-cases/) |
| **Architecture methods** | Memory systems and algorithms you can run, extend, or compare. | [methods/](methods/) |
| **Benchmarks** | Open evaluation suites for memory quality and agent self-evolution. | [benchmarks/](benchmarks/) |

At the center of EverOS is **EverCore**, a long-term memory operating system for agents. If you are new to the project, scan the use cases first to see what memory enables, then follow the [Quick Start](#quick-start) to run EverCore locally. The architecture and benchmark sections below give you the deeper reference material when you are ready to compare systems or reproduce results.

<br>

## Use Cases

Use cases show what persistent memory makes possible in real products and workflows. Some examples are packaged in this repository; others point to external demos or integrations you can study and adapt.

<table>
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

## Quick Start

Choose the path that matches your goal:

```bash
git clone https://github.com/EverMind-AI/EverOS.git
cd EverOS
```

| Goal | Component | Entry Point |
| :--- | :--- | :--- |
| Build agents with long-term memory | **EverCore** | [methods/EverCore/](methods/EverCore/) |
| Explore the hypergraph memory architecture | **HyperMem** | [methods/HyperMem/](methods/HyperMem/) |
| Evaluate memory system quality | **EverMemBench** | [benchmarks/EverMemBench/](benchmarks/EverMemBench/) |
| Measure agent self-evolution | **EvoAgentBench** | [benchmarks/EvoAgentBench/](benchmarks/EvoAgentBench/) |
| Adapt an example app or integration | **Use cases** | [use-cases/](use-cases/) |

> Each component has its own installation guide, dependency configuration, and usage examples.

### EverCore

The fastest way to run a memory system locally is to start with EverCore:

```bash
cd methods/EverCore

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

Server runs at `http://localhost:1995` · [Full Setup Guide](methods/EverCore/docs/installation/SETUP.md)

### Basic Usage

Store and retrieve memories with simple Python code:

```python
import requests

API_BASE = "http://localhost:1995/api/v1"

# 1. Store a conversation memory
requests.post(f"{API_BASE}/memories", json={
    "user_id": "user_001",
    "messages": [
        {
            "message_id": "msg_001",
            "timestamp": 1738404000000,
            "sender_id": "user_001",
            "content": "I love playing soccer on weekends",
            "role": "user",
        }
    ],
})

# 2. Search for relevant memories
response = requests.post(f"{API_BASE}/memories/search", json={
    "query": "What sports does the user like?",
    "method": "hybrid",
    "memory_types": ["episodic_memory"],
    "filters": {"user_id": "user_001"},
})

result = response.json().get("result", {})
for memory_group in result.get("memories", []):
    print(f"Memory: {memory_group}")
```

[More Examples](methods/EverCore/docs/usage/USAGE_EXAMPLES.md) · [API Reference](https://docs.evermind.ai/api-reference/introduction) · [Interactive Demos](methods/EverCore/docs/usage/DEMOS.md)

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Architecture Methods

These are the memory architectures currently included in EverOS. Use them as runnable systems, research references, or starting points for your own agent memory layer.

<table>
<tr>
<td width="50%" valign="top">

### EverCore

A self-organizing memory operating system inspired by biological imprinting. Extracts, structures, and retrieves long-term knowledge from conversations so agents can remember, understand, and continuously evolve.

LoCoMo **93.05%** · LongMemEval **83.00%**

[Paper](https://arxiv.org/abs/2601.02163) · [Docs](methods/EverCore/)

</td>
<td width="50%" valign="top">

### HyperMem

A hypergraph-based hierarchical memory architecture that captures high-order associations through hyperedges, with topic, event, and fact layers for coarse-to-fine conversation retrieval.

LoCoMo **92.73%**

[Paper](https://arxiv.org/abs/2604.08256) · [Docs](methods/HyperMem/)

</td>
</tr>
</table>

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Benchmarks

These benchmarks provide shared standards for measuring memory quality and agent self-evolution across systems.

<table>
<tr>
<td width="50%" valign="top">

### EverMemBench

Three-layer memory quality evaluation: factual recall, applied reasoning, and personalized generalization.

[Paper](https://arxiv.org/abs/2602.01313) · [Dataset](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) · [Docs](benchmarks/EverMemBench/)

</td>
<td width="50%" valign="top">

### EvoAgentBench

Agent self-evolution evaluation through longitudinal growth curves, transfer efficiency, error avoidance, and skill-hit quality.

[Dataset](https://huggingface.co/datasets/EverMind-AI/EvoAgentBench) · [Docs](benchmarks/EvoAgentBench/)

</td>
</tr>
</table>

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Evaluation

Use the evaluation runner to reproduce EverCore results or compare another memory system against the same benchmark tasks.

### Benchmark Results

![EverOS Benchmark Results](https://github.com/user-attachments/assets/41b656e7-6f82-41b7-891d-d6079d10dd39)

### Supported Benchmarks

- **[LoCoMo](https://github.com/snap-research/locomo)** — Long-context memory benchmark with single/multi-hop reasoning
- **[LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)** — Multi-session conversation evaluation
- **[PersonaMem](https://huggingface.co/datasets/bowen-upenn/PersonaMem)** — Persona-based memory evaluation

### Run Evaluations

```bash
cd methods/EverCore

# Install evaluation dependencies
uv sync --group evaluation

# Run smoke test (quick verification)
uv run python -m evaluation.cli --dataset locomo --system everos --smoke

# Run full evaluation
uv run python -m evaluation.cli --dataset locomo --system everos

# View results
cat evaluation/results/locomo-everos/report.txt
```

[Full Evaluation Guide](methods/EverCore/evaluation/README.md) · [Complete Results](https://huggingface.co/datasets/EverMind-AI/everos_Eval_Results)

<br>
<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>

## Citations

If EverOS helps your research, please cite the relevant paper:

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

### Contribution Guidelines

Read the [Contribution Guidelines](.github/CONTRIBUTING.md) for setup, pull request expectations, and use-case submission notes. For responsible disclosure, see the [Security Policy](.github/SECURITY.md).

![divider](https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only)
![divider](https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only)

### License, Conduct, and Acknowledgments

[Apache 2.0](https://github.com/EverMind-AI/EverOS/blob/main/LICENSE) • [Code of Conduct](.github/CODE_OF_CONDUCT.md) • [Acknowledgments](methods/EverCore/docs/ACKNOWLEDGMENTS.md)

<br>

<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>
