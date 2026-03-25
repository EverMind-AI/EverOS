<div align="center" id="readme-top">

![banner-gif][banner-gif]

[![][arxiv-badge]][arxiv-link]
[![Docker][docker-badge]][docker]
[![Ask DeepWiki][deepwiki-badge]][deepwiki]
[![License][license-badge]][license]


<!-- [![][arxiv-badge]][arxiv-link]
[![Python][python-badge]][python]
[![Docker][docker-badge]][docker]
[![FastAPI][fastapi-badge]][fastapi]
[![MongoDB][mongodb-badge]][mongodb]
[![Elasticsearch][elasticsearch-badge]][elasticsearch]
[![Milvus][milvus-badge]][milvus]
[![Ask DeepWiki][deepwiki-badge]][deepwiki]
[![License][license-badge]][license] -->

<!-- <p><strong>Share EverMemOS Repository</strong></p>

[![][share-x-shield]][share-x-link]
[![][share-linkedin-shield]][share-linkedin-link]
[![][share-reddit-shield]][share-reddit-link]
[![][share-telegram-shield]][share-telegram-link] -->


[Documentation][documentation] •
[API Reference][api-docs] •
[Demo][demo-section]

[![English][lang-en-badge]][lang-en-readme]
[![简体中文][lang-zh-badge]][lang-zh-readme]

</div>

<br>

<!-- [![Memory Genesis Competition 2026][competition-image]][competition-link] -->

> [!IMPORTANT]
>
> ### [Memory Sparse Attention](https://github.com/EverMind-AI/MSA)
>
> Check out our latest papar [Memory Sparse Attention](https://github.com/EverMind-AI/MSA) - A scalable, end-to-end trainable latent-memory framework for 100M token contexts
>
> Scalable sparse attention + document-wise RoPE (parallel/global) achieving near-linear complexity in both training and inference;
> KV cache compression with a Memory Parallel inference engine to deliver 100M token throughput on 2×A800 GPUs;
> Memory Interleave for multi-round, multi-hop reasoning across scattered memory segments.
>
> Join our [Discord][discord] to ask anything you want. AMA session is open to everyone and occurs biweekly.

<br>

<details open>
<summary><kbd>Table of Contents</kbd></summary>

<br>

- [Welcome to EverMemOS][welcome]
- [Use Cases][use-cases]
- [Quick Start][quick-start]
- [API Usage][api-usage]
- [Demo][demo-section]
- [Evaluation][evaluation-section]
- [Documentation][docs-section]
- [GitHub Codespaces][codespaces]
- [Questions][questions-section]
- [Contributing][contributing]

<br>

</details>

## Welcome to EverMemOS

Welcome to EverMemOS! Join our community to help improve the project and collaborate with talented developers worldwide.

| Community | Purpose |
| :-------- | :------ |
| [![Discord Members][discord-members-badge]][discord] | Join the EverMind Discord community to connect with other users |
| [![WeChat][wechat-badge]][wechat] | Join the EverMind WeChat group for discussion and updates |
<!-- | [![X][x-badge]][x] | Follow updates on X |
| [![LinkedIn][linkedin-badge]][linkedin] | Connect with us on LinkedIn |
| [![Hugging Face Space][hugging-face-badge]][hugging-face] | Join our Hugging Face community to explore our spaces and models |
| [![Reddit][reddit-badge]][reddit] | Join the Reddit community | -->

<br>

## Use Cases

[![EverMind + OpenClaw Agent Memory and Plugin][usecase-openclaw-image]][usecase-openclaw-link]

**EverMind + OpenClaw Agent Memory and Plugin**

Claw is putting the pieces of his memory together. Imagine a 24/7 agent with continuous learning memory that you can carry with you wherever you go next. Check out the [agent_memory][usecase-openclaw-link] branch and the [plugin][usecase-openclaw-plugin-link] for more details.

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

Use computer-use to launch screenshot to do analysis all in your memory.
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

[![EverMemOS Claude Code Plugin][usecase-claude-image]][usecase-claude-link]

**EverMemOS Claude Code Plugin**

Persistent memory for Claude Code. Automatically saves and recalls context from past coding sessions.
See the [code][usecase-claude-link] for more details.

![divider][divider-light]
![divider][divider-dark]

<br>

[![Visualize Memories with Graphs][usecase-graph-image]][usecase-graph-link]

**Visualize Memories with Graphs**

Memory Graph view that visualizes your stored entities and how they relate. This is a pure frontend demo which has not been plugged into the backend yet, and we are working on it.
See the [live demo][usecase-graph-link].

<!-- ## Introduction

> 💬 **More than memory — it's foresight.**

**EverMemOS** enables AI to not only remember what happened, but understand the meaning behind memories and use them to guide decisions. Achieving **93% reasoning accuracy** on the LoCoMo benchmark, EverMemOS provides long-term memory capabilities for conversational AI agents through structured extraction, intelligent retrieval, and progressive profile building.

![EverMemOS Architecture Overview][overview-image]

**How it works:** EverMemOS extracts structured memories from conversations (Encoding), organizes them into episodes and profiles (Consolidation), and intelligently retrieves relevant context when needed (Retrieval).

📄 [Paper][paper-link] • 📚 [Vision & Overview][overview-doc] • 🏗️ [Architecture][architecture-doc] • 📖 [Full Documentation][full-docs]

**Latest**: v1.2.0 with API enhancements + DB efficiency improvements ([Changelog][changelog-doc])

<br>

## Why EverMemOS?

- 🎯 **93% Accuracy** - Best-in-class performance on LoCoMo benchmark
- 🚀 **Production Ready** - Enterprise-grade with Milvus vector DB, Elasticsearch, MongoDB, and Redis
- 🔧 **Easy Integration** - Simple REST API, works with any LLM
- 📊 **Multi-Modal Memory** - Episodes, facts, preferences, relations
- 🔍 **Smart Retrieval** - BM25, embeddings, or agentic search

![EverMemOS Overall Benchmark Results][benchmark-summary-image]

*EverMemOS outperforms existing memory systems across all major benchmarks* -->

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Quick Start

### Prerequisites

- Python 3.10+ • Docker 20.10+ • uv package manager • 4GB RAM

**Verify Prerequisites:**

```bash
# Verify you have the required versions
python --version  # Should be 3.10+
docker --version  # Should be 20.10+
```

### Installation

```bash
# 1. Clone and navigate
git clone https://github.com/EverMind-AI/EverMemOS.git
cd EverMemOS

# 2. Start Docker services
docker compose up -d

# 3. Install uv and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 4. Configure API keys
cp env.template .env
# Edit .env and set:
#   - LLM_API_KEY (for memory extraction)
#   - VECTORIZE_API_KEY (for embedding/rerank)

# 5. Start server
uv run python src/run.py

# 6. Verify installation
curl http://localhost:1995/health
# Expected response: {"status": "healthy", ...}
```

✅ Server running at `http://localhost:1995` • [Full Setup Guide][setup-guide]

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Basic Usage

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

📖 [More Examples][usage-examples] • 📚 [API Reference][api-docs] • 🎯 [Interactive Demos][interactive-demos]

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

## Advanced Techniques

- **[Group Chat Conversations][group-chat-guide]** - Combine messages from multiple speakers
- **[Conversation Metadata Control][metadata-control-guide]** - Fine-grained control over conversation context
- **[Memory Retrieval Strategies][retrieval-strategies-guide]** - Lightweight vs Agentic retrieval modes
- **[Batch Operations][batch-operations-guide]** - Process multiple messages efficiently

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

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Evaluation & Benchmarking

EverMemOS achieves **93% overall accuracy** on the LoCoMo benchmark, outperforming comparable memory systems.

### Benchmark Results

![EverMemOS Benchmark Results][benchmark-image]

### Supported Benchmarks

- **[LoCoMo][locomo-link]** - Long-context memory benchmark with single/multi-hop reasoning
- **[LongMemEval][longmemeval-link]** - Multi-session conversation evaluation
- **[PersonaMem][personamem-link]** - Persona-based memory evaluation

### Quick Start

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

📊 [Full Evaluation Guide][evaluation-guide] • 📈 [Complete Results][evaluation-results-link]

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## GitHub Codespaces

EverMemOS supports [GitHub Codespaces][codespaces-link] for cloud-based development. This eliminates the need to set up Docker, manage local network configurations, or worry about environment compatibility issues.

[![Open in GitHub Codespaces][codespaces-badge]][codespaces-project-link]

![divider][divider-light]
![divider][divider-dark]

### Requirements

| Machine Type | Status | Notes |
| ------------ | ------ | ----- |
| 2-core (Free tier) | ❌ Not supported | Insufficient resources for infrastructure services |
| 4-core | ✅ Minimum | Works but may be slow under load |
| 8-core | ✅ Recommended | Good performance with all services |
| 16-core+ | ✅ Optimal | Best for heavy development workloads |

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

## Questions

EverMemOS is available on these AI-powered Q&A platforms. They can help you find answers quickly and accurately in multiple languages, covering everything from basic setup to advanced implementation details.

| Service | Link |
| ------- | ---- |
| DeepWiki | [![Ask DeepWiki][deepwiki-badge]][deepwiki] |

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

<br>

<a id="star-us"></a>
## 🌟 Star and stay tuned with us

![star us gif][star-gif]

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Contributing

We love open-source energy! Whether you are squashing bugs, shipping features, sharpening docs, or just tossing in wild ideas, every PR moves EverMemOS forward. Browse [Issues][issues-link] to find your perfect entry point, then show us what you have got. Let us build the future of memory together.

<br>

> [!TIP]
>
> **Welcome all kinds of contributions** 🎉
>
> Join us in building EverMemOS better! Every contribution makes a difference, from code to documentation. Share your projects on social media to inspire others!
>
> Connect with one of the EverMemOS maintainers [@elliotchen200][elliot-x-link] on 𝕏 or [@cyfyifanchen][cyfyifanchen-link] on GitHub for project updates, discussions, and collaboration opportunities.

![divider][divider-light]
![divider][divider-dark]

### Code Contributors

[![EverMemOS Contributors][contributors-image]][contributors]

![divider][divider-light]
![divider][divider-dark]

### Contribution Guidelines

Read our [Contribution Guidelines][contributing-doc] for code standards and Git workflow.

![divider][divider-light]
![divider][divider-dark]

### License & Citation & Acknowledgments

[Apache 2.0][license] • [Citation][citation-doc] • [Acknowledgments][acknowledgments-doc]

<br>

<div align="right">

[![][back-to-top]][readme-top]

</div>

<!-- Navigation -->
[readme-top]: #readme-top
[welcome]: #welcome-to-evermemos
[introduction]: #introduction
[why-evermemos]: #why-evermemos
[use-cases]: #use-cases
[quick-start]: #quick-start
[prerequisites]: #prerequisites
[installation]: #installation
[codespaces]: #github-codespaces
[run-demo]: #run-the-demo
[full-demo-experience]: #full-demo-experience
[api-usage]: #basic-usage
[evaluation-section]: #evaluation--benchmarking
[docs-section]: #documentation
[questions-section]: #questions
[contributing]: #contributing
[demo-section]: #demo

<!-- Dividers -->
[divider-light]: https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only
[divider-dark]: https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only

<!-- Images -->
[banner-gif]: https://github.com/user-attachments/assets/f661bf5b-9942-4142-8310-9d4c5cc57924
[competition-image]: https://github.com/user-attachments/assets/739a0939-ab1d-4659-81c4-0842466afde9
[usecase-openclaw-image]: https://github.com/user-attachments/assets/0e06da2b-0236-430f-89b4-980b8b6a855f
[usecase-live2d-image]: https://github.com/user-attachments/assets/a80bdab3-e5d0-43b9-9e8d-0a9605012a26
[usecase-computer-image]: https://github.com/user-attachments/assets/0d306b4c-bcd7-4e9e-a244-22fa3cb7b727
[usecase-got-image]: https://github.com/user-attachments/assets/d1efe507-4eb7-4867-8996-457497333449
[usecase-claude-image]: https://github.com/user-attachments/assets/b40b2241-b0e6-4fc9-9a35-92139f3a2d81
[usecase-graph-image]: https://github.com/user-attachments/assets/6586e647-dd5f-4f9f-9b26-66f930e8241c
[overview-image]: figs/overview.png
[benchmark-image]: figs/benchmark_2.png
[benchmark-summary-image]: https://github.com/user-attachments/assets/a6ff7523-db24-40f5-96ab-aa94f41b2392
[star-gif]: https://github.com/user-attachments/assets/0c512570-945a-483a-9f47-8e067bd34484

<!-- Header Badges -->
[arxiv-badge]: https://img.shields.io/badge/arXiv-2601.02163-b31b1b?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white
[license-badge]: https://img.shields.io/badge/License-Apache%202.0-blue?labelColor=gray&labelColor=F5C842&style=flat-square

<!-- Tech Stack Badges -->
[python-badge]: https://img.shields.io/badge/Python-3.10+-blue?labelColor=gray&style=flat-square&logo=python&logoColor=white&labelColor=F5C842
[docker-badge]: https://img.shields.io/badge/Docker-Supported-4A90E2?labelColor=gray&style=flat-square&logo=docker&logoColor=white&labelColor=F5C842
[fastapi-badge]: https://img.shields.io/badge/FastAPI-Latest-26A69A?labelColor=gray&style=flat-square&logo=fastapi&logoColor=white&labelColor=F5C842
[mongodb-badge]: https://img.shields.io/badge/MongoDB-7.0+-00C853?labelColor=gray&style=flat-square&logo=mongodb&logoColor=white&labelColor=F5C842
[elasticsearch-badge]: https://img.shields.io/badge/Elasticsearch-8.x-0084FF?labelColor=gray&style=flat-square&logo=elasticsearch&logoColor=white&labelColor=F5C842
[milvus-badge]: https://img.shields.io/badge/Milvus-2.4+-00A3E0?labelColor=gray&style=flat-square&labelColor=F5C842

<!-- Language Badges -->
[lang-en-badge]: https://img.shields.io/badge/English-lightgrey?style=flat-square
[lang-zh-badge]: https://img.shields.io/badge/简体中文-lightgrey?style=flat-square

<!-- Community Badges -->
[discord-members-badge]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FgYep5nQRZJ%3Fwith_counts%3Dtrue&query=%24.approximate_member_count&suffix=%20members&label=Discord&color=404EED&style=for-the-badge&logo=discord&logoColor=white
[hugging-face-badge]: https://img.shields.io/badge/Hugging_Face-EverMind-F5C842?style=flat&logo=huggingface&logoColor=white
[x-badge]: https://img.shields.io/badge/X/Twitter-EverMind-000000?style=flat&logo=x&logoColor=white
[linkedin-badge]: https://img.shields.io/badge/LinkedIn-EverMind-0A66C2?style=flat&logo=linkedin&logoColor=white
[reddit-badge]: https://img.shields.io/badge/Reddit-EverMind-FF4500?style=flat&logo=reddit&logoColor=white
[wechat-badge]: https://img.shields.io/badge/WeChat-EverMind%20社区-07C160?style=for-the-badge&logo=wechat&logoColor=white

<!-- Q&A Badges -->
[deepwiki-badge]: https://deepwiki.com/badge.svg

<!-- Misc Badges -->
[back-to-top]: https://img.shields.io/badge/-Back_to_top-gray?style=flat-square
[codespaces-badge]: https://github.com/codespaces/badge.svg

<!-- Primary Links -->
[arxiv-link]: https://arxiv.org/abs/2601.02163
[python]: https://www.python.org/
[docker]: https://www.docker.com/
[fastapi]: https://fastapi.tiangolo.com/
[mongodb]: https://www.mongodb.com/
[elasticsearch]: https://www.elastic.co/elasticsearch/
[milvus]: https://milvus.io/
[license]: https://github.com/EverMind-AI/EverMemOS/blob/main/LICENSE
[documentation]: docs/
[api-docs]: docs/api_docs/memory_api.md
[lang-en-readme]: README.md
[lang-zh-readme]: README.zh.md
[competition-link]: https://luma.com/n88icl03
[starter-kit]: docs/STARTER_KIT.md
[discord]: https://discord.gg/gYep5nQRZJ
[wechat]: https://github.com/EverMind-AI/EverMemOS/discussions/67
[deepwiki]: https://deepwiki.com/EverMind-AI/EverMemOS
[usecase-openclaw-link]: https://github.com/EverMind-AI/EverMemOS/tree/agent_memory
[usecase-openclaw-plugin-link]: https://github.com/EverMind-AI/EverMemOS/tree/agent_memory/evermemos-openclaw-plugin
[ten-framework-link]: https://github.com/TEN-framework/ten-framework
[usecase-live2d-link]: https://github.com/TEN-framework/ten-framework/tree/main/ai_agents/agents/examples/voice-assistant-with-EverMemOS
[usecase-computer-link]: https://screenshot-analysis-vercel.vercel.app/
[usecase-got-link]: https://github.com/EverMind-AI/evermem_got_demo
[usecase-claude-link]: https://github.com/EverMind-AI/evermem-claude-code
[usecase-graph-link]: https://main.d2j21qxnymu6wl.amplifyapp.com/graph.html
[paper-link]: https://arxiv.org/abs/2601.02163
[overview-doc]: docs/OVERVIEW.md
[architecture-doc]: docs/ARCHITECTURE.md
[full-docs]: docs/
[changelog-doc]: docs/CHANGELOG.md
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
[locomo-link]: https://github.com/snap-research/locomo
[longmemeval-link]: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
[personamem-link]: https://huggingface.co/datasets/bowen-upenn/PersonaMem
[evaluation-results-link]: https://huggingface.co/datasets/EverMind-AI/EverMemOS_Eval_Results
[codespaces-link]: https://github.com/features/codespaces
[codespaces-project-link]: https://codespaces.new/EverMind-AI/EverMemOS
[issues-link]: https://github.com/EverMind-AI/EverMemOS/issues
[elliot-x-link]: https://x.com/elliotchen200
[cyfyifanchen-link]: https://github.com/cyfyifanchen
[contributors-image]: https://contrib.rocks/image?repo=EverMind-AI/EverMemOS
[contributors]: https://github.com/EverMind-AI/EverMemOS/graphs/contributors
[contributing-doc]: CONTRIBUTING.md
[citation-doc]: docs/CITATION.md
[acknowledgments-doc]: docs/ACKNOWLEDGMENTS.md
[hugging-face]: https://huggingface.co/EverMind-AI
[x]: https://x.com/EverMindAI
[linkedin]: https://www.linkedin.com/company/ai-evermind
[reddit]: https://www.reddit.com/r/EverMindAI/

<!-- Share Badges -->
[share-linkedin-link]: https://linkedin.com/feed/?shareActive=true&text=Check%20this%20repo%20out.%0A%0AEverMemOS%3A%20persistent%20memory%20for%20all%20agents.%0A%0AOpen%20source%20and%20ready%20to%20use.%0A%0Ahttps%3A%2F%2Fgithub.com%2FEverMind-AI%2FEverMemOS
[share-linkedin-shield]: https://img.shields.io/badge/-Share%20on%20LinkedIn-555?labelColor=555&style=flat-square&logo=linkedin&logoColor=white
[share-mastodon-link]: https://mastodon.social/share?text=Check%20this%20repo%20out.%0A%0AEverMemOS%3A%20persistent%20memory%20for%20all%20agents.%0A%0AOpen%20source%20and%20ready%20to%20use.%0A%0Ahttps%3A%2F%2Fgithub.com%2FEverMind-AI%2FEverMemOS
[share-mastodon-shield]: https://img.shields.io/badge/-Share%20on%20Mastodon-555?labelColor=555&logo=mastodon&logoColor=white&style=flat-square
[share-reddit-link]: https://www.reddit.com/submit?title=EverMemOS%3A%20persistent%20memory%20for%20all%20agents.%20Open%20source%20and%20ready%20to%20use.&url=https%3A%2F%2Fgithub.com%2FEverMind-AI%2FEverMemOS
[share-reddit-shield]: https://img.shields.io/badge/-Share%20on%20Reddit-555?labelColor=555&logo=reddit&logoColor=white&style=flat-square
[share-telegram-link]: https://t.me/share/url?text=Check%20this%20repo%20out.%0A%0AEverMemOS%3A%20persistent%20memory%20for%20all%20agents.%0A%0AOpen%20source%20and%20ready%20to%20use.&url=https%3A%2F%2Fgithub.com%2FEverMind-AI%2FEverMemOS
[share-telegram-shield]: https://img.shields.io/badge/-Share%20on%20Telegram-555?labelColor=555&logo=telegram&logoColor=white&style=flat-square
[share-weibo-link]: https://service.weibo.com/share/share.php?sharesource=weibo&title=Check%20this%20repo%20out.%0A%0AEverMemOS%3A%20persistent%20memory%20for%20all%20agents.%0A%0AOpen%20source%20and%20ready%20to%20use.&url=https%3A%2F%2Fgithub.com%2FEverMind-AI%2FEverMemOS
[share-weibo-shield]: https://img.shields.io/badge/-Share%20on%20Weibo-555?labelColor=555&logo=sinaweibo&logoColor=white&style=flat-square
[share-whatsapp-link]: https://api.whatsapp.com/send?text=Check%20this%20repo%20out.%0A%0AEverMemOS%3A%20persistent%20memory%20for%20all%20agents.%0A%0AOpen%20source%20and%20ready%20to%20use.%0A%0Ahttps%3A%2F%2Fgithub.com%2FEverMind-AI%2FEverMemOS
[share-whatsapp-shield]: https://img.shields.io/badge/-Share%20on%20WhatsApp-555?labelColor=555&logo=whatsapp&logoColor=white&style=flat-square
[share-x-link]: https://x.com/intent/tweet?text=Check%20this%20repo%20out.%0A%0AEverMemOS%3A%20persistent%20memory%20for%20all%20agents.%0A%0AOpen%20source%20and%20ready%20to%20use.&url=https%3A%2F%2Fgithub.com%2FEverMind-AI%2FEverMemOS
[share-x-shield]: https://img.shields.io/badge/-Share%20on%20X-555?labelColor=555&logo=x&logoColor=white&style=flat-square
