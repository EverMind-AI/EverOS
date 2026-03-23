<div align="center" id="readme-top">

![banner-gif][banner-gif]

[![][arxiv-badge]][arxiv-link]
[![Python][python-badge]][python]
[![Docker][docker-badge]][docker]
[![FastAPI][fastapi-badge]][fastapi]
[![MongoDB][mongodb-badge]][mongodb]
[![Elasticsearch][elasticsearch-badge]][elasticsearch]
[![Milvus][milvus-badge]][milvus]
[![Ask DeepWiki][deepwiki-badge]][deepwiki]
[![License][license-badge]][license]

<p><strong>分享 EverMemOS 仓库</strong></p>

[![][share-x-shield]][share-x-link]
[![][share-linkedin-shield]][share-linkedin-link]
[![][share-reddit-shield]][share-reddit-link]
[![][share-telegram-shield]][share-telegram-link]
<!-- [![][share-whatsapp-shield]][share-whatsapp-link]
[![][share-mastodon-shield]][share-mastodon-link]
[![][share-weibo-shield]][share-weibo-link] -->

[文档][documentation] •
[API 参考][api-docs] •
[演示][demo-section]

[![English][lang-en-badge]][lang-en-readme]
[![简体中文][lang-zh-badge]][lang-zh-readme]

</div>

<br>

[![Memory Genesis Competition 2026][competition-image]][competition-link]

> [!IMPORTANT]
>
> ### Memory Genesis Competition 2026
>
> 欢迎参加我们的 AI [记忆竞赛][competition-link]。无论你是构建创新应用、平台插件，还是改进底层基础设施，只要是基于 EverMemOS 的作品都欢迎提交。
>
> **赛道：**
> - **Agent + Memory** - 构建具备长期、可演化记忆的智能体
> - **Platform Plugins** - 将 EverMemOS 集成到 VSCode、Chrome、Slack、Notion、LangChain 等平台
> - **OS Infrastructure** - 优化核心能力与系统性能
>
> **[从竞赛 Starter Kit 开始][starter-kit]**
>
> 欢迎加入我们的 [Discord][discord] 提问交流。AMA 面向所有人开放，双周举行一次。

<br>

<!-- <details>
<summary><kbd>目录</kbd></summary>

<br>

- [欢迎来到 EverMemOS][welcome]
- [介绍][introduction]
- [点亮 Star 并关注我们][star-us]
- [为什么选择 EverMemOS][why-evermemos]
- [快速开始][quick-start]
  - [环境要求][prerequisites]
  - [安装][installation]
- [API 使用][api-usage]
- [演示][demo-section]
  - [运行 Demo][run-demo]
  - [完整 Demo 体验][full-demo-experience]
- [评测][evaluation-section]
- [文档][docs-section]
- [GitHub Codespaces][codespaces]
- [提问与支持][questions-section]
- [参与贡献][contributing]

<br>

</details> -->

## 欢迎来到 EverMemOS

欢迎来到 EverMemOS。加入我们的社区，一起改进项目，并与来自世界各地的优秀开发者协作。

| 社区 | 用途 |
| :--- | :--- |
| [![Discord Members][discord-members-badge]][discord] | 加入 EverMind Discord 社区，与其他用户交流 |
| [![WeChat][wechat-badge]][wechat] | 加入 EverMind 微信群，参与讨论并获取更新 |
<!-- | [![X][x-badge]][x] | 在 X 上关注项目动态 |
| [![LinkedIn][linkedin-badge]][linkedin] | 在 LinkedIn 上与我们联系 |
| [![Hugging Face Space][hugging-face-badge]][hugging-face] | 加入我们的 Hugging Face 社区，体验相关 Space 与模型 |
| [![Reddit][reddit-badge]][reddit] | 加入 Reddit 社区 | -->

<br>

## 应用场景

[![EverMind + OpenClaw Agent Memory and Plugin][usecase-openclaw-image]][usecase-openclaw-link]

**EverMind + OpenClaw Agent Memory and Plugin**

Claw 正在把自己的记忆碎片重新拼起来。想象一个 24/7 在线、拥有持续学习记忆的 Agent，无论你走到哪里，它都能跟着你继续工作。详情可查看 [agent_memory][usecase-openclaw-link] 分支和对应的 [plugin][usecase-openclaw-plugin-link]。

![divider][divider-light]
![divider][divider-dark]

<br>

[![Live2D Character with Memory][usecase-live2d-image]][usecase-live2d-link]

**Live2D Character with Memory**

为你的二次元角色加上长期记忆，让它能够与你实时语音互动，底层由 [TEN Framework][ten-framework-link] 驱动。
更多细节请查看 [Live2D Character with Memory Example][usecase-live2d-link]。

![divider][divider-light]
![divider][divider-dark]

<br>

[![Computer-Use with Memory][usecase-computer-image]][usecase-computer-link]

**Computer-Use with Memory**

将 computer-use 与长期记忆结合起来，启动截图分析并把整个过程沉淀进记忆系统。
更多细节请查看 [在线演示][usecase-computer-link]。

![divider][divider-light]
![divider][divider-dark]

<br>

[![Game of Thrones Memories][usecase-got-image]][usecase-got-link]

**Game of Thrones Memories**

这是一个通过《冰与火之歌：权力的游戏》互动问答来展示 AI 记忆基础设施能力的示例。
更多细节请查看 [代码][usecase-got-link]。

![divider][divider-light]
![divider][divider-dark]

<br>

[![EverMemOS Claude Code Plugin][usecase-claude-image]][usecase-claude-link]

**EverMemOS Claude Code Plugin**

为 Claude Code 提供持久化记忆。它会自动保存并召回你过去编程会话中的上下文。
更多细节请查看 [代码][usecase-claude-link]。

![divider][divider-light]
![divider][divider-dark]

<br>

[![Visualize Memories with Graphs][usecase-graph-image]][usecase-graph-link]

**Visualize Memories with Graphs**

Memory Graph 视图会把你存储的实体及其关系可视化出来。这目前还是一个纯前端演示，暂时还没有接入后端，我们正在推进中。
可查看 [在线演示][usecase-graph-link]。

<!-- ## 介绍

> 💬 **不止于记忆，更是前瞻。**

**EverMemOS** 不只让 AI 记住发生过什么，还让它理解这些记忆背后的意义，并将其用于当前决策。EverMemOS 在 LoCoMo 基准测试上实现了 **93% 推理准确率**，通过结构化提取、智能检索与持续演化的画像构建，为对话式 AI Agent 提供长期记忆能力。

![EverMemOS Architecture Overview][overview-image]

**工作方式：** EverMemOS 从对话中提取结构化记忆（Encoding），将其整理为情景记忆和画像（Consolidation），并在需要时智能检索相关上下文（Retrieval）。

📄 [论文][paper-link] • 📚 [愿景与总览][overview-doc] • 🏗️ [架构][architecture-doc] • 📖 [完整文档][full-docs]

**最新版本**：v1.2.0，带来 API 增强与数据库效率优化（[Changelog][changelog-doc]）

<br>

## 为什么选择 EverMemOS？

- 🎯 **93% 准确率** - 在 LoCoMo 基准测试上达到同类领先水平
- 🚀 **生产可用** - 企业级架构，集成 Milvus、Elasticsearch、MongoDB 与 Redis
- 🔧 **易于集成** - 提供简洁的 REST API，可与任意 LLM 配合使用
- 📊 **多模态记忆** - 支持情景、事实、偏好、关系等多类记忆
- 🔍 **智能检索** - 支持 BM25、向量检索和 Agentic 检索

![EverMemOS Overall Benchmark Results][benchmark-summary-image]

*EverMemOS 在主要基准测试上整体优于现有记忆系统* -->

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## 快速开始

### 环境要求

- Python 3.10+ • Docker 20.10+ • `uv` 包管理器 • 4GB RAM

**验证环境：**

```bash
# 验证所需版本
python --version  # 应为 3.10+
docker --version  # 应为 20.10+
```

### 安装

```bash
# 1. 克隆仓库并进入目录
git clone https://github.com/EverMind-AI/EverMemOS.git
cd EverMemOS

# 2. 启动 Docker 依赖服务
docker compose up -d

# 3. 安装 uv 和项目依赖
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 4. 配置 API Key
cp env.template .env
# 编辑 .env 并设置：
#   - LLM_API_KEY（用于记忆提取）
#   - VECTORIZE_API_KEY（用于向量化 / rerank）
#
# 支持的 LLM 后端：OpenAI、Anthropic Claude、Google Gemini、
# MiniMax、Azure OpenAI、Ollama 或任何 OpenAI 兼容 API。
# 详细配置请参阅 src/config/llm_backends.yaml。

# 5. 启动服务
uv run python src/run.py

# 6. 验证安装
curl http://localhost:1995/health
# 期望响应：{"status": "healthy", ...}
```

✅ 服务运行地址：`http://localhost:1995` • [完整安装指南][setup-guide]

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## 基础使用

通过简单的 Python 代码即可写入并检索记忆：

```python
import requests

API_BASE = "http://localhost:1995/api/v1"

# 1. 存储一条对话记忆
requests.post(f"{API_BASE}/memories", json={
    "message_id": "msg_001",
    "create_time": "2025-02-01T10:00:00+00:00",
    "sender": "user_001",
    "content": "I love playing soccer on weekends"
})

# 2. 搜索相关记忆
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

📖 [更多示例][usage-examples] • 📚 [API 参考][api-docs] • 🎯 [交互式 Demo][interactive-demos]

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## Demo

### 运行 Demo

```bash
# 终端 1：启动 API 服务
uv run python src/run.py

# 终端 2：运行简单 Demo
uv run python src/bootstrap.py demo/simple_demo.py
```

**现在就试试**：按照 [Demo Guide][interactive-demos] 中的步骤操作即可。

### 完整 Demo 体验

```bash
# 从样例数据中提取记忆
uv run python src/bootstrap.py demo/extract_memory.py

# 启动带记忆的交互式聊天
uv run python src/bootstrap.py demo/chat_with_memory.py
```

更多细节请查看 [Demo Guide][interactive-demos]。

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## 进阶技巧

- **[群聊对话][group-chat-guide]** - 组合多个说话者的消息
- **[会话元数据控制][metadata-control-guide]** - 精细控制会话上下文
- **[记忆检索策略][retrieval-strategies-guide]** - 轻量检索与 Agentic 检索
- **[批量操作][batch-operations-guide]** - 高效处理多条消息

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## 文档

| 指南 | 说明 |
| ---- | ---- |
| [快速开始][getting-started] | 安装与配置 |
| [配置指南][config-guide] | 环境变量与服务配置 |
| [API 使用指南][api-usage-guide] | 接口与数据格式 |
| [开发指南][dev-guide] | 架构与最佳实践 |
| [Memory API][memory-api-doc] | 完整 API 参考 |
| [Demo Guide][demo-guide] | 交互式示例 |
| [评测指南][evaluation-guide] | 基准测试方法 |

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## 评测与基准测试

EverMemOS 在 LoCoMo 基准测试上实现了 **93% 总体准确率**，表现优于可比记忆系统。

### 基准结果

![EverMemOS Benchmark Results][benchmark-image]

### 支持的基准数据集

- **[LoCoMo][locomo-link]** - 面向单跳 / 多跳推理的长上下文记忆基准
- **[LongMemEval][longmemeval-link]** - 多会话对话记忆评测
- **[PersonaMem][personamem-link]** - 基于人格画像的记忆评测

### 快速开始

```bash
# 安装评测依赖
uv sync --group evaluation

# 运行 smoke test（快速验证）
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# 运行完整评测
uv run python -m evaluation.cli --dataset locomo --system evermemos

# 查看结果
cat evaluation/results/locomo-evermemos/report.txt
```

📊 [完整评测指南][evaluation-guide] • 📈 [完整结果][evaluation-results-link]

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## GitHub Codespaces

EverMemOS 支持使用 [GitHub Codespaces][codespaces-link] 进行云端开发。这意味着你不需要在本地手动配置 Docker、网络环境或兼容性问题。

[![Open in GitHub Codespaces][codespaces-badge]][codespaces-project-link]

![divider][divider-light]
![divider][divider-dark]

### 资源要求

| 机器规格 | 状态 | 说明 |
| -------- | ---- | ---- |
| 2-core（免费档） | ❌ 不支持 | 资源不足以运行基础设施服务 |
| 4-core | ✅ 最低可用 | 可以运行，但负载较高时可能较慢 |
| 8-core | ✅ 推荐 | 可较稳定地运行全部服务 |
| 16-core+ | ✅ 最佳 | 适合更重的开发与测试任务 |

> **说明：** 如果你的公司提供 GitHub Codespaces，通常不会受硬件限制影响，因为企业方案往往可以使用更高规格的机器。

### 开始使用 Codespaces

1. 点击上方 "Open in GitHub Codespaces" 按钮
2. 在弹窗中选择 **4-core 及以上** 的机器规格
3. 等待容器构建并自动启动服务
4. 在 `.env` 中填写 API Key（`LLM_API_KEY`、`VECTORIZE_API_KEY` 等）
5. 运行 `make run` 启动服务

MongoDB、Elasticsearch、Milvus、Redis 等基础设施服务都会自动启动，并已预配置为可协同工作。

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## 提问与支持

EverMemOS 已接入以下 AI 驱动问答平台。它们可以帮助你快速、准确地获取答案，并支持从基础安装到高级实现的多语言问题查询。

| 服务 | 链接 |
| ---- | ---- |
| DeepWiki | [![Ask DeepWiki][deepwiki-badge]][deepwiki] |

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

<br>

<a id="star-us"></a>
## 🌟 点亮 Star 并关注我们

![star us gif][star-gif]

<br>
<div align="right">

[![][back-to-top]][readme-top]

</div>

## 参与贡献

我们热爱开源社区的活力。无论你是在修 Bug、做功能、完善文档，还是提出一些大胆的新想法，每一个 PR 都会推动 EverMemOS 往前走。欢迎查看 [Issues][issues-link] 寻找适合你的切入点，然后把你的成果提交给我们。一起构建记忆系统的未来。

<br>

> [!TIP]
>
> **欢迎各种形式的贡献** 🎉
>
> 一起把 EverMemOS 做得更好。从代码到文档，每一份贡献都很重要。也欢迎你把自己的项目分享到社交平台，激发更多人的灵感。
>
> 欢迎通过 𝕏 联系 EverMemOS 维护者 [@elliotchen200][elliot-x-link]，或通过 GitHub 联系 [@cyfyifanchen][cyfyifanchen-link]，获取项目动态、参与讨论并展开合作。

![divider][divider-light]
![divider][divider-dark]

### 代码贡献者

[![EverMemOS Contributors][contributors-image]][contributors]

![divider][divider-light]
![divider][divider-dark]

### 贡献指南

请阅读 [Contribution Guidelines][contributing-doc] 了解代码规范与 Git 工作流。

![divider][divider-light]
![divider][divider-dark]

### 许可证、引用与鸣谢

[Apache 2.0][license] • [Citation][citation-doc] • [Acknowledgments][acknowledgments-doc]

<br>

<div align="right">

[![][back-to-top]][readme-top]

</div>

<!-- Navigation -->
[readme-top]: #readme-top
[welcome]: #欢迎来到-evermemos
[introduction]: #介绍
[why-evermemos]: #为什么选择-evermemos
[quick-start]: #快速开始
[prerequisites]: #环境要求
[installation]: #安装
[codespaces]: #github-codespaces
[run-demo]: #运行-demo
[full-demo-experience]: #完整-demo-体验
[api-usage]: #api-使用
[evaluation-section]: #评测与基准测试
[docs-section]: #文档
[questions-section]: #提问与支持
[contributing]: #参与贡献
[demo-section]: #demo

<!-- Dividers -->
[divider-light]: https://github.com/user-attachments/assets/2e2bbcc6-e6d8-4227-83c6-0620fc96f761#gh-light-mode-only
[divider-dark]: https://github.com/user-attachments/assets/d57fad08-4f49-4a1c-bdfc-f659a5d86150#gh-dark-mode-only

<!-- Images -->
[banner-gif]: https://github.com/user-attachments/assets/3f22c9a8-a8db-4061-accf-f04c055aa01b
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
[arxiv-badge]: https://img.shields.io/badge/arXiv-EverMemOS_Paper-F5C842?labelColor=gray&style=flat-square&logo=arxiv&logoColor=white
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
