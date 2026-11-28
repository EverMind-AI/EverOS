<div align="center" id="readme-top">

<h1>EverOS</h1>

**一份记忆，连接多个 Agent。**

面向 AI Agent 的本地优先、Markdown 原生记忆层。

<p align="center">
<a href="https://x.com/evermind"><img src="https://img.shields.io/badge/EverMind-000000?labelColor=gray&style=for-the-badge&logo=x&logoColor=white" alt="X"></a>
  <a href="https://huggingface.co/EverMind-AI"><img src="https://img.shields.io/badge/🤗_HuggingFace-EverMind-F5C842?labelColor=gray&style=for-the-badge" alt="HuggingFace"></a>
  <a href="https://discord.gg/QHQ2YYp7cH"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FQHQ2YYp7cH%3Fwith_counts%3Dtrue&query=%24.approximate_presence_count&suffix=%20online&label=Discord&color=404EED&labelColor=gray&style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/EverMind-AI/EverOS/discussions/67"><img src="https://img.shields.io/badge/WeCom-EverMind_社区-07C160?labelColor=gray&style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a> <!-- cjk-allow: community name -->
</p>

[官网](https://evermind.ai) · [文档](https://docs.evermind.ai) · [博客](https://evermind.ai/blogs) · [English](README.md)

[![LoCoMo: 94.61% re-judged](https://img.shields.io/badge/LoCoMo-94.61%25*-F5C842?style=flat-square&labelColor=282722)](#评测)
[![PyPI](https://img.shields.io/pypi/v/everos?style=flat-square&labelColor=282722&color=F5C842)](https://pypi.org/project/everos/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-F5C842?style=flat-square&labelColor=282722)](LICENSE)

</div>

<p align="center">
<a href="#快速开始">快速开始</a> · <a href="#为什么选择-everos">为什么选择 EverOS</a> · <a href="#评测">评测</a> · <a href="#生态集成">生态集成</a> · <a href="#使用场景">使用场景</a> · <a href="#文档">文档</a>
</p>

EverOS 将对话、文件和 Agent 执行轨迹保存为可读的 Markdown。SQLite 管理状态，LanceDB 建立检索索引；你可以直接阅读、编辑、比较和版本化自己的记忆文件。

## 快速开始

需要 **Python 3.12+**。国内推荐使用一个[阿里云百炼 DashScope API Key](https://bailian.console.aliyun.com/)，复用于记忆抽取、Embedding 和 Rerank。

### 1. 安装并初始化

```bash
pip install everos
# 或：uv pip install everos
everos init
```

### 2. 配置百炼

在百炼控制台创建华北 2（北京）地域的 Key，打开 `~/.everos/everos.toml`，配置以下三个 provider：

```toml
[llm]
model = "qwen-plus"
api_key = "<DASHSCOPE_API_KEY>"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

[embedding]
model = "text-embedding-v4"
api_key = "<DASHSCOPE_API_KEY>"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
dimensions = 1024

[rerank]
provider = "dashscope"
model = "gte-rerank-v2"
api_key = "<DASHSCOPE_API_KEY>"
base_url = "https://dashscope.aliyuncs.com"
```

同一个 Key 可用于三个 provider。只想先跑通关键词检索，也可以仅配置 `[llm]`，并在搜索时使用 `"method":"keyword"`。

### 3. 启动服务

```bash
everos server start
```

保持服务运行，在第二个终端执行 `curl http://127.0.0.1:8000/health`，确认响应中有 `"status":"ok"`。

### 4. 写入并召回记忆

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

在 session 结束时手动 flush 这条记忆：

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/flush \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo-001","app_id":"default","project_id":"default"}'
```

再把这条记忆搜索回来：

```bash
curl -X POST http://127.0.0.1:8000/api/v2/memory/search \
  -H 'Content-Type: application/json' \
  -d '{
    "user_id": "alice",
    "app_id": "default",
    "project_id": "default",
    "query": "Where do I like to climb?",
    "method": "hybrid",
    "top_k": 5
  }'
```

响应里应该能看到 Yosemite 相关记忆。完整百炼配置可以直接使用 hybrid search；
也可以省略 `method` 字段，因为 API 默认就是 hybrid。

记忆写入后，索引可能需要片刻完成同步。如果第一次搜索为空，请稍后重试。打开 `~/.everos`，可以直接查看生成的 Markdown 文件。

[完整操作指南](QUICKSTART.zh-CN.md) · [OpenRouter 配置](QUICKSTART.md) · [Windows / WSL2](docs/windows.zh.md)

## 为什么选择 EverOS

| 能力 | 实际用途 |
|---|---|
| **自己掌握记忆** | Markdown 文件保存在本机，可读、可编辑、可比较差异、可纳入 Git；本地索引可重建。 |
| **用户与 Agent 双轨** | 保存用户画像与经历，以及 Agent 的案例与技能；按用户、Agent、应用、项目和会话划分作用域。 |
| **多种检索方式** | 支持关键词、向量、混合与多轮 Agentic 检索，按需要添加 provider。 |
| **Knowledge Wiki** | 可编辑、可溯源的 Markdown 知识页，支持分类、增删改查和主题搜索。 |
| **Reflection** | 后台合并相关经历，整理画像与技能，让已有记忆持续得到维护。 |

本地存储使用 **Markdown + SQLite + LanceDB**，无需 MongoDB、Elasticsearch 或 Redis。
模型调用使用你配置的 LLM、Embedding 和 Rerank 服务。参见[记忆原理](docs/how-memory-works.md)和[能力配置](QUICKSTART.zh-CN.md#一个百炼-key-可以使用哪些能力)。

## 评测

| 评测 | 题量 | 准确率 | 答题模型 |
|---|---|---|---|
| [LoCoMo](benchmarks/locomo.zh.md) | 1,540 | **94.61%**\* | `openai/gpt-4.1-mini` |
| LongMemEval | 500 | **94.00%** | `google/gemini-3.6-flash` |
| EverMemBench | 2,400 | **66.67%** | `google/gemini-3-flash-preview` |
| SubtleMemory | 1,522 | **71.75%** | `openai/gpt-5.4` |

\* LoCoMo 使用公开的 `gpt-4.1-mini` 配置，取三次 `agentic` 完整运行中的最好成绩；99 道争议题按审计修正答案重判。[完整结果与评分口径](benchmarks/locomo.zh.md)。

**检索延迟：** LoCoMo 延迟测量中，`hybrid` 检索中位数为 **0.27 秒**，检索过程不调用 LLM。这与取得 94.61% 的 `agentic` 路径不同，也不代表 p95 或完整回答耗时。

**加压吞吐：** 发版前的 1.8.0 构建，在存有 168,821 条记忆的 4 核 8 GB 服务器上，持续支持每秒 **14–16 次 hybrid 检索**、**120–140 次关键词检索**。模型服务使用桩模拟，测量 EverOS 本身。[测试环境与限制](benchmarks/throughput.zh.md)。

[复现评测 →](benchmarks/README.zh.md)

## 生态集成

通过宿主插件、MCP 或 HTTP API 将 Agent 接入 EverOS。[Raven](https://github.com/EverMind-AI/Raven) 已内置 EverOS 记忆。

| 宿主 / 工作流 | 安装指南 | 发布状态 |
|---|---|---|
| Claude Code · Codex | [Claude Code 插件](https://github.com/EverMind-AI/plugins/tree/main/claude-code) · [Codex 插件](https://github.com/EverMind-AI/plugins/tree/main/codex) | 发布前验证中 |
| DeepSeek Harness · Hermes | [DSH 插件](https://github.com/EverMind-AI/plugins/tree/main/dsh) · [Hermes 插件](https://github.com/EverMind-AI/plugins/tree/main/hermes) | 发布前验证中 |
| OpenClaw | [插件与安装指南](https://github.com/EverMind-AI/plugins/tree/main/openclaw) | 包作用域迁移中，详见指南 |
| Dify | [自托管](https://github.com/EverMind-AI/plugins/tree/main/dify) · [云端](https://github.com/EverMind-AI/plugins/tree/main/dify_cloud) | 待提交 Marketplace |
| Claude Desktop · Cursor · 其他 MCP 宿主 | [MCP 接入指南](docs/mcp.md) | 独立的 `everos-mcp` 服务 |
| 自己的应用或 Agent | [HTTP API](docs/api.md) · [Python 库](src/everos/README.md) | EverOS 内置接口 |

宿主插件负责自动召回与记录；Dify 提供可放入工作流的显式工具。最新发布状态以[官方集成列表](https://github.com/EverMind-AI/plugins#plugins)为准。

在用 EverOS 构建产品？欢迎[分享集成或使用案例](https://github.com/EverMind-AI/EverOS/issues/new/choose)。

## 使用场景

以下是部分社区项目，更多示例见[完整案例集](docs/use-cases.md)。

<table>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/7282b38b-56bf-4356-aa7b-06a845e7683d)](https://github.com/tt-a1i/hive)

#### Hive Orchestrator

在浏览器中组织多个 CLI 编程 Agent 协作。

[代码](https://github.com/tt-a1i/hive)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/550071c1-dc39-4964-9f67-ffdfad792345)](https://chromewebstore.google.com/detail/ruminer-browser-agent/lbccjohfpdpimbhpckljimgolndfmfif)

#### Browser Agent for Personal Memory

为浏览器 Agent 提供持久记忆，让个人上下文跨网页任务保留。

[插件](https://chromewebstore.google.com/detail/ruminer-browser-agent/lbccjohfpdpimbhpckljimgolndfmfif)

</td>
</tr>
<tr>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/314c9126-8e08-4688-bbbb-8555ad58cf67)](https://github.com/onenewborn/StudyBuddy-public)

#### Study Buddy with Self-Evolving Memory

带长期记忆的学习助手。

[代码](https://github.com/onenewborn/StudyBuddy-public)

</td>
<td width="50%" valign="top">

[![banner-gif](https://github.com/user-attachments/assets/3a2357a1-c0c3-464a-8979-0d1cdfc9b0d4)](https://github.com/TEN-framework/ten-framework/tree/04cb80601374fa9e35b4e544b2dbd23286ca7763/ai_agents/agents/examples/voice-assistant-with-EverMemOS)

#### Live2D Character with Memory

基于 [TEN Framework](https://github.com/TEN-framework/ten-framework)，为实时 Live2D 角色添加长期记忆。

[代码](https://github.com/TEN-framework/ten-framework/tree/04cb80601374fa9e35b4e544b2dbd23286ca7763/ai_agents/agents/examples/voice-assistant-with-EverMemOS)

</td>
</tr>
</table>

## 文档

| 想做什么 | 从这里开始 |
|---|---|
| 跑通本地记忆闭环 | [快速开始](QUICKSTART.zh-CN.md) |
| 接入编程助手 | [MCP](docs/mcp.md) · [宿主插件](https://github.com/EverMind-AI/plugins) |
| 配置模型与可选能力 | [配置参考](docs/configuration.md) · [推荐模型](QUICKSTART.zh-CN.md#推荐模型) |
| 对接 API | [HTTP API](docs/api.md) · [OpenAPI Schema](docs/openapi.json) |
| 使用知识库、反思或多模态 | [Knowledge](docs/knowledge.md) · [Reflection](docs/reflection.md) · [Multimodal](docs/multimodal.md) |
| 理解架构或参与开发 | [架构](docs/architecture.md) · [工程指南](docs/engineering.md) |

[全部文档](docs/index.md) · [更新记录](CHANGELOG.md) · [迁移指南](docs/migration-to-1.0.0.md)

## 更多 EverMind 项目

[Raven](https://github.com/EverMind-AI/Raven) 是内置 EverOS 记忆的 Agent Harness；[EverMe](https://github.com/EverMind-AI/EverMe) 提供跨设备、跨 Agent 的个人记忆；[EverAlgo](https://github.com/EverMind-AI/EverAlgo) 提供 EverOS 使用的记忆算子。

研究与评测：[HyperMem](https://github.com/EverMind-AI/HyperMem) · [MSA](https://github.com/EverMind-AI/MSA) · [EverMemBench](https://github.com/EverMind-AI/EverMemBench) · [EvoAgentBench](https://github.com/EverMind-AI/EvoAgentBench) · [SkillCorpus](https://github.com/EverMind-AI/SkillCorpus)

## 参与贡献

欢迎提交代码、文档、评测报告和集成。请先阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，浏览 [Issues](https://github.com/EverMind-AI/EverOS/issues)，或加入 [Discord](https://discord.gg/QHQ2YYp7cH)。

[![EverOS 贡献者](https://contrib.rocks/image?repo=EverMind-AI/EverOS)](https://github.com/EverMind-AI/EverOS/graphs/contributors)

[Apache License 2.0](LICENSE) · [第三方声明](NOTICE) · [引用](CITATION.md)

[返回顶部 ↑](#readme-top)
