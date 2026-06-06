<div align="center" id="readme-top">

![EverOS banner](https://github.com/EverMind-AI/EverOS/releases/download/v1.0.0/everos-readme-banner.jpg)

<p align="center">
  <a href="https://x.com/evermind"><img src="https://img.shields.io/badge/EverMind-000000?labelColor=gray&style=for-the-badge&logo=x&logoColor=white" alt="X"></a>
  <a href="https://huggingface.co/EverMind-AI"><img src="https://img.shields.io/badge/🤗_HuggingFace-EverMind-F5C842?labelColor=gray&style=for-the-badge" alt="HuggingFace"></a>
  <a href="https://discord.gg/gYep5nQRZJ"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fdiscord.com%2Fapi%2Fv10%2Finvites%2FgYep5nQRZJ%3Fwith_counts%3Dtrue&query=%24.approximate_presence_count&suffix=%20online&label=Discord&color=404EED&labelColor=gray&style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/EverMind-AI/EverOS/discussions/67"><img src="https://img.shields.io/badge/WeCom-EverMind_社区-07C160?labelColor=gray&style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat"></a>
</p>

[Website](https://evermind.ai) · [Documentation](https://docs.evermind.ai) · [Blog](https://evermind.ai/blogs) · [English](README.md)

</div>

<br>

<details>
  <summary><kbd>目录</kbd></summary>

<br>

- [EverOS 1.0.0 亮点](#everos-100-亮点)
- [什么是 EverOS](#什么是-everos)
- [快速开始](#快速开始)
- [架构概览](#架构概览)
- [功能](#功能)
- [文档](#文档)
- [EverMind 生态](#evermind-生态)
- [Star History](#star-history)

<br>

</details>

## EverOS 1.0.0 亮点

> [!IMPORTANT]
>
> **EverOS 1.0.0 是面向自进化记忆的一次重要发布。** 它带来了
> local-first 运行时、Markdown 作为记忆源数据、混合检索、多模态
> 摄取、用户/Agent 记忆作用域，以及由
> [EverAlgo](https://github.com/EverMind-AI/EverAlgo) 支撑的模块化算法。
>
> **欢迎 Watch 这个仓库。** 后续我们会持续推进 Wiki 式知识层、
> Dreaming 离线进化，以及更多面向 Agent 的长期记忆能力。

<table>
<tr>
<td width="33%" valign="top">
<strong>Markdown-First Memory</strong><br>
记忆以 Markdown 持久化：可读、可审计、可手动编辑，也方便 Git 管理。
</td>
<td width="33%" valign="top">
<strong>Lightweight Local Stack</strong><br>
Python 即可安装。SQLite 负责运行时状态，LanceDB 负责向量、BM25 和结构化过滤检索。
</td>
<td width="33%" valign="top">
<strong>Self-Evolving Agents</strong><br>
Agent 可以从重复工作流中沉淀可复用的 cases 和 skills，并随着使用持续改进。
</td>
</tr>
</table>

<br>

## 什么是 EverOS

EverOS 是一个开源 Python 框架，用来构建**跨 Agent、跨平台的自进化长期记忆**。
它让 maker 可以为所有常用 Agent 维护同一套可携带的记忆层，例如
Claude Code、Codex、OpenClaw、Hermes 等，让上下文、决策、文件和
Agent 轨迹跟着工作流走，而不是被锁在某一个工具里。

EverOS 会把对话、Agent 轨迹和文件保存为可读的 Markdown，并同步本地
SQLite 与 LanceDB 索引，方便快速检索。Agent 可以复用过去的 cases
和 skills，从重复工作中自我改进，并逐渐变得更加主动。

EverOS 的核心边界：

1. **记忆内容保持可读** - Markdown 是长期记忆的 source of truth。
2. **运行时状态保持本地** - SQLite 负责状态，LanceDB 负责向量、BM25 和结构化过滤。
3. **算法保持模块化** - EverAlgo 负责记忆算法，EverOS 负责运行时、持久化、在线流程和离线进化。

<br>

## 快速开始

### 1. 安装 EverOS

```bash
uv pip install everos
# or: pip install everos
```

### 2. 初始化配置

```bash
everos init
```

`everos init` 默认写入 `./.env`。你也可以使用 `everos init --xdg`
写入 `${XDG_CONFIG_HOME:-~/.config}/everos/.env`。

### 3. 启动服务

```bash
everos --help
everos server start
```

EverOS 的端点兼容 OpenAI protocol，可以接入 OpenAI、OpenRouter、
vLLM、Ollama、DeepInfra 等服务。修改 `.env` 中的 `*__BASE_URL`
即可指向对应模型服务。

<br>

## 架构概览

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

DDD 5 层架构，单向依赖。详见 [docs/architecture.md](docs/architecture.md)。

<br>

## 功能

- **Hybrid Retrieval**: BM25 + vector + scalar filter，统一由 LanceDB 查询
- **Cascade Index Sync**: 修改 `.md` 后自动 diff 并同步到 LanceDB
- **Multi-Source Extraction**: 支持对话、Agent 轨迹、文件知识
- **Dual-Track Memory**: 用户记忆和 Agent 记忆分层管理
- **Multimodal Ingestion**: 支持文本、图片、音频、PDF、Office 文档等
- **Modular Algorithms**: EverAlgo 负责可复用的算法层

<br>

## 文档

- [QUICKSTART.md](QUICKSTART.md) - 快速上手
- [docs/overview.md](docs/overview.md) - 项目愿景与概览
- [docs/architecture.md](docs/architecture.md) - 架构设计
- [docs/api.md](docs/api.md) - API 文档
- [docs/use-cases.md](docs/use-cases.md) - 用例集合
- [CHANGELOG.md](CHANGELOG.md) - 版本记录

<br>

## EverMind 生态

EverMind 是面向长期记忆、自进化 Agent 和记忆评测的开源生态。
EverOS 是核心运行时；EverAlgo 提供算法引擎；EverMemBench 和
EvoAgentBench 提供评测基准；EverMe 和相关插件面向跨设备、跨 Agent
的个人记忆工作流。

相关仓库：

- [EverOS](https://github.com/EverMind-AI/EverOS)
- [EverAlgo](https://github.com/EverMind-AI/EverAlgo)
- [HyperMem](https://github.com/EverMind-AI/HyperMem)
- [EverMemBench](https://github.com/EverMind-AI/EverMemBench)
- [EvoAgentBench](https://github.com/EverMind-AI/EvoAgentBench)
- [EverMe](https://github.com/EverMind-AI/EverMe)

<br>

## Star History

关注 EverOS 的社区增长，也欢迎 Star 和 Watch 这个仓库，持续跟进
EverOS 1.0.0 之后的 Wiki 式知识层、Dreaming 离线进化和 Agent 记忆能力。

[![Star History Chart](https://api.star-history.com/svg?repos=EverMind-AI/EverOS&type=Date)](https://www.star-history.com/#EverMind-AI/EverOS&Date)

<br>

<div align="right">

[![](https://img.shields.io/badge/-Back_to_top-gray?style=flat-square)](#readme-top)

</div>
