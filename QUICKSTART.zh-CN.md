# 快速开始

[English](QUICKSTART.md) · [中文](QUICKSTART.zh-CN.md)

> 国内推荐使用一个阿里云百炼 DashScope API Key，即可启动 EverOS、写入
> 持久化记忆，并使用完整的文本检索能力。

## 前置条件

- Python 3.12+
- 一个[阿里云百炼 DashScope API Key](https://bailian.console.aliyun.com/)

## 1. 安装

```bash
uv pip install everos
# or: pip install everos
```

## 2. 初始化并配置百炼

```bash
everos init
```

这个命令会创建 `~/.everos/everos.toml` 和 `~/.everos/ome.toml`。打开
`~/.everos/everos.toml`，配置下面的百炼 provider。

在[百炼控制台](https://bailian.console.aliyun.com/)创建一个华北 2（北京）地域的
DashScope API Key，并把同一个 Key 复用到三个文本 provider。下面是推荐的百炼模型，
也可以换成任意 OpenAI 兼容模型：

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

Embedding 会启用 vector 和 user-hybrid retrieval；DashScope rerank 会进一步
启用 agentic search、默认 agent-hybrid search 和 Knowledge Wiki。共享 DashScope
host 仍支持北京地域的 Key；生产环境可以换成对应地域与业务空间的百炼专属 host。

如果希望更换 memory root，可以使用 `everos init --root <path>`。后续命令也要
传入同一个 `--root <path>`。

## 3. 启动 EverOS

```bash
everos server start
```

保持服务运行，然后打开第二个 terminal 检查：

```bash
curl http://127.0.0.1:8000/health
```

确认响应里有 `"status":"ok"`。上面的百炼配置会显示 `llm`、`embed` 和
`rerank` 三个文本 capability 均可用。

## 4. 写入并搜索第一条记忆

> [!NOTE]
> 业务接口位于 `/api/v2`。旧的 `/api/v1` 前缀仍然指向同一批 handler，已有集成
> 不会受影响；但它只是兼容用的 legacy alias，未来的大版本可能移除 —— 新代码请
> 直接使用 `/api/v2`。

添加一个很小的 conversation：

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

> [!TIP]
> **第一条记忆已经写入。**
> 你刚刚把一个事实交给 EverOS，把它整理进可持久化的 Markdown-backed memory，
> 并通过本地索引把它搜索回来。这就是 EverOS 的核心闭环。
> 打开 `~/.everos`，可以直接检查生成的 Markdown 文件。

记忆写入后，本地索引可能需要片刻完成同步。如果第一次搜索为空，请稍后重试。

## 一个百炼 Key 可以使用哪些能力？

| 配置 | 可用能力 |
| --- | --- |
| 百炼 `[llm]` + `[embedding]` + `[rerank]` | Keyword、vector、hybrid、agentic search；reflection、skill extraction、Knowledge Wiki |
| 再添加 `[multimodal]` 和 parser extra | 图片、PDF、音频、Office 文件摄取 |

`/health` 会列出缺失的可选能力。如果请求了尚未配置 provider 的功能，API 会
返回明确的 HTTP 422。

## 推荐模型

除了上面推荐的百炼模型，下表是 `everos init` 生成的 `everos.toml` 里预填的模型（decider
除外），都可以换成任意 OpenAI 兼容模型：

| 角色 | 配置段 | 推荐模型 |
| --- | --- | --- |
| 记忆抽取 | `[llm]` | `openai/gpt-4.1-mini` |
| `llm_multiround` 检索的决策模型（decider） | `[llm.decider]` | `qwen3.6-27B` 或 `deepseek/deepseek-v4-flash-0731`；留空时由 `[llm]` 的模型决策 |
| Embedding | `[embedding]` | `Qwen/Qwen3-Embedding-4B` |
| Rerank | `[rerank]` | `Qwen/Qwen3-Reranker-4B` |
| 多模态解析 | `[multimodal]` | `google/gemini-3.8-flash` |

其中抽取模型和 decider 就是我们已发表 benchmark 指标所用的模型；每个 benchmark 各角色用的
是哪个模型、怎么复现，见 [benchmarks/README.zh.md](benchmarks/README.zh.md)。想换用其他模型，
[LoCoMo 结果总表](benchmarks/locomo.zh.md#各模型和检索方式的分数与耗时) 按模型和检索方式对比了分数和检索耗时。

## 可选：摄取多模态文件

如果要通过 `/api/v2/memory/add` 的 `content` items 摄取非文本内容
（image / pdf / audio / office documents），安装可选 extra：

```bash
uv pip install 'everos[multimodal]'   # or: pip install 'everos[multimodal]'
```

这会为 parser 增加通过 cairosvg 提供 SVG 支持的 `[svg]` bundle；
基础 parser 已由知识摄取依赖。在 `everos.toml` 的 `[multimodal]` 中完成配置；配置里已预填推荐模型，
即通过 OpenRouter 使用的 `google/gemini-3.8-flash`。

**Office 文档支持需要 LibreOffice 作为系统依赖。** parser 会调用
`soffice`（LibreOffice 的 headless renderer），先把 `.doc` / `.docx` /
`.ppt` / `.pptx` / `.xls` / `.xlsx` 转换为 PDF，再交给多模态 LLM。
如果没有 LibreOffice，office 上传会返回 HTTP 415，并带有明确错误信息；
PDF / image / audio / HTML / email 解析不受影响。

在提供 office 文档服务前，请先在宿主机安装：

```bash
brew install --cask libreoffice              # macOS
sudo apt-get install -y libreoffice          # Debian / Ubuntu
```

[返回 README](README.zh-CN.md)
