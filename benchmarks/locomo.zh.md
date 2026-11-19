# LoCoMo 结果

EverOS 在 LoCoMo 上得分 **94.61%**，是用 `benchmarks/configs/locomo-step.toml` 跑 3 次
`agentic` 里最高的一次。本文按检索方式拆开这个分数，对比批量和逐条边界检测，
并展示各环节换用不同模型后分数和写入、检索耗时的变化。想自己跑 LoCoMo，从
[README.zh.md](README.zh.md) 开始。

> English version: [locomo.md](locomo.md).

本文所有分数都按重判口径：LoCoMo 审计认为标准答案有误的 99 道题，按审计修正后的答案判分；见
[每次运行给出两个分数](README.zh.md#每次运行给出两个分数)。

## 各检索方式

| 检索方式 | 逐条边界检测（`locomo-step`） | 批量边界检测（`locomo`，默认） | 检索耗时中位数 |
|---|---|---|---|
| `agentic` | **94.61** | 94.35 | 6.7 秒 |
| `llm_multiround` | 94.29 | 94.29 |  |
| `hybrid` | 93.64 | 93.38 | 0.26 秒 |
| `vector` | 92.73 | 92.08 | 0.24 秒 |
| `keyword` | 91.04 | 92.27 | 0.01 秒 |

每份配置的 `agentic` 跑 3 次取最高的一次，其余检索方式在那次运行建好的库上各跑 1 次。抽取和答题用
`openai/gpt-4.1-mini`，判分用 `openai/gpt-4o-mini` 判 3 次，`llm_multiround` 的 decider 用
`deepseek/deepseek-v4-flash-0731`（关思考）；everalgo-core 0.7.0、everalgo-boundary 0.5.0。
检索耗时是在 `locomo-step` 的库上检索 30 次的中位数，一次一个请求；`hybrid`、`vector`、`keyword`
不调用 LLM（见[检索耗时](#检索耗时)）。

## batch 还是 step

两份 LoCoMo 配置只差两项 EverOS 设置，EverOS 的默认值是第一组：

| | `locomo`（默认） | `locomo-step` |
|---|---|---|
| 边界检测，`[boundary_detection] strategy` | `batch`：每次写入调用一次 | `step`：每条消息调用一次 |
| 回复格式，`[llm] response_format_mode` | `json_schema`：由服务商保证每条回复符合 schema | `prompt_only`：只在 prompt 里要求 schema |
| `agentic`，3 次运行取最高 | 94.35 | **94.61** |
| 每次运行的边界检测调用 | 544 次 | 约 5,250 次 |
| ADD 耗时（10 个 server） | 约 4 分钟 | 约 17–22 分钟 |
| 存下的 episode | 约 320 个 | 约 550 个 |
| agent 对话（含工具调用） | 支持 | 只处理对话消息 |

- **`locomo` 更快、更稳。** batch 的边界检测调用只有 step 的十分之一，写入快约 5 倍。strict 回复
  一定符合所要求的 schema；需要服务商支持 `json_schema`。
- **`locomo-step` 分数更高。** 它 `agentic` 的最高分比 batch 高 0.26 个百分点，在 LoCoMo 单次运行
  之间约 1 个百分点的波动以内。`prompt_only` 的回复解析不了时会重新请求。

做对话记忆、能接受较慢的写入时选 step；agent 场景和需要快速写入时保持默认。在 EverOS 部署里见
[`[boundary_detection]`](../docs/configuration.md#boundary_detection) 和
[`[llm]`](../docs/configuration.md#llm)。

## 模型配置

已发表的 LoCoMo 分数里，EverOS 的每次模型调用都用 `openai/gpt-4.1-mini`，这也是记忆系统发表
LoCoMo 分数时常用的模型。下面这些配置换掉了一个或几个环节的模型，展示选不同模型对效果和耗时的影响。
它们的分数和用其他模型测出的已发表分数不可直接比较。除已发表的 `openai/gpt-4.1-mini` 用
`benchmarks/configs/locomo-step.toml` 外，每份配置都是 `benchmarks/configs/locomo.toml`（批量边界检测），
模型写在 `[llm]`（所有环节都继承）和
[`[llm.<task>]`](../docs/configuration.md#llmtask) 的 `boundary`、`episode`、`summary`、
`atomic_fact`、`search` 几节里。embedding 和重排始终用已发表配置里的 `Qwen/Qwen3-Embedding-4B`
和 `Qwen/Qwen3-Reranker-4B`。

### 分数

所有环节用同一个模型：

| 模型 | `agentic`，3 次取最高 | `hybrid`，3 次取最高 |
|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | **95.52** | 94.61 |
| `anthropic/claude-haiku-5.5` | 95.32 | **94.94** |
| `openai/gpt-4.1-mini`（已发表，`locomo-step`） | 94.61 | 93.64 |
| `google/gemini-2.5-flash-lite` | 94.29 | 93.25 |
| `qwen/qwen3-32b` | 93.38 | 92.99 |

各环节用不同模型：

| 边界检测 | episode 和 summary | 原子事实和 `agentic` 检索 | `agentic`，3 次取最高 | `hybrid`，3 次取最高 |
|---|---|---|---|---|
| `openai/gpt-5.4-mini` | `openai/gpt-5.4-mini` | `qwen/qwen3-235b-a22b-2507` | **95.78** | 94.22 |
| `qwen3-4b-instruct-2507`，自部署 | `openai/gpt-5.4-mini` | `qwen/qwen3-235b-a22b-2507` | 94.68 | 93.64 |

每份配置完整跑 3 次取最高；`hybrid` 在每次运行建好的库上检索。`openai/gpt-4.1-mini` 是已发表的配置
`benchmarks/configs/locomo-step.toml`：逐条边界检测、只在 prompt 里要求回复格式，其余配置都用批量
边界检测；它的 `hybrid` 只在最高那次运行的库上检索 1 次。答题和判分与已发表的运行相同；每次运行的
分数见[附录](#附录完整结果)。

这些配置大多相差不到 2 个百分点，3 次运行能分出高下的很少。用对运行和题目两层抽样的配对 bootstrap
比较 `agentic`，并对所有比较对做校正后，有 4 个差别成立：`deepseek/deepseek-v4.1-flash-1017` 高于
`google/gemini-2.5-flash-lite`（3 次平均高 2.0 个百分点）和 `qwen/qwen3-32b`（高 2.4），
`anthropic/claude-haiku-5.5` 高于 `qwen/qwen3-32b`（高 2.2），`openai/gpt-5.4-mini` 那份分环节配置
高于 `qwen/qwen3-32b`（高 2.1）。两份分环节配置只有边界检测模型不同，差 0.5 个百分点，这个差别
不成立。

### 写入耗时

对一整段会话做一次 `/memory/add`，再做一次 `/memory/flush`，在客户端计时。每格是 3 段会话的
中位数：LoCoMo 第 0、2、7 个对话各自的第一段会话（每段 16–18 条消息），每段都写进新库、用新启动的
server。

| 配置 | add | flush | add + flush | 全部抽取完成 |
|---|---|---|---|---|
| `google/gemini-2.5-flash-lite` | 1.1 秒 | 2.5 秒 | 3.6 秒 | 9.6 秒 |
| `deepseek/deepseek-v4.1-flash-1017` | 0.9 秒 | 3.1 秒 | 4.3 秒 | 10.3 秒 |
| 边界、episode 和 summary `gpt-5.4-mini`，原子事实 `qwen3-235b` | 1.6 秒 | 4.1 秒 | 5.7 秒 | 46.0 秒 |
| 边界 `qwen3-4b`，episode 和 summary `gpt-5.4-mini`，原子事实 `qwen3-235b` | 1.4 秒 | 4.7 秒 | 6.0 秒 | 27.0 秒 |
| `anthropic/claude-haiku-5.5` | 2.3 秒 | 5.5 秒 | 7.8 秒 | 16.9 秒 |
| `qwen/qwen3-32b` | 1.5 秒 | 9.2 秒 | 10.9 秒 | 21.4 秒 |
| `openai/gpt-4.1-mini`（已发表，`locomo-step`） | 21.8 秒 | 4.2 秒 | 26.9 秒 | 38.9 秒 |

- 每列各取各的中位数，所以 add 和 flush 加起来不等于 add + flush。
- **全部抽取完成**：从 add 开始，到这段会话的所有后台抽取都结束。flush 在 episode 写完后就返回，
  summary 和原子事实在它返回之后才抽取，所以这两个环节的模型只影响这一列，不影响 add 和 flush。
- 批量边界检测每次 add 调用一次，这几段会话里 add 只做了边界检测，episode 是在 flush 时切出来的。
  `openai/gpt-4.1-mini` 用的逐条边界检测每条消息调用一次，每次约 1.3 秒，所以一次 add 写入整段
  会话约需 22 秒，而且可能在 add 里就切出 episode。应用如果每来一条消息就调用一次 add，等待的是
  一次边界检测调用，这条消息结束一个 episode 时再加上抽取 episode 的那次调用。
- 每段会话的耗时和每次模型调用的耗时见[附录](#每段会话的写入耗时)。

### 检索耗时

每份配置在自己的 LoCoMo 库上检索：从第 0 个对话里抽 10 道题，每题 3 次，一次一个请求。`hybrid`、
`vector`、`keyword` 不调用 LLM，不论配置用什么模型，检索耗时中位数都在 0.30 秒以内；`agentic`
多了 LLM 调用，耗时取决于检索模型。

检索耗时中位数：

| 配置 | `hybrid` | `vector` | `keyword` | `agentic` |
|---|---|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | 0.29 秒 | 0.25 秒 | 0.01 秒 | 4.0 秒 |
| `google/gemini-2.5-flash-lite` | 0.26 秒 | 0.26 秒 | 0.01 秒 | 4.9 秒 |
| `openai/gpt-4.1-mini`（已发表，`locomo-step`） | 0.26 秒 | 0.24 秒 | 0.01 秒 | 6.7 秒 |
| `anthropic/claude-haiku-5.5` | 0.27 秒 | 0.28 秒 | 0.01 秒 | 8.5 秒 |
| 边界 `qwen3-4b`，episode 和 summary `gpt-5.4-mini`，原子事实和检索 `qwen3-235b` | 0.27 秒 | 0.25 秒 | 0.01 秒 | 12.5 秒 |
| 边界、episode 和 summary `gpt-5.4-mini`，原子事实和检索 `qwen3-235b` | 0.30 秒 | 0.25 秒 | 0.01 秒 | 14.7 秒 |
| `qwen/qwen3-32b` | 0.27 秒 | 0.27 秒 | 0.01 秒 | 14.8 秒 |

- 一次 `agentic` 检索里 EverOS 自身只花约 0.1 秒，其余都是模型调用。每次检索至少调用一次 LLM，
  判断召回的记忆是否足够；进入第二轮时，再调用一次 LLM 生成补充查询，并再做一次重排。各类模型调用
  的耗时、90 分位和进入第二轮的比例见[附录](#检索耗时明细)。
- `hybrid` 和 `vector` 的耗时几乎都花在查询的 embedding 上。

耗时测量时 LLM 走 OpenRouter，embedding 和重排走 DeepInfra，`qwen3-4b-instruct-2507` 用自部署的
端点；换服务商或网络会改变这些数字。边界检测用 `gpt-5.4-mini` 的那份配置是单独测的，当时
`qwen/qwen3-235b-a22b-2507` 每次调用的耗时是测边界 `qwen3-4b` 那份时的 2–3 倍（原子事实每次 34 秒
对 18 秒），所以这份配置的原子事实、全部抽取完成和 `agentic` 耗时反映的是服务商当时的速度。

## 附录：完整结果

上面各表背后的全部数字。

### 每次运行的分数

每格是「原始 / 重判」：原始分按数据集自带的标准答案判分，重判分把有争议的 99 道题改按审计修正后的
答案判分（[每次运行给出两个分数](README.zh.md#每次运行给出两个分数)）。每个分数都是全部 1,540 道题、
判分 3 次取多数的结果。加粗的是上面各表引用的那次运行。

`agentic`，每份配置完整跑 3 次：

| 配置 | 第 1 次 | 第 2 次 | 第 3 次 | 平均 |
|---|---|---|---|---|
| `openai/gpt-4.1-mini`（已发表，`locomo-step`） | 92.86 / 93.70 | 92.99 / 93.70 | 93.25 / **94.61** | 93.03 / 94.00 |
| `deepseek/deepseek-v4.1-flash-1017` | 94.61 / **95.52** | 94.68 / 95.26 | 93.70 / 95.00 | 94.33 / 95.26 |
| `anthropic/claude-haiku-5.5` | 94.55 / 95.26 | 94.55 / **95.32** | 93.77 / 94.74 | 94.29 / 95.11 |
| `google/gemini-2.5-flash-lite` | 91.36 / 92.60 | 93.51 / **94.29** | 92.08 / 92.99 | 92.32 / 93.29 |
| `qwen/qwen3-32b` | 91.30 / 92.40 | 92.40 / **93.38** | 92.34 / 92.92 | 92.01 / 92.90 |
| 边界、episode 和 summary `gpt-5.4-mini`，原子事实和检索 `qwen3-235b` | 93.31 / 94.42 | 94.74 / **95.78** | 94.09 / 94.74 | 94.05 / 94.98 |
| 边界 `qwen3-4b`，episode 和 summary `gpt-5.4-mini`，原子事实和检索 `qwen3-235b` | 93.51 / 94.61 | 93.25 / 94.22 | 93.83 / **94.68** | 93.53 / 94.50 |

`hybrid`，在每次运行建好的库上检索：

| 配置 | 第 1 次 | 第 2 次 | 第 3 次 | 平均 |
|---|---|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | 93.44 / **94.61** | 92.86 / 93.90 | 93.83 / 94.48 | 93.38 / 94.33 |
| `anthropic/claude-haiku-5.5` | 93.57 / 94.61 | 93.90 / **94.94** | 93.70 / 94.68 | 93.72 / 94.74 |
| `google/gemini-2.5-flash-lite` | 91.82 / **93.25** | 92.08 / 93.05 | 91.82 / 92.86 | 91.90 / 93.05 |
| `qwen/qwen3-32b` | 91.69 / 92.47 | 91.75 / **92.99** | 91.75 / 92.66 | 91.73 / 92.71 |
| 边界、episode 和 summary `gpt-5.4-mini`，原子事实和检索 `qwen3-235b` | 92.60 / 93.38 | 93.12 / **94.22** | 92.47 / 93.57 | 92.73 / 93.72 |
| 边界 `qwen3-4b`，episode 和 summary `gpt-5.4-mini`，原子事实和检索 `qwen3-235b` | 92.53 / 93.57 | 92.66 / **93.64** | 92.47 / 93.38 | 92.55 / 93.53 |

`openai/gpt-4.1-mini` 的其他检索方式，各在 `agentic` 最高的那次运行（第 3 次）建好的库上检索 1 次：

| 检索方式 | 原始 / 重判 |
|---|---|
| `llm_multiround` | 93.77 / 94.29 |
| `hybrid` | 92.60 / 93.64 |
| `vector` | 91.75 / 92.73 |
| `keyword` | 89.74 / 91.04 |

### 每段会话的写入耗时

每格是这个对话第一段会话的「add / flush / 全部抽取完成」，单位秒。

| 配置 | 第 0 个对话 | 第 2 个对话 | 第 7 个对话 |
|---|---|---|---|
| `google/gemini-2.5-flash-lite` | 1.1 / 3.1 / 10.2 | 1.0 / 2.1 / 9.1 | 1.1 / 2.5 / 9.6 |
| `deepseek/deepseek-v4.1-flash-1017` | 0.9 / 3.4 / 10.3 | 0.9 / 1.9 / 8.8 | 7.0 / 3.1 / 16.1 |
| 边界、episode 和 summary `gpt-5.4-mini`，原子事实 `qwen3-235b` | 1.6 / 4.1 / 50.8 | 1.4 / 4.0 / 35.5 | 2.5 / 4.4 / 46.0 |
| 边界 `qwen3-4b`，episode 和 summary `gpt-5.4-mini`，原子事实 `qwen3-235b` | 1.4 / 4.7 / 21.1 | 1.1 / 4.8 / 27.0 | 1.4 / 4.6 / 36.1 |
| `anthropic/claude-haiku-5.5` | 2.4 / 5.7 / 17.1 | 2.3 / 5.5 / 16.9 | 1.8 / 5.5 / 16.4 |
| `qwen/qwen3-32b` | 1.2 / 5.1 / 21.4 | 2.4 / 9.2 / 26.6 | 1.5 / 9.4 / 19.9 |
| `openai/gpt-4.1-mini`（已发表，`locomo-step`） | 21.8 / 5.2 / 45.0 | 19.2 / 3.1 / 34.4 | 22.7 / 4.2 / 38.9 |

每次模型调用的耗时，取 3 段会话里所有调用的中位数。批量边界检测这一列不含 add 时的那次调用：
它是 server 启动后的第一次模型调用，比较慢。逐条边界检测每条消息调用一次。

| 配置 | 边界检测 | episode | summary | 原子事实 |
|---|---|---|---|---|
| `google/gemini-2.5-flash-lite` | 0.7 秒 | 1.2 秒 | 1.1 秒 | 1.3 秒 |
| `deepseek/deepseek-v4.1-flash-1017` | 0.6 秒 | 1.4 秒 | 0.8 秒 | 1.9 秒 |
| 边界、episode 和 summary `gpt-5.4-mini`，原子事实 `qwen3-235b` | 1.2 秒 | 2.9 秒 | 1.6 秒 | 34.1 秒 |
| 边界 `qwen3-4b`，episode 和 summary `gpt-5.4-mini`，原子事实 `qwen3-235b` | 0.8 秒 | 3.8 秒 | 1.3 秒 | 17.6 秒 |
| `anthropic/claude-haiku-5.5` | 1.2 秒 | 4.3 秒 | 2.0 秒 | 3.7 秒 |
| `qwen/qwen3-32b` | 0.8 秒 | 7.9 秒 | 3.2 秒 | 9.4 秒 |
| `openai/gpt-4.1-mini`（已发表，`locomo-step`） | 1.3 秒 | 3.1 秒 | 1.4 秒 | 3.8 秒 |

### 检索耗时明细

`agentic`，每份配置 30 次检索。LLM、重排、embedding 和 EverOS 自身（模型调用之外的时间）各取各的
中位数，加起来不等于总耗时。

| `agentic` 检索模型 | 中位数 | 平均 | 90 分位 | LLM | 重排 | embedding | EverOS 自身 | 进入第二轮 |
|---|---|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | 4.0 秒 | 5.1 秒 | 8.7 秒 | 1.2 秒 | 1.6 秒 | 0.8 秒 | 0.1 秒 | 30 次中 12 次 |
| `google/gemini-2.5-flash-lite` | 4.9 秒 | 4.7 秒 | 6.8 秒 | 1.8 秒 | 1.9 秒 | 0.5 秒 | 0.1 秒 | 30 次中 15 次 |
| `openai/gpt-4.1-mini`（已发表，`locomo-step`） | 6.7 秒 | 6.4 秒 | 9.1 秒 | 3.8 秒 | 1.9 秒 | 0.8 秒 | 0.1 秒 | 30 次中 12 次 |
| `anthropic/claude-haiku-5.5` | 8.5 秒 | 8.2 秒 | 11.2 秒 | 4.7 秒 | 2.3 秒 | 1.2 秒 | 0.1 秒 | 30 次中 20 次 |
| `qwen/qwen3-235b-a22b-2507`，边界 `qwen3-4b` 的库 | 12.5 秒 | 12.4 秒 | 19.1 秒 | 8.2 秒 | 2.3 秒 | 1.0 秒 | 0.1 秒 | 30 次中 18 次 |
| `qwen/qwen3-235b-a22b-2507`，边界 `gpt-5.4-mini` 的库 | 14.7 秒 | 15.3 秒 | 26.5 秒 | 11.6 秒 | 1.7 秒 | 1.0 秒 | 0.1 秒 | 30 次中 13 次 |
| `qwen/qwen3-32b` | 14.8 秒 | 12.7 秒 | 17.1 秒 | 10.4 秒 | 2.5 秒 | 1.0 秒 | 0.1 秒 | 30 次中 21 次 |

`hybrid`、`vector`、`keyword` 的「中位数 / 90 分位」，单位秒：

| 配置 | `hybrid` | `vector` | `keyword` |
|---|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | 0.29 / 0.66 | 0.25 / 1.48 | 0.01 / 0.01 |
| `google/gemini-2.5-flash-lite` | 0.26 / 0.63 | 0.26 / 0.54 | 0.01 / 0.01 |
| `openai/gpt-4.1-mini`（已发表，`locomo-step`） | 0.26 / 0.58 | 0.24 / 0.40 | 0.01 / 0.01 |
| `anthropic/claude-haiku-5.5` | 0.27 / 0.53 | 0.28 / 0.69 | 0.01 / 0.01 |
| 边界 `qwen3-4b`，episode 和 summary `gpt-5.4-mini`，原子事实和检索 `qwen3-235b` | 0.27 / 0.61 | 0.25 / 0.56 | 0.01 / 0.01 |
| 边界、episode 和 summary `gpt-5.4-mini`，原子事实和检索 `qwen3-235b` | 0.30 / 1.97 | 0.25 / 3.23 | 0.01 / 0.01 |
| `qwen/qwen3-32b` | 0.27 / 0.63 | 0.27 / 1.20 | 0.01 / 0.01 |
