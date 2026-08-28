# 评测运行指南

一个运行器,四个评测,四个阶段:ADD、SEARCH、ANSWER、JUDGE。
各评测自己的规则在 `benchmarks/adapters/<name>.py`。

> English version: [README.md](README.md).

---

## 1. 安装依赖

```bash
uv sync
```

## 2. 准备数据

| 评测 | 目标路径 | 来源 |
|---|---|---|
| LoCoMo | `benchmarks/data/locomo10.json` | [snap-research/locomo](https://github.com/snap-research/locomo) |
| LongMemEval | `benchmarks/data/longmemeval_s.json` | [xiaowu0162/longmemeval](https://huggingface.co/datasets/xiaowu0162/longmemeval) |
| EverMemBench | `benchmarks/data/evermembench.json` | [EverMind-AI/EverMemBench-Dynamic](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) |
| SubtleMemory | `benchmarks/data/subtlememory/` | [Yummytanmo/SubtleMemory](https://huggingface.co/datasets/Yummytanmo/SubtleMemory) |

> **EverMemBench** —— 将快照下载到 `benchmarks/data/raw/EverMemBench-Dynamic/`,
> 然后执行一次 `python -m benchmarks.adapters.evermembench`。
>
> **SubtleMemory** —— 保留 `persona_0` .. `persona_9` 目录结构。

## 3. 配置环境

```bash
cp benchmarks/.env.example benchmarks/.env
```

| 角色 | 键 | 说明 |
|---|---|---|
| 抽取模型 | `EVEROS_LLM__MODEL` / `__API_KEY` / `__BASE_URL` | ADD 阶段使用 |
| 检索模型 | `BENCH_DECIDER_MODEL` / `BENCH_DECIDER_BASE_URL` | 多轮 decider;默认 `qwen3.6-27B`,未设端点时改用抽取模型 |
| 答题模型 | `ANSWER_API_KEY` / `ANSWER_BASE_URL` | 模型名在配置文件中 |
| 判分模型 | `JUDGE_API_KEY` / `JUDGE_BASE_URL` | 模型名在配置文件中 |

模型、`top_k`、检索设置和并发在 `benchmarks/configs/<dataset>.toml` 中设置。
`reproduce.sh` 不传任何模型覆盖参数。

## 4. 运行

```bash
# 全部对话、四个阶段
DATASET=locomo bash benchmarks/reproduce.sh

# 冒烟测试:单个对话,抽样 10 题
DATASET=locomo CONV=0 bash benchmarks/reproduce.sh --smoke
```

| 变量 | 默认 | 取值 |
|---|---|---|
| `DATASET` | `locomo` | `locomo` \| `longmemeval` \| `subtlememory` \| `evermembench` |
| `CONV` | `all` | 对话下标 |
| `STAGES` | `add search answer judge` | 任意子集,按顺序 |
| `RUN` | 自动推导 | 结果目录名 |

`reproduce.sh` 自行启动和关闭 EverOS server。

| 阶段 | 动作 |
|---|---|
| ADD | 将对话流式送入 EverOS,由其抽取记忆 |
| SEARCH | 每题一次检索,记录注入的 episode |
| ANSWER | 基于这些 episode 回答每题 |
| JUDGE | 对照参考答案判分 |

各阶段按对话追加 JSONL 并跳过已存在的条目,中断后可续跑。

## 5. 输出

```
benchmarks/results/<Dataset>/<run>/
├── report.txt            汇总
├── report.json           机器可读汇总
├── run_spec.json         实际服务的模型、端点、包版本、参数
├── store/                ADD 建立的库;--everos-root 指向别处时不存在
├── conv<N>/
│   ├── search_<method>.jsonl
│   ├── answer_<method>.jsonl
│   └── judge_<method>.jsonl
└── traces/               逐轮检索 trace
```

`run_spec.json` 记录实际提供服务的模型,以及决定库内容的 `everalgo` 包版本。

## 6. 结果

各评测的已发表数字,由 `benchmarks/configs/<dataset>.toml` 的配置产出。

| 评测 | 题量 | 准确率 | decider | 答题模型 |
|---|---|---|---|---|
| LoCoMo | 1,540 | **94.42** | `deepseek/deepseek-v4-flash-0731` | `openai/gpt-4.1-mini` |
| LongMemEval | 500 | **94.00** | `qwen3.6-27B` | `google/gemini-3.6-flash` |
| EverMemBench | 2,400 | **66.67** | `qwen3.6-27B` | `google/gemini-3-flash-preview` |
| SubtleMemory | 1,522 | **71.75** | `qwen3.6-27B` | `openai/gpt-5.4` |

> **EverMemBench** —— 按其九个类目列的均值计分,这是该评测自身的报告方式。
>
> **SubtleMemory** —— 按事实抽取是否发现冲突,在两套答题契约之间逐题路由,
> 适配器自动完成。


---

## run.py 参数

| 参数 | 含义 |
|---|---|
| `--conv` | 对话下标,或 `all` |
| `--stages` | `add search answer judge` 的任意组合 |
| `--everos-root` | 使用的库;默认 `<results>/<run>/store` |
| `--servers` | 并行的 EverOS server 数 |
| `--results-root` | 输出根目录;默认 `benchmarks/results/<Dataset>` |
| `--data-path` | 单次运行的数据集路径 |
| `--methods` | `llm_multiround` \| `hybrid` \| `agentic` |
| `--answer-model` / `--judge-model` | 单次运行的模型覆盖 |
| `--decider-model` / `--decider-base-url` | decider;两项须同时设置 |
| `--smoke` | 每个对话抽样 10 题 |
