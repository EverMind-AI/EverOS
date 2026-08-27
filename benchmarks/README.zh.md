# 评测运行指南

EverOS 自带一套四个长期记忆评测基准的运行器:
[LoCoMo](https://github.com/snap-research/locomo)
([Maharana et al., 2024](https://arxiv.org/abs/2402.17753))、
[LongMemEval](https://github.com/xiaowu0162/LongMemEval)、SubtleMemory、
EverMemBench。

四者走同一条流水线,唯一的差别是各自的 adapter。每个基准自己的规则 —— 如何命名
owner、什么算 gold 证据、judge 如何判分 —— 都在
`benchmarks/adapters/<name>.py` 里。运行器的其余部分不知道自己在跑哪个基准。

> English version: [README.md](README.md)

---

## 快速开始

```bash
# 1. 安装
uv sync

# 2. 配置
cp benchmarks/.env.example benchmarks/.env
$EDITOR benchmarks/.env          # answer/judge 的 API key、抽取 backbone

# 3. 把数据集放到配置期望的位置
curl -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# 4. 复现
DATASET=locomo bash benchmarks/reproduce.sh
```

最后这条命令会按**已发布数字所用的配置**跑完整流水线(ADD → SEARCH → ANSWER →
JUDGE),报告写到 `benchmarks/results/LoCoMo/locomo/`。

---

## 目录结构

```
benchmarks/
├── reproduce.sh          入口:端到端跑一个基准
├── run.py                流水线(ADD → SEARCH → ANSWER → JUDGE)
├── config.py             冻结的配置模型
├── configs/
│   ├── default.toml      共享默认值
│   └── <dataset>.toml    每个基准一份:模型、top_k、并发
├── adapters/
│   ├── base.py           一个 adapter 要回答的四个问题
│   └── <dataset>.py      每个基准一份:owner、gold、prompt、judge
├── metrics/              排序检索指标 + core 选择指标
├── data/                 ← 数据集放这里(已 git-ignore)
├── results/              ← 跑批产物写这里(已 git-ignore)
└── .env.example          复制成 .env
```

`data/` 和 `results/` 是你唯二需要写入的目录,两个都已 git-ignore ——
数据集只有其原作者有权再分发,而跑批产物又大又可重新生成。

---

## 1. 准备数据集

每个基准的输入路径来自它自己的配置,默认就指向 `benchmarks/data/`。
**把文件放进去就够了** —— 不需要环境变量,不需要命令行参数:

| 基准 | 文件放在 | 环境变量(可选覆盖) |
|---|---|---|
| `locomo` | `benchmarks/data/locomo10.json` | `BENCH_DATA_LOCOMO` |
| `longmemeval` | `benchmarks/data/longmemeval_s.json` | `BENCH_DATA_LONGMEMEVAL` |
| `subtlememory` | `benchmarks/data/subtlememory/`(内含 `persona_0..9/` 的目录) | `BENCH_DATA_SUBTLEMEMORY` |
| `evermembench` | `benchmarks/data/evermembench.json` | `BENCH_DATA_EVERMEMBENCH` |

只有当数据放在别处(共享挂载、另一块盘)时才需要设环境变量,写在
`benchmarks/.env` 里。

### 各数据集怎么拿

**LoCoMo** —— 10 段多 session 对话,每段约 50 个 session,每段约 150 个 QA,
覆盖四个类别(adversarial 类别被排除)。

```bash
curl -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

**LongMemEval** —— 取 `longmemeval_s` 划分,来自
[xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval),
放成 `benchmarks/data/longmemeval_s.json`。

**SubtleMemory** 和 **EverMemBench** —— 由各自作者发布,按其项目说明获取。
EverMemBench 需要先把原始发布版转换一次;把 `EVERMEMBENCH_RAW_ROOT` 指向原始目录,
adapter 会从中恢复转换后文件里没有的 session 名。

### 路径不对时会怎样

路径缺失或配错会**让这次跑失败**,而不是载入 0 行然后打出 0%。
未设置且没有默认值的变量会以字面量 `${BENCH_DATA_LOCOMO}` 到达 loader 并被点名报出,
所以报错说的是"该设哪个变量",而不是"找不到一个叫 `${...}` 的文件"。

---

## 2. 配置

```bash
cp benchmarks/.env.example benchmarks/.env
```

第一次跑至少要填:

| 键 | 是什么 |
|---|---|
| `ANSWER_API_KEY` / `ANSWER_BASE_URL` | 根据检索到的记忆回答问题的模型 |
| `JUDGE_API_KEY` / `JUDGE_BASE_URL` | 给上面这些回答判分的模型 |
| `EVEROS_LLM__MODEL` / `__API_KEY` / `__BASE_URL` | ADD 阶段 EverOS 跑的抽取 backbone |

**决定数字的一切** —— 模型、`top_k`、检索旋钮、并发 —— 都在
`benchmarks/configs/<dataset>.toml` 里,不在命令行上。这是刻意的:
一次跑要能从它的配置复现,所以 `reproduce.sh` 不传任何模型覆盖参数。

### 多轮 decider(可选)

`llm_multiround` 检索会跑一个 **decider**:读候选记忆、挑出 core 集合、
提出补充子查询。默认它跑和抽取相同的模型,无需额外配置。

要给它单独的模型,**两个必须成对设置**:

```bash
BENCH_DECIDER_MODEL=<模型名>
BENCH_DECIDER_BASE_URL=<提供该模型的端点>
```

只设一个,正是这个运行器在启动时专门检查的那种故障:
把模型名发到不提供该模型的端点,每次调用都返回 404,而检索循环会**退回到固定的
top-N core 并照常报出一个完整结果**。`run.py` 会在第一个问题之前探测 decider,
不应答就拒绝启动。

---

## 3. 启动 server(只在你想自己管理时)

`reproduce.sh` 会自己起停 server。只有当你想跨多次跑复用同一个库时才手动启:

```bash
ulimit -n 10240        # 并发检索会同时打开大量 LanceDB 分片文件
everos server start [--root <path>]
```

把同一个路径用 `--everos-root` 传给运行器。运行器要靠轮询该 root 下的 cascade 和
OME 数据库来判断数据是否就绪,路径不一致会造成**静默的假就绪**。

---

## 4. 运行

```bash
# 完整复现:全部对话、四个阶段
DATASET=locomo bash benchmarks/reproduce.sh

# 只跑一个切片做冒烟
DATASET=locomo CONV=0 STAGES=search bash benchmarks/reproduce.sh \
  --everos-root <一个已有的库>

# run.py 的任何参数都会被转发
DATASET=longmemeval bash benchmarks/reproduce.sh --servers 4
```

| 变量 | 默认 | 含义 |
|---|---|---|
| `DATASET` | `locomo` | `locomo` \| `longmemeval` \| `subtlememory` \| `evermembench` |
| `CONV` | `all` | 对话下标;**切片不算复现** |
| `STAGES` | `add search answer judge` | 跑哪些阶段 |
| `RUN` | 自动推导 | 结果目录名 |

### 四个阶段

| 阶段 | 做什么 |
|---|---|
| **ADD** | 把每段对话流式送入 EverOS,由它抽取记忆 |
| **SEARCH** | 每题一次检索;写下答题模型将会看到的 episodes |
| **ANSWER** | 拿这些 episodes 让答题模型逐题作答 |
| **JUDGE** | 对照参考答案给每个回答判分 |

阶段可断点续跑:每个阶段按对话写 JSONL,已在文件里的会跳过。
在第 199 题里的第 190 题崩溃,只丢一题,不是 190 题。
用 `--stages answer judge` 重跑会直接复用已有的检索结果。

---

## 5. 输出

```
benchmarks/results/<Dataset>/<run>/
├── report.txt            人读的汇总
├── report.json           同样的数字,机器可读
├── run_spec.json         用到的每个模型、端点、包版本、旋钮
├── conv<N>/
│   ├── search_<method>.jsonl
│   ├── answer_<method>.jsonl
│   └── judge_<method>.jsonl
└── traces/               开启时:逐轮检索轨迹
```

`run_spec.json` 是一个数字日后还能复现的依据。它记录的是**实际被服务的模型**
(不是配置里写的那个),外加包版本 —— 因为抽取算法的版本会改变一个库里装的是什么。

---

## 可复现性

一个数字能不能和已发布的比,取决于两件事:

**库的抽取 backbone。** 用不同抽取模型建的库是不同的库。
`run_spec.json` 记录了是谁建的,共享库里的 `store_spec.json` 记录了产出它的
`everalgo` 版本。**检索跨版本可复现,抽取不行。**

**judge。** 每个基准的 judge 是它自己的 —— LongMemEval 会按类别追加宽容条款,
EverMemBench 的选择题按字母判分、完全不调 LLM。adapter 逐字搬运了各基准参考实现里的
judge,并有一道 parity 门禁逐字节比对。

---

## 命令行参考

`reproduce.sh` 覆盖了常规路径。`run.py` 直接接受这些参数:

| 参数 | 含义 |
|---|---|
| `--conv` | 对话下标,或 `all` |
| `--stages` | `add search answer judge` 的任意组合 |
| `--everos-root` | 用哪个库;默认 `<results>/<run>/store` |
| `--servers` | 并行起几个 EverOS server |
| `--results-root` | 写到哪;默认 `benchmarks/results/<Dataset>` |
| `--data-path` | 单次覆盖数据集路径 |
| `--methods` | `llm_multiround` \| `hybrid` \| `agentic` |
| `--answer-model` / `--judge-model` | 单次覆盖 |
| `--decider-model` / `--decider-base-url` | 多轮 decider,**两个一起设** |
| `--smoke` | 每段对话抽 10 题 |

---

## 排障

| 现象 | 原因 | 处理 |
|---|---|---|
| `benchmarks/.env not found` | 第一次跑 | `cp benchmarks/.env.example benchmarks/.env` |
| `data directory not found: benchmarks/data/...` | 数据集没下载 | 见[「1. 准备数据集」](#1-准备数据集) |
| `decider ... did not answer` | 设了 `BENCH_DECIDER_MODEL` 却没设端点 | 两个都设,或都不设 |
| `decider ... produced no usable content` | 推理型模型把预算全花在思考上 | `EVEROS_DECIDER__EXTRA='{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'` |
| wait_ready 里 `Timeout after 1800s` | 抽取还没跑完 | 调大配置里的 `cascade_timeout`,或去看 server 日志 |
| `Too many open files` | 并发下 LanceDB 文件描述符耗尽 | 调低 `search_concurrency`,或调大 `ulimit -n` |
| 检索到 0 条 episode 且不报错 | 库是用不同的分区键建的 | `app_id` / `project_id` 必须和建库时一致 |
