# 评测运行指南

一个运行器，四个评测，四个阶段：ADD、SEARCH、ANSWER、JUDGE。
各评测自己的规则在 `benchmarks/adapters/<name>.py`。

> English version: [README.md](README.md).

## 快速开始

从一个全新 clone 跑 LoCoMo 冒烟测试。所有命令都在仓库根目录执行。

```bash
uv sync
uv run python -m nltk.downloader -d .venv/nltk_data punkt_tab stopwords
bash benchmarks/download_data.sh locomo
cp benchmarks/.env.example benchmarks/.env   # 然后填好 REQUIRED 下的各行：
                                             # OpenRouter key 填三处，DeepInfra key 填两处
DATASET=locomo bash benchmarks/reproduce.sh --smoke
# 只有 OpenRouter key——改用关键词检索：
DATASET=locomo bash benchmarks/reproduce.sh --smoke --methods keyword
```

每种检索方法需要哪些 key，见 [需要哪些 key](#需要哪些-key)。

下面逐步说明，每一步都附有「怎样算成功」的检查。运行在启动任何东西之前会先检查数据和
key，一次列出所有缺项，所以第一次失败不会花一分钱。

**需要准备：**

- Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)
- `bash` 和 `curl`（macOS 或 Linux；Windows 请用 WSL2，见
  [docs/windows.zh.md](../docs/windows.zh.md)）
- 一个 [OpenRouter](https://openrouter.ai) key（抽取、答题、判分模型）。只有它就能跑
  `--methods keyword`。
- 一个 [DeepInfra](https://deepinfra.com) key，用于 EverOS 默认的 embedding 和 rerank 模型。
  LoCoMo 用 `agentic` 检索，两者都要；另外三个评测用 `llm_multiround` 检索，只要 embedding。

---

## 1. 安装依赖

```bash
uv sync
uv run python -m nltk.downloader -d .venv/nltk_data punkt_tab stopwords
```

LoCoMo 的建索引和检索都用 EverAlgo 评测框架的 NLTK 分词器（`configs/locomo.toml` 里的
`EVEROS_TOKENIZER__NAME = "nltk"`），需要上面这份 NLTK 数据；`make install-deps` 也会下载。
用别的分词器建的库不能用这个分词器检索，只能复用同一设置下建的库。

**成功标志：** `uv sync` 无报错退出，且 `.venv/bin/python` 存在。

## 2. 准备数据

```bash
bash benchmarks/download_data.sh locomo   # 或 longmemeval | evermembench | subtlememory | all
```

脚本只下载 adapter 实际读取的文件，直接放到配置期望的路径；已存在的文件会保留，
所以下载中断后重跑同一条命令即可续完。EverMemBench 还会顺带完成一次性的格式转换。

| 评测 | 落盘位置 | 下载量 | 来源 |
|---|---|---|---|
| LoCoMo | `benchmarks/data/locomo10.json`，外加金标审计 `locomo-audit-errors.json` | 2.8 MB + 151 KB | [snap-research/locomo](https://github.com/snap-research/locomo)、[dial481/locomo-audit](https://github.com/dial481/locomo-audit) |
| LongMemEval | `benchmarks/data/longmemeval_s.json` | 278 MB | [xiaowu0162/longmemeval](https://huggingface.co/datasets/xiaowu0162/longmemeval) |
| EverMemBench | `benchmarks/data/evermembench.json` | 10 个文件 | [EverMind-AI/EverMemBench-Dynamic](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) |
| SubtleMemory | `benchmarks/data/subtlememory/persona_{0..9}/` | 20 个文件 | [Yummytanmo/SubtleMemory](https://huggingface.co/datasets/Yummytanmo/SubtleMemory) |

EverMemBench 还会把原始数据保留在 `benchmarks/data/raw/EverMemBench-Dynamic/`；adapter 运行时
要从中读取 session 名，请不要删除。

**成功标志：** 输出中出现 `Data ready.`

数据已经放在别处？改为在 `benchmarks/.env` 里把 `BENCH_DATA_LOCOMO`、`BENCH_DATA_LONGMEMEVAL`、
`BENCH_DATA_EVERMEMBENCH`、`BENCH_DATA_SUBTLEMEMORY` 或 `BENCH_DATA_LOCOMO_AUDIT` 指向它。

## 3. 配置

```bash
cp benchmarks/.env.example benchmarks/.env
```

填好文件顶部 `REQUIRED` 区块里的 key，其余保持默认即可。

| 变量 | 填什么 | 谁在用 |
|---|---|---|
| `EVEROS_LLM__API_KEY` | OpenRouter key | EverOS server：抽取（ADD）和检索 decider |
| `EVEROS_EMBEDDING__API_KEY` | DeepInfra key | EverOS server：embedding（默认 Qwen3-Embedding-4B）——除 `keyword` 外的所有检索方法 |
| `EVEROS_RERANK__API_KEY` | DeepInfra key | EverOS server：rerank（默认 Qwen3-Reranker-4B）——LoCoMo 用的 `agentic`；没有它，每次 agentic 检索都会失败 |
| `ANSWER_API_KEY` | OpenRouter key | ANSWER |
| `JUDGE_API_KEY` | OpenRouter key | JUDGE |

### 需要哪些 key

EverOS 本身只强依赖 LLM；embedding 和 rerank 是可选能力，配上之后能用更多检索方法（见根目录
[README.zh-CN.md](../README.zh-CN.md) 的「一个百炼 Key 可以使用哪些能力？」一节）。因此评测需要
什么，取决于它的检索方法——`configs/<dataset>.toml` 里的 `methods`（LoCoMo 是 `agentic`，
另外三个是 `llm_multiround`），或单次运行的 `--methods`：

| 检索方法 | OpenRouter key | DeepInfra key：embedding | DeepInfra key：rerank |
|---|---|---|---|
| `agentic`（LoCoMo） | 要 | 要 | 要 |
| `llm_multiround`（另外三个） | 要 | 要 | — |
| `vector`、`hybrid` | 要 | 要 | — |
| `keyword`（仅 BM25） | 要 | — | — |

没有 embedding 时，EverOS 在 ADD 阶段也会跳过反思（reflection）和技能抽取，所以 `keyword` 运行
测的是一条更短的流水线，分数不能和 [6. 结果](#6-结果) 对比。运行会在启动任何东西之前，按你选的
方法检查这些要求。

模型、`top_k`、检索设置和并发在 `benchmarks/configs/<dataset>.toml` 中设置，不在 `.env` 里；
下面可选的 decider 覆盖是 `.env` 唯一接受的模型设置。`reproduce.sh` 不传任何模型覆盖参数。
后台策略开关（`[ome_strategies]`，如 LoCoMo 关掉画像）会写进每个启动的服务的 `ome.toml`，
EverOS 从那里读取。LoCoMo 的配置还把每条消息都记在被提问的那位说话人名下
（`message_owner = "eval_owner"`），分享了图片的那一轮会把图片描述放在原文前面再交给抽取；
两者都决定库里存什么，所以都属于入库身份。

`flush_scope` 决定一段对话的多少内容进同一个 EverOS session、在结尾 flush 一次：
`"session"`（默认，所有自带配置都用它）按数据集的每个 session 分别 flush；`"conversation"`
把整段对话当成一个 session，与 EverAlgo 参考评测框架的做法相同，于是一条记忆可能跨越两个
数据集 session。session 自带 id 的数据集不接受 `"conversation"`。

**可选：单独的检索 decider**（只用于 `llm_multiround`）。各配置已预填已发表数字所用的 decider
（见 [6. 结果](#6-结果) 的 decider 列；LoCoMo 见 [LoCoMo 各检索方式](#locomo-各检索方式)）。
要使用它，把 `BENCH_DECIDER_BASE_URL` 设为一个提供该模型的 OpenAI 兼容端点，`BENCH_DECIDER_API_KEY`
填它的 key；只有想换别的模型时才设 `BENCH_DECIDER_MODEL`。LoCoMo 的 decider 在 OpenRouter 上：
`BENCH_DECIDER_BASE_URL=https://openrouter.ai/api/v1`，key 填你的 OpenRouter key；它的配置让 decider 关着思考运行。不设端点时由
抽取模型担任 decider，运行会提示这不是已发表的配置。运行通过 EverOS 的 `[llm.decider]` 使用它，
并在第一道题之前先确认它能应答。

**可选：自托管 embedding。** 把 `EVEROS_EMBEDDING__BASE_URL` 设为你的服务地址，
`EVEROS_EMBEDDING__MODEL` 设为该服务实际提供的模型名，`EVEROS_EMBEDDING__API_KEY` 填任意非空值。
模型输出至少要有 1024 维（EverOS 的 `[embedding] dim`）；维度不够的，server 启动时会拒绝。

**成功标志：** `benchmarks/.env` 里 `REQUIRED` 下的各行都已填值（你要跑的方法都用不到的服务商
key 可以留空）。第 4 步还会再查一遍：缺任何一个，运行都会以
`Not started -- nothing has been spent yet` 停下，并点名缺的变量。

## 4. 运行

先跑冒烟测试：

```bash
DATASET=locomo bash benchmarks/reproduce.sh --smoke
```

冒烟测试只灌入对话 0 的前 50 条消息，最多抽样回答 10 题（LongMemEval 每个对话只有一题，所以只答
一题），每题判分一次。它端到端跑通每个阶段，但分数不能和 [6. 结果](#6-结果) 对比，`report.txt`
开头几行也会注明这一点。用它确认配置无误，并在跑全量之前看清一个对话在你的服务商那里花多少钱。
报告只统计答题和判分的 token；抽取和 embedding 的用量请看各服务商的控制台。

冒烟分数能说明多少，因评测而异。LoCoMo 抽到的题大多引用前几个会话（10 题里有 9 题只引用前
50 条消息），所以分数大致有参考意义。LongMemEval、EverMemBench 和 SubtleMemory 抽到的题，证据都在
前 50 条消息之后，所以冒烟分数偏低是正常的；它说明流程跑通了，不说明检索效果。

ADD 开始之前，运行会按顺序检查：

1. 数据路径存在、所选检索方法需要的 key 都已设置——在启动任何 server 之前；通过时会打印
   `Preflight: data and keys ready`；
2. 它启动的每个 EverOS server 都具备该方法需要的能力（embedding；`agentic` 还要
   rerank）——在 ADD 写入任何东西之前；通过时会打印 `Embedding answered on 1 server(s)`
   （以及 `Rerank configured ...`）。`keyword` 两者都不需要，跳过这一步。

任一检查不通过，运行都会停下，并说明怎么修。默认的冒烟运行中还会出现一行属于正常现象、不是问题：
`--servers 10 exceeds 1 conversation(s); using 1`。

**成功标志：** `benchmarks/results/LoCoMo/locomo_smoke/report.txt` 存在。结果目录是
`benchmarks/results/<Dataset>/<RUN>/`：`<Dataset>` 为 `LoCoMo`、`LongMemEval`、`EverMemBench`
或 `SubtleMemory`；`RUN` 在全量运行时是数据集名，只跑部分对话时是 `<dataset>_conv<CONV>`，
冒烟运行再加 `_smoke` 后缀——所以冒烟的库永远不会和正式运行共用目录。另外，如果一次运行灌入的
消息数和建库时不同，结果目录会拒绝续跑。

然后跑全量：

```bash
DATASET=locomo bash benchmarks/reproduce.sh
```

| 变量 | 默认 | 取值 |
|---|---|---|
| `DATASET` | `locomo` | `locomo` \| `locomo-step` \| `longmemeval` \| `subtlememory` \| `evermembench` |
| `CONV` | `all`（带 `--smoke` 时为对话 0） | 对话下标 |
| `STAGES` | `add search answer judge` | 任意子集，按顺序 |
| `RUN` | 自动推导 | 结果目录名 |

命令后面可以接任何 `run.py` 参数，见 [run.py 参数](#runpy-参数)。
`reproduce.sh` 调用的 `run.py` 会自行启动和关闭 EverOS server：数量取配置里的 `servers`，
且不超过对话数。每个 server 都是一个独立的 EverOS 实例，有自己的库；对话按序号分片到各个
实例上（对话 `c` 分给第 `c mod N` 个 server）。EverOS 本身每个库只跑一个进程，所以这是在
多个独立的库上并行，不是把一套部署水平扩展。`--servers N` 只影响跑得快慢，不影响结果。
可以同时跑多次、跑不同的 benchmark：每次运行只用自己启动的 server，两次运行不会共用同一个库。

`locomo-step` 是用逐条边界检测、且 EverOS 每次调用都只在 prompt 里要求回复格式的 LoCoMo：
`configs/locomo-step.toml` 就是 `locomo.toml` 改了这两项 server 设置。两者各自的分数和代价见
[batch 还是 step](#batch-还是-step)。

不用这个包装脚本时，同样的冒烟测试是：

```bash
uv run python benchmarks/run.py locomo --run-name locomo_smoke --smoke
```

`run.py` 会在读配置之前自己加载 `benchmarks/.env`，所以那里的 `BENCH_*` 变量对直接运行同样生效。

| 阶段 | 动作 |
|---|---|
| ADD | 将对话流式送入 EverOS，由其抽取记忆 |
| SEARCH | 每题一次检索，记录注入的 episode |
| ANSWER | 基于这些 episode 回答每题 |
| JUDGE | 对照参考答案判分；有金标审计时，还会重判有争议的题（见 [每次运行给出两个分数](#每次运行给出两个分数)） |

各阶段按对话追加 JSONL 并跳过已存在的条目，中断后重跑同一条命令即可续跑。

### 在已建好的库上给其他检索方式打分

一次运行建好的库可以直接给任何检索方式用，不必重新写入。每种检索方式单独起一个运行，跳过 ADD，
指向那些库：

```bash
DATASET=locomo RUN=locomo_hybrid STAGES="search answer judge" \
  bash benchmarks/reproduce.sh --everos-root benchmarks/results/LoCoMo/locomo/store \
  --methods hybrid
```

多个 server 时，`--everos-root` 写库的基础路径，运行会读它旁边的 `store_s0` 到
`store_s<N-1>`；只跑检索不会改动这些库。`DATASET` 要和建库时一致（step 建的库用 `locomo-step`），
每种检索方式用一个新的 `RUN`：结果目录换了检索设置重跑时，会把之前的检索、回答和判分文件归档成
`*.superseded.jsonl`。

### 运行被拦下时

| 提示 | 怎么办 |
|---|---|
| `benchmarks/.env not found.` | `cp benchmarks/.env.example benchmarks/.env`，然后填好 `REQUIRED` 下的 key |
| `.venv not found` | `uv sync` |
| `Not started -- nothing has been spent yet. Fix these first:` | 每一行点名一个缺失的数据集（附下载命令）或一个未设置的 key |
| `The embedding service did not answer the startup probe` | 检查 `EVEROS_EMBEDDING__BASE_URL` / `__MODEL` / `__API_KEY`，或改用 `--methods keyword`；对提示里给出的 server 日志 `grep embedding_` 可看到 server 自己报的原因 |
| `No rerank provider is configured` | 设置 `EVEROS_RERANK__API_KEY`，或换一个不是 `agentic` 的方法 |
| `decider ... did not answer` | `llm_multiround` 运行的 decider 没有回答探测请求，提示里写了原因；见 [3. 配置](#3-配置) 中的 decider 选项 |
| `llm_multiround round(s) fell back to a fixed core` | decider 在这些轮里失败了，server 日志里的 `llm_multiround_decider_error` 写明了原因；处理好后换一个新的 `RUN` 重新检索 |
| `decider  no endpoint for ...` | 是警告，不是错误（只在 `llm_multiround` 出现）：运行会继续，改用抽取模型，分数不能和 [6. 结果](#6-结果) 对比 |
| `Refusing to start: these stores already hold memories` | 删除列出的库后重跑，或加 `--reuse-store` |
| `No SubtleMemory personas under ...` | `bash benchmarks/download_data.sh subtlememory` |

## 5. 输出

```
benchmarks/results/<Dataset>/<run>/
├── report.txt            汇总；开头几行会标出冒烟或截断运行
├── report.json           机器可读汇总
├── run_spec.json         实际服务的模型、端点、包版本、参数
├── store/                单个 server 时 ADD 建立的库（里面有该 server 的 server.log）
├── store_s0/ .. store_s<N-1>/   N 个 server 时改为每个 server 一个库
├── conv<N>/
│   ├── add.done          标记：该对话的 ADD 已完成
│   ├── search_<method>.jsonl     例如 search_agentic.jsonl
│   ├── answer_<method>.jsonl
│   ├── judge_<method>.jsonl
│   └── rejudge_<method>.jsonl    有争议的题，按审计答案重判
└── traces/               逐轮检索 trace
```

`run_spec.json` 记录实际提供服务的模型（decider 按 EverOS 的解析结果记录，附每项设置的生效值和
来源段，`extra` 里请求头和查询参数的值会遮掉）、决定库内容的 `everalgo` 包版本，以及这次运行是不是
冒烟或截断运行（`smoke`、`max_messages`）。只有用 `--everos-root` 指向别处时，库才不在这里；
运行开头会以 `store` 一行打印库的位置。

### 每次运行给出两个分数

`report.txt`、`report.json` 和终端汇总为每种检索方法并列给出两个准确率，总分和分类别都有：

| 行 | 怎么判 | 用途 |
|---|---|---|
| `Accuracy` | 每个回答都对照数据集的金标 | 和其他系统已发表的结果比较，它们用这个口径 |
| `Re-judged` | 同样的回答，其中第三方审计认为金标有问题的题改为对照审计给出的正确答案 | [6. 结果](#6-结果) 里的 LoCoMo 数字（标 \*）：金标本身错了的题，测的是数据集而不是系统 |

重判只换金标：不重新检索、不重新答题，有争议的回答用同一个判分模型、同样的判分次数和同样的
多数票再判一次。JUDGE 把结果写进 `rejudge_<method>.jsonl`，已判过的会跳过；审计文件下载之前
就判完分的运行，重跑 `--stages judge` 即可补上，这时只判有争议的题。在所有有争议的题都重判完
之前，报告不给 `Re-judged` 的数字，只写明还差几题。

LoCoMo 用的审计是 [dial481/locomo-audit](https://github.com/dial481/locomo-audit) 的
`errors.json`：1540 道题里有 99 道金标错误、含糊或没有依据，每道都附了正确答案。只修正证据编号
的条目，金标本身没问题，不重判。`download_data.sh locomo` 会把它下载到数据集旁边，并固定在读取时
的那个提交；它和 LoCoMo 一样采用 CC BY-NC 4.0 许可。没有这个文件时运行照常进行，报告写明未重判。
另外三个评测没有审计，报告会在 `Re-judged` 一行写明。

LoCoMo 每道题判 3 次，按多数票定对错（见 `benchmarks/configs/<dataset>.toml` 的
`[judge] runs`）；`Mean accuracy` 是各次判分的平均，它和 `Accuracy` 的差距就是判分本身的波动。
`--smoke` 仍只判 1 次。

### 清理

一次运行写下的所有东西都在它的结果目录里，删掉这个目录，库、日志、trace 和报告就一起清掉了。
结果目录之外，运行会留下：

| 内容 | 位置 | 怎么处理 |
|---|---|---|
| 下载的数据集 | `benchmarks/data/` | 留给下次复用，或者删掉（保留 `.gitkeep`） |
| 你的 key | `benchmarks/.env` | 跑完删掉，或者自己保管好 |
| NLTK 数据 | `.venv/nltk_data/` | 随 `.venv` 一起处理 |
| 残留的 server | 进程；只有 `run.py` 被 `kill -9` 时才会有 | 先查出来，再按 PID 停掉 |

`benchmarks/results/`、`benchmarks/data/` 和 `benchmarks/.env` 都已被 git 忽略。

```bash
rm -rf benchmarks/results/LoCoMo/locomo_smoke        # 删一次运行
rm -rf benchmarks/results                            # 删全部运行
find benchmarks/data -mindepth 1 ! -name .gitkeep -delete
rm -f benchmarks/.env
ps -eo pid,command | grep "[e]veros server start"    # 查残留的 server
```

## 6. 结果

各评测的已发表数字。LoCoMo 由 `benchmarks/configs/locomo-step.toml` 产出，其余由
`benchmarks/configs/<dataset>.toml` 产出。

| 评测 | 题量 | 准确率 | decider | 答题模型 |
|---|---|---|---|---|
| LoCoMo | 1,540 | **94.61\*** | --（`agentic` 不用） | `openai/gpt-4.1-mini` |
| LongMemEval | 500 | **94.00** | `qwen3.6-27B` | `google/gemini-3.6-flash` |
| EverMemBench | 2,400 | **66.67** | `qwen3.6-27B` | `google/gemini-3-flash-preview` |
| SubtleMemory | 1,522 | **71.75** | `qwen3.6-27B` | `openai/gpt-5.4` |

> **准确率**——表中每个评测都是微平均准确率：所有已判分题目的 `sum(correct) / sum(total)`。
> `report.txt` 里的分类目各行只做报告，从不平均进总分，所以题量少的类目对总分的影响小于
> 题量大的类目。
>
> **EverMemBench**——它已发表的对比表里，大多数行是各类目列的宏平均，和这里的微平均是两种
> 不同的统计量。
>
> **SubtleMemory**——按事实抽取是否发现冲突，在两套答题契约之间逐题路由，
> 适配器自动完成。

\* 重判口径：LoCoMo 审计认为金标有问题的 99 道题，改为对照审计给出的正确答案判分（见
[每次运行给出两个分数](#每次运行给出两个分数)）。

### LoCoMo 各检索方式

每份 LoCoMo 配置完整跑 3 次取最高的一次，再用这次运行建好的库跑其余每种检索方式各 1 次（做法见
[在已建好的库上给其他检索方式打分](#在已建好的库上给其他检索方式打分)）。测量环境：everalgo-core 0.7.0、
everalgo-boundary 0.5.0，提交 `ccb402af`（`llm_multiround` 为 `d278a2cb`）。抽取和答题用
`openai/gpt-4.1-mini`，判分用 `openai/gpt-4o-mini` 判 3 次，`llm_multiround` 的 decider 用
`deepseek/deepseek-v4-flash-0731`，关着思考。

| 检索方式 | `locomo-step` | `locomo`（默认） |
|---|---|---|
| `agentic` | **94.61\*** | 94.35\* |
| `llm_multiround` | 94.29\* | 94.29\* |
| `hybrid` | 93.18\* | 92.08\* |
| `vector` | 92.73\* | 92.08\* |
| `keyword` | 91.04\* | 92.27\* |

\* 重判口径，同上。

### batch 还是 step

两份 LoCoMo 配置只差两项 EverOS 设置，EverOS 的默认值是第一组：

| | `locomo`（默认） | `locomo-step` |
|---|---|---|
| 边界检测，`[boundary_detection] strategy` | `batch`：每次写入调用一次 | `step`：每条消息调用一次 |
| 回复格式，`[llm] response_format_mode` | `json_schema`：由服务商保证每条回复符合 schema | `prompt_only`：只在 prompt 里要求 schema |
| `agentic`，3 次运行取最高 | 94.35\* | **94.61\*** |
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

\* 重判口径，同上。


---

## run.py 参数

| 参数 | 含义 |
|---|---|
| `--conv` | 对话下标，或 `all` |
| `--stages` | `add search answer judge` 的任意组合 |
| `--everos-root` | 使用的库；默认 `<results>/<run>/store`（或多 server 建的 `store_s<i>`） |
| `--servers` | 要启动的独立 EverOS 实例数，每个一个库，对话分片到各实例；默认取配置，且不超过对话数 |
| `--results-root` | 输出根目录；默认 `benchmarks/results/<Dataset>` |
| `--data-path` | 单次运行的数据集路径 |
| `--methods` | `agentic` \| `llm_multiround` \| `hybrid` \| `vector` \| `keyword` |
| `--answer-model` / `--judge-model` | 单次运行的模型覆盖 |
| `--decider-model` / `--decider-base-url` | 单次运行的 decider；URL 必须提供该模型 |
| `--smoke` | 不给 `--conv` 时只跑对话 0；每个对话只灌前 50 条消息、最多抽样 10 题、判分一次 |
| `--max-messages` | 每个对话最多灌入 N 条消息，不抽样题目 |
| `--reuse-store` | 向已有记忆的库继续 ADD |
