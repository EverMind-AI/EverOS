# EverOS 图归因代码审查：与 MemTrace 主实现的对照

审查日期：2026-10-11。审查方式：源码对照、现有单元测试、虚构数据的离线复现。第 1–5 节保留修改前的审查结果；用户随后授权实施修复，实施状态见第 6 节。

## 1. 结论与范围

**当前 EverOS 已有可运行的 MemTrace 风格图归因闭环，但图的版本、实际输入和依赖关系仍存在足以改变归因结论的问题。不能仅凭归因返回 `status=success`，就认定它与 MemTrace 主实现具有等价的诊断能力。**

优先处理的三项是：跨图版本错位（R1）、回答模型实际上下文与图中上下文不一致（R2）、skill bridge 的错误依赖连边（R4）。它们会直接改变归因算法看到的事实。其次处理请求隔离、占位节点、请求身份和大图探索问题。

本次以工作区实际代码为准：

| 对象 | 位置 | HEAD | 范围 |
| --- | --- | --- | --- |
| EverOS | `EverOS/` | `6ff07ef` | `infra/memtrace`、service hooks、search collector、`examples/memtrace/graphtrace` 及驱动脚本 |
| MemTrace 主实现 | `MemTrace/` | `d027ed0` | `graphtrace`、`toolkits/memtrace_utils.py`、`bench_utils.py`、图合并工具、membase 部分插桩 |
| smartcomment | `smartcomment/` | `6683bf8` | identity、version、`identity_only`、操作子图语义 |

这些 checkout 均存在预先已有的本地改动，HEAD 仅用于定位，**不代表被审查内容全部已提交**。目标是 `EverOS/`，不是 `EverOS-1/`。已有 `problem.md` 描述的部分旧问题已经被后续代码修复，本文没有将其直接作为当前缺陷依据。

实际验证使用 EverOS `.venv`：`everos` 指向本 checkout 的 `src/everos`，`smartcomment` 来自该 `.venv` 的 site-packages。已核对其 `api/_helpers.py` 与本地 smartcomment checkout 的同名文件 SHA-256 一致，所以下文关于版本分配和 `identity_only` 的分析与验证环境一致。

## 2. 算法对照：已经实现什么，尚不等价在哪里

| 环节 | MemTrace 主实现 | EverOS 当前实现 | 判断 |
| --- | --- | --- | --- |
| 归因目标 | 最早、局部错误且对最终错误具有决定性的操作 | 保留同一判定原则与五类错误 | 核心目标一致 |
| 起始证据 | 原始证据、伪证据或 synthetic root；伪证据支持 sparse/dense/hybrid，默认 hybrid | 对解码后的 message content 做 BM25，使用问题和参考答案 | 是稀疏检索子集，不能直接等同于主实现默认配置 |
| 图探索 | 七个 notebook tools、版本化节点、frontier 和每轮 hint | 保留七个工具与相同工作流 | 已实现核心机制 |
| active node | 主实现以排序容器首项表示 active | active 单独存储 | EverOS 避免了新增较早节点挤掉 active 的问题，不是缺陷 |
| 模型驱动 | AgentScope ReAct 与结构化输出 | OpenAI function calling、串行工具执行 | 可接受的实现替换 |
| 上下文管理 | token 计数、LLM 压缩、保留近期完整工具组 | 字符预算、删除旧工具组、保留最新组和全部完成记录 hint | 大图下存在 R8；不是同等的压缩策略 |
| QA 识别 | `llm-judge`、独立 `judge_score` | 同时支持 native QA 和 EverOS `judge_answer_*` | 兼容入口已补齐，但实际 QA 数据链有 R2、R7 |
| 输出 | `{error_type, op_id, reason}` | 同一 schema，加 transcript、失败状态和 inspected checks | 结构兼容；不能保证诊断正确，见 R10 |
| 批处理与实验 | `MemTraceRunner.arun`、缓存、多案例与统计 | CLI 默认选择 `cases[0]`，一次一例 | README 已明确，属于功能范围差异 |
| 图输入 | 消费已有 execution graph | 依赖文件差分、检索阶段快照及离线 QA 注入 | 归因可信度主要受 R1–R7、R9 限制 |

对照位置：MemTrace `toolkits/memtrace_utils.py:286,933,1009,1065,1200`、`graphtrace/graph_trace_agent.py:164,189`；EverOS `examples/memtrace/graphtrace/{agent,retrieval,models,notebook,prompts}.py`、`examples/memtrace/attribute.py:124`。

英文 tokenizer `[a-z0-9_]+` 是从主实现继承的行为。纯中文问题和消息可能全部得到零分，进而按创建时间选起点；这不是本次移植新增的 BM25 错误，但 EverOS 没有 dense/hybrid 补偿路径。主实现是否表现更好仍需同数据、同模型对照，本文不作准确率推断。

## 3. 问题清单

P1：可能直接改变记忆历史、真实依赖或故障归属，应在将归因结果用于实验结论前处理。P2：特定条件下丢失证据、误连请求、终止探索或接受不合规结果。

“复现”表示已经执行虚构数据验证；“源码确认”表示控制流可直接确定；“条件性限制”表示明确触发条件下存在信息缺口，不等于已证明真实业务运行出现该故障。

| 编号 | 级别 | 问题 | 证据 |
| --- | --- | --- | --- |
| R1 | P1 | 同一记忆在不同图中独立编号，合并后接错历史版本 | 复现 |
| R2 | P1 | 图中的回答上下文并非示例真正交给模型的上下文 | 源码确认 |
| R3 | P1 | 文件差分无法保证恢复真实操作历史和检索时刻状态 | 条件性限制、源码确认 |
| R4 | P1 | skill bridge 与 fusion 的输入连边不符合实际计算 | 复现、源码确认 |
| R5 | P2 | 持久记忆节点复用使不同 query 的下游相互串联 | 复现；主代码部分插桩也有同类模式 |
| R6 | P2 | candidate stub 后续成为 hit，仍不保存已经可得的正文 | 复现 |
| R7 | P2 | request id 重复与重启时的 query 身份、QA 解析不可靠 | 复现、源码确认 |
| R8 | P2 | 无值预览不分页，与保护最新工具组的策略共同导致大图直接失败 | 复现 |
| R9 | P2 | 失败的 add 没有记录本次输入消息，错误操作与源证据脱离 | 源码确认、离线验证 |
| R10 | P2 | 提交校验仍接受 judge 等不应被归因的操作 | 复现；校验能力缺口 |

### R1：跨图版本号不是同一条版本时间线

**位置**：`src/everos/infra/memtrace/tracker.py:1170–1181,1389–1421,1776–1885`；smartcomment `api/_helpers.py:214–288`；MemTrace `toolkits/graph_merge_utils.py:124–150`。

写入按 `(app, project, owner, session)` 分图，而不带 session filter 的查询通常进入 `global` 图。smartcomment 只在当前图内查找最新版本：新图第一次看到一个 identity，版本就是 `@1`。相同 identity 并不意味着不同图内的 `@1` 对应同一历史内容。

**已复现**：

```text
写入图：alice_ep_1@1 = Friday；alice_ep_1@2 = Monday
搜索图：第一次搜索发生在更新后，alice_ep_1@1 = Monday
合并冲突：alice_ep_1@1
```

主仓库 merge 工具按 `full_node_id` 合并，冲突保留较早图的快照，只在 metadata 报告冲突。因此搜索边仍指向 `@1`，但合并后的 `@1` 是 Friday；真正被检索的 Monday 成了写入图中的另一版本。图能成功 import，也不代表版本语义正确。

**影响**：丢失信息的更新可能被绕过，或把一次正确检索解释成返回旧记忆。针对 UpdateError、DeletionError、RetrievalError 的结论均受影响。仅增加“合并成功”检查不够。

另一个边界是 app/project：图文件名有租户隔离，但 entry identity 仅为 `{owner}_{entry_id}`。若将不同 app/project 的图放进同一 merge 输入，相同 owner/entry id 仍可能误合并。

**建议**：建立带 app/project 的稳定逻辑身份和跨图一致的 revision 引用；或采用统一的构建版本图并显式引用其版本。合并前拒绝无法消解的内容冲突，不能仅凭局部版本号认定相同节点。回归样例必须覆盖“先更新、后首次跨会话搜索”。

### R2：图中声称模型看到了实际上没有送入提示词的事实

**位置**：`tracker.py:1316–1334,1467–1508`；`examples/memtrace/filetrace_locomo_drive.py:167–176`；`src/everos/infra/memtrace/qa.py:147–158`。对照 MemTrace `membase/runners/evaluation.py:170–175,193–205,241–265`。

tracker 的 context 包含 episode、atomic facts、profile、agent case、skill 对应的 passages，并将返回的 unprocessed message 连到 context。可是示例 `_qa_record()` 的实际 prompt 只拼接 `episodes[].episode`，缺失时才退回 subject/id；没有拼接 atomic facts、profile 或 unprocessed message。QA 注入又把 tracker context 作为 answer generation 的输入。

例如：episode 摘要没有日期，atomic fact 含 `Friday`。归因图的 context 包含 Friday，但实际回答 prompt 不包含它。若回答失败，模型可能据此选择 ResponseError；真正需要检查的却是 response 数据到 QA prompt 的组装过程。

MemTrace evaluation 会记录实际用于 `qa_operator` 的 `contexts[idx]`，并保留 context builder 的来源信息。这正是 EverOS 此处需要对齐的契约。

**建议**：分别记录检索响应和实际 `answer_context`，显式记录从前者到后者的组装操作；QA 只连接实际使用的 context。unprocessed message 的正文也应与 context 的值一致。prompt 已存入图，人工能发现冲突，但这不能抵消错误的输入连边。

### R3：观察时间和磁盘终态不能恢复真实操作历史

**位置**：`tracker.py:493–547,619–645,1679–1749,1820–1857,2122–2202`；`src/everos/service/search.py:108–124`。对照 MemTrace `membase/layers/naive_rag.py:146–249` 的运行路径内插桩。

当前文件差分在业务执行之后记录状态。它捕获的是两个扫描时刻之间的终态变化，存在以下明确边界：

1. 一次扫描间隔内经历“创建正确记忆 → 错误更新”，图可能只看到错误的首次创建，无法区分 ExtractionError 与 UpdateError；若经历“创建 → 删除”，甚至没有 entry 节点。
2. 一个 memcell 没有产生 Markdown entry 时，不会进入 `_apply_entry()`，也就不会因该 entry 调用 `_record_parent()`。空抽取、全部输出被过滤等情况无法得到与成功 entry 同等粒度的输入、操作和输出。显式 flush op 已存在，但其 `outputs=[]` 和请求级状态不等于真实抽取结果。
3. 新 profile/skill 等 whole-file artifact 只记录变量，没有抽取操作及输入来源。归因不能沿构建链判断关键事实最初是否被保存。
4. 搜索完成后才 `_sync_files()`，随后 hits 优先使用 `_entries` 的磁盘快照。若索引滞后，或磁盘在搜索期间更新，hit 节点可能描述较新的磁盘内容，而 context passage 仍是本次搜索实际返回的旧文本。

**影响**：`event_seq` 可以排列记录动作，却不能还原被观察前的真实事件顺序，无法普遍保证“最早决定性故障”。文档和 prompt 已承认 file-diff 的观察性质；这是已知架构限制，但仍是达到用户目标的重要缺口。

**建议**：在 boundary/extraction/update/delete、索引投影、实际 context assembly 的必要边界补充真实事件与输入输出；至少保留操作尝试、空结果和实际读到的 revision。无法补齐时，应输出“不足以区分”的范围或 InsufficientEvidence，不能用终态推断唯一真实故障操作。

### R4：skill bridge 的阶段排序被错误地当成依赖关系

**位置**：`tracker.py:1235–1286`；`src/everos/memory/search/manager.py:302–321,546–604`。

实际计算是：

```text
agent_case 最终结果 + skill dense → bridged_dense
skill sparse + bridged_dense → skill fusion/rerank
```

tracker 按 kind 维护 `pending`。遇到 `bridged_dense` 时，将同 kind 累积的 sparse/dense 都接为输入，再将 pending 替换为 bridged_dense。因此记录成：

```text
skill sparse + skill dense → bridged_dense
bridged_dense → skill fusion
```

离线复现确认了这两组实际图边。agent_case 到 skill bridge 的跨 kind 依赖丢失；sparse 被误认为参与 dense 合并；fusion 又丢失实际的独立 sparse 输入。

**影响**：归因无法准确判断错误来自 case 检索、bridge 扩展还是 sparse/fusion，也可能把未参与该步计算的数据作为该步输入证据。

**建议**：stage record 提供显式 stage id 和 input stage ids，包含跨 kind 的依赖；不能根据列表出现顺序推断通用 DAG。验证时应检查每个 op 的输入集合与 manager 的实参一致。

### R5：共享记忆节点产生跨 query 的虚假可达路径

**位置**：`tracker.py:1411–1508,1617–1675`；`examples/memtrace/graphtrace/notebook.py:181–201`。

两个查询命中同一持久记忆版本时，都会连到同一个 entry 节点，随后该节点又连向各自的 context：

```text
Q1 → pool1 → E@1 → context1
Q2 → pool2 → E@1 → context2
```

因此同时存在 `Q2 → E@1 → context1`。离线复现中，后发生的 Q2 可以到达先前 Q1 的 context。notebook 又会展示该节点参与的全部操作，没有以当前 failed case 做请求范围或时间截断。

**影响**：追踪“沿最终输出继续向下游”时可能进入其他问题的回答链，或者把当前问题之后才发生的操作当作候选故障。此处未必产生图论意义上的环，但已经不再是严格的请求因果图。

**对照说明**：MemTrace 的 `membase/layers/naive_rag.py:307` 也存在 query 输出到稳定 memory identity 的模式，因此这不能宣称是 EverOS 独有回归。EverOS 的长期、多请求合并图使这一风险更突出；照搬主代码本身并不能解决。

**建议**：为每次检索建立请求独立的 hit/result 节点，将持久 entry 作为其来源输入或显式引用；context 使用本次 hit。归因探索同时约束 case、请求和历史截止点，保留必要的构建祖先。

### R6：占位节点不会因后续命中而补全

**位置**：`tracker.py:1395–1404,1617–1658`；smartcomment `api/_helpers.py:240–255`；MemTrace `toolkits/graph_merge_utils.py:57–78`。

当未知 candidate 首次出现，会记录 `{"id": ..., "source": "candidate_pool"}`。之后它真正出现在 search response 中，只要当前图已有版本，tracker 就设置 `identity_only=True`，复用旧节点，不使用新返回的正文。

**已复现**：先在 terminal pool 中出现 `ep_late`，再让它成为包含正文的 hit；最终 `alice_ep_late` 仍只有 candidate_pool stub。context 虽另存 response 文本，但 memory 节点和其历史链缺少这份已知证据。

此外，主仓库 merge 工具只把键集合恰好为 `{"id"}` 的快照识别为占位。带 `source` 的 stub 不满足条件，不能依赖“合并时 rich snapshot 覆盖 stub”的既有规则自动补救。

**建议**：用显式占位标记并统一合并契约；将“补全一次观察内容”与“业务记忆更新”区分开，或记录独立的请求 hit。避免永久冻结缺正文的节点，也避免把观察补全伪造成 UpdateError。

### R7：query 的碰撞处理与 QA 回连没有使用同一个稳定请求身份

**位置**：`tracker.py:261–283,1183–1204,1310–1311,2384–2454`；`qa.py:88–94`。

有两个问题：

1. **重启**：`_next_query_identity()` 在 `_update_graph()` 恢复磁盘图之前调用，此时 `_graphs.get(key)` 可能为空；`_query_seq` 又从零开始。同 request id，或未传 request id 的首个直接调用，可能复用历史 query identity。复现得到同一请求名下的 `@1`、`@2`，而不是不同请求节点。context/pool/QA identity 也由 query identity 派生，风险会扩散。
2. **同进程重试**：记录侧给重复 request id 添加 `-NNNN` 后缀，但 `resolve_query_node()` 一旦发现无后缀 base，就直接返回 base 的最新版本。正常碰撞时 base 本来就存在，suffix 分支不会被使用；新的 QA 很容易回连到第一次查询。

**影响**：把回答、参考答案或 judge 接到错误的检索上下文。现有 `query_id` 自洽检查仅检查快照字符串是否一致，不能证明绑定的是正确请求实例。

同进程复现中同时存在 `query-alice-q2` 和 `query-alice-q2-0006`，QA resolver 返回前者，确认没有定位到新一次请求。

**建议**：先恢复图，再分配请求 identity；向 QA 返回和传递确切的 query full node id 或不可重复的 execution id。对确实属于同一次请求的重试明确幂等规则；无法唯一解析时拒绝注入，不能静默选择 base。

### R8：推荐的无值预览可以使归因在下一轮直接失败

**位置**：`examples/memtrace/graphtrace/notebook.py:158–202,316–352`；`agent.py:62–90,190–201`。对照 MemTrace `graphtrace/graph_trace_agent.py:189`。

`include_variable_value=False` 会返回所有相关操作，忽略 `offset/limit`；但 `trim_history()` 不能裁剪最新工具组，也不能裁剪 fresh hint。大图一次预览就可能超过默认 `context_chars=100000`，下一轮调用模型前直接返回 ContextBudgetError。

**已复现**：用虚构图给同一 message 增加 180 个操作，无值预览长度为 **215149 字符**。脚本模型只调用一次预览，归因便在第 2 轮准备阶段失败：`iterations=1`、`failure.type=ContextBudgetError`。模型没有机会遵循报错建议缩小工具页；这个模式本身还忽略页大小。

另外，hint 包含所有已完成节点的 outcome，随探索持续增长，即使旧对话已删除，保护部分仍可能溢出。主实现有 token 计数和摘要压缩，当前移植的直接裁剪不是等价替换。

**建议**：无值预览也按操作分页并遵守预算；对历史 outcome 做有界摘要或按需查询；单个超大工具结果应给可恢复的分页提示。保留工具请求/响应配对的现有约束。

### R9：失败 add 的错误记录没有保留导致错误的本次消息

**位置**：`tracker.py:549–617`；`src/everos/service/memorize.py:236–250`。

成功路径先 `_record_messages()`，失败路径却先扫描文件，然后把非 flush 错误记录到 `(app, project, "unknown", session)`。失败 hook 不调用 `_record_messages()`，通常在 unknown owner 图里也找不到输入 buffer，最终只是一个没有本次源消息的错误操作。

**触发**：首次 add 在 ingest 之后、抽取或写入过程中抛错，随后用户对本应被保存的事实提问。即使 payload 中有关键消息，归因侧也无法从 message 起点走到该失败操作；部分 Markdown 写入时还可能出现 source stub 与 unknown error op 分离。

离线向错误 hook 传入含一条虚构消息的 payload 后，失败请求图中的 `category=message` 节点数量为 **0**。

**对照**：主代码内联记录在执行过程中建立 message、buffer 和中间输出。它也不保证所有异常都完整可归因，但当前 EverOS 的错误 hook 明确拥有 payload 却没有保存它。

**建议**：将本次输入作为“提交/尝试处理的消息”记录，并与正确 owner/session 的失败请求操作相连；区分提交、接受、成功落盘，避免把失败请求误记成成功构建。新增回归样例应检查输入可达性，不能只检查错误 op 存在。

### R10：结果通过校验，不代表满足已声明的归因约束

**位置**：`examples/memtrace/graphtrace/agent.py:228–246`、`notebook.py:201`、`prompts.py:127–136`。

目前校验包括 schema、op 存在、探索已结束、非 insufficient evidence、op 标题出现在工具输出。但没有排除 judge/reference injection，也不要求 reason 包含节点证据；“inspected”来自对返回文本中标题的正则匹配，标题出现不代表读到了该操作的输入输出值。

**已复现**：查看 query 的无值预览后，结束探索，提交：

```json
{"error_type":"DeletionError","op_id":"judge-op","reason":"Test guard"}
```

结果为 `status=success`。这违反本地 prompt 中“不要把 judge/reference injection 当成 memory-system fault”的要求。这里只证明校验接受该结果，不声称真实模型必然如此回答。

**对照说明**：主实现的结构化输出也不是诊断正确性证明；EverOS 新增的 inspected guard 是改进，但仍未覆盖可机械验证的限制。不要简单禁止所有 `category=evaluation`：主实现的 `question-answering` 操作也属于 evaluation。

**建议**：按操作角色明确排除 judge/reference injection，保留 native answer generation 兼容；记录结构化的已展示 operation ids 和已读取证据，而不是从正文识别标题；输出区分 schema-valid 与经过证据规则检查。诊断语义仍需人工标签或受控故障实验评估。

## 4. 其他需要保留的边界

以下不另列为主问题，避免把设计取舍全部当作新增 bug：

- **扫描重试不是完整事务**：`tracker.py:1744` 在 graph apply/export 成功之前更新 `_file_state`；`1821–1822` 也提前写 entry cache。若后续构图失败，下一次未变动文件可能直接跳过。解析异常已支持重试，但这不覆盖 apply 异常。建议后续以失败注入测试验证提交/回滚边界。
- **不完整候选集**：pool cap 为 50，已有 `total_count/truncated` 标记；不能据此把“未出现”解释成“从未召回”。中间 pool 只保留 id/score/source，只有 terminal pool 展开 candidate，无法普遍仅凭图恢复中途被过滤记忆的当时正文。
- **多进程写图**：当前 portalocker 能防止半文件，不能合并各 worker 的内存图。单 writer/独立输出目录是现有明确限制，不应写成“当前没有文件锁”。
- **QA 注入需按既有流程停服执行**：运行中的 tracker 保留内存图；离线注入后的边并不会自动进入它的内存。重复注入也会增加 judge 操作，收集案例时可能出现同 query 多次评估；应明确注入幂等键和评估版本。
- **没有伪造主实现等价性**：只支持 BM25、单案例 CLI、没有 OBS 和缓存，是 README 已说明的范围差异。是否需要移植取决于实验目标，不能单凭缺少这些功能判定核心算法错误。

## 5. 验证结果与覆盖空白

现有测试命令（EverOS 根目录）：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  tests/unit/test_memtrace_example \
  tests/unit/test_infra/test_memtrace_tracker.py \
  tests/unit/test_infra/test_memtrace_qa.py \
  -q -p no:cacheprovider \
  --basetemp=/tmp/everos-attribution-review-pytest-escalated
```

结果：**94 passed in 8.29s**。最初沙箱内运行出现 asyncio 线程收尾卡住；允许在沙箱外执行后通过，未将该环境现象计为项目 bug。

额外的临时离线脚本使用真实 tracker/graphtrace API、现有虚构图 fixture 和 scripted provider；输出图只写入 `/tmp/everos-review-*`。没有调用线上 LLM、没有读取凭据内容、没有启动或修改 EverOS 服务、没有改写已有实验图。关键观测如下：

| 验证 | 观测 |
| --- | --- |
| 写入 v2 后首次跨图搜索 | 写入图 v1=Friday、v2=Monday；搜索图 v1=Monday；merge 报同 id 冲突 |
| 两次查询复用同一 entry | Q2 到 Q1 context 的有向可达性为 True |
| skill 阶段依赖 | bridge 输入为 skill sparse+dense；fusion 输入只有 bridge |
| stub 后续命中 | 节点仍为 `id + source=candidate_pool` |
| 重启后重复 request id | 同 query identity 出现 @1/@2 |
| 同进程重复 request id | 新 query 有后缀，QA resolver 仍选择无后缀的旧 query |
| 首次 add 失败 | 已传入消息，但失败请求图的 message 数为 0 |
| 大图无值预览 | 215149 字符；仅完成一次模型请求便因上下文预算停止 |
| 错误提交 judge-op | 以 DeletionError 提交仍被接受为 success |

这些样例验证的是图和运行时契约，**不是归因准确率评测**。94 个测试通过与上述问题存在并不矛盾：当前 scripted end-to-end 测试预先指定探索路线，主要证明工具调用闭环，不能证明真实 tracker 生成的跨图、跨请求版本和依赖正确。

建议补充的验收顺序：

1. **先验证图事实**：单租户及多租户 identity、更新后的跨图命中、真实 QA context、bridge 输入集合、失败 add 的源消息、重启后的精确 query 回连。
2. **再验证探索能力**：大图分页、历史摘要、当前 case 的范围限制、非法 judge 归因的拒绝。
3. **最后比较归因结果**：构建 Extraction/Update/Deletion/Retrieval/Response 五类受控故障，在同一份语义正确的图、同一模型、同一 sparse 起点条件下比较 EverOS 和主实现；另行比较 hybrid 起点效果。分别统计归因正确、证据不足、图不完整和运行失败，不能把这些结果混为一个准确率。

当前更有价值的下一步是先修正图的事实与连接，再扩充检索策略或模型能力。否则更强的归因模型仍可能依据错误的图得出看似合理的结论。

## 6. 修改实施状态（2026-10-11）

本次修改针对图事实、连接和可机械验证的归因约束，不宣称达到 MemTrace 主实现的全部功能或归因准确率。

| 编号 | 实施结果 |
| --- | --- |
| R1 | 搜索图引用构建图的原始 revision，保留 `@2` 等版本号；新增 `examples/memtrace/merge.py`，按 app/project 隔离节点并拒绝内容冲突；CLI 拒绝带未解决冲突标记的旧合并图。 |
| R2 | 新增实际 `answer_context` 及组装操作；LoCoMo 驱动传入真正使用的 episode context；回答只消费该节点。未记录的 context 保持 unknown，不能据此提交 ResponseError。 |
| R3 | boundary cell、episode extractor 尝试/结果、持久化关联改为采集真实事件；空结果可见。持久化关联只在正文与实际抽取输出一致时成立，后续磁盘改写不能冒充 extractor 输出。hit 保留搜索响应正文，无法匹配历史内容时 revision 明确未解决。其余文件差分标为 `observation_only`，提交时拒绝把观察动作当作执行故障。**背景 OME 的所有中间更新/删除尚未实现逐步事件采集，证据不足时必须停止，不能从终态推断唯一故障。** |
| R4 | stage 增加显式 id 和 input ids；真实 LLM bridge 路径验证 case fusion + skill dense，随后 sparse + bridged dense。普通路径不虚构 case 输入。 |
| R5 | 检索 hit 按请求独立，context 消费本次 hit；探索和起始证据选择限制租户、当前请求和历史截止点。 |
| R6 | candidate 使用请求内的显式占位观察，后续请求 hit 独立保存正文，不再冻结持久记忆节点或伪造业务更新。 |
| R7 | 恢复图后再分配 query id；重复 request id 默认拒绝歧义，QA 支持确切 `query_full_node_id`；离线注入也检查跨文件/租户唯一性。 |
| R8 | 无值预览、value、history 均分页；工具页随预算缩小；跨页累计真实 operation 阅读范围；hint 仅保留有界摘要，完整历史可按需查询。 |
| R9 | 失败 add 的本次消息保存为 attempted，与各参与 owner/session 的失败操作相连，不伪装为接受成功。 |
| R10 | 检查实际已展示的操作和已读节点，reason 必须引用相关 full node id；明确拒绝 judge、reference injection、其他请求/未来操作及纯文件观察归因，保留 native QA 角色兼容。 |

额外修复：相同 QA record 用摘要标识实现幂等（包括仅 answer、仅 golden）；构图或导出异常会恢复扫描指纹，未变化文件仍可重试，entry/artifact/deletion cache 在成功后提交。这不是整个图的强事务回滚，仍需区分局部记录异常与完整业务执行历史。

运行顺序：重新采集 → 停止服务 → 注入 QA → 用新增 safe merger 合并构建图和搜索图 → 列出并执行归因。旧图中缺失的输入、正文、版本历史和背景中间状态无法通过事后合并恢复，不应覆盖旧实验结果或猜测修复版本。

新增验证均使用虚构数据和 scripted provider，没有运行线上 LLM、修改真实实验数据或启动服务。最终受影响测试 **483 passed in 14.71s**：

```bash
.venv/bin/python -m pytest -q \
  tests/unit/test_infra/test_memtrace_tracker.py \
  tests/unit/test_infra/test_memtrace_qa.py \
  tests/unit/test_infra/test_memtrace_regressions.py \
  tests/unit/test_memtrace_example tests/unit/test_service \
  tests/unit/test_memory/test_search tests/unit/test_memory/test_extract \
  tests/unit/test_entrypoints/test_api/test_lifespans/test_memtrace.py
```

其他检查结果：

- `ruff check src tests examples/memtrace`、`ruff format --check`：通过，673 个文件格式一致。
- `lint-imports`：3 个依赖契约全部通过；datetime discipline、OpenAPI drift、`git diff --check`：通过。
- 新增 safe merge CLI → 离线 `--list-cases`：通过，无凭据及外部模型请求。
- `scripts/check_docs.py`：未通过，错误全部位于已有、gitignored 的 `smartcomment_docs/` 本地镜像，其中 README/code reading guide 的相对源码链接不存在；本次修改文档没有新增该检查报告的失效链接。

这些验证证明图契约和工具行为，不代表真实模型在 LoCoMo 或五类受控故障上的归因准确率。完整项目测试、真实 LLM 对照和背景 OME 全事件历史不在本次验证结果内。
