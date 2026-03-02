# 两阶段记忆提取（Two-Phase Memory Extraction）

[Home](../../README.md) > [Docs](../README.md) > [Advanced](README.md) > Two-Phase Extraction

---

## 背景：问题与动机

EverMemOS 默认通过 LLM **边界检测**决定何时将对话片段切分为 MemCell，再触发 episodic memory 提取。这种方式能保证 episode LLM 看到完整的多轮上下文，生成的叙事质量高。

但存在一个痛点：**如果对话中途结束，边界从未触发，积压在 Redis 队列里的消息会在 60 分钟 TTL 过期后永久丢失**，一条记忆都不会生成。

常见的应对方式是设置 `DISABLE_BOUNDARY_DETECTION=true`，让每条消息立即生成 MemCell——但这样 episode LLM 每次只看到 1–2 条孤立消息，叙事质量明显下降。

**两阶段提取**解决了这一张力：

```
实时阶段（每条消息后，毫秒级）
  MemCell 立即生成 ──→ 只提取 event_log（原子事实）
                         原子事实不依赖多轮上下文，单条消息已足够

批处理阶段（后台 Worker，默认 15 分钟）
  BatchEpisodeWorker 扫描同一 group 的多个 MemCell
       └─ 合并为完整多轮对话上下文
       └─ 调用 episode LLM ──→ 生成高质量叙事记忆
```

---

## 快速上手

在 EverMemOS 的 `.env` 中添加以下两行（**缺一不可**）：

```bash
# 1. 每条消息立即生成 MemCell，不等 LLM 判断边界
DISABLE_BOUNDARY_DETECTION=true

# 2. 实时阶段只提取 event_log，episode 交给批处理 Worker
REALTIME_EVENT_LOG_ONLY=true
```

重启 EverMemOS，启动日志出现以下内容即表示生效：

```
✅ BatchEpisodeWorker started
```

---

## 参数配置

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DISABLE_BOUNDARY_DETECTION` | `false` | 设为 `true` 跳过 LLM 边界检测，每条消息立即生成 MemCell |
| `REALTIME_EVENT_LOG_ONLY` | `false` | 设为 `true` 开启两阶段模式 |
| `BATCH_EPISODE_INTERVAL` | `15` | Worker 运行间隔（分钟） |
| `BATCH_EPISODE_LOOKBACK` | `120` | 每次向前扫描多少分钟内的 MemCell |
| `BATCH_EPISODE_MIN_CELLS` | `3` | 一个 group 至少积累几个 MemCell 才触发 episode 生成 |

### 调优建议

| 场景 | 建议 |
|---|---|
| 对话节奏慢（每天只聊几条） | `BATCH_EPISODE_MIN_CELLS=2` |
| 希望 episode 尽快出现 | `BATCH_EPISODE_INTERVAL=5` |
| 限制 Worker 扫描量 | `BATCH_EPISODE_LOOKBACK=60` |
| 使用上下文窗口较小的 LLM | 同时降低 `BATCH_EPISODE_MIN_CELLS` 和 `BATCH_EPISODE_LOOKBACK`，减少合并消息数 |

---

## 工作流程详解

### 实时阶段

```
用户发送消息
  └─ CountBot (或其他客户端) POST /api/v1/memories
        └─ MemorizeRequest → preprocess_conv_request
              └─ ConvMemCellExtractor
                    DISABLE_BOUNDARY_DETECTION=true
                    → should_end=True（立即）
                    → 生成 MemCell（含本条消息 original_data）
                          └─ process_memory_extraction
                                REALTIME_EVENT_LOG_ONLY=true
                                → 仅调用 EventLogExtractor
                                → event_log 以 parent_type="memcell" 写入数据库
                                → 跳过 EpisodeMemoryExtractor / ForesightExtractor
                                → 更新会话状态
```

### 批处理阶段

```
BatchEpisodeWorker.run_once()（每 BATCH_EPISODE_INTERVAL 分钟）
  └─ 查询最近 BATCH_EPISODE_LOOKBACK 分钟内所有 MemCell
        └─ 过滤 extend.batch_ep_done=true（已处理）
        └─ 按 group_id 分组
              每组 ≥ BATCH_EPISODE_MIN_CELLS
              └─ 合并所有 original_data（按时间升序）
              └─ 构建合并 MemCell（完整上下文）
              └─ EpisodeMemoryExtractor → group episode + 每个用户 episode
              └─ 保存 episode 到数据库（MongoDB + ES + Milvus）
              └─ 标记每个 MemCell: extend.batch_ep_done=true
```

---

## 与默认模式的对比

| | 默认模式 | `DISABLE_BOUNDARY_DETECTION` | 两阶段模式（本功能） |
|---|---|---|---|
| 数据丢失风险 | ⚠️ 对话中断后 60 分钟丢失 | ✅ 无 | ✅ 无 |
| event_log 质量 | ✅ 高 | ✅ 高 | ✅ 高 |
| episode 质量 | ✅ 高 | ❌ 低（上下文不足） | ✅ 高 |
| episode 生成延迟 | 实时 | 实时 | ≤ BATCH_EPISODE_INTERVAL 分钟 |
| LLM 调用次数 | 边界检测 + 提取 | 仅提取 | 仅提取（分离时机） |

---

## 已知限制

- **Clustering 未在批处理阶段触发**：BatchEpisodeWorker 保存 episode 后不会调用 `_trigger_clustering`。Clustering 主要影响**用户画像（Profile）提取**，若你的场景高度依赖 Profile，建议暂时仍使用默认模式或手动触发 clustering。

- **`BATCH_EPISODE_MIN_CELLS` 的影响**：若用户在 lookback 窗口内的 MemCell 数量始终低于该阈值（如新用户只发了 1 条消息），该周期内不会生成 episode。可通过降低 `BATCH_EPISODE_MIN_CELLS=1` 解决（但会降低 episode 上下文丰富度）。
