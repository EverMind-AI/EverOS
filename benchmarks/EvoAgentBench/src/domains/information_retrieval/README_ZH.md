# BrowseComp-Plus Domain

[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) 评测。Agent 通过 MCP 搜索本地 FAISS 语料库回答问题，LLM Judge 验证正确性。

## 快速开始

### 1. 环境

```bash
conda env create -f environment.yml
conda activate evoagentbench
```

### 2. 数据准备

```bash
python src/utils/browsecomp-plus-tools/setup_data.py
```

自动完成：
- 从 HuggingFace 下载并解密 `Tevatron/browsecomp-plus` 数据集
- 下载预构建 FAISS 索引（默认 `qwen3-embedding-8b`）

数据存储在 `data/BrowseComp-Plus/`：

```
data/BrowseComp-Plus/
├── browsecomp_plus_decrypted.jsonl    # 解密后的数据集（830 queries）
├── queries.tsv                         # query_id → query
├── task_split.json                     # 12 clusters, train/test 划分
└── indexes/
    └── qwen3-embedding-8b/
        └── corpus.shard{1..4}_of_4.pkl
```

可选参数：

```bash
python setup_data.py --index qwen3-embedding-0.6b   # 不同 embedding 索引
python setup_data.py --skip-index                     # 只下数据不下索引
```

### 3. 配置

`config.yaml`：

```yaml
agent:
  name: nanobot          # 或 openclaw
  command: nanobot

domain:
  name: browsecomp_plus
  config: ./src/domains/browsecomp_plus/browsecomp_plus.yaml

job_dir: ./jobs
trials: 1
parallel: 4
```

`browsecomp_plus.yaml` 中需要关注的配置：

| 字段 | 说明 |
|------|------|
| `dataset_file` | 解密后的 JSONL 路径 |
| `split_file` | train/test 划分文件 |
| `mcp_server.index_path` | FAISS 索引 glob pattern |
| `mcp_server.model_name` | Embedding 模型路径（需与索引匹配） |
| `judge.model` | LLM Judge 模型名 |
| `judge.api_base` | LLM Judge API 地址 |

### 4. 运行

```bash
# 跑指定 task
python src/run.py --domain browsecomp_plus --task 784

# 跑指定 split
python src/run.py --domain browsecomp_plus --split test --parallel 4

# 跑指定 cluster
python src/run.py --domain browsecomp_plus --split ACTOR_INDIAN_test --parallel 8

# 跑全部
python src/run.py --domain browsecomp_plus --split all --parallel 8
```

### 5. 结果

每个 task 的输出在 `jobs/{job_name}/{qid}__trial_1/`：

| 文件 | 内容 |
|------|------|
| `result.json` | 完整结果（agent 回答、验证结果、耗时等） |
| `session.jsonl` | Agent session 轨迹 |
| `verifier/details.json` | LLM Judge 判定详情 |

汇总在 `jobs/{job_name}/summary.json`。

## 数据划分

`task_split.json` 包含 12 个 cluster，每个 cluster 有 train/test 划分：

| Split 名称 | 说明 |
|------------|------|
| `train` | 所有 cluster 的训练集合并（134 cases） |
| `test` | 所有 cluster 的测试集合并（117 cases） |
| `all` | train + test（251 cases） |
| `{CLUSTER}_train` | 单个 cluster 训练集，如 `ACTOR_INDIAN_train` |
| `{CLUSTER}_test` | 单个 cluster 测试集，如 `ACTOR_INDIAN_test` |
| 数字 | 前 N 条，如 `--split 10` |

## Agent 支持

| Agent | MCP 配置方式 | 模型配置 |
|-------|-------------|---------|
| **nanobot** | 临时 workspace + `--config`/`--workspace` 参数 | `config.yaml` 的 `agent.model`/`agent.provider` |
| **openclaw** | `mcporter config add` 自动注册 | openclaw 自身配置 |

## MCP 搜索服务

评测开始时自动启动，结束时自动关闭（`mcp_server.auto_start: true`）。

手动管理：

```bash
# 手动启动（读取 browsecomp_plus.yaml 配置）
python src/utils/browsecomp-plus-tools/start_mcp.py

# 关闭后在 yaml 中设置 auto_start: false
```

## 搜索服务代码

`src/utils/browsecomp-plus-tools/searcher/` 中的代码来自 [texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus)，做了以下适配：

- Embedding 模型使用 `device_map="auto"` 分散到多 GPU（原版单卡加载）
- FAISS index 保留在 CPU（避免与 LLM 推理服务争抢 GPU 显存）
