# LiveCodeBench Domain Adapter

[LiveCodeBench](https://livecodebench.github.io/) 竞赛编程评测适配器。无需 Docker，Agent 直接解题，使用 LCB 官方测试用例验证。

## 评测流程

1. **构建 Prompt** — 按 LCB 官方格式生成题目描述 + 格式要求（`prompt.md`）
2. **Agent 解题** — Agent 通过 exec 工具编写和调试代码，输出 ```python``` 代码块
3. **代码提取** — 从 Agent 响应中提取代码（支持 openclaw JSON payloads / nanobot 纯文本），失败时回退到 session.jsonl
4. **验证评分** — 使用 LCB `check_correctness` 跑全部测试用例，全部通过 reward=1.0

## 前置依赖

```bash
# 克隆 LiveCodeBench（用于导入 evaluation 模块）
git clone https://github.com/LiveCodeBench/LiveCodeBench.git

# 安装依赖
pip install datasets
```

## 配置

Domain 配置（`livecode.yaml`）：

```yaml
lcb_repo: ./LiveCodeBench
release_version: release_v6
cache_dir: ./data/livecodebench
split_file: ./data/livecodebench/split.json
agent_timeout: 900
test_timeout: 6
```

运行配置（`config.yaml`）：

```yaml
agent:
  name: openclaw
domain:
  name: livecode
  config: ./src/domains/livecode/livecode.yaml
job_dir: ./livecodebench/jobs
trials: 1
parallel: 8
```

## Split 文件格式

支持两种格式：

**统一 cluster 格式**（与 `src/skill_evolution/evermemos` 一致）：

```json
{"clusters": {"CLUSTER_A": {"train": ["id1", "id2"], "test": ["id3"]}, ...}}
```

**扁平格式**：

```json
{"train": ["id1", "id2", ...], "test": ["id3", ...]}
```

## 运行

```bash
# 跑 test 集
python -u run.py --config config.yaml --split test --parallel 8 --job baseline

# 跑 train 集
python -u run.py --config config.yaml --split train --parallel 8 --job train-run

# 指定单题调试
python -u run.py --config config.yaml --task 2757 --job debug

# 按难度筛选
python -u run.py --config config.yaml --split easy --job easy-run
```

## Skill 注入

Skill 注入通过框架统一入口 `src/skill_evolution/evermemos/eval_with_skills.py` 实现，不在 domain 内部处理。详见 `src/skill_evolution/evermemos/README.md`。

## 结果

每个任务保存在 `job_dir/{task_id}__trial_1/`：

```
result.json          # agent_result + verifier_result
verifier/
  details.json       # 代码、来源、逐用例结果
  solution.py        # 提取的代码
session.jsonl        # Agent 会话记录
```

`verifier_result.reward` = 1.0 全部通过，0.0 未通过。
