# EverMemOS Skill Evaluation

使用 EverMemOS 从 agent session 中自动提取 skills，注入到测试集评测中，验证 skills 对准确率的提升效果。

支持所有已注册的 domain，通过 `--domain` 参数指定。

## Split File 格式

所有 domain 必须提供统一格式的 split file：

```json
{
  "clusters": {
    "FOOTBALL": {"train": ["id1", "id2"], "test": ["id3", "id4"]},
    "BAND": {"train": [...], "test": [...]},
    ...
  }
}
```

没有天然分簇的 domain，用一个 `"default"` 簇：

```json
{
  "clusters": {
    "default": {"train": ["task1", "task2"], "test": ["task3", "task4"]}
  }
}
```

## 评估流程

```
Step 1  跑训练集，收集 agent session
Step 2  用 EverMemOS 按簇提取 skills
Step 3  跑测试集 baseline（无 skill）
Step 4  跑测试集（注入 skill）
Step 5  对比结果
```

## 前置条件

- 已完成对应 domain 的环境安装和数据准备
- 对应 domain 的 split file 已准备好（统一格式）
- EverMemOS 服务已部署并可访问

## Step 1: 跑训练集

```bash
python src/run.py --domain <name> --split train --parallel 8
```

结果保存在 `jobs/{agent}-{domain}-{timestamp}/`，每个 task 目录下包含 `session.jsonl`。

## Step 2: 提取 Skills

启动 EverMemOS API server：

```bash
cd evermemos-opensource
uv run python src/run.py --port 1996
```

运行提取脚本：

```bash
python src/skill_evolution/evermemos/extract_skills.py \
    --domain <name> \
    --job-dir jobs/{agent}-{domain}-{timestamp} \
    --api-url http://localhost:1996
```

参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--job-dir` | Step 1 的 job 输出目录 | **必填** |
| `--domain` | Domain 名称 | config.yaml 中的值 |
| `--split-file` | Split file 路径 | domain config 中的值 |
| `--output-dir` | skill 输出目录 | `src/skill_evolution/evermemos/skills` |
| `--api-url` | EverMemOS API 地址 | `http://localhost:1995` |
| `--clusters` | 只处理指定簇（空格分隔） | 全部 |
| `--split` | 使用哪个 split | `train` |

提取结果保存为 `{cluster}/{skill_name}/SKILL.md` 格式：

```
src/skill_evolution/evermemos/skills/
└── FOOTBALL/
    ├── skill_name_1/SKILL.md
    └── skill_name_2/SKILL.md
```

支持 nanobot 和 openclaw 两种 session 格式。

## Step 3: 跑测试集 Baseline

```bash
python src/run.py --domain <name> --split test --parallel 8
```

## Step 4: 跑测试集（注入 Skill）

每个簇提取的 skills 自动分配给该簇的所有 test cases：

```bash
python src/skill_evolution/evermemos/eval_with_skills.py \
    --domain <name> \
    --skills-dir src/skill_evolution/evermemos/skills \
    --parallel 8
```

参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--domain` | Domain 名称 | config.yaml 中的值 |
| `--split-file` | Split file 路径 | domain config 中的值 |
| `--skills-dir` | skill 文件目录 | `src/skill_evolution/evermemos/skills` |
| `--clusters` | 只测指定簇 | 全部 |
| `--split` | 使用哪个 split | `test` |
| `--parallel` | 并行数 | `1` |
| `--job` | Job 名称 | `evermemos-with-skills-{domain}` |

Skills 注入方式：在 domain 原始 prompt 末尾追加 `## Domain-Specific Strategies` 段落。

## Step 5: 对比结果

```bash
cat jobs/{agent}-{domain}-{timestamp}/summary.json | python -m json.tool
cat jobs/evermemos-with-skills-{domain}/summary.json | python -m json.tool
```

## 参考数据（BrowseComp-Plus）

历史实验中 skills 对各 cluster 的增益（Qwen3.5-397B, FAISS qwen3-embedding-8b）：

| Cluster | Baseline | With Skills | Delta |
|---------|----------|-------------|-------|
| MOVIE_INDIAN | 0/5 | 5/5 | +5 |
| ACTOR_INDIAN | 0/11 | 7/11 | +7 |
| BAND | 0/11 | 5/11 | +5 |
| FOOTBALL | 1/11 | 5/11 | +4 |
| GEO_LANDMARK | 0/15 | 4/15 | +4 |
| **Total** | **4/117 (3.4%)** | **40/117 (34.2%)** | **+36** |

## 文件说明

| 文件 | 说明 |
|------|------|
| `extract_skills.py` | 从 train session 提取 skills（调 EverMemOS API） |
| `eval_with_skills.py` | 按簇自动分配 skills，注入 prompt 跑 test split |
| `skills/` | 提取的 skill 文件，按 `{cluster}/{skill_name}/SKILL.md` 组织 |
| `skills_sample/` | 手写 skill 样例（供参考） |
