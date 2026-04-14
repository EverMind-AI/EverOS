# BrowseComp-Plus Domain

[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) evaluation. Agent searches a local FAISS corpus via MCP and answers questions, verified by LLM Judge.

## Quick Start

### 1. Environment

```bash
conda env create -f environment.yml
conda activate evoagentbench
```

### 2. Data Preparation

```bash
python src/utils/browsecomp-plus-tools/setup_data.py
```

This will:
- Download and decrypt `Tevatron/browsecomp-plus` dataset from HuggingFace
- Download pre-built FAISS index (default `qwen3-embedding-8b`)

Data stored in `data/BrowseComp-Plus/`:

```
data/BrowseComp-Plus/
├── browsecomp_plus_decrypted.jsonl    # Decrypted dataset (830 queries)
├── queries.tsv                         # query_id → query
├── task_split.json                     # 12 clusters, train/test split
└── indexes/
    └── qwen3-embedding-8b/
        └── corpus.shard{1..4}_of_4.pkl
```

Options:

```bash
python setup_data.py --index qwen3-embedding-0.6b   # Different embedding index
python setup_data.py --skip-index                     # Dataset only, no index
```

### 3. Configuration

`config.yaml`:

```yaml
agent:
  name: nanobot          # or openclaw
  command: nanobot

domain:
  name: browsecomp_plus
  config: ./src/domains/browsecomp_plus/browsecomp_plus.yaml

job_dir: ./jobs
trials: 1
parallel: 4
```

Key fields in `browsecomp_plus.yaml`:

| Field | Description |
|-------|-------------|
| `dataset_file` | Path to decrypted JSONL |
| `split_file` | Train/test split file |
| `mcp_server.index_path` | FAISS index glob pattern |
| `mcp_server.model_name` | Embedding model path (must match index) |
| `judge.model` | LLM Judge model name |
| `judge.api_base` | LLM Judge API endpoint |

### 4. Run

```bash
# Single task
python src/run.py --domain browsecomp_plus --task 784

# By split
python src/run.py --domain browsecomp_plus --split test --parallel 4

# By cluster
python src/run.py --domain browsecomp_plus --split ACTOR_INDIAN_test --parallel 8

# All
python src/run.py --domain browsecomp_plus --split all --parallel 8
```

### 5. Output

Each task output in `jobs/{job_name}/{qid}__trial_1/`:

| File | Content |
|------|---------|
| `result.json` | Full result (agent response, verification, timing) |
| `session.jsonl` | Agent session trajectory |
| `verifier/details.json` | LLM Judge verdict details |

Summary in `jobs/{job_name}/summary.json`.

## Data Splits

`task_split.json` contains 12 clusters, each with train/test split:

| Split Name | Description |
|------------|-------------|
| `train` | All clusters' train sets merged (134 cases) |
| `test` | All clusters' test sets merged (117 cases) |
| `all` | train + test (251 cases) |
| `{CLUSTER}_train` | Single cluster train, e.g. `ACTOR_INDIAN_train` |
| `{CLUSTER}_test` | Single cluster test, e.g. `ACTOR_INDIAN_test` |
| number | First N items, e.g. `--split 10` |

## Agent Support

| Agent | MCP Configuration | Model Configuration |
|-------|------------------|-------------------|
| **nanobot** | Temp workspace + `--config`/`--workspace` flags | `config.yaml` `agent.model`/`agent.provider` |
| **openclaw** | Auto-register via `mcporter config add` | openclaw's own config |

## MCP Search Server

Auto-starts at evaluation begin, auto-stops at end (`mcp_server.auto_start: true`).

Manual management:

```bash
# Start manually (reads browsecomp_plus.yaml)
python src/utils/browsecomp-plus-tools/start_mcp.py

# Set auto_start: false in yaml to disable
```

## Searcher Code

Code in `src/utils/browsecomp-plus-tools/searcher/` is from [texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus), with these adaptations:

- Embedding model uses `device_map="auto"` for multi-GPU distribution (original loads on single GPU)
- FAISS index kept on CPU (avoids GPU memory contention with LLM inference services)
