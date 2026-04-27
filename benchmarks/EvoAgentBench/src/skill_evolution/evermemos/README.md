# EverCore Skill Evaluation

Extract reusable skills from agent training sessions via EverCore, inject them into test evaluation, and measure accuracy improvement.

Works with any domain. Adding a new domain requires one line in `domain_info.py`.

## Architecture

```
domain_info.py     # Registry: per-domain query field + injection strategy
extract_skills.py     # Send sessions → EverCore v1 API → save SKILL.md files
eval_with_skills.py   # Load skills (API search / static files / cache) → inject → run
```

## Split File Formats

Two formats supported (auto-detected):

**Cluster format** (browsecomp_plus, swebench, etc.):
```json
{"clusters": {"FOOTBALL": {"train": ["id1", "id2"], "test": ["id3"]}, ...}}
```

**Flat format** (omnimath, swebench, etc.):
```json
{"train": ["id1", "id2"], "test": ["id3", "id4"]}
```

Domains without a split file (gdpval, officeqa) are loaded directly from the domain adapter — no split file needed.

## Workflow

```
Step 1  Run train split → collect agent sessions
Step 2  Extract skills from sessions via EverCore
Step 3  Evaluate test split with skills injected
Step 4  Compare against baseline (test without skills)
```

## Prerequisites

- Domain data prepared (see each domain's README)
- EverCore v1 API server running (default: `http://localhost:1997`)

## Step 1: Run Train Split

```bash
python src/run.py --domain reasoning --split train --parallel 8
```

## Step 2: Extract Skills

```bash
python src/skill_evolution/evermemos/extract_skills.py \
    --domain reasoning \
    --job-dir jobs/{job-name} \
    --api-url http://localhost:1997
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--job-dir` | Step 1 job output directory | **required** |
| `--domain` | Domain name | from config.yaml |
| `--config` | Path to config.yaml | `config.yaml` |
| `--split-file` | Path to split file | from domain config |
| `--output-dir` | Skill output directory | `src/skill_evolution/evermemos/skills` |
| `--api-url` | EverCore v1 API address | `http://localhost:1997` |
| `--clusters` | Process only specified clusters | all |
| `--split` | Which split to extract from | `train` |
| `--parallel` | Concurrent session sends | `16` |
| `--poll-interval` | Seconds between skill stability polls | `120` |
| `--feedback` | Append task evaluation feedback to sessions | off |
| `--success-only` | Only send sessions with reward > 0 | off |

### Session Normalization

Both nanobot and openclaw session formats are supported. Reasoning/thinking content is handled automatically:

- **Multi-turn sessions** (has tool calls): thinking is dropped — it's about tool selection, not domain knowledge.
- **Single-turn sessions** (no tools, e.g. omnimath): thinking is the core content, chunked into 1000-char simulated tool_call/tool_result pairs for EverCore extraction.

### Output

Skills saved as SKILL.md files, grouped by EverCore cluster:

```
src/skill_evolution/evermemos/skills/
├── metadata_{user_id}.json
├── cluster_000/
│   ├── skill_name_1/SKILL.md
│   └── skill_name_2/SKILL.md
└── cluster_001/
    └── .../SKILL.md
```

## Step 3: Evaluate with Skills

Three skill sources (mutually exclusive):

### API Search (recommended)

Query EverCore per task — dynamically finds relevant skills:

```bash
python src/skill_evolution/evermemos/eval_with_skills.py \
    --domain reasoning \
    --api-url http://localhost:1997 \
    --user-id extract_abc123 \
    --top-k 2 \
    --parallel 8
```

### Static Files

Load pre-extracted SKILL.md files from disk:

```bash
python src/skill_evolution/evermemos/eval_with_skills.py \
    --domain information_retrieval \
    --skills-dir src/skill_evolution/evermemos/skills \
    --parallel 8
```

GLOBAL/ skills (if present) are automatically prepended to all tasks.

### Skill Cache

Reuse previously searched skills (reproducibility):

```bash
python src/skill_evolution/evermemos/eval_with_skills.py \
    --domain reasoning \
    --skill-cache jobs/prev-run/skill_cache.json \
    --parallel 8
```

A `skill_cache.json` is saved in the job directory after every run.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--domain` | Domain name | from config.yaml |
| `--config` | Path to config.yaml | `config.yaml` |
| `--api-url` | EverCore API (enables search mode) | `http://localhost:1997` |
| `--user-id` | EverCore user_id (from extract output) | auto-detected |
| `--top-k` | Max skills per task (search mode) | `5` |
| `--search-method` | vector / hybrid | `vector` |
| `--skills-dir` | Skills directory (static mode) | `src/skill_evolution/evermemos/skills` |
| `--skill-cache` | Path to skill_cache.json (cache mode) | — |
| `--task` | Specific task(s), comma-separated | all |
| `--clusters` | Only these clusters | all |
| `--split` | Which split to evaluate | `test` |
| `--parallel` | Parallelism | `4` |
| `--job` | Job name | auto-generated |

### Injection Strategy

Determined automatically per domain via `domain_info.py`:

| Domain | Strategy | How |
|-----------|----------|-----|
| browsecomp_plus, swebench, gdpval, ... | `prompt_append` | Appends skills to prompt via monkey-patch |
| omnimath | `task_field` | Sets `task["skill_text"]` for built-in injection |

## Adding a New Domain

Add one entry to `domain_info.py`:

```python
BENCHMARK_DESCRIPTORS = {
    ...
    "my_domain": DomainInfo(
        query_field="question",           # field in task dict for search query
        # skill_injection="prompt_append", # default, or "task_field"
    ),
}
```

That's it. `extract_skills.py` and `eval_with_skills.py` will handle the rest.

## Files

| File | Description |
|------|-------------|
| `domain_info.py` | Per-domain registry (query field, injection strategy) |
| `extract_skills.py` | Extract skills via EverCore v1 API |
| `eval_with_skills.py` | Evaluate with skills (API search / static / cache) |
| `config.yaml` | Default API URL, skill directory, split settings |
| `skills/` | Extracted skill files |
| `skills_sample/` | Hand-written skill examples |
