# EverMemOS Evaluation Framework

A unified, modular evaluation framework for benchmarking memory systems on standard datasets.

## 📖 Overview

### Evaluation Scope

In addition to **EverMemOS**, this framework supports evaluation of several influential memory systems in the industry:
- **Mem0** 
- **MemOS** 
- **Zep** 
- **MemU** 

These systems were selected based on recent industry benchmarks and their prominence in global markets. Since many commercial systems have web-based optimizations not available in their open-source versions, we evaluate them through their **online API interfaces** to ensure fair comparison with production-grade capabilities.

### Implementation

Our adapter implementations are based on:
- **Official open-source repositories**: Mem0, MemOS, Zep on GitHub
- **Official documentation**: Mem0, MemOS, MemU, Zep quick start guide and API documentation
- **Consistent methodology**: All systems evaluated using the same pipeline, datasets, and metrics
- **Unified answer generation**: All systems use **GPT-4.1-mini** as the answer LLM to ensure fair comparison across different memory backends

During our evaluation, we identified several issues in existing open-source reference implementations for benchmarking these systems that could negatively impact their performance. We addressed these implementation gaps to ensure each system is evaluated at its best potential:

- **Mem0 timezone handling**: The latest version returns timestamps in PDT format in search results, requiring additional timezone conversion for accurate temporal reasoning.

- **MemU retrieval enhancement**: While some memories are visible in the backend dashboard, the `/memory/retrieve/related-memory-items` API likely relies on simple vector-based retrieval, which may miss relevant context. Following the official documentation examples, we included category summaries as additional context to improve recall.

- **Zep API migration**: Zep's official open-source evaluation implementation was based on the earlier v2 API. Since Zep has officially upgraded to v3 API, we migrated the evaluation code to v3 following the official documentation to benchmark the latest capabilities.

- **Zep timestamp semantics**: Unlike most memory systems that record conversation timestamps, Zep records event occurrence timestamps. For example, a conversation on March 2nd mentioning "Anna ate a burger yesterday" would be timestamped March 1st, with the memory content preserving the original phrasing. Using standard answer prompts leads to significant errors on temporal questions. Zep's team provides optimized prompts in their open-source evaluation code to handle this. This informed one of our evaluation principles: **each memory system uses its own official answer prompts** rather than a unified prompt template, ensuring fair evaluation of each system's intended usage.
 
 


### Evaluation Results
**Results on Locomo**

| Locomo    | single hop | multi hop | temporal | open domain | Overall | Average Tokens | Version                                         | Answer LLM |
|-----------|------------|-----------|----------|-------------|---------|----------------|----------------------------------------------|-----------------|
| Full-context | 94.93      | 90.43     | 87.95    | 71.88       | 91.21   | 20281          |                                              | gpt-4.1-mini    |
| Mem0      | 68.97      | 61.70     | 58.26    | 50.00       | 64.20   | 1016           | web API/v1.0.0 (2025.11)                   | gpt-4.1-mini    |
| Zep       | 90.84      | 81.91     | 77.26    | 75.00       | 85.22   | 1411           | web API/v3 (2025.11)                       | gpt-4.1-mini    |
| MemOS     | 85.37      | 79.43     | 75.08    | 64.58       | 80.76   | 2498           | web API/v1 (2025.11)                       | gpt-4.1-mini    |
| MemU      | 74.91      | 72.34     | 43.61    | 54.17       | 66.67   | 3964           | web API/v1 (2025.11)                      | gpt-4.1-mini    |
| EverMemOS | 96.08      | 91.13     | 89.72    | 70.83       | 92.32   | 2298           | open-source EverMemOS v1.0.0 companion | gpt-4.1-mini    |

*Full-context: using the whole conversation as context for answering questions.


**Results on Longmemeval**

| Longmemeval | Single-session-user  | Single-session-assistant  | Single-session-preference  | Multi-session  | Knowledge-update  | Temporal-reasoning  | Overall |
|-------------|----------------------|---------------------------|----------------------------|----------------|-------------------|---------------------|---------|
| EverMemOS   | 100.00               | 78.57                     | 96.67                      | 78.45          | 87.18             | 71.18               | 82.00   |

> **Note on Reproducibility**: To ensure the reproducibility of our evaluation, we provide full evaluation intermediate data for all methods. You can access the data at [EverMind-AI/EverMemOS_Eval_Results](https://huggingface.co/datasets/EverMind-AI/EverMemOS_Eval_Results).


## 🌟 Key Features

### Unified & Modular Framework
- **One codebase for all**: No need to write separate code for each dataset or system
- **Plug-and-play systems**: Support multiple memory systems (EverMemOS, Mem0, MemOS, MemU, etc.)
- **Multiple benchmarks**: LoCoMo, LongMemEval, PersonaMem out of the box
- **Consistent evaluation**: All systems evaluated with the same pipeline and metrics

### Automatic Compatibility Detection
The framework automatically detects and adapts to:
- **Multi-user vs Single-user conversations**: Handles both conversation types seamlessly
- **Q&A vs Multiple-choice questions**: Adapts evaluation approach based on question format  
- **With/without timestamps**: Works with or without temporal information

### Robust Checkpoint System
- **Cross-stage checkpoints**: Resume from any pipeline stage (add → search → answer → evaluate)
- **Fine-grained resume**: Saves progress every conversation (search) and every 400 questions (answer)


## 🏗️ Architecture Overview

### Code Structure

```
evaluation/
├── src/
│   ├── core/           # Pipeline orchestration and data models
│   ├── adapters/       # System-specific implementations
│   ├── evaluators/     # Answer evaluation (LLM Judge, Exact Match)
│   ├── converters/     # Dataset format converters
│   └── utils/          # Configuration, logging, I/O
├── config/
│   ├── datasets/       # Dataset configurations (locomo.yaml, etc.)
│   ├── systems/        # System configurations (evermemos.yaml, etc.)
│   └── prompts.yaml    # Prompt templates
├── data/               # Benchmark datasets
└── results/            # Evaluation results and logs
```

### Pipeline Flow

The evaluation consists of 4 sequential stages:

1. **Add**: Ingest conversations and build indexes
2. **Search**: Retrieve relevant memories for each question
3. **Answer**: Generate answers using retrieved context
4. **Evaluate**: Assess answer quality with LLM Judge or Exact Match

Each stage saves its output and can be resumed independently.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- EverMemOS environment configured (see main project's `env.template`)

### Data Preparation

Place your dataset files in the `evaluation/data/` directory:

**LoCoMo** (native format, no conversion needed):
Get data from: https://github.com/snap-research/locomo/tree/main/data

```
evaluation/data/locomo/
└── locomo10.json
```

**LongMemEval** (auto-converts to LoCoMo format):
Get data from: https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

```
evaluation/data/longmemeval/
└── longmemeval_s_cleaned.json  # Original file
# → Will auto-generate: longmemeval_s_locomo_style.json
```

**PersonaMem** (auto-converts to LoCoMo format):
Get data from: https://huggingface.co/datasets/bowen-upenn/PersonaMem

```
evaluation/data/personamem/
├── questions_32k.csv           # Original file
└── shared_contexts_32k.jsonl   # Original file
# → Will auto-generate: personamem_32k_locomo_style.json
```

The framework will automatically detect and convert non-LoCoMo formats on first run. You don't need to manually run any conversion scripts.

### Installation

Install evaluation-specific dependencies:

```bash
# For evaluating local systems (EverMemOS)
uv sync --group evaluation

# For evaluating online API systems (Mem0, MemOS, MemU, etc.)
uv sync --group evaluation-full
```

### Environment Configuration

The evaluation framework reuses most environment variables from the main EverMemOS `.env` file:
- `LLM_API_KEY`, `LLM_BASE_URL` (for answer generation with GPT-4.1-mini)
- `VECTORIZE_API_KEY` and  `RERANK_API_KEY` (for embeddings/reranker)

**⚠️ Important**: For OpenRouter API (used by gpt-4.1-mini), make sure `LLM_API_KEY` is set to your OpenRouter API key (format: `sk-or-v1-xxx`). The system will look for API keys in this order:
1. Explicit `api_key` parameter in config
2. `LLM_API_KEY` environment variable

For testing EverMemOS, please first configure the whole .env file.

**Additional variables for online API systems** (add to `.env` if testing these systems):

```bash
# Mem0
MEM0_API_KEY=your_mem0_api_key

# MemOS
MEMOS_KEY=your_memos_api_key

# MemU
MEMU_API_KEY=your_memu_api_key
```

### Quick Test (Smoke Test)

Run a quick test with limited data to verify everything works:

```bash
# Navigate to project root
cd /path/to/memsys-opensource

# Default: first conversation, first 10 messages, first 3 questions 
uv run python -m evaluation.cli --dataset locomo --system evermemos --smoke

# Custom: first conversation, 20 messages, 5 questions
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --smoke --smoke-messages 20 --smoke-questions 5

# You can also evaluate specific conversations with `--from-conv` and `--to-conv` (0-based, end exclusive):
uv run python -m evaluation.cli --dataset locomo --system evermemos_custom --from-conv 0 --to-conv 1
```


### Full Evaluation

Run the complete benchmark:

```bash
# Evaluate EverMemOS on LoCoMo
uv run python -m evaluation.cli --dataset locomo --system evermemos

# Evaluate EverMemOS via local API (start server first)
uv run python src/run.py
# Use --clean-groups to clear existing data before Add stage
uv run python -m evaluation.cli --dataset locomo --system evermemos_local_api --clean-groups

# Evaluate other systems
uv run python -m evaluation.cli --dataset locomo --system memos
uv run python -m evaluation.cli --dataset locomo --system memu
# For mem0, it's recommended to run add first, check the memory status on the web console to make sure it's finished and then following stages.
uv run python -m evaluation.cli --dataset locomo --system mem0 --stages add
uv run python -m evaluation.cli --dataset locomo --system mem0 --stages search answer evaluate

# Evaluate on other datasets
uv run python -m evaluation.cli --dataset longmemeval --system evermemos
uv run python -m evaluation.cli --dataset personamem --system evermemos

# Use --run-name to distinguish multiple runs (useful for A/B testing)
# Results will be saved to: results/{dataset}-{system}-{run-name}/
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name baseline
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name experiment1
uv run python -m evaluation.cli --dataset locomo --system evermemos --run-name 20241107

# Resume from checkpoint if interrupted (automatic)
# Just re-run the same command - it will detect and resume from checkpoint
uv run python -m evaluation.cli --dataset locomo --system evermemos

```

### View Results

Results are saved to `evaluation/results/{dataset}-{system}[-{run-name}]/`:

```bash
# View summary report
cat evaluation/results/locomo-evermemos/report.txt

# View detailed evaluation results
cat evaluation/results/locomo-evermemos/eval_results.json

# View pipeline execution log
cat evaluation/results/locomo-evermemos/pipeline.log
```

**Result files:**
- `report.txt` - Summary metrics (accuracy, total questions)
- `eval_results.json` - Per-question evaluation details
- `answer_results.json` - Generated answers and retrieved context
- `search_results.json` - Retrieved memories for each question
- `pipeline.log` - Detailed execution logs

## 📊 Understanding Results

### Metric

- **Accuracy**: Percentage of correct answers (QA judged by LLM, multiple choice questions judged by exact match)

### Detailed Results

Check `eval_results.json` for per-question breakdown:

**LoCoMo example (Q&A format, evaluated by LLM Judge):**

```json
{
  "total_questions": ...,
  "correct": ...,
  "accuracy": ...,
  "detailed_results": {
      "locomo_exp_user_0": [
         {
            "question_id": "locomo_0_qa0",
            "question": "What is my favorite food?",
            "golden_answer": "Pizza",
            "generated_answer": "Your favorite food is pizza.",
            "judgments": [
               true,
               true,
               true
            ],
            "category": "1"
         }
         ...
      ]
  }
}
```

**PersonaMem example (Multiple-choice format, evaluated by Exact Match):**

```json
{
  "overall_accuracy": ...,
  "total_questions": ...,
  "correct_count": ...,
  "detailed_results": [
    {
      "question_id": "acd74206-37dc-4756-94a8-b99a395d9a21",
      "question": "I recently attended an event where there was a unique blend of modern beats with Pacific sounds.",
      "golden_answer": "(c)",
      "generated_answer": "(c)",
      "is_correct": true,
      "category": "recall_user_shared_facts"
    }
    ...
  ]
}
```

## 🔧 Advanced Usage

### Run Specific Stages

Skip completed stages to iterate faster:

```bash
# Only run search stage (if add is already done)
uv run python -m evaluation.cli --dataset locomo --system evermemos --stages search

# Run search, answer, and evaluate (skip add)
uv run python -m evaluation.cli --dataset locomo --system evermemos \
    --stages search answer evaluate
```
If you have already done search, and you want to do it again, please remove the "search" (and following stages from the completed_stages in the checkpoint_default.json file):
```
  "completed_stages": [
    "answer",
    "search",
    "evaluate",
    "add"
  ]
```


### Custom Configuration

Modify system or dataset configurations:

```bash
# Copy and edit configuration
cp evaluation/config/systems/evermemos.yaml evaluation/config/systems/evermemos_custom.yaml
# Edit evermemos_custom.yaml with your changes

# Run with custom config
uv run python -m evaluation.cli --dataset locomo --system evermemos_custom
```

## 📄 License

Same as the parent project.
