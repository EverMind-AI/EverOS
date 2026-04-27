# Multi-Person Group Chat Evaluation Framework

[![arXiv](https://img.shields.io/badge/arXiv-2602.01313-b31b1b.svg)](https://arxiv.org/pdf/2602.01313)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-EverMemBench--Dynamic-yellow)](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic)

A comprehensive evaluation framework for multi-person group chat datasets, supporting **Memory Systems** (Memos, Mem0, Memobase, EverCore, Zep) and **LLM Long-Context Evaluation**.

📄 **Paper**: [EverMemBench: A Comprehensive Benchmark for Long-Term Memory in Conversational AI](https://arxiv.org/pdf/2602.01313)

🤗 **Dataset**: [EverMind-AI/EverMemBench-Dynamic](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic)

## Features

- **Multi-person group chat support**: Handles datasets with multiple speakers across multiple groups and days
- **5 Memory Systems**: Memos, Mem0, Memobase, EverCore, Zep
- **LLM Long-Context Evaluation**: Direct LLM evaluation using full dialogue as context
- **Full Evaluation Pipeline**: Add → Search → Answer → Evaluate
- **Two Question Types**: Multiple choice (direct comparison) and open-ended (LLM judge)
- **Unified message format**: All messages include group/speaker attribution
- **LLM Integration**: Uses OpenRouter for answer generation and evaluation
- **Batch processing**: Efficient API calls with configurable batch sizes and rate limiting
- **Smoke test mode**: Quick validation with limited data

## Pipeline Stages

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│   Add   │ -> │  Search  │ -> │  Answer  │ -> │ Evaluate  │
└─────────┘    └──────────┘    └──────────┘    └───────────┘
     │              │               │               │
     v              v               v               v
  Ingest       Retrieve LLM      Generate       Assess
 memories     memories        answers       accuracy
```

| Stage | Description | Output |
|-------|-------------|--------|
| **Add** | Ingest conversation data into memory system | - |
| **Search** | Retrieve relevant memories for QA questions | `search_results_{user_id}.json` |
| **Answer** | Generate answers using LLM with retrieved context | `answer_results_{user_id}.json` |
| **Evaluate** | Assess answer quality (MC: direct, OE: LLM judge) | `evaluation_results_{user_id}.json` |

## Supported Systems

### Memory Systems

| System | Timestamp Support | Message Format | Environment Variables |
|--------|-------------------|----------------|----------------------|
| **Memos** | Native `chat_time` | `[Group: X][Speaker: Y]content` | `MEMOS_API_KEY`, `MEMOS_BASE_URL` |
| **Mem0** | Native `timestamp` (Unix, per-batch) | `run_id="${user_id}_${groupId}"`, `name=<Speaker>` | `MEM0_API_KEY` |
| **Memobase** | Native `created_at` | `[Group: X][Speaker: Y]content`, `alias=<Speaker>` | `MEMOBASE_BASE_URL`, `MEMOBASE_API_TOKEN` |
| **EverCore** | Native `create_time` | `sender=<Speaker>`, `group_id=${user_id}_${groupId}` | `EVERMEMOS_BASE_URL`, `EVERMEMOS_API_KEY` |
| **Zep** | Native `created_at` | `[Group: X][Speaker: Y]content` | `ZEP_API_KEY` |

### LLM System

| System | Context | Use Case | Environment Variables |
|--------|---------|----------|----------------------|
| **LLM** | Full dialogue (no retrieval) | Test LLM long-context comprehension | `LLM_BASE_URL`, `LLM_API_KEY` |

**Key Differences: Memory Systems vs LLM System**

| Aspect | Memory Systems | LLM System |
|--------|---------------|------------|
| Context | Retrieved memories (top-k) | Full dialogue |
| Add Stage | Ingest into memory system | No-op (stores dialogue) |
| Search Stage | Query memory system | Returns full dialogue |
| Answer Stage | Answer with retrieved context | Answer with full dialogue |
| Use Case | Test memory retrieval | Test LLM long-context |

## Directory Structure

```
eval/
├── cli.py                    # CLI entry point
├── config/
│   ├── pipeline.yaml        # Pipeline settings (answer/evaluate/search/retry/debug)
│   ├── prompts.yaml         # LLM prompts for answer/evaluate
│   ├── memos.yaml           # Memos configuration (connection + add + search)
│   ├── mem0.yaml            # Mem0 configuration (connection + add + search)
│   ├── memobase.yaml        # Memobase configuration (connection + add + search)
│   ├── evermemos.yaml       # EverCore configuration (connection + add + search)
│   └── zep.yaml             # Zep configuration (connection + add + search)
├── src/
│   ├── core/
│   │   ├── data_models.py   # Data classes (QAItem, SearchResult, etc.)
│   │   ├── loaders.py       # Dataset loading utilities
│   │   ├── qa_loader.py     # QA data loader
│   │   ├── pipeline.py      # Evaluation pipeline orchestrator
│   │   ├── answerer.py      # Answer generation with LLM
│   │   └── evaluator.py     # Evaluation with LLM judge
│   ├── adapters/
│   │   ├── base.py          # Base adapter abstract class
│   │   ├── memos_adapter.py # Memos implementation
│   │   ├── mem0_adapter.py  # Mem0 implementation
│   │   ├── memobase_adapter.py   # Memobase implementation
│   │   ├── evermemos_adapter.py  # EverCore implementation
│   │   ├── zep_adapter.py   # Zep Graph API implementation
│   │   └── llm_adapter.py   # LLM system adapter (full dialogue as context)
│   └── utils/
│       ├── config.py        # YAML config loader with env var support
│       └── logger.py        # Rich console logging
└── results/{system}/        # Output: eval/results/{system}/*.json
│                            #   LLM: eval/results/llm/{model}/*.json
tools/
└── analyze_results.py       # Analyze evaluation results by category
```

## Installation

**Requires Python >= 3.11**.

```bash
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Copy the template and fill in your API keys:

```bash
cp env.template .env
```

The LLM variables (OpenRouter) are required for answer generation and evaluation across all systems. Memory system variables only need to be configured for the systems you intend to use. See `env.template` for details.

### Pipeline Configuration

Pipeline settings are in `eval/config/pipeline.yaml`.

```yaml
# eval/config/pipeline.yaml

# Answer generation (answerer.py)
answer:
  model: "openai/gpt-4.1-mini"
  provider:
    order: ["openai"]
    allow_fallbacks: false
  temperature: 0
  max_tokens: 1000
  timeout: 300
  concurrency: 1

# LLM judge evaluation (evaluator.py)
evaluate:
  model: "google/gemini-3-flash-preview"
  provider:
    order: ["google-ai-studio"]
    allow_fallbacks: false
  concurrency: 20

# Search stage (pipeline.py)
search:
  concurrency: 3
  timeout: 120

# Retry (shared)
retry:
  max_retries: 20
  retry_delay: 1.0
  max_delay: 300

# Debug
debug:
  show_usage: true

# Cache warmup (LLM system only)
warmup:
  enabled: true
  delay_seconds: 15
```

### System Search Configuration

Each memory system has its own config file (`eval/config/{system}.yaml`) with a `search:` section for system-specific search parameters. CLI `--top-k` overrides the config `top_k` when provided.

```yaml
# eval/config/memos.yaml
search:
  top_k: 10                        # Number of memories to retrieve
  preference_limit_number: 6        # Number of preference memories

# eval/config/mem0.yaml
search:
  top_k: 10
  group_ids: ["1", "2", "3"]       # Group IDs to search across

# eval/config/memobase.yaml
search:
  max_token_size: 3000              # Max token size for search results
  event_similarity_threshold: 0.2   # Similarity threshold for event matching

# eval/config/evermemos.yaml
search:
  top_k: 10
  retrieve_method: "hybrid"         # Retrieval method: hybrid/semantic/keyword

# eval/config/zep.yaml
search:
  top_k: 10
  reranker_edges: "cross_encoder"   # Edge reranking strategy
  reranker_nodes: "rrf"             # Node reranking strategy
  max_query_length: 400             # Max query length for search
```

### Prompt Templates

```yaml
# eval/config/prompts.yaml
llm_answer:
  multiple_choice: |
    ...
  open_ended: |
    ...
llm_judge:
  system_prompt: |
    ...
  user_prompt: |
    ...
```

## Usage

### Memory Systems Evaluation

Memory systems follow a two-phase workflow: **Add** (ingest data), then **Search → Answer → Evaluate** (run evaluation).

#### Memos

```bash
# Add
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --system memos \
    --user-id 004 \
    --stages add

# Search -> Answer -> Evaluate
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --qa dataset/004/qa_004.json \
    --system memos \
    --user-id 004 \
    --stages search answer evaluate \
    --top-k 10
```

#### Mem0

```bash
# Add
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --system mem0 \
    --user-id 004 \
    --stages add

# Search -> Answer -> Evaluate
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --qa dataset/004/qa_004.json \
    --system mem0 \
    --user-id 004 \
    --stages search answer evaluate \
    --top-k 10
```

#### Memobase

```bash
# Add
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --system memobase \
    --user-id 004 \
    --stages add

# Search -> Answer -> Evaluate
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --qa dataset/004/qa_004.json \
    --system memobase \
    --user-id 004 \
    --stages search answer evaluate
```

#### EverCore

EverCore requires **separate data isolation per batch** (user ID):
- **Cloud service**: Create a new memspace for each batch via the EverCore dashboard, then use the corresponding `--base-url`.
- **Local deployment**: Start a separate service instance per batch, each on its own port (e.g., port `19004` for user `004`, port `19005` for user `005`). API key is not required for local deployment.

```bash
# Add (local deployment, port per batch)
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --system evermemos \
    --user-id 004 \
    --stages add \
    --base-url http://0.0.0.0:19004

# Search -> Answer -> Evaluate
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --qa dataset/004/qa_004.json \
    --system evermemos \
    --user-id 004 \
    --stages search answer evaluate \
    --top-k 10 \
    --base-url http://0.0.0.0:19004
```

#### Zep

```bash
# Add
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --system zep \
    --user-id 004 \
    --stages add

# Search -> Answer -> Evaluate
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --qa dataset/004/qa_004.json \
    --system zep \
    --user-id 004 \
    --stages search answer evaluate \
    --top-k 10
```

### LLM Long-Context Evaluation

The LLM system uses the **full dialogue** as context (no memory retrieval). Add/search stages are auto-injected.

```bash
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --qa dataset/004/qa_004.json \
    --system llm \
    --user-id 004 \
    --stages answer evaluate
```

### Evaluate Only (re-evaluate existing answer results)

```bash
python -m eval.cli \
    --qa dataset/004/qa_004.json \
    --system mem0 \
    --user-id 004 \
    --stages evaluate
```

### Smoke Test

```bash
# Smoke test add stage
python -m eval.cli --dataset dataset/004/dialogue.json --system memos --smoke

# Smoke test with specific date
python -m eval.cli --dataset dataset/004/dialogue.json --system memos --smoke --smoke-date 2025-01-16

# LLM smoke test with limited questions
python -m eval.cli \
    --dataset dataset/004/dialogue.json \
    --qa dataset/004/qa_004.json \
    --system llm \
    --user-id 004 \
    --stages answer evaluate \
    --qa-limit 3
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--dataset` | Path to dataset JSON file (required for add stage) | - |
| `--system` | System (memos/mem0/memobase/evermemos/zep/llm) | Required |
| `--stages` | Stages to run: add, search, answer, evaluate | `["add"]` |
| `--qa` | Path to QA JSON file (required for search/answer/evaluate) | - |
| `--user-id` | User ID for memory system | Auto-generated |
| `--top-k` | Number of memories to retrieve | From system config |
| `--output-dir` | Results base directory (output goes to `{output-dir}/{system}/`) | `eval/results` |
| `--base-url` | Override base URL for memory system | - |
| `--start-date` | Resume add from this date (YYYY-MM-DD) | - |
| `--smoke` | Enable smoke test mode | False |
| `--smoke-days` | Days to process in smoke test | 1 |
| `--smoke-date` | Specific date for smoke test (YYYY-MM-DD) | - |
| `--qa-limit` | Limit number of QA questions | - |

## Output Structure

Results are organized by system under `eval/results/`:

```
eval/results/
├── memos/
│   ├── search_results_004.json
│   ├── answer_results_004.json
│   └── evaluation_results_004.json
├── mem0/
│   └── ...
├── memobase/
│   └── ...
├── evermemos/
│   └── ...
├── zep/
│   └── ...
└── llm/
    └── openai/
        └── gpt-4.1-mini/          # LLM results include model name in path
            ├── answer_results_004.json
            └── evaluation_results_004.json
```

## Analysis Tools

`tools/analyze_results.py` analyzes evaluation results by question_id categories (major/minor/hierarchical). Supports single-file analysis and multi-batch aggregation.

```bash
# Single file analysis
python tools/analyze_results.py eval/results/evermemos/evaluation_results_004.json

# Aggregate all batches for a system
python tools/analyze_results.py --system mem0

# Specify results directory directly
python tools/analyze_results.py --results-dir eval/results/memos/

# Save JSON report
python tools/analyze_results.py --system evermemos -o report.json

# Quiet mode (JSON output only)
python tools/analyze_results.py --system zep -o report.json -q
```

## Dataset Batches

Supported user IDs: `004`, `005`, `010`, `011`, `016`

Each batch has:
- `dataset/{batch_id}/dialogue.json` - Conversation data
- `dataset/{batch_id}/qa_{batch_id}.json` - QA questions for evaluation
