# Running the LoCoMo Benchmark

EverOS ships a self-contained runner for the
[LoCoMo](https://github.com/snap-research/locomo) (Long Conversation Memory)
benchmark ([Maharana et al., 2024](https://arxiv.org/abs/2402.17753)).
LoCoMo evaluates how well a memory system retrieves facts from long
multi-session dialogues across four question categories: **single-hop**,
**multi-hop**, **open-domain**, and **temporal**. This guide walks through
reproducing EverOS's LoCoMo retrieval scores locally.

## Pipeline at a glance

Each conversation runs through a four-stage pipeline:

```
ADD ──► wait_ready ──► SEARCH ──► ANSWER ──► JUDGE
 │         │              │          │          │
 │  ingest msgs &    query EverOS  generate   LLM-as-judge
 │  flush per-session  per QA pair  answers    majority vote
 │  into EverOS                     from       (judge_runs×)
 │                                  context
 ▼
 cascade + OME drain
 (per-conv polling)
```

- **ADD** — sends LoCoMo sessions to EverOS (`/add` + `/flush`), then polls
  cascade and OME queues until the conversation's data is fully indexed.
- **SEARCH** — queries EverOS `/search` for each QA question.
- **ANSWER** — feeds retrieved episodes to an LLM to generate an answer.
- **JUDGE** — an LLM judge scores each answer as CORRECT or WRONG against
  the gold answer; runs `judge_runs` times per question and majority-votes.

Stages are **independently re-runnable** — each reads from and writes to
JSONL files, so you can re-judge with a different model without re-ingesting
or re-searching.

Multiple conversations run **in parallel** via `conversations_concurrency`.
Within each conversation, search and eval questions run concurrently via
`search_concurrency` and `eval_concurrency`.

## Contents

- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [1. Prepare the dataset](#1-prepare-the-dataset)
- [2. Start the server](#2-start-the-server)
- [3. Run the benchmark](#3-run-the-benchmark)
- [4. Output](#4-output)
- [CLI reference](#cli-reference)
- [Notes](#notes)

---

## Prerequisites

- A working EverOS installation — complete **all steps** in
  [QUICKSTART.md](../QUICKSTART.md) (configure providers, start server,
  verify search works — not just `/health`)
- Python 3.12+ with `tqdm` installed (`pip install tqdm`)
- EverOS configured for chat-only extraction — in your `everos.toml`:

  ```toml
  [memorize]
  mode = "chat"
  ```

  And in `ome.toml`, disable strategies the benchmark does not use:

  ```toml
  [strategies.extract_foresight]
  enabled = false

  [strategies.extract_user_profile]
  enabled = false
  ```

  This keeps episode extraction, `extract_atomic_facts`, and
  `trigger_profile_clustering` (agentic search relies on clusters),
  while cutting unnecessary LLM calls from foresight and profile
  extraction.

- Copy `benchmarks/.env.example` → `benchmarks/.env` and fill in your API
  keys:

```bash
cp benchmarks/.env.example benchmarks/.env
# Edit benchmarks/.env:
ANSWER_API_KEY=sk-...                           # LLM for generating answers
ANSWER_BASE_URL=https://openrouter.ai/api/v1
JUDGE_API_KEY=sk-...                            # LLM for judging answers
JUDGE_BASE_URL=https://openrouter.ai/api/v1
```

Keys are comma-separated for round-robin failover (e.g.
`ANSWER_API_KEY=sk-aaa,sk-bbb`). More keys raise the effective RPM
ceiling, which lets you increase `eval_concurrency` in `config.toml` for
faster answer/judge throughput.

## Configuration

The only required configuration is provider credentials — copy
`benchmarks/.env.example` → `benchmarks/.env` and fill in your API keys
(already done in [Prerequisites](#prerequisites)).

Everything else has sensible defaults in `benchmarks/config.toml` — see
the comments in that file for tunable parameters.

## 1. Prepare the dataset

LoCoMo 10 contains 10 multi-session conversations (~50 sessions each,
~150 QA pairs per conversation across 4 categories, adversarial
category excluded).

```bash
mkdir -p data
curl -o data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

## 2. Start the server

Raise the file descriptor limit **before** starting — concurrent agentic
searches open many LanceDB segment files simultaneously (EverOS compacts
segments automatically, but burst concurrency during benchmark can exceed
the default macOS limit of 256):

```bash
ulimit -n 10240
everos server start [--root <path>]
```

> **Important:** if you use a custom `--root`, pass the same path to the
> benchmark runner via `--everos-root` — the runner polls the cascade and
> OME databases under that root to know when data is ready. A mismatch
> causes silent readiness false-positives.

## 3. Run the benchmark

All runs require `--run-name`, which becomes the `project_id` used for data
isolation (see [Run isolation](#run-isolation) below).

**Smoke test first** — verify end-to-end connectivity before a full run:

```bash
python benchmarks/run.py --run-name smoke --smoke [--everos-root <path>]
```

**Full run (all 10 conversations):**

```bash
python benchmarks/run.py --run-name locomo-agentic [--everos-root <path>]
```

**Single conversation:**

```bash
python benchmarks/run.py --run-name locomo-agentic --conv 0 [--everos-root <path>]
```

**Skip ingest, re-run search + answer + judge:**

```bash
python benchmarks/run.py --run-name locomo-agentic --stages search answer judge
```

**Re-judge only (reuse existing answer JSONL):**

```bash
python benchmarks/run.py --run-name locomo-agentic --stages judge
```

## 4. Output

Output root is `benchmarks/results/<run-name>/`:

```
benchmarks/results/<run-name>/
├── run_spec.json                  # reproducibility snapshot (git hash, config, stages)
├── conv0/
│   ├── search_<method>.jsonl      # per-question search results
│   ├── answer_<method>.jsonl      # per-question generated answers
│   ├── judge_<method>.jsonl       # per-question judge verdicts
│   └── error.log                  # only on failure — full traceback
├── conv1/ … conv9/
├── report.json                    # aggregate accuracy by method + category
└── report.txt                     # human-readable accuracy table
```

`report.json` and `report.txt` are written after all conversations finish
(only when the `judge` stage is included).

**Sample `report.txt`:**

```
================================================================
  EverOS LoCoMo Benchmark Report
================================================================

Run Info
  Run name:       locomo-agentic
  Generated:      2026-06-28T14:30:00+00:00
  Git hash:       abc1234
  EverOS version: 1.1.0
  Python:         3.12.11
  Conversations:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  Stages:         ['add', 'search', 'answer', 'judge']

Configuration
  Answer model:   gpt-4.1-mini
  Judge model:    gpt-4o-mini
  Judge runs:     3
  Top-k:          10
  Eval owner:     speaker_a

----------------------------------------------------------------
  Method: agentic
----------------------------------------------------------------

  Max accuracy:     93.4% (best of 3 judge runs / mean / majority)
  Majority:         93.3% (1437/1540)
  Mean accuracy:    93.3% (avg across 3 judge runs)

  Per category:
    1. single-hop     94.0% (265/282)
    2. multi-hop      91.0% (292/321)
    3. open-domain    80.2% (77/96)
    4. temporal       95.5% (803/841)

  Per conversation:
    conv0    93.4% (142/152)
    conv1    96.3% (78/81)
    ...

  Search: 1540 queries, avg 23.1s, p50 19.4s, max 142.3s
  Answer: 1540 questions, avg 4.7s, 7,224,168 tokens
  Judge:  1540 questions × 3 runs, 2,335,683 tokens, unanimous 95.2%

  Total tokens: 9,559,851
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--run-name` | *(required)* | Run name — maps to `project_id` for data isolation |
| `--conv` | `0 1 2 … 9` | Conversation indices to run |
| `--stages` | `add search answer judge` | Pipeline stages to execute |
| `--config` | `config` | TOML config name (without `.toml` extension) |
| `--base-url` | `http://localhost:8000` | EverOS server address |
| `--everos-root` | `~/.everos` | EverOS root path (for cascade/OME queue polling) |
| `--data-path` | `data/locomo10.json` | Path to LoCoMo dataset JSON |
| `--smoke` | off | Smoke mode: 2 convs, first 50 msgs each, 10 QA (stratified), `judge_runs=1` |

## Notes

### Evaluation methodology

The runner uses an **LLM-as-Judge** approach: a judge LLM receives the
question, the gold answer, and the generated answer, then outputs `CORRECT`
or `WRONG`. Each question is judged `judge_runs` times (default 3); the
final verdict is a **majority vote**. Accuracy = correct / total per method
and per category.

The four LoCoMo question categories test different retrieval capabilities:

| Category | Name | Tests |
|---|---|---|
| 1 | single-hop | Direct fact retrieval from one episode |
| 2 | multi-hop | Reasoning across multiple episodes |
| 3 | open-domain | General knowledge grounded in conversation |
| 4 | temporal | Time-sensitive questions requiring date reasoning |

Category 5 (adversarial — questions with no answer in the conversation) is
excluded from evaluation.

### Run isolation

Each benchmark run is scoped by three identifiers:

| Scope | Value | Purpose |
|---|---|---|
| `app_id` | `locomo_benchmark` | Fixed; separates benchmark data from production |
| `project_id` | `--run-name` value | Per-experiment isolation |
| `owner_id` | `<speaker>_conv<N>` | Per-conversation memory partition |

Two runs with the **same** `--run-name` share the same memory corpus —
useful when re-running later stages, but problematic if you want a clean
ingest. Use distinct names (e.g. `locomo-agentic`, `locomo-hybrid`) for
independent experiments.

### Stage independence

Each stage reads from and writes to JSONL files in `conv<N>/`. This means:

- `--stages search` reads from the EverOS server (requires prior `add`).
- `--stages answer` reads `search_<method>.jsonl` (requires prior `search`).
- `--stages judge` reads `answer_<method>.jsonl` (requires prior `answer`).

You can swap the judge model and re-run `--stages judge` without touching
ingest or search.

### Smoke mode

`--smoke` is a **pipeline sanity check**, not a scored run. It forces:

- 2 conversations (conv 0, 1) running in parallel
- First 50 messages each (across however many sessions that covers)
- 10 QA pairs per conversation, stratified-sampled to cover all categories
- `judge_runs=1` (no majority vote)

Use it to verify end-to-end connectivity before committing to a full run.

### Runtime estimates

Rough estimates with default settings (varies by provider latency):

| Scope | Time | Token cost (approx.) |
|---|---|---|
| Smoke | 2–5 min | ~80k tokens |
| Single conv (full) | 15–30 min | ~1M tokens |
| Full 10-conv run | 2–4 hours | ~10M tokens |

The `add` + `wait_ready` phase dominates wall-clock time; LLM calls
(answer + judge) dominate token cost.

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Connection refused` on run | EverOS server not running | `everos server start` |
| `ANSWER_API_KEY not set` | Missing `.env` | Copy `.env.example` → `.env`, fill keys |
| `Timeout after 1800s` in wait_ready | Cascade/OME still processing | Increase `cascade_timeout` in config.toml or check server logs |
| `OME task(s) failed` warning | OME strategy crashed | Check `everos cascade status`; data may be incomplete |
| `Missing search_*.jsonl` | Running `answer` without prior `search` | Add `search` to `--stages` or run it first |
| `Too many open files (os error 24)` | LanceDB FD exhaustion from concurrent searches | Lower `search_concurrency` in config.toml (agentic needs more FDs per query) or raise `ulimit -n` |
| Low accuracy across all categories | Embedding/rerank not configured | Verify `everos.toml` has working embedding + rerank providers |
| `conv<N>/error.log` exists | Unhandled exception in that conversation | Read the traceback; other conversations are unaffected |
