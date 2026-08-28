# Running the benchmarks

EverOS ships a self-contained runner for four long-term-memory benchmarks:
[LoCoMo](https://github.com/snap-research/locomo)
([Maharana et al., 2024](https://arxiv.org/abs/2402.17753)),
[LongMemEval](https://github.com/xiaowu0162/LongMemEval), SubtleMemory, and
EverMemBench.

All four run through one pipeline and differ only in a per-dataset adapter.
`benchmarks/adapters/<name>.py` is where each benchmark's own rules live — how it
names an owner, what counts as gold evidence, how its judge grades. Nothing else
in the runner knows which benchmark it is running.

> Also available in Chinese: [README.zh.md](README.zh.md).

---

## Quickstart

```bash
# 1. install
uv sync

# 2. configure
cp benchmarks/.env.example benchmarks/.env
$EDITOR benchmarks/.env          # answer/judge API keys, extraction backbone

# 3. put the dataset where the config expects it
curl -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json

# 4. reproduce
DATASET=locomo bash benchmarks/reproduce.sh
```

That last command runs the full pipeline — ADD, SEARCH, ANSWER, JUDGE — at the
configuration the published number was produced with, and writes a report under
`benchmarks/results/LoCoMo/locomo/`.

---

## Layout

```
benchmarks/
├── reproduce.sh          the entry point; runs one benchmark end to end
├── run.py                the pipeline (ADD → SEARCH → ANSWER → JUDGE)
├── config.py             the frozen config model
├── configs/
│   ├── default.toml      shared defaults
│   └── <dataset>.toml    one per benchmark: models, top_k, concurrency
├── adapters/
│   ├── base.py           the four questions an adapter answers
│   └── <dataset>.py      one per benchmark: owners, gold, prompts, judge
├── metrics/              ranked-retrieval and core-selection metrics
├── data/                 ← put the datasets here (git-ignored)
├── results/              ← runs are written here (git-ignored)
└── .env.example          copy to .env
```

`data/` and `results/` are the only two directories you write to. Both are
git-ignored; the datasets are redistributable only by their own authors, and a
run's output is large and regenerable.

---

## 1. Prepare the dataset

Each benchmark's input path comes from its config, which defaults to
`benchmarks/data/`. **Putting the file there is all that is required** — no
environment variable, no flag:

| Benchmark | Put the file at | Environment override |
|---|---|---|
| `locomo` | `benchmarks/data/locomo10.json` | `BENCH_DATA_LOCOMO` |
| `longmemeval` | `benchmarks/data/longmemeval_s.json` | `BENCH_DATA_LONGMEMEVAL` |
| `subtlememory` | `benchmarks/data/subtlememory/` (a directory of `persona_0..9/`) | `BENCH_DATA_SUBTLEMEMORY` |
| `evermembench` | `benchmarks/data/evermembench.json` | `BENCH_DATA_EVERMEMBENCH` |

Set the environment variable only when the data lives somewhere else — a shared
mount, another disk. It is read from `benchmarks/.env`.

### Getting each dataset

**LoCoMo** — 10 multi-session conversations, ~50 sessions each, ~150 QA pairs per
conversation across four categories (the adversarial category is excluded).

```bash
curl -o benchmarks/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
```

**LongMemEval** — the `longmemeval_s` split from
[xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval). Place it as
`benchmarks/data/longmemeval_s.json`.

**SubtleMemory** and **EverMemBench** — released by their own authors; follow each
project's instructions. EverMemBench needs its raw release converted once; point
`EVERMEMBENCH_RAW_ROOT` at the raw directory, which the adapter reads to recover
session names the converted file does not carry.

### When a path is wrong

A missing or misconfigured path **fails the run**; it does not load zero rows and
score 0%. An unset variable with no default reaches the loader as the literal
`${BENCH_DATA_LOCOMO}` and is reported by name, so the error says which variable
to set rather than describing a file called `${...}`.

---

## 2. Configure

```bash
cp benchmarks/.env.example benchmarks/.env
```

The minimum for a first run:

| Key | What it is |
|---|---|
| `ANSWER_API_KEY` / `ANSWER_BASE_URL` | the model that answers questions from retrieved memories |
| `JUDGE_API_KEY` / `JUDGE_BASE_URL` | the model that grades those answers |
| `EVEROS_LLM__MODEL` / `__API_KEY` / `__BASE_URL` | the extraction backbone EverOS runs during ADD |

Everything that decides the number — models, `top_k`, retrieval knobs, concurrency
— lives in `benchmarks/configs/<dataset>.toml`, not in the command line. That is
deliberate: a run is reproducible from its config, and `reproduce.sh` passes no
model overrides.

### The multi-round decider (optional)

`llm_multiround` retrieval runs a *decider* that reads candidate memories, picks
the core set, and asks follow-up sub-queries. The published numbers used a
**separate** decider, which each config names:

```toml
decider_model    = "${BENCH_DECIDER_MODEL:-qwen3.6-27B}"
decider_base_url = "${BENCH_DECIDER_BASE_URL}"
```

The model has a default; **the endpoint does not**, because there is no endpoint
we can name that serves it for you. So out of the box the run prints:

```
decider  no endpoint for qwen3.6-27B (set BENCH_DECIDER_BASE_URL); falling back to the [llm] model.
         ⚠ This is NOT the published configuration -- the published numbers were
         produced with a separate decider, so this run is not comparable to them.
```

and continues with the extraction model as the decider. That is a working run; it
is simply not the published arm. To reproduce the published arm, serve
`qwen3.6-27B` (or another decider) and set **both**:

```bash
BENCH_DECIDER_MODEL=<model name>
BENCH_DECIDER_BASE_URL=<endpoint serving it>
```

Setting only the model, with an endpoint that does not serve it, is the failure
this runner will not let you have: the model 404s on every call, the retrieval
loop falls back to a fixed top-ranked core, and the run **still reports a complete
result**. `run.py` sends one real completion to the decider before the first
question and refuses to start if it does not answer.

---

## 3. Start the server (only if you want to manage it yourself)

`reproduce.sh` starts and stops its own servers. Start one manually only when you
want to reuse a store across runs:

```bash
ulimit -n 10240        # concurrent searches open many LanceDB segment files
everos server start [--root <path>]
```

Pass the same path to the runner with `--everos-root`. The runner polls the
cascade and OME databases under that root to know when data is ready, so a
mismatch causes silent readiness false-positives.

---

## 4. Run

```bash
# full reproduction, all conversations, all four stages
DATASET=locomo bash benchmarks/reproduce.sh

# a slice, for a smoke test
DATASET=locomo CONV=0 STAGES=search bash benchmarks/reproduce.sh \
  --everos-root <an existing store>

# any run.py flag is forwarded
DATASET=longmemeval bash benchmarks/reproduce.sh --servers 4
```

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | `locomo` | `locomo` \| `longmemeval` \| `subtlememory` \| `evermembench` |
| `CONV` | `all` | conversation indices; a slice is not a reproduction |
| `STAGES` | `add search answer judge` | which stages to run |
| `RUN` | derived | result directory name |

### The four stages

| Stage | What it does |
|---|---|
| **ADD** | streams each conversation into EverOS, which extracts memories |
| **SEARCH** | one retrieval per question; writes the episodes the answer model will see |
| **ANSWER** | asks the answer model each question against those episodes |
| **JUDGE** | grades each answer against the reference |

Stages are resumable: each writes a JSONL artefact per conversation and skips
what is already in it. A crash at question 190 of 199 loses one question, not
190. Re-running with `--stages answer judge` reuses the existing search results.

---

## 5. Output

```
benchmarks/results/<Dataset>/<run>/
├── report.txt            human-readable summary
├── report.json           the same numbers, machine-readable
├── run_spec.json         every model, endpoint, package version and knob used
├── conv<N>/
│   ├── search_<method>.jsonl
│   ├── answer_<method>.jsonl
│   └── judge_<method>.jsonl
└── traces/               per-round retrieval traces, when enabled
```

`run_spec.json` is what makes a number reproducible later. It records the models
*actually served* — not what the config asked for — along with package versions,
because the extraction algorithm's version changes what a store contains.

---

## Reproducibility

Two things decide whether a number can be compared to a published one:

**The store's extraction backbone.** A store built with a different extraction
model is a different store. `run_spec.json` records which one built it, and
`store_spec.json` inside a shared store records the `everalgo` versions that
produced it. Retrieval reproduces across those versions; extraction does not.

**The judge.** Each benchmark's judge is its own — LongMemEval adds per-category
leniency clauses, EverMemBench grades multiple-choice by letter with no LLM call
at all. The adapters carry each benchmark's judge verbatim from its reference
harness, and a parity check compares them byte for byte.

---

## CLI reference

`reproduce.sh` covers the normal path. `run.py` takes these directly:

| Flag | Meaning |
|---|---|
| `--conv` | conversation indices, or `all` |
| `--stages` | any of `add search answer judge` |
| `--everos-root` | store to use; defaults to `<results>/<run>/store` |
| `--servers` | how many EverOS servers to run in parallel |
| `--results-root` | where to write; defaults to `benchmarks/results/<Dataset>` |
| `--data-path` | overrides the dataset path for one run |
| `--methods` | `llm_multiround` \| `hybrid` \| `agentic` |
| `--answer-model` / `--judge-model` | override for one run |
| `--decider-model` / `--decider-base-url` | the multi-round decider; set both |
| `--smoke` | 10 sampled questions per conversation |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `benchmarks/.env not found` | first run | `cp benchmarks/.env.example benchmarks/.env` |
| `data directory not found: benchmarks/data/...` | dataset not downloaded | see [1. Prepare the dataset](#1-prepare-the-dataset) |
| `decider ... did not answer` | `BENCH_DECIDER_MODEL` set without its endpoint | set both, or neither |
| `decider ... produced no usable content` | a reasoning model spent its budget thinking | `EVEROS_DECIDER__EXTRA='{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'` |
| `Timeout after 1800s` in wait_ready | extraction still running | raise `cascade_timeout` in the config, or check the server log |
| `Too many open files` | LanceDB FD exhaustion under concurrency | lower `search_concurrency`, or raise `ulimit -n` |
| 0 episodes retrieved, no error | store built under different partition keys | `app_id` / `project_id` must match how the store was built |
