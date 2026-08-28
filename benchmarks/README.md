# Running the benchmarks

One runner, four benchmarks, four stages: ADD, SEARCH, ANSWER, JUDGE.
Per-benchmark rules live in `benchmarks/adapters/<name>.py`.

> Also available in Chinese: [README.zh.md](README.zh.md).

---

## 1. Install dependencies

```bash
uv sync
```

## 2. Get the data

| Benchmark | Target path | Source |
|---|---|---|
| LoCoMo | `benchmarks/data/locomo10.json` | [snap-research/locomo](https://github.com/snap-research/locomo) |
| LongMemEval | `benchmarks/data/longmemeval_s.json` | [xiaowu0162/longmemeval](https://huggingface.co/datasets/xiaowu0162/longmemeval) |
| EverMemBench | `benchmarks/data/evermembench.json` | [EverMind-AI/EverMemBench-Dynamic](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) |
| SubtleMemory | `benchmarks/data/subtlememory/` | [Yummytanmo/SubtleMemory](https://huggingface.co/datasets/Yummytanmo/SubtleMemory) |

> **EverMemBench** — download the snapshot to `benchmarks/data/raw/EverMemBench-Dynamic/`,
> then run `python -m benchmarks.adapters.evermembench` once.
>
> **SubtleMemory** — keep the `persona_0` .. `persona_9` directory layout.

## 3. Configure the environment

```bash
cp benchmarks/.env.example benchmarks/.env
```

| Role | Keys | Notes |
|---|---|---|
| Extraction model | `EVEROS_LLM__MODEL` / `__API_KEY` / `__BASE_URL` | used by ADD |
| Retrieval model | `BENCH_DECIDER_MODEL` / `BENCH_DECIDER_BASE_URL` | the multi-round decider; defaults to `qwen3.6-27B`, and falls back to the extraction model when no endpoint is set |
| Answer model | `ANSWER_API_KEY` / `ANSWER_BASE_URL` | model name in the config |
| Judge model | `JUDGE_API_KEY` / `JUDGE_BASE_URL` | model name in the config |

Models, `top_k`, retrieval settings and concurrency are set in
`benchmarks/configs/<dataset>.toml`. `reproduce.sh` passes no model overrides.

## 4. Run

```bash
# all conversations, all four stages
DATASET=locomo bash benchmarks/reproduce.sh

# smoke test: one conversation, 10 sampled questions
DATASET=locomo CONV=0 bash benchmarks/reproduce.sh --smoke
```

| Variable | Default | Values |
|---|---|---|
| `DATASET` | `locomo` | `locomo` \| `longmemeval` \| `subtlememory` \| `evermembench` |
| `CONV` | `all` | conversation indices |
| `STAGES` | `add search answer judge` | any subset, in order |
| `RUN` | derived | result directory name |

`reproduce.sh` starts and stops its own EverOS servers.

| Stage | Action |
|---|---|
| ADD | streams conversations into EverOS, which extracts memories |
| SEARCH | one retrieval per question; records the injected episodes |
| ANSWER | answers each question from those episodes |
| JUDGE | grades each answer against the reference |

Each stage appends per-conversation JSONL and skips entries already present, so
an interrupted run resumes.

## 5. Output

```
benchmarks/results/<Dataset>/<run>/
├── report.txt            summary
├── report.json           machine-readable summary
├── run_spec.json         models served, endpoints, package versions, knobs
├── store/                built by ADD; absent when --everos-root points elsewhere
├── conv<N>/
│   ├── search_<method>.jsonl
│   ├── answer_<method>.jsonl
│   └── judge_<method>.jsonl
└── traces/               per-round retrieval traces
```

`run_spec.json` records the models actually served and the `everalgo` package
versions, which determine what a store contains.

## 6. Results

Each benchmark's published number, produced by the configuration in
`benchmarks/configs/<dataset>.toml`.

| Benchmark | Questions | Accuracy | Decider | Answer model |
|---|---|---|---|---|
| LoCoMo | 1,540 | **94.42** | `deepseek/deepseek-v4-flash-0731` | `openai/gpt-4.1-mini` |
| LongMemEval | 500 | **94.00** | `qwen3.6-27B` | `google/gemini-3.6-flash` |
| EverMemBench | 2,400 | **66.67** | `qwen3.6-27B` | `google/gemini-3-flash-preview` |
| SubtleMemory | 1,522 | **71.75** | `qwen3.6-27B` | `openai/gpt-5.4` |

> **EverMemBench** — scores as the mean of its nine category columns, which is
> how the benchmark reports it.
>
> **SubtleMemory** — routes each question between two answer contracts on whether
> fact extraction found a conflict; the adapter does this automatically.


---

## run.py flags

| Flag | Meaning |
|---|---|
| `--conv` | conversation indices, or `all` |
| `--stages` | any of `add search answer judge` |
| `--everos-root` | store to use; default `<results>/<run>/store` |
| `--servers` | EverOS servers to run in parallel |
| `--results-root` | output root; default `benchmarks/results/<Dataset>` |
| `--data-path` | dataset path for one run |
| `--methods` | `llm_multiround` \| `hybrid` \| `agentic` |
| `--answer-model` / `--judge-model` | model override for one run |
| `--decider-model` / `--decider-base-url` | decider; set both |
| `--smoke` | 10 sampled questions per conversation |
