# Running the LoCoMo Benchmark

This guide walks through reproducing EverOS's LoCoMo retrieval scores
locally using the `hybrid` and `agentic` search methods.

## Contents

- [Prerequisites](#prerequisites)
- [1. Prepare the dataset](#1-prepare-the-dataset)
- [2. Start the server](#2-start-the-server)
- [3. Run `hybrid`](#3-run-hybrid)
- [4. Run `agentic`](#4-run-agentic)
- [5. Where the results land](#5-where-the-results-land)
- [Notes](#notes)

---

## Prerequisites

- Python **3.12**, [uv](https://docs.astral.sh/uv/)
- A `.env` at the repo root with the LLM / embedding credentials EverOS
  needs:
  - `EVEROS_LLM__MODEL`, `EVEROS_LLM__API_KEY`, `EVEROS_LLM__BASE_URL`
  - `EVEROS_EMBEDDING__*`
  - `EVEROS_RERANK__*`
  - The benchmark driver also reads `LLM_API_KEY` / `ANSWER_MODEL` /
    `JUDGE_MODEL` for the answer + judge passes.

Install the project:

```bash
uv sync
```

## 1. Prepare the dataset

Place the LoCoMo file at `data/locomo10.json` (the dataset is
distributed by the LoCoMo authors, not this repo). Override the path
later with `--data-path` if you keep it elsewhere.

## 2. Start the server

```bash
EVEROS_MEMORY__ROOT=~/.everos \
uv run python -m everos.entrypoints.cli.main server start --port 8000
```

`EVEROS_MEMORY__ROOT` isolates one benchmark's corpus from another —
change it (or `rm -rf` it) whenever you want a clean run.

Leave the server running in one terminal; run the benchmark from
another.

## 3. Run `hybrid`

Single conversation:

```bash
bash tests/run_locomo_batch.sh \
  --conv-indices 0 \
  --methods hybrid \
  --base-url http://localhost:8000 \
  --top-k 10
```

All 10 conversations, 2-way parallel:

```bash
bash tests/run_locomo_batch.sh \
  --conv-indices 0-9 \
  --methods hybrid \
  --base-url http://localhost:8000 \
  --top-k 10 \
  --concurrency 2
```

The wrapper picks up `EVEROS_MEMORY__ROOT` from the environment so the
cascade poll path matches the server's data root. If you set them
differently, pass `--corpus-path` explicitly.

## 4. Run `agentic`

Same wrapper, swap `--methods`:

```bash
bash tests/run_locomo_batch.sh \
  --conv-indices 0-9 \
  --methods agentic \
  --base-url http://localhost:8000 \
  --top-k 10 \
  --concurrency 2
```

You can also benchmark multiple methods in one go — they share the
same ingested corpus:

```bash
bash tests/run_locomo_batch.sh \
  --conv-indices 0-9 \
  --methods hybrid,agentic \
  --base-url http://localhost:8000 \
  --top-k 10 \
  --concurrency 2
```

## 5. Where the results land

Default output root is `benchmark_results/run_<timestamp>/`. Override
with `--output-root`:

```
<output_root>/
├── conv0.json … conv9.json          # per-conv summary + per-question details
├── conv0.log  … conv9.log           # per-conv stdout (only in --concurrency >1 mode)
└── conv0_checkpoints/ …             # incremental search/answer/eval JSON
```

An aggregate accuracy table prints at the end of the wrapper run.

## Notes

- **Re-running on the same corpus**: add `--skip-add` to skip ingest and
  reuse what's already in `~/.everos`. Useful when comparing methods
  side by side.
- **Judge variance**: `--judge-runs 3` runs the judge three times per
  question and majority-votes; slower but reduces LLM-judge noise.
