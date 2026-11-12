# Running the benchmarks

One runner, four benchmarks, four stages: ADD, SEARCH, ANSWER, JUDGE.
Per-benchmark rules live in `benchmarks/adapters/<name>.py`.

> Also available in Chinese: [README.zh.md](README.zh.md).

## Quick start

A smoke test on LoCoMo, from a fresh clone. Run every command from the repository
root.

```bash
uv sync
uv run python -m nltk.downloader -d .venv/nltk_data punkt_tab stopwords
bash benchmarks/download_data.sh locomo
cp benchmarks/.env.example benchmarks/.env   # then fill in its REQUIRED lines:
                                             # OpenRouter key x3, DeepInfra key x2
DATASET=locomo bash benchmarks/reproduce.sh --smoke
# With only an OpenRouter key -- keyword search instead:
DATASET=locomo bash benchmarks/reproduce.sh --smoke --methods keyword
```

Which keys each retrieval method needs is in [Which keys you need](#which-keys-you-need).

Each step is explained below, with a check that tells you it worked. Before it starts
anything, the run checks the data and the keys and lists everything missing at once,
so a failed first attempt costs nothing.

**You need:**

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- `bash` and `curl` (macOS or Linux; on Windows use WSL2, see
  [docs/windows.md](../docs/windows.md))
- An [OpenRouter](https://openrouter.ai) key (extraction, answer and judge models).
  With it alone you can run `--methods keyword`.
- A [DeepInfra](https://deepinfra.com) key for EverOS's default embedding and rerank
  models. LoCoMo searches with `agentic`, which needs both; the other three benchmarks
  search with `llm_multiround`, which needs embedding only.

---

## 1. Install dependencies

```bash
uv sync
uv run python -m nltk.downloader -d .venv/nltk_data punkt_tab stopwords
```

LoCoMo indexes and searches with the NLTK tokenizer of the EverAlgo harness
(`EVEROS_TOKENIZER__NAME = "nltk"` in `configs/locomo.toml`), which needs the
NLTK data above; `make install-deps` fetches it too. A store indexed with another
tokenizer cannot be searched with this one, so reuse only stores built under the
same setting.

**Done when** `uv sync` exits without an error and `.venv/bin/python` exists.

## 2. Get the data

```bash
bash benchmarks/download_data.sh locomo   # or longmemeval | evermembench | subtlememory | all
```

The script fetches only the files the adapter reads, straight to the path the config
expects, and keeps files that are already there, so running it again finishes an
interrupted download. For EverMemBench it also runs the one-time conversion.

| Benchmark | Lands at | Download | Source |
|---|---|---|---|
| LoCoMo | `benchmarks/data/locomo10.json`, plus the gold audit `locomo-audit-errors.json` | 2.8 MB + 151 KB | [snap-research/locomo](https://github.com/snap-research/locomo), [dial481/locomo-audit](https://github.com/dial481/locomo-audit) |
| LongMemEval | `benchmarks/data/longmemeval_s.json` | 278 MB | [xiaowu0162/longmemeval](https://huggingface.co/datasets/xiaowu0162/longmemeval) |
| EverMemBench | `benchmarks/data/evermembench.json` | 10 files | [EverMind-AI/EverMemBench-Dynamic](https://huggingface.co/datasets/EverMind-AI/EverMemBench-Dynamic) |
| SubtleMemory | `benchmarks/data/subtlememory/persona_{0..9}/` | 20 files | [Yummytanmo/SubtleMemory](https://huggingface.co/datasets/Yummytanmo/SubtleMemory) |

EverMemBench also keeps its raw release in `benchmarks/data/raw/EverMemBench-Dynamic/`;
the adapter reads session names from it at run time, so leave it in place.

**Done when** the output contains `Data ready.`

Already have a copy elsewhere? Point `BENCH_DATA_LOCOMO`, `BENCH_DATA_LONGMEMEVAL`,
`BENCH_DATA_EVERMEMBENCH`, `BENCH_DATA_SUBTLEMEMORY` or `BENCH_DATA_LOCOMO_AUDIT` in
`benchmarks/.env` at it instead.

## 3. Configure

```bash
cp benchmarks/.env.example benchmarks/.env
```

Fill in the keys in the `REQUIRED` block at the top of the file. Everything else
works as shipped.

| Variable | Value | Used by |
|---|---|---|
| `EVEROS_LLM__API_KEY` | OpenRouter key | the EverOS servers: extraction (ADD) and the retrieval decider |
| `EVEROS_EMBEDDING__API_KEY` | DeepInfra key | the EverOS servers: embedding (Qwen3-Embedding-4B by default) -- every retrieval method except `keyword` |
| `EVEROS_RERANK__API_KEY` | DeepInfra key | the EverOS servers: rerank (Qwen3-Reranker-4B by default) -- `agentic`, LoCoMo's method; without it every agentic search fails |
| `ANSWER_API_KEY` | OpenRouter key | ANSWER |
| `JUDGE_API_KEY` | OpenRouter key | JUDGE |

### Which keys you need

EverOS needs only an LLM; embedding and rerank are optional providers that unlock more
retrieval methods (see [What works with one key?](../README.md#what-works-with-one-key)).
A benchmark therefore needs what its retrieval method needs -- `methods` in
`configs/<dataset>.toml` (`agentic` for LoCoMo, `llm_multiround` for the other three),
or `--methods` for one run:

| Retrieval method | OpenRouter key | DeepInfra key: embedding | DeepInfra key: rerank |
|---|---|---|---|
| `agentic` (LoCoMo) | yes | yes | yes |
| `llm_multiround` (the other three) | yes | yes | -- |
| `vector`, `hybrid` | yes | yes | -- |
| `keyword` (BM25 only) | yes | -- | -- |

Without an embedding provider EverOS also skips reflection and skill extraction during
ADD, so a `keyword` run measures a smaller pipeline, and its score is not comparable
to [6. Results](#6-results). The run checks exactly these requirements for the
methods you chose, before anything starts.

Models, `top_k`, retrieval settings and concurrency are set in
`benchmarks/configs/<dataset>.toml`, not in `.env`; the optional decider override below
is the only model setting `.env` takes. `reproduce.sh` passes no model overrides.
Strategy switches (`[ome_strategies]`, e.g. LoCoMo's profile off) are written as
each started server's `ome.toml`, where EverOS reads them. LoCoMo's config also files
every message under the queried speaker (`message_owner = "eval_owner"`), and a turn
that shares an image reaches the extractor with the image's caption in front of its
text; both decide what is stored, so both are part of the ingest identity.

`flush_scope` decides how much of a conversation goes into one EverOS session, flushed
once at its end: `"session"` (the default, every shipped config) flushes each dataset
session apart; `"conversation"` sends the whole conversation as one session, as the
EverAlgo reference harness did, so a memory may span two dataset sessions. It is
refused for datasets whose sessions carry their own ids.

**Optional -- a separate retrieval decider** (`llm_multiround` only). We recommend
the deciders behind the published numbers (the Decider column in
[6. Results](#6-results), and for LoCoMo [its methods table](#locomo-by-retrieval-method));
each config comes pre-filled with its recommended one. To
use it, set `BENCH_DECIDER_BASE_URL` to an OpenAI-compatible endpoint that serves it,
plus `BENCH_DECIDER_API_KEY` for a hosted API; set `BENCH_DECIDER_MODEL` as well only
to run a different model. Without an endpoint the extraction model decides, and the
run warns that it is not the published configuration. LoCoMo's decider is served by
OpenRouter: set `BENCH_DECIDER_BASE_URL=https://openrouter.ai/api/v1` and
`BENCH_DECIDER_API_KEY` to the OpenRouter key. Serving the others is outside this guide.

A reasoning model must decide without thinking. The decider's output cap is 512 tokens
unless `EVEROS_LLM__DECIDER__MAX_TOKENS` sets one, and a model that thinks can spend it
all before it answers; a round whose decider never answers falls back to a fixed core,
which is a different method. LoCoMo's configs turn thinking off for theirs
(`EVEROS_LLM__DECIDER__EXTRA` in `[retrieval_env]`, OpenRouter's spelling). A run whose
decider still falls back stops that conversation right after SEARCH, before paying for
its answers, and the message names the fix.

The decider reaches the servers a run starts as `EVEROS_LLM__DECIDER__MODEL` /
`__BASE_URL` / `__API_KEY` -- EverOS's `[llm.decider]`. Before the first question of
an `llm_multiround` run, `run.py` resolves the decider the way those servers do
(`[llm.decider]`, then the deprecated `[decider]`, then the decider's own defaults,
then `[llm]`) and sends it one request through the same client; a decider that does
not answer stops the run.

**Optional -- self-hosted embedding.** Set `EVEROS_EMBEDDING__BASE_URL` to your server,
`EVEROS_EMBEDDING__MODEL` to the model name it serves, and `EVEROS_EMBEDDING__API_KEY`
to any non-empty value. The model must return at least 1024 dimensions (EverOS's
`[embedding] dim`); a narrower one is refused when the server starts.

**Done when** the lines under `REQUIRED` in `benchmarks/.env` each have a value (a
provider key may stay empty when no method you run needs it). Step 4 checks them
again: a missing one stops the run with `Not started -- nothing has been spent yet`
and names the variable.

## 4. Run

Start with the smoke test:

```bash
DATASET=locomo bash benchmarks/reproduce.sh --smoke
```

A smoke run ingests the first 50 messages of conversation 0, answers up to 10 sampled
questions (LongMemEval has one question per conversation, so one) and judges each
once. It exercises every stage end to end; its score is not comparable to
[6. Results](#6-results), and `report.txt` says so on its first lines. Use it to check
the setup and to watch what one conversation costs with your provider before a full
run. The report counts answer and judge tokens only; extraction and embedding usage
shows on your providers' dashboards.

What a smoke score can tell you depends on the benchmark. On LoCoMo the sampled
questions mostly cite the first sessions (9 of 10 cite only the first 50 messages),
so the score is roughly meaningful. On LongMemEval, EverMemBench and SubtleMemory the
sampled questions' evidence lies past the first 50 messages, so expect a low smoke
score there; it shows the pipeline works, not how well it retrieves.

Before ADD, the run checks, in order:

1. the data path exists and every key its retrieval method needs is set -- before any
   server starts; a pass prints `Preflight: data and keys ready`;
2. each EverOS server it started reports the providers that method needs (embedding;
   rerank too for `agentic`) -- before ADD writes anything; a pass prints
   `Embedding answered on 1 server(s)` (and `Rerank configured ...`). A `keyword` run
   needs neither and skips this.

Either check failing stops the run with a message naming the fix. One line is expected
on a default smoke run and is not a problem: `--servers 10 exceeds 1 conversation(s);
using 1`.

**Done when** `benchmarks/results/LoCoMo/locomo_smoke/report.txt` exists. Results go to
`benchmarks/results/<Dataset>/<RUN>/`, where `<Dataset>` is `LoCoMo`, `LongMemEval`,
`EverMemBench` or `SubtleMemory`, and `RUN` is the dataset name for a full run,
`<dataset>_conv<CONV>` for a slice, with `_smoke` appended for a smoke run -- so a smoke
store never shares a directory with a real run. A results directory also refuses to
resume a run that ingests a different number of messages than the one that built it.

Then the full run:

```bash
DATASET=locomo bash benchmarks/reproduce.sh
```

| Variable | Default | Values |
|---|---|---|
| `DATASET` | `locomo` | `locomo` \| `locomo-step` \| `longmemeval` \| `subtlememory` \| `evermembench` |
| `CONV` | `all` (with `--smoke`: conversation 0) | conversation indices |
| `STAGES` | `add search answer judge` | any subset, in order |
| `RUN` | derived | result directory name |

Any `run.py` flag can follow the command; see [run.py flags](#runpy-flags).
`run.py`, which `reproduce.sh` calls, starts and stops its own EverOS servers: the
config's `servers`, capped at the number of conversations. Each is an independent
EverOS instance with its own store, and the conversations are sharded across them
(conversation `c` goes to server `c mod N`). EverOS itself runs one process per store,
so this is parallelism across separate stores, not one deployment scaled out.
`--servers N` changes how fast a run goes, not its result.
Several runs can go at once, on any mix of benchmarks: each picks free ports for its
own servers and accepts an answer only from a server it started (`/health` reports the
answering process's `pid`), so two runs never share a store.

`locomo-step` is LoCoMo with step boundary detection and prompt-only replies on every
EverOS call: `configs/locomo-step.toml` is `locomo.toml` with those two server settings
changed. See [Batch or step](#batch-or-step) for what each one costs and scores.

Without the wrapper, the same smoke test is:

```bash
uv run python benchmarks/run.py locomo --run-name locomo_smoke --smoke
```

`run.py` loads `benchmarks/.env` itself, before it reads the config, so the
`BENCH_*` variables there apply to a direct run too.

| Stage | Action |
|---|---|
| ADD | streams conversations into EverOS, which extracts memories |
| SEARCH | one retrieval per question; records the injected episodes |
| ANSWER | answers each question from those episodes |
| JUDGE | grades each answer against the reference; with a gold audit, also re-judges the disputed questions (see [Two scores per run](#two-scores-per-run)) |

Each stage appends per-conversation JSONL and skips entries already present, so
rerunning the same command resumes an interrupted run.

### Scoring another retrieval method on stores you built

A run's stores serve any retrieval method without ingesting again. Give the new run a
name of its own, skip ADD, and point it at the stores:

```bash
DATASET=locomo RUN=locomo_hybrid STAGES="search answer judge" \
  bash benchmarks/reproduce.sh --everos-root benchmarks/results/LoCoMo/locomo/store \
  --methods hybrid
```

With a fleet, `--everos-root` names the base and the run reads `store_s0` ..
`store_s<N-1>` next to it; search-only servers run with the cascade off, so the stores
are not changed. Use the `DATASET` the stores were built with (`locomo-step` for a step
store): a store from another results directory is not checked against the config. Do not
change `--methods` under the original run's name instead -- a results directory re-run
with other read settings sets its earlier search, answer and judge files aside as
`*.superseded.jsonl`, and its report then shows only the new method.

### When something stops the run

| Message | What to do |
|---|---|
| `benchmarks/.env not found.` | `cp benchmarks/.env.example benchmarks/.env`, then fill in the keys under `REQUIRED` |
| `.venv not found` | `uv sync` |
| `Not started -- nothing has been spent yet. Fix these first:` | each line names a missing dataset (with its download command) or an unset key |
| `The embedding service did not answer the startup probe` | check `EVEROS_EMBEDDING__BASE_URL` / `__MODEL` / `__API_KEY`, or run `--methods keyword`; `grep embedding_` on the server log it names shows the server's reason |
| `No rerank provider is configured` | set `EVEROS_RERANK__API_KEY`, or choose a method other than `agentic` |
| `decider ... did not answer` | the decider of an `llm_multiround` run refused the probe request; the message says why. See the decider option in [3. Configure](#3-configure) |
| `llm_multiround round(s) fell back to a fixed core` | the decider failed every attempt on those rounds; for a reasoning model, turn thinking off as the message says. Then search again under a new `RUN`: resuming the same one keeps the fallen-back searches |
| `decider  no endpoint for ...` | a warning, not an error (`llm_multiround` only): the run continues on the extraction model and its score is not comparable to [6. Results](#6-results) |
| `Refusing to start: these stores already hold memories` | delete the listed store for a clean run, or pass `--reuse-store` |
| `No SubtleMemory personas under ...` | `bash benchmarks/download_data.sh subtlememory` |

## 5. Output

```
benchmarks/results/<Dataset>/<run>/
├── report.txt            summary; its first lines flag a smoke or trimmed run
├── report.json           machine-readable summary
├── run_spec.json         models served, endpoints, package versions, knobs
├── store/                built by ADD with one server (each holds its server.log)
├── store_s0/ .. store_s<N-1>/   instead, with N servers: one store per server
├── conv<N>/
│   ├── add.done          marker: ADD finished for this conversation
│   ├── search_<method>.jsonl     e.g. search_agentic.jsonl
│   ├── answer_<method>.jsonl
│   ├── judge_<method>.jsonl
│   └── rejudge_<method>.jsonl    the disputed questions, judged against the audit
└── traces/               per-round retrieval traces
```

`run_spec.json` records the models actually served -- the decider as EverOS
resolves it, with each setting's effective value and the section it came from
(`extra`'s header and query values masked) -- the `everalgo` package versions, which
determine what a store contains, and whether the run was a smoke or trimmed one
(`smoke`, `max_messages`). The stores are absent here only when `--everos-root`
pointed somewhere else; the run prints their location as `store`.

### Two scores per run

`report.txt`, `report.json` and the terminal summary give each method two accuracies
side by side, overall and per category:

| Line | Grades | Use it for |
|---|---|---|
| `Accuracy` | every answer against the dataset's gold | comparing with other systems' published results, which use it |
| `Re-judged` | the same answers, with the questions a third-party audit disputes graded against the audit's corrected answers | the LoCoMo numbers in [6. Results](#6-results) (marked \*): a question with a wrong gold answer measures the dataset, not the system |

Re-judging changes only the gold: nothing is searched or answered again, and the
disputed answers are graded by the same judge model, the same number of times, with
the same majority vote. JUDGE writes them to `rejudge_<method>.jsonl` and skips the
ones already there; a run judged before the audit was downloaded gets them by
rerunning `--stages judge`, which judges only the disputed questions. Until every
disputed question has been re-judged, the report gives no `Re-judged` number and
says how many are missing.

For LoCoMo the audit is [dial481/locomo-audit](https://github.com/dial481/locomo-audit)
`errors.json`: 99 of the 1,540 questions have a wrong, ambiguous or unsupported gold
answer, each with a corrected one. Entries that only correct the evidence pointer
keep their gold and are not re-judged. `download_data.sh locomo` fetches the file
next to the dataset, pinned to the commit it was read at; like LoCoMo it is licensed
CC BY-NC 4.0. Without the file the run goes on and the report says it was not
re-judged. The other three benchmarks have no audit, and their reports say so on the
`Re-judged` line.

LoCoMo judges every question three times and grades by majority vote (`[judge] runs`
in `benchmarks/configs/<dataset>.toml`); `Mean accuracy` averages the passes, so a
gap between it and `Accuracy` is judge noise. `--smoke` keeps a single pass.

### Cleaning up

Everything a run writes is in its results directory, so deleting that directory
removes the stores, logs, traces and reports in one go. Outside it, a run leaves:

| What | Where | What to do |
|---|---|---|
| Downloaded datasets | `benchmarks/data/` | keep for the next run, or delete (keep `.gitkeep`) |
| Your keys | `benchmarks/.env` | delete when done, or keep it private |
| NLTK data | `.venv/nltk_data/` | goes with `.venv` |
| Stray servers | processes, only if `run.py` was killed with `kill -9` | find them, then stop them by PID |

`benchmarks/results/`, `benchmarks/data/` and `benchmarks/.env` are git-ignored.

```bash
rm -rf benchmarks/results/LoCoMo/locomo_smoke        # one run
rm -rf benchmarks/results                            # every run
find benchmarks/data -mindepth 1 ! -name .gitkeep -delete
rm -f benchmarks/.env
ps -eo pid,command | grep "[e]veros server start"    # stray servers, if any
```

## 6. Results

Each benchmark's published number. LoCoMo's comes from `benchmarks/configs/locomo-step.toml`, the
others from `benchmarks/configs/<dataset>.toml`.

| Benchmark | Questions | Accuracy | Decider | Answer model |
|---|---|---|---|---|
| LoCoMo | 1,540 | **94.61\*** | -- (`agentic` uses none) | `openai/gpt-4.1-mini` |
| LongMemEval | 500 | **94.00** | `qwen3.6-27B` | `google/gemini-3.6-flash` |
| EverMemBench | 2,400 | **66.67** | `qwen3.6-27B` | `google/gemini-3-flash-preview` |
| SubtleMemory | 1,522 | **71.75** | `qwen3.6-27B` | `openai/gpt-5.4` |

> **Accuracy** is a micro accuracy for every benchmark in this table:
> `sum(correct) / sum(total)` over all graded questions. The per-category lines in
> `report.txt` are reported, never averaged into it, so a category with few questions
> moves the headline less than a large one does. This note used to say EverMemBench
> scored as the mean of its nine category columns — a macro average, which is a
> different statistic and not what `run.py` computes.
>
> Two things follow, both worth knowing before comparing the number to anyone else's.
> Five of the six rows in the published EverMemBench comparison table are the macro
> mean of their own nine columns, so reading our figure alongside that column compares
> two different statistics. And the "nine categories" is not something this code can
> confirm: the adapter names three (`F_SH`, `F_MH`, `F_TP` in
> `adapters/evermembench.py`) and derives the rest from a question-id prefix, so any
> unlabelled id is reported under its raw key.
>
> **SubtleMemory** — routes each question between two answer contracts on whether
> fact extraction found a conflict; the adapter does this automatically.

\* Re-judged: the 99 questions whose gold answer the LoCoMo audit disputes are graded against
its corrected answers (see [Two scores per run](#two-scores-per-run)).

### LoCoMo by retrieval method

The best of three full runs of each LoCoMo config, then every other retrieval method searching
that run's stores, one run each (see
[Scoring another retrieval method on stores you built](#scoring-another-retrieval-method-on-stores-you-built)).
Measured with everalgo-core 0.7.0 and everalgo-boundary 0.5.0 at commit `ccb402af`, and
`llm_multiround` at `d278a2cb`, where the configs turn its decider's thinking off: extraction and
answers by `openai/gpt-4.1-mini`, judging by `openai/gpt-4o-mini` three times, and
`llm_multiround`'s decider `deepseek/deepseek-v4-flash-0731`.

| Retrieval method | `locomo-step` | `locomo` (default) |
|---|---|---|
| `agentic` | **94.61\*** | 94.35\* |
| `llm_multiround` | 94.29\* | 94.29\* |
| `hybrid` | 93.18\* | 92.08\* |
| `vector` | 92.73\* | 92.08\* |
| `keyword` | 91.04\* | 92.27\* |

\* Re-judged, as above.

### Batch or step

`locomo` cuts episodes with EverOS's default batch boundary detection. `locomo-step` asks once per
message whether it starts a new episode, and requests every EverOS reply in the prompt alone
(`response_format_mode = "prompt_only"`). EverOS keeps batch as its default:

| | `locomo` (batch, default) | `locomo-step` (step) |
|---|---|---|
| `agentic`, best of three runs | 94.35\* | 94.61\* |
| Boundary calls per run | 544 | about 5,250, one per message |
| ADD with 10 servers | about 4 minutes | about 17–22 minutes |
| Episodes stored | about 320 | about 550 |
| Agent conversations (tool calls) | yes | no: chat messages only |

With `agentic`, step scored 0.26 points higher at its best, which is less than LoCoMo's
run-to-run spread (each config's three runs spanned about one point). On the same stores the
other methods split, one run each: step was ahead with `hybrid` and `vector`, tied with
`llm_multiround` and behind with `keyword`. Step costs about ten times the boundary calls and
five times the ingest time. Choose it for chat memory when that cost is acceptable; keep batch
for agents and for fast ingest. The same choice in an EverOS deployment is
`[boundary_detection] strategy` ([docs/configuration.md](../docs/configuration.md)).

\* Re-judged, as above.


---

## run.py flags

| Flag | Meaning |
|---|---|
| `--conv` | conversation indices, or `all` |
| `--stages` | any of `add search answer judge` |
| `--everos-root` | store to use; default `<results>/<run>/store` (or the `store_s<i>` a fleet built) |
| `--servers` | independent EverOS instances to start, one store each, with conversations sharded across them; default from the config, capped at the number of conversations |
| `--results-root` | output root; default `benchmarks/results/<Dataset>` |
| `--data-path` | dataset path for one run |
| `--methods` | `agentic` \| `llm_multiround` \| `hybrid` \| `vector` \| `keyword` |
| `--answer-model` / `--judge-model` | model override for one run |
| `--decider-model` / `--decider-base-url` | decider for one run; the URL must serve the model |
| `--smoke` | conversation 0 unless `--conv`; first 50 messages and up to 10 sampled questions per conversation, one judge pass |
| `--max-messages` | ingest at most N messages per conversation, without sampling questions |
| `--reuse-store` | add into a store that already holds memories |
