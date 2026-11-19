# LoCoMo results

EverOS scores **94.61%** on LoCoMo, the best of three `agentic` runs with
`benchmarks/configs/locomo-step.toml`. This page breaks the score down by retrieval method,
compares batch and step boundary detection, and shows how the model of each stage changes
accuracy and write / search latency. To run LoCoMo yourself, start with
[README.md](README.md).

> Also available in Chinese: [locomo.zh.md](locomo.zh.md).

All scores on this page are re-judged: the 99 questions whose gold answer the LoCoMo audit
disputes are graded against its corrected answers; see [Two scores per run](README.md#two-scores-per-run).

## By retrieval method

| Retrieval method | Step boundary detection (`locomo-step`) | Batch boundary detection (`locomo`, default) | Search time, median |
|---|---|---|---|
| `agentic` | **94.61** | 94.35 | 6.7 s |
| `llm_multiround` | 94.29 | 94.29 |  |
| `hybrid` | 93.64 | 93.38 | 0.26 s |
| `vector` | 92.73 | 92.08 | 0.24 s |
| `keyword` | 91.04 | 92.27 | 0.01 s |

Best of three `agentic` runs per config; every other method searched that run's stores once.
Extraction and answers by `openai/gpt-4.1-mini`, judging by `openai/gpt-4o-mini` three times,
`llm_multiround`'s decider `deepseek/deepseek-v4-flash-0731` with thinking off; everalgo-core
0.7.0, everalgo-boundary 0.5.0. Search time is the median of 30 searches on the `locomo-step`
stores, one request at a time; `hybrid`, `vector` and `keyword` call no LLM
([Search latency](#search-latency)).

## Batch or step

The two LoCoMo configs differ in two EverOS settings, and EverOS ships the first pair as its
defaults:

| | `locomo` (default) | `locomo-step` |
|---|---|---|
| Boundary detection, `[boundary_detection] strategy` | `batch`: one call per add | `step`: one call per message |
| Reply format, `[llm] response_format_mode` | `json_schema`: the provider holds each reply to its schema | `prompt_only`: the prompt asks for the schema |
| `agentic`, best of three runs | 94.35 | **94.61** |
| Boundary calls per run | 544 | about 5,250 |
| ADD with 10 servers | about 4 minutes | about 17–22 minutes |
| Episodes stored | about 320 | about 550 |
| Agent conversations (tool calls) | supported | chat messages only |

- **`locomo` is faster and more stable.** Batch needs a tenth of step's boundary calls and
  ingests about five times faster. Strict replies always match the schema they were asked for;
  they need a provider that supports `json_schema`.
- **`locomo-step` scores higher.** Its best `agentic` run is 0.26 points above batch's, within
  LoCoMo's run-to-run spread of about one point. A prompt-only reply that does not parse is
  asked for again.

Choose step for chat memory when slower ingest is acceptable; keep the defaults for agents and
for fast ingest. In an EverOS deployment, see
[`[boundary_detection]`](../docs/configuration.md#boundary_detection) and
[`[llm]`](../docs/configuration.md#llm).

## Model configurations

The published LoCoMo score runs every EverOS call on `openai/gpt-4.1-mini`, the model memory
systems commonly report LoCoMo with. The configurations below change the model of one or more
stages, to show what a model choice does to accuracy and latency. Their scores do not
compare with published scores that use another model.
Each is `benchmarks/configs/locomo.toml` (batch boundary detection), except the published
`openai/gpt-4.1-mini` on `benchmarks/configs/locomo-step.toml`, with its models set in
`[llm]`, which every stage inherits, and in the
[`[llm.<task>]`](../docs/configuration.md#llmtask) sections `boundary`, `episode`, `summary`,
`atomic_fact` and `search`. Embedding and rerank are the published `Qwen/Qwen3-Embedding-4B`
and `Qwen/Qwen3-Reranker-4B` throughout.

### Accuracy

One model for every stage:

| Model | `agentic`, best of three | `hybrid`, best of three |
|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | **95.52** | 94.61 |
| `anthropic/claude-haiku-5.5` | 95.32 | **94.94** |
| `openai/gpt-4.1-mini` (published, `locomo-step`) | 94.61 | 93.64 |
| `google/gemini-2.5-flash-lite` | 94.29 | 93.25 |
| `qwen/qwen3-32b` | 93.38 | 92.99 |

A different model per stage:

| Boundary detection | Episode and summary | Atomic facts and `agentic` search | `agentic`, best of three | `hybrid`, best of three |
|---|---|---|---|---|
| `openai/gpt-5.4-mini` | `openai/gpt-5.4-mini` | `qwen/qwen3-235b-a22b-2507` | **95.78** | 94.22 |
| `qwen3-4b-instruct-2507`, self-hosted | `openai/gpt-5.4-mini` | `qwen/qwen3-235b-a22b-2507` | 94.68 | 93.64 |

Best of three full runs per configuration; `hybrid` searched each run's stores.
`openai/gpt-4.1-mini` is the published configuration, `benchmarks/configs/locomo-step.toml`: step
boundary detection and prompt-only replies, where every other configuration detects boundaries
in batch; its `hybrid` searched its best run's stores once. Answers and judging as in the
published runs; every run is in the [appendix](#appendix-full-results).

Most configurations sit within two points of each other, and three runs separate few of them.
Compared on `agentic` with a paired bootstrap over runs and questions, corrected for every pair
compared, four differences hold: `deepseek/deepseek-v4.1-flash-1017` above
`google/gemini-2.5-flash-lite` (+2.0 points on the mean of three runs) and above
`qwen/qwen3-32b` (+2.4), `anthropic/claude-haiku-5.5` above `qwen/qwen3-32b` (+2.2), and the
`openai/gpt-5.4-mini` per-stage configuration above `qwen/qwen3-32b` (+2.1). The two per-stage
configurations, which differ only in the boundary detection model, do not differ measurably
(0.5 points).

### Write latency

One `/memory/add` of a whole session followed by one `/memory/flush`, timed at the client. Each
cell is the median of three sessions, the first of LoCoMo conversations 0, 2 and 7 (16–18
messages each), each written to a fresh store on a freshly started server.

| Configuration | add | flush | add + flush | All extraction done |
|---|---|---|---|---|
| `google/gemini-2.5-flash-lite` | 1.1 s | 2.5 s | 3.6 s | 9.6 s |
| `deepseek/deepseek-v4.1-flash-1017` | 0.9 s | 3.1 s | 4.3 s | 10.3 s |
| `gpt-5.4-mini` boundary, episode and summary, `qwen3-235b` atomic facts | 1.6 s | 4.1 s | 5.7 s | 46.0 s |
| `qwen3-4b` boundary, `gpt-5.4-mini` episode and summary, `qwen3-235b` atomic facts | 1.4 s | 4.7 s | 6.0 s | 27.0 s |
| `anthropic/claude-haiku-5.5` | 2.3 s | 5.5 s | 7.8 s | 16.9 s |
| `qwen/qwen3-32b` | 1.5 s | 9.2 s | 10.9 s | 21.4 s |
| `openai/gpt-4.1-mini` (published, `locomo-step`) | 21.8 s | 4.2 s | 26.9 s | 38.9 s |

- Every column is its own median, so add and flush do not sum to add + flush.
- **All extraction done** runs from the add's start until every background extraction of the
  session has finished. The flush returns once the episode is written; summary and atomic
  facts are extracted after it returns, so their models change this column, not add or flush.
- Batch detection makes one boundary call per add; in these sessions the add only detected
  boundaries and the flush cut the episode. Step detection, which `openai/gpt-4.1-mini` runs,
  makes one call per message, about 1.3 s each, so one add of a whole session took about 22 s
  and could close an episode inside the add. An application that adds each message as it
  arrives waits for one boundary call per add, plus the episode call when that message closes
  an episode.
- Each session's times and the model time per call are in the
  [appendix](#write-latency-per-session).

### Search latency

Each configuration searched its own LoCoMo stores: 10 questions sampled from conversation 0,
three times each, one request at a time. `hybrid`, `vector` and `keyword` call no LLM and take
0.30 s or less at the median, whatever models the configuration uses; `agentic` adds LLM
calls, so its time follows the search model.

Median search time:

| Configuration | `hybrid` | `vector` | `keyword` | `agentic` |
|---|---|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | 0.29 s | 0.25 s | 0.01 s | 4.0 s |
| `google/gemini-2.5-flash-lite` | 0.26 s | 0.26 s | 0.01 s | 4.9 s |
| `openai/gpt-4.1-mini` (published, `locomo-step`) | 0.26 s | 0.24 s | 0.01 s | 6.7 s |
| `anthropic/claude-haiku-5.5` | 0.27 s | 0.28 s | 0.01 s | 8.5 s |
| `qwen3-4b` boundary; `gpt-5.4-mini` episode and summary; `qwen3-235b` atomic facts and search | 0.27 s | 0.25 s | 0.01 s | 12.5 s |
| `gpt-5.4-mini` boundary, episode and summary; `qwen3-235b` atomic facts and search | 0.30 s | 0.25 s | 0.01 s | 14.7 s |
| `qwen/qwen3-32b` | 0.27 s | 0.27 s | 0.01 s | 14.8 s |

- EverOS's own work in an `agentic` search took about 0.1 s; the rest is model calls. Each
  search makes at least one LLM call, to judge whether the recalled memories suffice; a second
  round adds an LLM call for follow-up queries and another rerank. The time of each kind of
  model call, the 90th percentiles and the share of second rounds are in the
  [appendix](#search-latency-in-detail).
- Almost all of a `hybrid` or `vector` search is the query embedding.

Latency was measured through OpenRouter for LLMs and DeepInfra for embedding and rerank, with
`qwen3-4b-instruct-2507` on a self-hosted endpoint; your providers and network change it. The
`gpt-5.4-mini`-boundary configuration was measured separately, when
`qwen/qwen3-235b-a22b-2507` took two to three times as long per call as in the
`qwen3-4b`-boundary measurements (34 s against 18 s per atomic-fact call), so its atomic-fact,
all-extraction-done and `agentic` times reflect that provider's speed at the time.

## Appendix: full results

Every number behind the tables above.

### Accuracy per run

Each cell reads original / re-judged: original grades every answer against the dataset's gold
answer, re-judged grades the 99 disputed questions against the audit's corrected answers
([Two scores per run](README.md#two-scores-per-run)). Each score is the majority of three judge
runs over all 1,540 questions. Bold marks the run the tables above report.

`agentic`, three full runs per configuration:

| Configuration | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| `openai/gpt-4.1-mini` (published, `locomo-step`) | 92.86 / 93.70 | 92.99 / 93.70 | 93.25 / **94.61** | 93.03 / 94.00 |
| `deepseek/deepseek-v4.1-flash-1017` | 94.61 / **95.52** | 94.68 / 95.26 | 93.70 / 95.00 | 94.33 / 95.26 |
| `anthropic/claude-haiku-5.5` | 94.55 / 95.26 | 94.55 / **95.32** | 93.77 / 94.74 | 94.29 / 95.11 |
| `google/gemini-2.5-flash-lite` | 91.36 / 92.60 | 93.51 / **94.29** | 92.08 / 92.99 | 92.32 / 93.29 |
| `qwen/qwen3-32b` | 91.30 / 92.40 | 92.40 / **93.38** | 92.34 / 92.92 | 92.01 / 92.90 |
| `gpt-5.4-mini` boundary, episode and summary; `qwen3-235b` atomic facts and search | 93.31 / 94.42 | 94.74 / **95.78** | 94.09 / 94.74 | 94.05 / 94.98 |
| `qwen3-4b` boundary; `gpt-5.4-mini` episode and summary; `qwen3-235b` atomic facts and search | 93.51 / 94.61 | 93.25 / 94.22 | 93.83 / **94.68** | 93.53 / 94.50 |

`hybrid`, searching each run's stores:

| Configuration | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | 93.44 / **94.61** | 92.86 / 93.90 | 93.83 / 94.48 | 93.38 / 94.33 |
| `anthropic/claude-haiku-5.5` | 93.57 / 94.61 | 93.90 / **94.94** | 93.70 / 94.68 | 93.72 / 94.74 |
| `google/gemini-2.5-flash-lite` | 91.82 / **93.25** | 92.08 / 93.05 | 91.82 / 92.86 | 91.90 / 93.05 |
| `qwen/qwen3-32b` | 91.69 / 92.47 | 91.75 / **92.99** | 91.75 / 92.66 | 91.73 / 92.71 |
| `gpt-5.4-mini` boundary, episode and summary; `qwen3-235b` atomic facts and search | 92.60 / 93.38 | 93.12 / **94.22** | 92.47 / 93.57 | 92.73 / 93.72 |
| `qwen3-4b` boundary; `gpt-5.4-mini` episode and summary; `qwen3-235b` atomic facts and search | 92.53 / 93.57 | 92.66 / **93.64** | 92.47 / 93.38 | 92.55 / 93.53 |

The other retrieval methods with `openai/gpt-4.1-mini`, each searching the stores of its
best `agentic` run (run 3) once:

| Retrieval method | Original / re-judged |
|---|---|
| `llm_multiround` | 93.77 / 94.29 |
| `hybrid` | 92.60 / 93.64 |
| `vector` | 91.75 / 92.73 |
| `keyword` | 89.74 / 91.04 |

### Write latency per session

Each cell reads add / flush / all extraction done, in seconds, for the first session of that
conversation.

| Configuration | Conversation 0 | Conversation 2 | Conversation 7 |
|---|---|---|---|
| `google/gemini-2.5-flash-lite` | 1.1 / 3.1 / 10.2 | 1.0 / 2.1 / 9.1 | 1.1 / 2.5 / 9.6 |
| `deepseek/deepseek-v4.1-flash-1017` | 0.9 / 3.4 / 10.3 | 0.9 / 1.9 / 8.8 | 7.0 / 3.1 / 16.1 |
| `gpt-5.4-mini` boundary, episode and summary, `qwen3-235b` atomic facts | 1.6 / 4.1 / 50.8 | 1.4 / 4.0 / 35.5 | 2.5 / 4.4 / 46.0 |
| `qwen3-4b` boundary, `gpt-5.4-mini` episode and summary, `qwen3-235b` atomic facts | 1.4 / 4.7 / 21.1 | 1.1 / 4.8 / 27.0 | 1.4 / 4.6 / 36.1 |
| `anthropic/claude-haiku-5.5` | 2.4 / 5.7 / 17.1 | 2.3 / 5.5 / 16.9 | 1.8 / 5.5 / 16.4 |
| `qwen/qwen3-32b` | 1.2 / 5.1 / 21.4 | 2.4 / 9.2 / 26.6 | 1.5 / 9.4 / 19.9 |
| `openai/gpt-4.1-mini` (published, `locomo-step`) | 21.8 / 5.2 / 45.0 | 19.2 / 3.1 / 34.4 | 22.7 / 4.2 / 38.9 |

Model time per call, median over every call in the three sessions. For batch detection the
boundary column leaves out the add's call: it is the first model call after the server
starts, and slower. Step detection makes one boundary call per message.

| Configuration | Boundary detection | Episode | Summary | Atomic facts |
|---|---|---|---|---|
| `google/gemini-2.5-flash-lite` | 0.7 s | 1.2 s | 1.1 s | 1.3 s |
| `deepseek/deepseek-v4.1-flash-1017` | 0.6 s | 1.4 s | 0.8 s | 1.9 s |
| `gpt-5.4-mini` boundary, episode and summary, `qwen3-235b` atomic facts | 1.2 s | 2.9 s | 1.6 s | 34.1 s |
| `qwen3-4b` boundary, `gpt-5.4-mini` episode and summary, `qwen3-235b` atomic facts | 0.8 s | 3.8 s | 1.3 s | 17.6 s |
| `anthropic/claude-haiku-5.5` | 1.2 s | 4.3 s | 2.0 s | 3.7 s |
| `qwen/qwen3-32b` | 0.8 s | 7.9 s | 3.2 s | 9.4 s |
| `openai/gpt-4.1-mini` (published, `locomo-step`) | 1.3 s | 3.1 s | 1.4 s | 3.8 s |

### Search latency in detail

`agentic`, 30 searches per configuration. LLM, rerank, embedding and EverOS (the time outside
model calls) are separate medians and do not sum to the total.

| `agentic` search model | Median | Mean | 90th percentile | LLM | Rerank | Embedding | EverOS | Second round |
|---|---|---|---|---|---|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | 4.0 s | 5.1 s | 8.7 s | 1.2 s | 1.6 s | 0.8 s | 0.1 s | 12 of 30 |
| `google/gemini-2.5-flash-lite` | 4.9 s | 4.7 s | 6.8 s | 1.8 s | 1.9 s | 0.5 s | 0.1 s | 15 of 30 |
| `openai/gpt-4.1-mini` (published, `locomo-step`) | 6.7 s | 6.4 s | 9.1 s | 3.8 s | 1.9 s | 0.8 s | 0.1 s | 12 of 30 |
| `anthropic/claude-haiku-5.5` | 8.5 s | 8.2 s | 11.2 s | 4.7 s | 2.3 s | 1.2 s | 0.1 s | 20 of 30 |
| `qwen/qwen3-235b-a22b-2507`, `qwen3-4b`-boundary stores | 12.5 s | 12.4 s | 19.1 s | 8.2 s | 2.3 s | 1.0 s | 0.1 s | 18 of 30 |
| `qwen/qwen3-235b-a22b-2507`, `gpt-5.4-mini`-boundary stores | 14.7 s | 15.3 s | 26.5 s | 11.6 s | 1.7 s | 1.0 s | 0.1 s | 13 of 30 |
| `qwen/qwen3-32b` | 14.8 s | 12.7 s | 17.1 s | 10.4 s | 2.5 s | 1.0 s | 0.1 s | 21 of 30 |

`hybrid`, `vector` and `keyword`, median / 90th percentile in seconds:

| Configuration | `hybrid` | `vector` | `keyword` |
|---|---|---|---|
| `deepseek/deepseek-v4.1-flash-1017` | 0.29 / 0.66 | 0.25 / 1.48 | 0.01 / 0.01 |
| `google/gemini-2.5-flash-lite` | 0.26 / 0.63 | 0.26 / 0.54 | 0.01 / 0.01 |
| `openai/gpt-4.1-mini` (published, `locomo-step`) | 0.26 / 0.58 | 0.24 / 0.40 | 0.01 / 0.01 |
| `anthropic/claude-haiku-5.5` | 0.27 / 0.53 | 0.28 / 0.69 | 0.01 / 0.01 |
| `qwen3-4b` boundary; `gpt-5.4-mini` episode and summary; `qwen3-235b` atomic facts and search | 0.27 / 0.61 | 0.25 / 0.56 | 0.01 / 0.01 |
| `gpt-5.4-mini` boundary, episode and summary; `qwen3-235b` atomic facts and search | 0.30 / 1.97 | 0.25 / 3.23 | 0.01 / 0.01 |
| `qwen/qwen3-32b` | 0.27 / 0.63 | 0.27 / 1.20 | 0.01 / 0.01 |
