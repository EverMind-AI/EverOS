# LoCoMo results

EverOS scores **94.61%** on LoCoMo, the best of three `agentic` runs with
`benchmarks/configs/locomo-step.toml`. Across the models below, `hybrid` calls no LLM and returns
in 0.27 s at the median while scoring up to 94.94; `agentic` scores up to
95.78, and with `deepseek-v4.1-flash` returns in 4.0 s at 95.52.
The sections after the table compare batch and step boundary detection and break down write and
search latency. To run LoCoMo yourself, start with [README.md](README.md).

> Also available in Chinese: [locomo.zh.md](locomo.zh.md).

All scores on this page are re-judged: the 99 questions whose gold answer the LoCoMo audit
disputes are graded against its corrected answers; see [Two scores per run](README.md#two-scores-per-run).

## Results by model and retrieval method

Accuracy for each retrieval method (rows) and model configuration (columns). `hybrid`, `vector`
and `keyword` call no LLM, so every model shares one search time; `agentic`'s search time follows
its model and stands in brackets.

| Retrieval method | Search time, median | `gpt-4.1-mini` (published) | `deepseek-v4.1-flash` | `claude-haiku-5.5` | `gemini-2.5-flash-lite` | `qwen3-32b` | `gpt-5.4-mini` + `qwen3-235b` | `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` |
|---|---|---|---|---|---|---|---|---|
| `agentic` | by model, in brackets | 94.61 (6.7 s) | 95.52 (4.0 s) | 95.32 (8.5 s) | 94.29 (4.9 s) | 93.38 (14.8 s) | **95.78** (14.7 s†) | 94.68 (12.5 s) |
| `hybrid` | 0.27 s | 93.64 | 94.61 | **94.94** | 93.25 | 92.99 | 94.22 | 93.64 |
| `vector` | 0.25 s | 92.73 | **94.22** | 93.64 | 92.60 | 92.73 |  |  |
| `keyword` | 0.01 s | 91.04 | **92.86** | 92.34 | 91.43 | 91.17 |  |  |

† Measured separately, when `qwen/qwen3-235b-a22b-2507` took two to three times as long per call as in the `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` measurements (34 s against 18 s per atomic-fact call); the time reflects that provider's speed then.

What each column runs:

| Column | Models |
|---|---|
| `gpt-4.1-mini` (published) | `openai/gpt-4.1-mini` for every stage |
| `deepseek-v4.1-flash` | `deepseek/deepseek-v4.1-flash-1017` for every stage |
| `claude-haiku-5.5` | `anthropic/claude-haiku-5.5` for every stage |
| `gemini-2.5-flash-lite` | `google/gemini-2.5-flash-lite` for every stage |
| `qwen3-32b` | `qwen/qwen3-32b` for every stage |
| `gpt-5.4-mini` + `qwen3-235b` | boundary detection `openai/gpt-5.4-mini`; episode and summary `openai/gpt-5.4-mini`; atomic facts and `agentic` search `qwen/qwen3-235b-a22b-2507` |
| `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` | boundary detection `qwen3-4b-instruct-2507`, self-hosted; episode and summary `openai/gpt-5.4-mini`; atomic facts and `agentic` search `qwen/qwen3-235b-a22b-2507` |

What each retrieval method does:

| Retrieval method | What it does | Calls an LLM |
|---|---|---|
| `agentic` | Recalls memories, reranks them with a cross-encoder and has the LLM judge whether they answer the question; if not, the LLM writes follow-up queries for a second round. | yes |
| `hybrid` | Fuses keyword and vector rankings, plus a lane of atomic facts for episodes. | no |
| `vector` | Semantic match on embeddings only. | no |
| `keyword` | BM25 full-text match only. | no |

- **Bold** marks the highest score in a row. Its gap to the others in the row is not
  necessarily real: the differences that hold statistically are in the paragraph that ends
  this section.
- **Accuracy.** `agentic` and `hybrid`: best of three full runs, except `openai/gpt-4.1-mini`'s
  `hybrid`, which searched its best run's stores once. `vector` and `keyword`: one search of one
  run's stores; the per-stage columns were not measured with them. Every column answers with
  `openai/gpt-4.1-mini` and judges with `openai/gpt-4o-mini` three times; every run is in the
  [appendix](#appendix-full-results).
- **Search time.** Median of single requests, one at a time: 10 questions from conversation 0,
  three times each, on each configuration's own stores. The `hybrid`, `vector` and `keyword`
  times pool every configuration's searches ([Search latency](#search-latency)).
- **Configurations.** `gpt-4.1-mini` is the published configuration,
  `benchmarks/configs/locomo-step.toml`: step boundary detection and prompt-only replies. Every
  other column is `benchmarks/configs/locomo.toml`, with batch boundary detection, and its models
  set in `[llm]`, which every stage inherits, and in the
  [`[llm.<task>]`](../docs/configuration.md#llmtask) sections `boundary`, `episode`, `summary`,
  `atomic_fact` and `search`. Embedding and rerank are `Qwen/Qwen3-Embedding-4B` and
  `Qwen/Qwen3-Reranker-4B` throughout. `openai/gpt-4.1-mini` is the model memory systems
  commonly report LoCoMo with; scores with another model do not compare with those.

Most configurations sit within two points of each other, and three runs separate few of them.
Four differences hold in a statistical test (a paired bootstrap over runs and questions,
corrected for every pair compared): `deepseek-v4.1-flash` above `gemini-2.5-flash-lite`
(+2.0 points on the mean of three runs) and above `qwen3-32b` (+2.4), `claude-haiku-5.5` above
`qwen3-32b` (+2.2), and `gpt-5.4-mini` + `qwen3-235b` above `qwen3-32b` (+2.1). The two per-stage
columns, which differ only in the boundary detection model, do not differ measurably (0.5 points).

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

Every retrieval method on the two configs' stores:

| Retrieval method | Step boundary detection (`locomo-step`) | Batch boundary detection (`locomo`, default) |
|---|---|---|
| `agentic` | **94.61** | 94.35 |
| `llm_multiround` | 94.29 | 94.29 |
| `hybrid` | **93.64** | 93.38 |
| `vector` | **92.73** | 92.08 |
| `keyword` | 91.04 | **92.27** |

Bold marks the higher score of the two. Best of three `agentic` runs per config; every other
method searched that run's stores once. `llm_multiround` lets an LLM decider run several
retrieval rounds; its decider is `deepseek/deepseek-v4-flash-0731` with thinking off.
everalgo-core 0.7.0, everalgo-boundary 0.5.0.

## Write latency

A write runs four model stages. Boundary detection decides where a stretch of conversation ends;
episode extraction writes that stretch up as one episode, and the request returns once it is on
disk; the summary and the atomic facts (single-sentence facts) are then extracted from the
episode in the background.

Each configuration wrote three sessions, the first of LoCoMo conversations 0, 2 and 7 (16–18
messages each), each to a fresh store on a freshly started server: one `/memory/add` of the whole
session and one `/memory/flush`, timed at the client. Each cell is the median of the three.

| Configuration | add | flush | add + flush | All extraction done |
|---|---|---|---|---|
| `deepseek-v4.1-flash` | 0.9 s | 3.1 s | 4.3 s | 10.3 s |
| `claude-haiku-5.5` | 2.3 s | 5.5 s | 7.8 s | 16.9 s |
| `gemini-2.5-flash-lite` | 1.1 s | 2.5 s | 3.6 s | 9.6 s |
| `qwen3-32b` | 1.5 s | 9.2 s | 10.9 s | 21.4 s |
| `gpt-5.4-mini` + `qwen3-235b` | 1.6 s | 4.1 s | 5.7 s | 46.0 s† |
| `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` | 1.4 s | 4.7 s | 6.0 s | 27.0 s |

† Measured separately, when `qwen/qwen3-235b-a22b-2507` took two to three times as long per call as in the `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` measurements (34 s against 18 s per atomic-fact call); the time reflects that provider's speed then.

- Every column is its own median, so add and flush do not sum to add + flush.
- **add** detects boundaries: with batch detection, one model call per add. In these sessions
  the flush cut the episode, so **flush** is mostly the episode call.
- **All extraction done** runs from the add's start until the summary and the atomic facts are
  written too. Their models change only this column.

The published configuration detects boundaries per message, one model call each, so its add is
timed per message:

| Configuration | add per message: one boundary call | flush | All extraction done |
|---|---|---|---|
| `gpt-4.1-mini` (published) | 1.3 s | 4.2 s | 38.9 s |

An add of one message waits for one boundary call, plus the episode call (3.1 s) when that
message closes an episode. Its flush and all-extraction-done times are for a whole session; an
add of the whole session at once took 22 s, one boundary call per message.

Each session's times and the model time per call are in the [appendix](#write-latency-per-session).
LLMs ran through OpenRouter, embedding and rerank through DeepInfra, and `qwen3-4b-instruct-2507`
on a self-hosted endpoint; other providers and networks give other times.

## Search latency

Each configuration searched its own LoCoMo stores: 10 questions sampled from conversation 0,
three times each, one request at a time, through the same providers as the writes. The results
are in the [table above](#results-by-model-and-retrieval-method).

- `hybrid`, `vector` and `keyword` call no LLM, so their time does not depend on the models.
  Pooled over every configuration, 210 searches each, their medians were 0.27 s, 0.25 s and
  0.01 s, almost all of it the query embedding for the first two.
- EverOS's own work in an `agentic` search took about 0.1 s; the rest is model calls. Each
  search makes at least one LLM call, to judge whether the recalled memories suffice; a second
  round adds an LLM call for follow-up queries and another rerank. The time of each kind of
  model call, the 90th percentiles and the share of second rounds are in the
  [appendix](#search-latency-in-detail).

## Appendix: full results

Every number behind the results table and the latency sections, with the results table's
columns.

### Accuracy per run

Each cell reads original / re-judged: original grades every answer against the dataset's gold
answer, re-judged grades the 99 disputed questions against the audit's corrected answers
([Two scores per run](README.md#two-scores-per-run)). Each score is the majority of three judge
runs over all 1,540 questions. Bold marks the run the tables above report.

`agentic`, three full runs per configuration:

| Configuration | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| `gpt-4.1-mini` (published) | 92.86 / 93.70 | 92.99 / 93.70 | 93.25 / **94.61** | 93.03 / 94.00 |
| `deepseek-v4.1-flash` | 94.61 / **95.52** | 94.68 / 95.26 | 93.70 / 95.00 | 94.33 / 95.26 |
| `claude-haiku-5.5` | 94.55 / 95.26 | 94.55 / **95.32** | 93.77 / 94.74 | 94.29 / 95.11 |
| `gemini-2.5-flash-lite` | 91.36 / 92.60 | 93.51 / **94.29** | 92.08 / 92.99 | 92.32 / 93.29 |
| `qwen3-32b` | 91.30 / 92.40 | 92.40 / **93.38** | 92.34 / 92.92 | 92.01 / 92.90 |
| `gpt-5.4-mini` + `qwen3-235b` | 93.31 / 94.42 | 94.74 / **95.78** | 94.09 / 94.74 | 94.05 / 94.98 |
| `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` | 93.51 / 94.61 | 93.25 / 94.22 | 93.83 / **94.68** | 93.53 / 94.50 |

`hybrid`, searching each run's stores:

| Configuration | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| `deepseek-v4.1-flash` | 93.44 / **94.61** | 92.86 / 93.90 | 93.83 / 94.48 | 93.38 / 94.33 |
| `claude-haiku-5.5` | 93.57 / 94.61 | 93.90 / **94.94** | 93.70 / 94.68 | 93.72 / 94.74 |
| `gemini-2.5-flash-lite` | 91.82 / **93.25** | 92.08 / 93.05 | 91.82 / 92.86 | 91.90 / 93.05 |
| `qwen3-32b` | 91.69 / 92.47 | 91.75 / **92.99** | 91.75 / 92.66 | 91.73 / 92.71 |
| `gpt-5.4-mini` + `qwen3-235b` | 92.60 / 93.38 | 93.12 / **94.22** | 92.47 / 93.57 | 92.73 / 93.72 |
| `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` | 92.53 / 93.57 | 92.66 / **93.64** | 92.47 / 93.38 | 92.55 / 93.53 |

`vector` and `keyword`, one search of one run's stores:

| Configuration | Stores of | `vector`, original / re-judged | `keyword`, original / re-judged |
|---|---|---|---|
| `gpt-4.1-mini` (published) | run 3 | 91.75 / 92.73 | 89.74 / 91.04 |
| `deepseek-v4.1-flash` | run 1 | 93.57 / 94.22 | 91.62 / 92.86 |
| `claude-haiku-5.5` | run 2 | 92.99 / 93.64 | 91.43 / 92.34 |
| `gemini-2.5-flash-lite` | run 2 | 91.69 / 92.60 | 90.32 / 91.43 |
| `qwen3-32b` | run 2 | 91.82 / 92.73 | 89.81 / 91.17 |
| `gpt-5.4-mini` + `qwen3-235b` |  |  |  |
| `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` |  |  |  |

### Write latency per session

Each cell reads add / flush / all extraction done, in seconds, for the first session of that
conversation. `gpt-4.1-mini` added each whole session at once, one boundary call per message.

| Configuration | Conversation 0 | Conversation 2 | Conversation 7 |
|---|---|---|---|
| `gpt-4.1-mini` (published) | 21.8 / 5.2 / 45.0 | 19.2 / 3.1 / 34.4 | 22.7 / 4.2 / 38.9 |
| `deepseek-v4.1-flash` | 0.9 / 3.4 / 10.3 | 0.9 / 1.9 / 8.8 | 7.0 / 3.1 / 16.1 |
| `claude-haiku-5.5` | 2.4 / 5.7 / 17.1 | 2.3 / 5.5 / 16.9 | 1.8 / 5.5 / 16.4 |
| `gemini-2.5-flash-lite` | 1.1 / 3.1 / 10.2 | 1.0 / 2.1 / 9.1 | 1.1 / 2.5 / 9.6 |
| `qwen3-32b` | 1.2 / 5.1 / 21.4 | 2.4 / 9.2 / 26.6 | 1.5 / 9.4 / 19.9 |
| `gpt-5.4-mini` + `qwen3-235b` | 1.6 / 4.1 / 50.8 | 1.4 / 4.0 / 35.5 | 2.5 / 4.4 / 46.0 |
| `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` | 1.4 / 4.7 / 21.1 | 1.1 / 4.8 / 27.0 | 1.4 / 4.6 / 36.1 |

Model time per call, median over every call in the three sessions. For batch detection the
boundary column leaves out the add's call: it is the first model call after the server starts,
and slower. Step detection makes one boundary call per message.

| Configuration | Boundary detection | Episode | Summary | Atomic facts |
|---|---|---|---|---|
| `gpt-4.1-mini` (published) | 1.3 s | 3.1 s | 1.4 s | 3.8 s |
| `deepseek-v4.1-flash` | 0.6 s | 1.4 s | 0.8 s | 1.9 s |
| `claude-haiku-5.5` | 1.2 s | 4.3 s | 2.0 s | 3.7 s |
| `gemini-2.5-flash-lite` | 0.7 s | 1.2 s | 1.1 s | 1.3 s |
| `qwen3-32b` | 0.8 s | 7.9 s | 3.2 s | 9.4 s |
| `gpt-5.4-mini` + `qwen3-235b` | 1.2 s | 2.9 s | 1.6 s | 34.1 s† |
| `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` | 0.8 s | 3.8 s | 1.3 s | 17.6 s |

† Measured while `qwen/qwen3-235b-a22b-2507` answered slowly; see the note under the first table.

### Search latency in detail

`agentic`, 30 searches per configuration. LLM, rerank, embedding and EverOS (the time outside
model calls) are separate medians and do not sum to the total.

| Configuration | Median | Mean | 90th percentile | LLM | Rerank | Embedding | EverOS | Second round |
|---|---|---|---|---|---|---|---|---|
| `gpt-4.1-mini` (published) | 6.7 s | 6.4 s | 9.1 s | 3.8 s | 1.9 s | 0.8 s | 0.1 s | 12 of 30 |
| `deepseek-v4.1-flash` | 4.0 s | 5.1 s | 8.7 s | 1.2 s | 1.6 s | 0.8 s | 0.1 s | 12 of 30 |
| `claude-haiku-5.5` | 8.5 s | 8.2 s | 11.2 s | 4.7 s | 2.3 s | 1.2 s | 0.1 s | 20 of 30 |
| `gemini-2.5-flash-lite` | 4.9 s | 4.7 s | 6.8 s | 1.8 s | 1.9 s | 0.5 s | 0.1 s | 15 of 30 |
| `qwen3-32b` | 14.8 s | 12.7 s | 17.1 s | 10.4 s | 2.5 s | 1.0 s | 0.1 s | 21 of 30 |
| `gpt-5.4-mini` + `qwen3-235b`† | 14.7 s | 15.3 s | 26.5 s | 11.6 s | 1.7 s | 1.0 s | 0.1 s | 13 of 30 |
| `qwen3-4b` + `gpt-5.4-mini` + `qwen3-235b` | 12.5 s | 12.4 s | 19.1 s | 8.2 s | 2.3 s | 1.0 s | 0.1 s | 18 of 30 |

† Measured while `qwen/qwen3-235b-a22b-2507` answered slowly; see the note under the first table.

`hybrid`, `vector` and `keyword` call no LLM, so every configuration's searches are pooled:

| Retrieval method | Median | 90th percentile | Searches |
|---|---|---|---|
| `hybrid` | 0.27 s | 0.65 s | 210 |
| `vector` | 0.25 s | 1.11 s | 210 |
| `keyword` | 0.01 s | 0.01 s | 210 |
