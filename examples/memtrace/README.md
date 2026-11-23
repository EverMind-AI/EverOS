# MemTrace graph attribution for EverOS

This example ports the **core MemTrace graph-attribution algorithm** to an
independent package under `graphtrace/`. It consumes existing SmartComment
execution graphs and diagnoses a failed question-answering case. It does not
require a running EverOS server or change the server's instrumentation.

MemTrace is an LLM-driven graph exploration algorithm. BM25 selects candidate
source messages; the model uses graph tools to inspect operations and follows
the evidence through memory construction, preservation, retrieval and response
generation. The goal is the **earliest locally faulty, decisive operation**:
correcting that operation, with earlier operations unchanged and ideal downstream
behavior, should prevent the wrong answer.

## Setup

Run commands from the EverOS checkout. Use its existing environment, or install
the locked environment with the existing tracing extra:

```bash
uv sync --frozen --extra memtrace
```

Tracing is off by default: start the server with `EVEROS_MEMTRACE_ENABLED=1`
to record graphs. See [the cost and privacy notes](../../docs/memtrace-graphs.md#enabling-tracing)
before enabling it anywhere but a test root.

The runtime uses `smartcomment`, `openai`, Pydantic and the standard library.
It does not import the sibling MemTrace checkout, install AgentScope, or require
`rank_bm25`, `networkx`, an embedding endpoint, or a vector database.

## List and attribute failed cases

For newly collected EverOS traces, stop the server, inject the QA records, and
merge construction and search exports with the tenant-aware merger:

```bash
.venv/bin/python examples/memtrace/inject_qa.py \
  --graph-dir /path/to/memtrace --qa-results /path/to/qa_results.json
.venv/bin/python examples/memtrace/merge.py /path/to/memtrace \
  --output /path/to/merged.json
.venv/bin/python examples/memtrace/attribute.py \
  --graph /path/to/merged.json --list-cases
```

The merger namespaces identities by app/project and rejects conflicting values
for the same immutable revision. Search exports preserve canonical construction
revision numbers, even when their first observed revision is `@2`. Merge both
construction and search graphs to retain the revision's producing operations.
Older exports with conflicting local versions need a fresh replay; the merger
does not guess which content was used. Graphs flagged with unresolved merge
conflicts are rejected by the attribution CLI.

The existing local LoCoMo trace can be inspected without credentials:

```bash
.venv/bin/python examples/memtrace/attribute.py \
  --graph ../MemTrace/outputs/everos_locomo_adapt/merged/merged.json \
  --list-cases
```

Run one case with the already-existing MemTrace credential configuration:

```bash
.venv/bin/python examples/memtrace/attribute.py \
  --graph ../MemTrace/outputs/everos_locomo_adapt/merged/merged.json \
  --query-id query-caroline-cfa8035aae4845cda7729e6dc4e00877 \
  --api-config ../MemTrace/input_files/api_config.json \
  --model qwen3.8-flash \
  --output ../MemTrace/outputs/everos_graphtrace_core/caroline_qwen3_8_flash.json
```

`--query-id` accepts the exact versioned node identifier, the unversioned full
name, or the query identity shown above. Without it, one case is selected in
judge-operation creation order. The input graph is never rewritten. The output
must differ from the input graph and credential file.

Alternatively, omit `--api-config` and set `OPENAI_API_KEY`, optionally
`OPENAI_BASE_URL` and `OPENAI_MODEL`. A configuration file contains paired,
nonempty `api_keys` and `base_urls` lists; this single-case runner uses the first
pair. Credentials are only given to the client and never included in reports.
Provider exception text is also excluded from reports.

In the local workspace, `all_proxy` points to a SOCKS proxy but `socksio` is not
installed. If the OpenAI client fails at startup with that import error, prefix
the command with `env -u all_proxy`. This retains the configured HTTP/HTTPS
proxies; no package installation is necessary.

### Defaults and limits

| Option | Default | Meaning |
| --- | --- | --- |
| `--model` | `qwen3.8-flash` | Model, unless `OPENAI_MODEL` is set |
| `--max-iters` | `200` | Maximum model requests per case |
| `--max-trace-nodes` | `16` | Queued variables, excluding the active node |
| `--num-starting-points` | `8` | BM25 candidate messages, capped to leave query space |
| `--context-chars` | `100000` | Serialized messages and tools character budget |

Tool pages default to 16,000 characters. Offsets
are one-based character positions. A truncated result gives the next offset.
Previews, value reads and exploration history all paginate, including previews
that omit values. The agent reduces the page cap to fit its character budget and
the number of tool calls in the current batch. Continue through pages to inspect
a large operation, or use previews followed by regex value searches.

The character budget is a lightweight estimate, not the provider's token count.
Oldest complete assistant/tool-response groups are removed when necessary.
System instructions, the case, the fresh notebook hint and the newest turn are
retained. If these alone exceed the budget, the run fails explicitly instead
of breaking tool-call pairing. The transcript retains all removed turns.

## Workflow and tools

1. Import the graph and inspect `judge_answer_*` evaluation operations.
   Decode the query, prediction, reference-answer list and judge snapshots from
   the same subgraph. Only complete cases with score `0.0` are selected.
   Malformed or mismatched QA chains produce diagnostics. Native `llm-judge`
   chains with a `judge_score` node are also supported.
2. Rank raw `message` content by BM25Okapi using question plus reference answer.
   Initialize with the query and candidate messages. These are **pseudo evidence**,
   not human-labeled source evidence. If there are no message candidates, initialize
   with a synthetic root and let the model select graph roots.
3. Before each model request, provide the current active node, ordered frontier
   and the last four completed outcomes with bounded summaries. Full history
   stays available through a paginated tool. Execute tool calls serially, even if
   the model emits several calls in one response. Tool errors let the model correct
   invalid actions.
4. Follow locally correct operations downstream. Inspect input/produced variables,
   direct successors, final output variables, versioned values and observation
   ordering. Finish graph exploration, then submit one structured attribution.

| Tool | Purpose |
| --- | --- |
| `initialize_execution_graph` | Initialize selected nodes or a synthetic root; restart after finishing |
| `explore_graph_trace_node` | Inspect operations producing or consuming the active/earliest variable |
| `update_to_explore_nodes` | Add/delete queued nodes; record discovery provenance |
| `view_node_value` | Read a value or search regex matches with character pagination |
| `finish_node_exploration` | Archive the active node and its findings |
| `finish_graph_exploration` | Archive graph state, or stop because evidence is insufficient |
| `view_exploration_history` | Read completed exploration records |
| `submit_attribution` | Validate and return one inspected operation's fault |

Queued nodes use `(created_at, full_node_id)` ordering; adding an older node does
not displace the active node. Recorded operations use unique `event_seq` values;
offline QA operations without sequences follow those observations. Duplicate
sequences trigger timestamp ordering instead. Sequence values from independent
processes do not establish a reliable global execution order. File-diff timestamps
are observation times, not actual extraction/update execution times.

An operation's structured rendering must have been read, possibly across pages,
before submission. The reason must cite a relevant versioned node whose value
was read. Judge and reference-injection operations are excluded. Exploration and
source-message selection respect the failed query's tenant and exclude later
construction and other requests' retrieval/answer operations.
The result schema is the original `{error_type, op_id, reason}`, with these types:
`ExtractionError`, `UpdateError`, `DeletionError`, `RetrievalError`, `ResponseError`.
Schema and graph membership checks do not establish that an LLM diagnosis is correct.

When critical source evidence is absent, the model can call
`finish_graph_exploration(insufficient_evidence=True, outcome=...)`. This returns
an `InsufficientEvidence` failure with no fabricated attribution. API failures,
context overflow and iteration exhaustion likewise retain the trace and state.

Search `context` and actual `answer_context` are separate snapshots. The latter
must be supplied by the caller; LoCoMo QA records its exact episode-only context.
An omitted answer context stays `null` and cannot establish a `ResponseError`.
File-diff operations carry `observation_only=true` and cannot be submitted as
executed faults. Boundary and episode extraction attempts/results record actual
values, including empty extractor results. Background OME updates/deletions are
still file observations; the tracer cannot reconstruct intermediate states that
never reached a scan. In that situation, stop with `InsufficientEvidence`.

## Reports and Python API

The output contains `status`, decoded `case`, `graph_id`, `model`, runtime `config`,
`pseudo_evidence_nodes`, `attribution`, `failure`, `iterations`, token `usage`,
`inspected_operations`, `final_state`, `diagnostics` and the complete `transcript`.
The transcript includes initialization, fresh hints, model/tool messages and
context-trimming events. Reports contain conversation data and should stay in
local output directories.

Exit codes: `0` for a valid attribution or successful case listing, `1` for a run
without attribution (including insufficient evidence), and `2` for invalid input
or setup. An insufficient-evidence result is not an accuracy score.

```python
from examples.memtrace.graphtrace import (
    AttributionConfig,
    OpenAIChatProvider,
    attribute,
    collect_failed_queries,
)

cases, diagnostics = collect_failed_queries(graph)
report = await attribute(
    graph,
    cases[0],
    OpenAIChatProvider(async_openai_client, "qwen3.8-flash"),
    AttributionConfig(max_iters=200),
)
```

The public facade also exports `FailedQueryCase`, `GraphTraceState`,
`GraphTraceNotebook`, `AttributionResult` and the injectable `ChatProvider` protocol.
The script and module entry points both work:
`python examples/memtrace/attribute.py` or
`python -m examples.memtrace.attribute`.

## Verification and fidelity

```bash
.venv/bin/python -m pytest tests/unit/test_memtrace_example -q
.venv/bin/ruff check examples/memtrace/graphtrace \
  examples/memtrace/attribute.py tests/unit/test_memtrace_example
.venv/bin/ruff format --check examples/memtrace/graphtrace \
  examples/memtrace/attribute.py tests/unit/test_memtrace_example
.venv/bin/python -m compileall -q examples/memtrace/graphtrace \
  examples/memtrace/attribute.py
```

Offline tests use invented graphs and scripted model responses. The end-to-end
test verifies a known retrieval failure: extraction preserves `Friday` in a memory
entry but retrieval produces unrelated context. It also verifies that a nonexistent
operation submission is rejected and corrected. Other tests cover frontier ordering,
capacity, restarts, graph immutability, pagination/regex search, QA parsing, source
ranking, complete tool-call pairs during trimming, and explicit failure paths.

This port preserves the attribution instructions, seven-tool exploration workflow,
versioned-node state and per-request hints. Differences from the original:

- OpenAI function calling replaces AgentScope's agent and final-response tool.
- Standard-library mappings replace sorted containers while keeping frontier order.
- BM25 uses decoded EverOS message content rather than encoded snapshot metadata.
  It retains the original English tokenizer and BM25Okapi parameters.
- Dense/hybrid evidence retrieval and LLM summary compression are deferred.
- Deterministic context pruning replaces summary compression; default page sizes
  are smaller. A fresh hint replaces accumulated hint messages.
- EverOS QA decoding, observation ordering, submission evidence checks and the
  insufficient-evidence stop are additional integration behavior.

The attribution instructions are adapted from
`MemTrace/toolkits/memtrace_utils.py`; notebook/state behavior follows
`MemTrace/graphtrace/`. The original MIT notice is preserved in
[`graphtrace/MEMTRACE_LICENSE.txt`](graphtrace/MEMTRACE_LICENSE.txt).

## Existing tracing helpers

The existing scripts remain separate from attribution:

- `filetrace_locomo_drive.py`: replay LoCoMo against a running EverOS server;
  `--qa` also answers and judges queries.
- `inject_qa.py`: append an out-of-process QA chain to existing exports.
- `merge.py`: merge tenant-scoped exports and reject revision conflicts.
- `export_png.py`: render exported graphs.

When collecting a new attribution case, replay the conversation containing the
actual source evidence before querying and injecting QA. A reference answer
about a later, unreplayed session is not evidence of an extraction failure.

Inject QA before merging, with the server stopped. Records identify a search by
`user_id` and `request_id`; repeated ids require `query_full_node_id`, and ambiguous
tenants require `app_id`/`project_id`. Missing or ambiguous matches are skipped.
Reinjecting an identical record is idempotent, including answer-only records.
Changed records retain their separate evaluation operations. Use a single trace
writer or separate per-worker directories: a file lock does not merge workers'
in-memory graphs.

## Recorded verification (2026-10-03)

The 35 offline tests passed; branch-aware coverage over the new example code was
86.79%. Ruff checks, formatting checks and syntax compilation passed.

A real `qwen3.8-flash` run on the invented, known-fault retrieval graph completed
in 12 model requests. It attributed `RetrievalError` to `retrieve-op`, correctly
distinguishing the preserved source fact from the missing retrieval context.
Reported usage was 58,369 input and 1,669 output tokens (60,038 total).
The local input and full tool transcript are stored under
`../MemTrace/outputs/everos_graphtrace_core/` as
`synthetic_retrieval_graph.json` and `synthetic_retrieval_qwen3_8_flash.json`.
This checks the model/tool integration on invented data, not benchmark accuracy.

To generate the same invented input from the offline test fixture and run it:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from tests.unit.test_memtrace_example.conftest import make_graph

Path('/tmp/memtrace_synthetic.json').write_text(
    json.dumps(make_graph().export_graph(), indent=2)
)
PY

.venv/bin/python examples/memtrace/attribute.py \
  --graph /tmp/memtrace_synthetic.json \
  --api-config ../MemTrace/input_files/api_config.json \
  --model qwen3.8-flash --max-iters 40 \
  --output /tmp/memtrace_synthetic_report.json
```

The existing LoCoMo merged graph contains two failed QA cases and no human
operation-attribution labels. Its recorded source messages end on July 3, 2023,
while Caroline's reference answer concerns the Friday before July 15. The raw
messages mention researching adoption agencies but do not establish the meeting
date. This graph cannot support a confident earliest-fault diagnosis for that
question; source absence alone does not establish an extraction or retrieval error.

The `qwen3.8-flash` LoCoMo run stopped at request 79 with `APIConnectionError`,
after 78 complete model/tool turns. The partial report retains the exploration
state and 21 context-pruning events in `caroline_qwen3_8_flash.json`. A bounded
continuation replayed the saved model responses through the read-only tools
without repeating paid requests; its first new model request was rejected with
HTTP 403 (`PermissionDeniedError`). That separate result is preserved in
`caroline_qwen3_8_flash_resumed.json`. Neither report contains an attribution.
This run validates import, QA selection, exploration, pruning and failure
reporting, but does not establish a LoCoMo fault or attribution accuracy.
