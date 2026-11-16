# MemTrace graph export

## Recorded topology

Search graphs retain every stage and its candidate-list snapshot, but only the
last pool of each memory kind expands candidates into request-local
`retrieval_hit` nodes. A returned hit stores the response text and, when an exact
content match exists, references its canonical `memory_entry` revision through
`source_revision` and a source-to-hit edge. Version numbers are preserved across
graphs. Unknown candidates carry `identity_only=true`; they do not create
persistent entry stubs or freeze later requests' observations.
If response shaping introduces an episode absent from the last pool's
snapshot, or a capped snapshot omits a returned entry, its membership edge
carries `response_only: true`; the actual pool list stays unchanged. Searches
without stage records retain the query-to-hit chain. A failed search expands
only the last observed pool of each kind.

Construction graphs remain partitioned by sender, but each participating
sender's buffer records the complete add batch, including assistant replies.
Both messages therefore feed the same buffer version in a two-message chat
request. Later adds link the previous buffer version and both new messages to
the next buffer version. The first observed empty buffer is a node too.
Boundary detection follows `message_buffer -> boundary_detection_result -> memcell`.
There is no raw-response node or direct buffer-to-memcell edge. Membership
IDs remain on the memcell snapshot and match SQLite. Derived episodes and foresights
reuse that boundary operation; missing child `sender_ids` never creates a
new memcell version. SQLite supplies the boundary's message and sender IDs.

The boundary service captures the real loaded, deduplicated and persisted buffer
IDs. Non-final detection records the persisted tail; a successful flush records
the consumed buffer-to-tail transition as its request-level operation. Empty
flushes also receive their own version. Calls without a captured buffer state
use a request-specific completion receipt, rather than sharing the global
`COMMENT:NONE` output sentinel. `message_count` in a flush response counts fresh
input messages and does not describe how many buffered messages were drained.

Profile tracing is temporarily disabled: searches do not record profile-read
operations, pools, or hits, including when callers supply older stage records.
The actual profile fetch and response are unchanged, and the context snapshot
still includes returned profile content.

Pending-message retrieval is an independent `query -> retrieval_results -> context`
branch. Its single `retrieval_result` variable contains the complete returned
message collection; search no longer expands individual message nodes. Reads
require a top-level `filters.session_id` equality predicate. The operation and
result preserve success, empty reads, skips, errors and the completion timestamp.
Empty and skipped results still connect directly to the context; failed searches
record the result without fabricating a context. Responses without collected
stages also receive this aggregate result when they contain pending messages.

The service also captures actual boundary cells and episode extractor results,
including empty results, through a request-local collector.
Every successful deferred add also records a boundary outcome with
`decision=deferred` and `detection_performed=false`; it does not imply an LLM call.
Deferred adds retain `message_buffer -> boundary_detection_result -> no_split_marker`.
An observed model call records `message_buffer -> boundary_detection_result`;
the result then feeds either MemCells or a no-split
marker. The result preserves the actual batch response under `model_result`
(boundary indices, waiting flag, and reasoning), alongside the applied message
groups and `is_final`. A final flush can close a cell even when the model reports
no topic boundary; these two decisions are not conflated. Model boundary indices
refer to the detector's chat input after any forced prefix splits, whereas the
applied groups use canonical message IDs. No incremental-detector confidence or
should-end fields are fabricated. Unobserved model calls remain service outcomes.
File-only observations use the same path through a recovered grouping result,
marked `observation_only=true` with unknown `detection_performed`; they do not
fabricate model decisions. Messages feed that result directly when the matching
buffer version is unavailable.
These operations carry `source=construction_event` and the original observation
timestamp. The episode chain is `memcell -> cell-input -> extracted -> episode`.
Extraction-start events do not create placeholder attempt nodes or operations.
Failed adds keep the attempted messages connected to the owning session's error
operation without treating them as accepted buffer contents.

Skill stages declare their input stage ids. With the case bridge enabled, dense
skills and the case fusion pool feed `bridged_dense`; sparse skills and that
bridged pool feed skill fusion. The non-bridge path uses only direct dense input.

The retrieval response's `context` includes the actual returned unprocessed
messages. QA uses a separate `answer_context` snapshot containing exactly the
text supplied to the answer model, connected through a context-assembly operation.
If the caller omits it, the value remains unknown rather than borrowing the
retrieval response.

Foresight entries consume the same original input as episode extraction through
`memcell -> cell-input -> foresight`, rather than depending on the episode result.
The input node and its incoming edge are shared across derived foresights and
survive restarts. Missing input snapshots are restored from the matching app,
project and session's SQLite `Memcell.payload_json`; unavailable payloads remain
null with `content_observed=false`. Generation is disabled by default; enable
`[strategies.extract_foresight]`
with `enabled = true` in the memory root's `ome.toml`. These asynchronous OME
writes are observed at subsequent requests or the final shutdown sync.
Such file-diff operations carry `observation_only=true`; their timestamps do not
establish when extraction, update or deletion actually executed. The attribution
runner rejects them as executed fault operations. Background transformations
between scans still require more event instrumentation to distinguish reliably.
Foresights are currently construction artifacts and are not search results.
The `reranked_results` class on an episode `hierarchy` pool denotes the
hierarchy scoring stage, rather than a separate reranker invocation.

Nodes, operations and edges carry concise English comments describing their
contents or action without function names. Search-stage identifiers remain in
metadata. Live and offline QA exports also describe their context-assembly steps;
exported system placeholders receive fixed descriptions. These recording changes
apply to newly collected graphs and do not rewrite historical experiments.

`examples/memtrace/export_png.py` converts smartcomment execution-graph JSON
files into report-ready PNGs. Run it from the EverOS checkout after a replay
or server run:

```bash
python examples/memtrace/export_png.py /path/to/memtrace \
  --out /path/to/memtrace/png
```

## Default presentation

- Nodes, borders, labels, and edges use a white/black palette so the graph is
  readable in papers and logs.
- A graph with multiple `query-*` nodes produces one image per query. The
  filename contains the query number and request-id prefix, for example
  `...__search-01-a55a090d.png`.
- Every candidate pool displays only its first two items. The pool still shows
  its original `total_count`. Membership edges show those displayed candidates
  plus every actual returned entry, including response-only episode wrappers.
  This limits candidate fan-out without disconnecting the assembled context.
- Construction-only graphs (without query nodes) are rendered as one image.
  Input JSON files are read-only and are never rewritten.

## Options

Use `--top-k N` to choose the displayed pool size (`--top-k 2` is the default;
`--top-k 0` keeps every item). Use `--combined` when a single whole-graph
image is needed instead of query-specific images. The exporter validates each
display graph through `ExecNetwork.import_graph` before invoking Graphviz.

The `dot` executable must be available on `PATH`. The exporter intentionally
does not change how EverOS records searches: a `hybrid` pool is the RRF merge
of sparse and dense candidates, and remains visible as one stage in the
query-specific graph.

## Merging and QA

Stop the server before offline QA injection; an idle server still holds a live
graph that can overwrite injected edges. Duplicate identical QA records are
idempotent. Reused request ids require an exact `query_full_node_id`; ambiguous
tenant matches additionally require `app_id`/`project_id`.

Use the safe merger on the original construction and search exports:

```bash
.venv/bin/python examples/memtrace/merge.py /path/to/memtrace \
  --output /path/to/merged.json
```

It namespaces every variable by tenant, adds scope metadata for exploration, and
rejects conflicting immutable revisions. Older exports with inconsistent local
versions require recollection. File locking protects atomic writes but does not
merge independent workers' live graphs; use one writer or separate directories.

## Repairing older construction graphs

Older trackers repeated boundary detection for every derived foresight and
used child-specific sender metadata, creating duplicate message edges and
false memcell versions. After stopping the server, inspect a repair against
the memory root's real SQLite boundary records:

```bash
uv run --frozen --no-sync python examples/memtrace/repair_construction.py \
  /path/to/experiment/everos_root
```

Add `--apply` to merge only repeated exact boundaries, retain their derived
memory edges, and connect the buffer version observed at the first boundary.
Real membership changes are preserved. Original graph files are backed up in
`experiment/construction_before_fix/memtrace/`; `construction_repair.json`
records changes and before/after hashes. Markdown and indexes are read-only.
Rerun the experiment's renderer to refresh its display copies and images.
