# Latency Alignment Plan

## Problem

Each memory-system adapter (OpenClaw, EverMemOS, mem0, zep, memos, ...)
currently decides where its own latency timers start and stop. The
evaluation pipeline only aggregates whatever `*_latency_ms` field
adapters happen to emit. As a result:

- Identically-named fields measure different things:
  - EverMemOS `retrieval_latency_ms` spans the full `search()` method
    (index load + BM25 + embedding lookup + optional rerank + LLM check).
  - OpenClaw `retrieval_latency_ms` covers only the backend RPC.
- Stage-internal decomposition differs (`add` in OpenClaw = ingest +
  flush + index settle; in EverMemOS = memcell extraction + event log +
  clustering + profile + BM25 index build + embedding index build).
- Retries and fallbacks are absorbed silently by each adapter's own
  retry loop; they corrupt mean latency without appearing in the
  summary.
- Concurrency is baked into per-call means in different ways across
  adapters.

Comparing means or even p50/p95 of these fields across adapters is not
apples-to-apples.

## Design goal

Move latency measurement authority from adapters to the pipeline. The
adapter interface (`add`, `search`, `answer`) becomes the canonical
measurement boundary. A new memory system can be plugged in without its
author needing to understand or align any latency semantics.

## Three-layer contract

### Layer 1 — Adapter boundary (canonical, cross-comparable)

The harness times every call to `adapter.add`, `adapter.search`,
`adapter.answer`. This produces `wall_ms` for every call, independent of
what the adapter does internally. Each stage emits four distributions:

| View | Sample set | Question answered |
|---|---|---|
| `realistic` | all calls | user-perceived latency in production |
| `clean` | calls with `attempts==1`, not failed, not fallback | latency when nothing went wrong |
| `first_attempt` | duration of `attempts[0]` for every call | pure first-attempt cost, retries stripped |
| `successful_attempt` | duration of each attempt whose outcome is `success` | cost of a successful attempt regardless of prior failures |

### Layer 2 — Adapter sub-phases (adapter-specific, diagnostic only)

Adapters may report sub-phase timings via
`ctx.record_subphase(name, duration_ms)`. Sub-phase names are
adapter-specific. Reports render them in a separate section labelled
"not cross-comparable". Adapters that do not implement this interface
simply contribute an empty dict.

### Layer 3 — Work-unit normalization (hard contract)

- `add` → per conversation
- `search` → per `question_id`
- `answer` → per `question_id`

The pipeline asserts `N(add) == len(conversations)` and
`N(search) == N(answer) == len(qa_pairs)`; batch interfaces are
un-batched by the harness so the measurement denominator is constant.

## Retry policy

`BenchmarkContext.retry_policy` is a pipeline-level switch, cascaded
uniformly to every adapter call in a run.

| Policy | Adapter behaviour | Use case |
|---|---|---|
| `strict_no_retry` | first failure raises, no retry | pure latency baseline |
| `retry_once` | at most 1 retry | minimal retry-overhead probe |
| `realistic` | adapter's native retry strategy | production-aligned |

Adapters read `ctx.retry_policy` to decide whether to enter their own
retry loop. Every attempt calls
`ctx.record_attempt(n, duration_ms, outcome, wait_ms_before_next)`.

`outcome` is a closed enum: `success | http_5xx | http_429 | timeout |
upstream_unavailable | invalid_response | quota_exceeded`.

## Derived end-to-end metrics

Per-stage wall times compose without additional measurement:

- `e2e_query_ms = search_wall_ms + answer_wall_ms`
- `e2e_with_add_ms = add_wall_ms / questions_per_conv + search_wall_ms + answer_wall_ms`

## Pipeline invariants

Harness asserts on run completion:

1. `abs(wall_ms - (Σ attempt.duration_ms + Σ wait_ms_before_next)) < 0.05 * wall_ms`
   — else warn "adapter under-reported attempts".
2. `N(add) == len(conversations)` and `N(search) == N(answer) == len(qa)`
   — else abort.
3. `retry_policy == strict_no_retry` implies `len(attempts) == 1` for
   every call — else error.
4. If sub-phases reported, `Σ subphase_ms <= wall_ms` — else warn.

## Phased rollout

- **Phase 0** — narrow review fixes on the interim
  `retrieval_latency_ms` + `add_summary.json` instrumentation.
- **Phase 1** — introduce `BenchmarkContext`, `AttemptRecord`,
  `latency_views.py`; harness wraps adapter calls; four views emitted.
  Old `*_mean` / `*_stats` fields kept one release for compatibility.
- **Phase 2** — wire retry policy end-to-end; OpenClaw and EverMemOS
  adapters respect `ctx.retry_policy`; `--retry-policy` CLI flag.
- **Phase 3** — enforce invariants, un-batch work units.
- **Phase 4** — run matched baselines (concurrency=1 + strict_no_retry
  smoke; concurrency=N + realistic full) across EverMemOS / OpenClaw
  embed / OpenClaw noembed; publish baseline report under
  `evaluation/docs/`.
- **Phase 5** — optional: sub-phase instrumentation in both adapters.

## Migration guarantees

- Legacy `*_latency_ms_mean` and `*_stats` fields in
  `benchmark_summary.json` remain emitted for one release after Phase 1
  lands.
- An adapter that does not adopt `BenchmarkContext` still runs end-to-end;
  its summary simply carries `retry_observability: false` and is not
  used for cross-adapter latency claims.

## Minimum obligations for a new memory-system adapter

1. Implement `add / search / answer`.
2. Read `ctx.retry_policy` and skip the adapter's native retry loop
   when it is `strict_no_retry`.
3. Call `ctx.record_attempt(...)` once per attempt.

No other latency-specific code is required. Layer 1 metrics are derived
automatically; Layer 2 sub-phases are optional; Layer 3 work-unit
accounting is enforced by the pipeline.
