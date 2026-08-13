# EverOS × Langfuse (native OpenTelemetry)

EverOS emits OpenTelemetry spans for its own memory operations — write, memcell
boundary + episode extraction (LLM), search with recall-quality scores, and OME
reflection — and exports them over OTLP to any backend, including
[Langfuse](https://langfuse.com). There is **no wrapper and no extra
instrumentation code**: enable it in config and the traces appear.

Two ways to look at it:

| | What it is | What you need |
| --- | --- | --- |
| [Replay a recording](#replay-a-recording-no-everos-needed) | A trace a real EverOS server produced, pushed into your Langfuse project | Langfuse keys only |
| [Trace your own server](#trace-your-own-server) | Your EverOS, your data, live | An EverOS server |

## Replay a recording (no EverOS needed)

`recorded_trace.json` is a capture of one real `demo.py` run against EverOS
1.2.1: 237 spans over 60 traces. Eleven conversations are ingested and flushed,
each with its LLM extraction and OME strategies nested underneath; reflection
then consolidates two of them and deprecates what they superseded; and five
questions are asked of the resulting memory, with their recall scores.
`replay.py` pushes it into your own Langfuse project, so you can see what the
integration looks like before deploying anything.

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"   # US: https://us.cloud.langfuse.com
python replay.py
```

Then open Langfuse → **Tracing** and filter on the `replay` tag.

Span names, attributes, token usage, parent/child structure and durations are
EverOS's own output, replayed verbatim. Three things are rewritten: trace and
span ids are minted fresh so repeated runs do not collide, timestamps are
shifted so the trace lands at the current time, and root spans carry a `replay`
tag so a recording is never mistaken for live traffic.

Two things in the trace list are not self-explanatory. The short keyword
searches beyond the five questions are `demo.py` waiting for each conversation
to become searchable. And the OME spans outlast the `flush` span they hang
under, because reflection continues after the request returns and re-attaches
to the originating trace through its `traceparent`.

Recall scores are per-method scales: read HYBRID against HYBRID, not against
AGENTIC. Agent cases and skills are not in this recording; the span and score
contract is the same when they appear.

## Trace your own server

1. Install the optional OpenTelemetry extra:

   ```bash
   pip install "everos[otel]"
   ```

2. Add `[observability]` to your `everos.toml`. The Langfuse keys derive the
   OTLP endpoint and auth automatically:

   ```toml
   [observability]
   enabled             = true
   langfuse_public_key = "pk-lf-..."
   langfuse_secret_key = "sk-lf-..."
   langfuse_host       = "https://us.cloud.langfuse.com"   # EU: https://cloud.langfuse.com
   # capture_content   = true   # opt-in: also record query / extracted memory text
   ```

   Container/CI equivalent via env vars: `EVEROS_OBSERVABILITY__ENABLED=true`,
   `EVEROS_OBSERVABILITY__LANGFUSE_PUBLIC_KEY=...`, and so on.

3. Run EverOS normally, then drive one memory lifecycle through it:

   ```bash
   everos server start
   python demo.py            # add -> flush -> search against 127.0.0.1:8000
   ```

   `demo.py` uses only the standard library and contains no instrumentation
   code; the spans come from the server. It ingests eleven conversations, nudges
   reflection (a weekly cron otherwise), then asks five questions, so the
   traces show recall choosing between memories rather than returning the only
   one there is.

Off by default — with `enabled = false` (or the `otel` extra absent) there is
zero tracing overhead.

The signal is plain OTLP/HTTP and vendor-neutral, so the same config exports to
an OpenTelemetry Collector or any other OTLP backend. The `langfuse_*` keys are
just a shortcut that fills in the endpoint and auth header for you.

## What you get

| EverOS operation | Langfuse observation |
| --- | --- |
| `POST /api/v2/memory/add` · `flush` | span `everos.memory.add` / `everos.memory.flush` |
| memcell boundary detection (LLM) | generation `everos.memcell.boundary` (model + tokens) |
| episode extraction (LLM) | generation `everos.extract` |
| markdown persistence | span `everos.persist.markdown` |
| `POST /api/v2/memory/search` | retriever `everos.memory.search` → `recall` / `rank` |
| query / recall embedding | embedding `everos.embedding` |
| OME extraction strategies | agent `everos.ome.<strategy>` (linked to the triggering request's trace) |
| reflection consolidating a cluster | span `everos.reflect.consolidate` under `everos.ome.reflect_episodes` |

`langfuse.session.id` / `langfuse.user.id` group the traces. Recall quality is
pushed as Langfuse scores, split by whether the method's score is calibrated:
`recall_top_score` plus `recall_hit` for HYBRID / AGENTIC (comparable `[0, 1]`),
and `recall_top_score_raw` for KEYWORD / single-route VECTOR, whose raw BM25 or
cosine values are on a different scale and must not be averaged in with the
calibrated ones. Query and memory text are captured only when
`capture_content = true`.

## Re-recording the fixture

`record_trace.py` is the maintainer-side tool that produced
`recorded_trace.json`. It stands in for Langfuse's two ingestion endpoints on
localhost, so a real EverOS server exports its spans *and* its recall scores
there instead of to Langfuse. Nothing about the recording is synthesized.

```bash
python record_trace.py     # sink on :4318; writes the fixture on Ctrl-C
```

Point `[observability].langfuse_host` at `http://127.0.0.1:4318`, start the
server, run `demo.py`, then stop the sink. Only worth redoing when the span
contract changes (a span added, renamed, or given new attributes); ordinary
releases do not invalidate a recording.

## Learn more

- Langfuse OpenTelemetry: https://langfuse.com/integrations/native/opentelemetry
- Config reference: the `[observability]` block in `src/everos/config/default.toml`.
