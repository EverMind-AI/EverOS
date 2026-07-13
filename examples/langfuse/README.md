# EverOS × Langfuse (OpenTelemetry)

Trace EverOS memory operations — writes, LLM extraction, recall with quality
scores, and reflection — into [Langfuse](https://langfuse.com) as OpenTelemetry
spans, so an agent's memory layer becomes visible and evaluable next to the rest
of its traces.

This is a thin, dependency-light wrapper (pure OpenTelemetry SDK, no Langfuse
package dependency). The same spans work with Langfuse Cloud, self-hosted
Langfuse, or any other OTLP backend.

## Files

- `everos_langfuse.py` — the instrumentation wrapper (`init_tracing`,
  `InstrumentedEverOS`, `HTTPTransport`, recall-score push).
- `demo.py` — a runnable end-to-end example. Ships a mock transport, so it runs
  with **no EverOS server required**; set `EVEROS_BASE_URL` to trace a real one.

## Span model

| EverOS operation | Langfuse observation |
| --- | --- |
| `POST /api/v1/memory/add` | span `everos.memory.add` |
| `POST /api/v1/memory/flush` → extraction | span + generation `everos.extract` (model + tokens) |
| markdown persistence | span `everos.persist.markdown` |
| async index sync | span `everos.cascade.index` (separate correlated trace) |
| `POST /api/v1/memory/search` | retriever `everos.memory.search` |
| ↳ embedding / hybrid recall / rerank | embedding / retriever / span |
| `POST /api/v1/ome/trigger` | agent `everos.ome.<strategy>` + generation |

`langfuse.session.id` / `langfuse.user.id` are set on every span; recall quality
is pushed as Langfuse scores (`recall_top_score`, `recall_hit`).

## Run

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp requests

export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_HOST="https://us.cloud.langfuse.com"   # EU: https://cloud.langfuse.com

python demo.py
```

- With no keys set, `demo.py` still runs against the built-in mock and writes a
  local `spans.jsonl` (offline inspection) — nothing is sent anywhere.
- With Langfuse keys set, the same spans and recall scores flow into your
  Langfuse project. Open **Tracing → Traces** (filter by tag `everos` / `memory`).
- To trace a real deployment, set `EVEROS_BASE_URL` to a running EverOS server
  (see the [EverOS quickstart](../../README.md)); the instrumentation is identical.

## Privacy

Spans carry non-sensitive metadata (latency, token counts, model names, scores)
by default. Capturing raw query or memory content as span input/output is opt-in.
The demo uses synthetic data, and its `public_traces` flag (safe only for
synthetic data) marks the resulting traces as publicly shareable.

## Learn more

- Langfuse OpenTelemetry docs: https://langfuse.com/integrations/native/opentelemetry
- Native, opt-in instrumentation inside EverOS core is planned; this wrapper is
  the interim path and mirrors the same span model.
