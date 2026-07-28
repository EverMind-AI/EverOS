# EverOS × Langfuse (native OpenTelemetry)

EverOS emits OpenTelemetry spans for its own memory operations — write, memcell
boundary + episode extraction (LLM), search with recall-quality scores, and OME
reflection — and exports them over OTLP to any backend, including
[Langfuse](https://langfuse.com). There is **no wrapper and no extra
instrumentation code**: enable it in config and the traces appear.

## Enable

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

3. Run EverOS normally:

   ```bash
   everos server start
   ```

Off by default — with `enabled = false` (or the `otel` extra absent) there is
zero tracing overhead.

## What you get

| EverOS operation | Langfuse observation |
| --- | --- |
| `POST /api/v1/memory/add` · `flush` | span `everos.memory.add` / `everos.memory.flush` |
| memcell boundary detection (LLM) | generation `everos.memcell.boundary` (model + tokens) |
| episode extraction (LLM) | generation `everos.extract` |
| markdown persistence | span `everos.persist.markdown` |
| `POST /api/v1/memory/search` | retriever `everos.memory.search` → `recall` / `rank` |
| query / recall embedding | embedding `everos.embedding` |
| OME reflection strategies | agent `everos.ome.<strategy>` (linked to the triggering request's trace) |

`langfuse.session.id` / `langfuse.user.id` group the traces. Recall quality is
pushed as Langfuse scores, split by whether the method's score is calibrated:
`recall_top_score` plus `recall_hit` for HYBRID / AGENTIC (comparable `[0, 1]`),
and `recall_top_score_raw` for KEYWORD / single-route VECTOR, whose raw BM25 or
cosine values are on a different scale and must not be averaged in with the
calibrated ones. Query and memory text are captured only when
`capture_content = true`.

## Try it

With a server running and `[observability]` enabled:

```bash
python demo.py
```

It drives one add → flush → search (keyword / hybrid / agentic) cycle against
`http://127.0.0.1:8000` using only the standard library, then tells you to open
Langfuse → **Tracing** filtered to `session.id = langfuse_demo`.

## Learn more

- Langfuse OpenTelemetry: https://langfuse.com/integrations/native/opentelemetry
- Config reference: the `[observability]` block in `src/everos/config/default.toml`.
