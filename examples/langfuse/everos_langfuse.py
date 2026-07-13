"""EverOS -> Langfuse OpenTelemetry instrumentation (prototype).

Emits EverOS memory operations as OpenTelemetry spans following Langfuse's
attribute conventions (https://langfuse.com/integrations/native/opentelemetry),
so that an agent's memory layer becomes visible — and evaluable — inside
Langfuse, next to the rest of the trace.

Span model (mirrors EverOS's documented write/read paths):

    POST /api/v1/memory/add        span        "everos.memory.add"
    POST /api/v1/memory/flush      span        "everos.memory.flush"
      |- extraction (LLM)          generation  "everos.extract"        model/tokens/cost
      |- markdown persistence      span        "everos.persist.markdown"
      |- index sync                span        "everos.index.sqlite+lancedb"
    POST /api/v1/memory/search     retriever   "everos.memory.search"  query/top_k -> episodes+scores
      |- query embedding           embedding   "everos.search.embed_query"
      |- hybrid recall             retriever   "everos.search.hybrid_recall"  (BM25 + vector ANN + fusion)
      |- rerank                    span        "everos.search.rerank"  scores
    POST /api/v1/ome/trigger       agent       "everos.ome.<strategy>" (reflection / self-evolution)
      |- consolidation (LLM)       generation  "everos.reflect.consolidate"

Design notes:
  * Pure OpenTelemetry SDK — no Langfuse package dependency. The same spans
    can go to any OTLP backend (incl. an OpenTelemetry Collector);
    Langfuse ingests them natively on /api/public/otel (HTTP/protobuf).
  * `langfuse.session.id` / `langfuse.user.id` are set on EVERY span, per
    Langfuse's attribute-propagation guidance.
  * Recall-quality signals (fused retrieval score of the top hit, hit/miss)
    are pushed as Langfuse *scores* via POST /api/public/scores, attached to
    the search trace + retriever observation, so they can be plotted and
    filtered in Langfuse evals. (Scores are not part of the OTel span model.)
  * EverOS request-ids are already W3C trace-context format (32-hex), see
    everos.core.observability.tracing — so server-side adoption is a thin,
    additive layer.

This file is written to be read: it doubles as the integration sketch for
the EverOS <> Langfuse proposal.
"""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Callable, Optional

import requests
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

try:  # OTLP/HTTP exporter (protobuf) — what Langfuse's endpoint expects
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ImportError:  # pragma: no cover
    OTLPSpanExporter = None

DEFAULT_LANGFUSE_HOST = "https://us.cloud.langfuse.com"

# Attribute keys we flatten into the local JSONL dump (offline inspection)
_FLAT_KEYS = {
    "langfuse.observation.type": "obs_type",
    "langfuse.session.id": "session_id",
    "langfuse.user.id": "user_id",
    "gen_ai.request.model": "model",
    "gen_ai.usage.input_tokens": "input_tokens",
    "gen_ai.usage.output_tokens": "output_tokens",
    "everos.search.top_score": "top_score",
    "everos.search.hit": "recall_hit",
    "everos.op": "op",
}


class JsonLinesSpanExporter(SpanExporter):
    """Dump every finished span as one JSON line — a transparent, local record
    of exactly what would be sent to Langfuse (handy for offline inspection)."""

    def __init__(self, path: str):
        # one file per run — a deterministic offline record
        self._fh = open(path, "w", encoding="utf-8")

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        for s in spans:
            ctx = s.get_span_context()
            attrs = dict(s.attributes or {})
            row: dict[str, Any] = {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
                "parent_span_id": format(s.parent.span_id, "016x") if s.parent else "",
                "name": s.name,
                "start_ts": s.start_time // 1_000_000,  # ms epoch
                "duration_ms": round((s.end_time - s.start_time) / 1_000_000, 3),
                "status": s.status.status_code.name,
            }
            for k, col in _FLAT_KEYS.items():
                if k in attrs:
                    row[col] = attrs[k]
            row["attributes"] = {k: v for k, v in attrs.items()}
            self._fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._fh.close()


def init_tracing(
    service_name: str = "everos",
    spans_jsonl: str = "spans.jsonl",
) -> bool:
    """Configure OTel. Returns True if a live Langfuse exporter is attached.

    Reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from env.
    Offline (no keys): spans still go to the local JSONL file, so you can
    inspect exactly what would be sent to Langfuse without an account.
    """
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "1.1.0",  # everos PyPI version this models
        }
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(JsonLinesSpanExporter(spans_jsonl)))

    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST).rstrip("/")
    live = bool(pk and sk and OTLPSpanExporter)
    if live:
        auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        exporter = OTLPSpanExporter(
            endpoint=f"{host}/api/public/otel/v1/traces",
            headers={
                "Authorization": f"Basic {auth}",
                "x-langfuse-ingestion-version": "4",
            },
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return live


def force_flush() -> None:
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()


# --------------------------------------------------------------------------
# Langfuse scores (recall quality) — pushed via the public API, since scores
# are first-class objects in Langfuse rather than span attributes.
# --------------------------------------------------------------------------


def push_score(
    trace_id: str,
    name: str,
    value: float,
    observation_id: Optional[str] = None,
    comment: Optional[str] = None,
) -> bool:
    pk = os.getenv("LANGFUSE_PUBLIC_KEY")
    sk = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", DEFAULT_LANGFUSE_HOST).rstrip("/")
    if not (pk and sk):
        return False
    payload: dict[str, Any] = {
        "traceId": trace_id,
        "name": name,
        "value": value,
        "dataType": "NUMERIC",
    }
    if observation_id:
        payload["observationId"] = observation_id
    if comment:
        payload["comment"] = comment
    try:
        r = requests.post(
            f"{host}/api/public/scores", auth=(pk, sk), json=payload, timeout=15
        )
        return r.status_code in (200, 201, 207)
    except requests.RequestException as exc:  # never break the caller's flow
        print(f"[everos-langfuse] score push failed ({type(exc).__name__}); "
              "spans are still recorded locally")
        return False


# --------------------------------------------------------------------------
# Instrumented EverOS client
# --------------------------------------------------------------------------

Transport = Callable[[str, dict], dict]
_TRUNC = 4000  # keep span payloads bounded


def _j(obj: Any) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    return s if len(s) <= _TRUNC else s[:_TRUNC] + "…"


class InstrumentedEverOS:
    """Wraps an EverOS transport (real HTTP server or mock) and emits the
    spans that the proposed server-side instrumentation would emit.

    Every public method == one EverOS API call == one Langfuse trace.
    """

    def __init__(self, transport: Transport, tracer_name: str = "everos",
                 public_traces: bool = False):
        """public_traces: mark every trace as publicly shareable via URL
        (langfuse.trace.public). Only enable for synthetic/demo data —
        never for real memory content."""
        self._t = transport
        self._tracer = trace.get_tracer(tracer_name)
        self._public = public_traces

    # -- helpers ------------------------------------------------------------

    def _common(self, span, *, session_id=None, user_id=None, agent_id=None,
                app_id="default", project_id="default", obs_type="span", op=""):
        span.set_attribute("langfuse.observation.type", obs_type)
        span.set_attribute("everos.op", op)
        if self._public:
            span.set_attribute("langfuse.trace.public", True)
        if session_id:
            span.set_attribute("langfuse.session.id", session_id)
        if user_id:
            span.set_attribute("langfuse.user.id", user_id)
        if agent_id:
            span.set_attribute("langfuse.trace.metadata.agent_id", agent_id)
        span.set_attribute("langfuse.trace.metadata.app_id", app_id)
        span.set_attribute("langfuse.trace.metadata.project_id", project_id)
        span.set_attribute("langfuse.trace.tags", ["everos", "memory"])

    # -- write path ----------------------------------------------------------

    def add(self, session_id: str, messages: list[dict], user_id: str | None = None,
            app_id: str = "default", project_id: str = "default") -> dict:
        with self._tracer.start_as_current_span("everos.memory.add") as span:
            self._common(span, session_id=session_id, user_id=user_id,
                         app_id=app_id, project_id=project_id, op="add")
            span.set_attribute("langfuse.observation.input", _j(messages))
            resp = self._t("/api/v1/memory/add", {
                "session_id": session_id, "app_id": app_id,
                "project_id": project_id, "messages": messages,
            })
            span.set_attribute("langfuse.observation.output", _j(resp["data"]))
            span.set_attribute("everos.buffer.status", resp["data"]["status"])
            return resp

    def flush(self, session_id: str, user_id: str | None = None,
              app_id: str = "default", project_id: str = "default") -> dict:
        """Boundary -> LLM extraction -> markdown persist -> index sync."""
        with self._tracer.start_as_current_span("everos.memory.flush") as span:
            self._common(span, session_id=session_id, user_id=user_id,
                         app_id=app_id, project_id=project_id, op="flush")
            resp = self._t("/api/v1/memory/flush", {
                "session_id": session_id, "app_id": app_id, "project_id": project_id,
            })
            detail = resp.get("_detail", {})

            # 1. LLM extraction, a *generation*: model + token usage.
            #    EverOS does not compute cost; Langfuse derives it from
            #    model + usage in its model-usage views.
            with self._tracer.start_as_current_span("everos.extract") as g:
                self._common(g, session_id=session_id, user_id=user_id,
                             app_id=app_id, project_id=project_id,
                             obs_type="generation", op="extract")
                g.set_attribute("gen_ai.request.model", detail.get("model", "gpt-4.1-mini"))
                g.set_attribute("langfuse.observation.input", _j(detail.get("buffered_messages", [])))
                g.set_attribute("langfuse.observation.output", _j(detail.get("memory_cell", {})))
                usage = detail.get("usage", {})
                g.set_attribute("gen_ai.usage.input_tokens", usage.get("input", 0))
                g.set_attribute("gen_ai.usage.output_tokens", usage.get("output", 0))
                time.sleep(detail.get("extract_s", 0.05))

            # 2. Markdown persistence (atomic tmp+fsync+rename), strong consistency
            with self._tracer.start_as_current_span("everos.persist.markdown") as p:
                self._common(p, session_id=session_id, user_id=user_id,
                             app_id=app_id, project_id=project_id, op="persist")
                p.set_attribute("langfuse.observation.output",
                                _j({"md_files": detail.get("md_files", [])}))
                time.sleep(0.008)

            span.set_attribute("langfuse.observation.output", _j(resp["data"]))

        # 3. Index sync runs AFTER the API call returns, in EverOS's async
        #    "cascade" daemon (file watcher + debounce + entry diff -> LanceDB).
        #    It is therefore emitted as its OWN short-lived trace, correlated
        #    to the originating write by session_id, not as a child span.
        with self._tracer.start_as_current_span("everos.cascade.index") as ix:
            self._common(ix, session_id=session_id, user_id=user_id,
                         app_id=app_id, project_id=project_id, op="index")
            ix.set_attribute("langfuse.observation.input",
                             _j({"triggered_by": "markdown change",
                                 "correlates_to_session": session_id}))
            ix.set_attribute("langfuse.observation.output",
                             _j({"rows_indexed": detail.get("rows_indexed", 0),
                                 "index_lag_ms": detail.get("index_lag_ms", 500)}))
            time.sleep(0.02)

        return resp

    # -- read path -----------------------------------------------------------

    def search(self, query: str, user_id: str | None = None, agent_id: str | None = None,
               top_k: int = 5, app_id: str = "default", project_id: str = "default",
               session_id: str | None = None, hit_threshold: float = 0.6) -> dict:
        with self._tracer.start_as_current_span("everos.memory.search") as span:
            self._common(span, session_id=session_id, user_id=user_id, agent_id=agent_id,
                         app_id=app_id, project_id=project_id,
                         obs_type="retriever", op="search")
            span.set_attribute("langfuse.observation.input",
                               _j({"query": query, "top_k": top_k, "method": "hybrid"}))
            ctx = span.get_span_context()
            trace_id_hex = format(ctx.trace_id, "032x")
            retriever_obs_id = format(ctx.span_id, "016x")

            payload = {"query": query, "method": "hybrid", "top_k": top_k,
                       "app_id": app_id, "project_id": project_id}
            if user_id:
                payload["user_id"] = user_id
            if agent_id:
                payload["agent_id"] = agent_id
            resp = self._t("/api/v1/memory/search", payload)
            detail = resp.get("_detail", {})

            # 1. Query embedding
            with self._tracer.start_as_current_span("everos.search.embed_query") as e:
                self._common(e, session_id=session_id, user_id=user_id, agent_id=agent_id,
                             app_id=app_id, project_id=project_id,
                             obs_type="embedding", op="embed")
                e.set_attribute("gen_ai.request.model",
                                detail.get("embed_model", "Qwen/Qwen3-Embedding-4B"))
                e.set_attribute("langfuse.observation.input", _j(query))
                # compact output — never dump the raw vector into telemetry
                e.set_attribute("langfuse.observation.output",
                                _j({"embedding_dims": detail.get("embed_dims", 2560)}))
                e.set_attribute("gen_ai.usage.input_tokens", detail.get("embed_tokens", 0))
                time.sleep(detail.get("embed_s", 0.03))

            # 2. Hybrid recall: single LanceDB query = BM25 + vector ANN + filter
            with self._tracer.start_as_current_span("everos.search.hybrid_recall") as h:
                self._common(h, session_id=session_id, user_id=user_id, agent_id=agent_id,
                             app_id=app_id, project_id=project_id,
                             obs_type="retriever", op="recall")
                h.set_attribute("langfuse.observation.input",
                                _j({"bm25": True, "vector_ann": True, "filters": None}))
                h.set_attribute("langfuse.observation.output",
                                _j({"candidates": detail.get("candidates", 0)}))
                time.sleep(detail.get("recall_s", 0.03))

            # 3. Rerank (cross-encoder) — scores become Langfuse scores
            with self._tracer.start_as_current_span("everos.search.rerank") as r:
                self._common(r, session_id=session_id, user_id=user_id, agent_id=agent_id,
                             app_id=app_id, project_id=project_id, op="rerank")
                r.set_attribute("langfuse.observation.metadata.rerank_model",
                                detail.get("rerank_model", "Qwen/Qwen3-Reranker-4B"))
                r.set_attribute("langfuse.observation.output", _j(detail.get("ranked", [])))
                time.sleep(detail.get("rerank_s", 0.05))

            # Compact result summary on the retriever span
            hits = detail.get("ranked", [])
            top_score = float(hits[0]["score"]) if hits else 0.0
            span.set_attribute("langfuse.observation.output", _j(resp["data"]))
            span.set_attribute("everos.search.top_score", top_score)
            span.set_attribute("everos.search.hit", top_score >= hit_threshold)

        # Recall-quality -> Langfuse scores (visible in evals/dashboards).
        # Pushed AFTER the span closes so exporter/network time never
        # inflates the measured search latency.
        pushed = push_score(trace_id_hex, "recall_top_score", top_score,
                            observation_id=retriever_obs_id,
                            comment="fused+reranked score of top memory hit")
        push_score(trace_id_hex, "recall_hit",
                   1.0 if top_score >= hit_threshold else 0.0,
                   observation_id=retriever_obs_id,
                   comment=f"top_score >= {hit_threshold}")
        resp["_scores_pushed"] = pushed
        resp["_trace_id"] = trace_id_hex
        return resp

    # -- self-evolution (OME / reflection) ------------------------------------

    def trigger_ome(self, strategy: str = "reflect_episodes",
                    user_id: str | None = None, session_id: str | None = None) -> dict:
        with self._tracer.start_as_current_span(f"everos.ome.{strategy}") as span:
            self._common(span, session_id=session_id, user_id=user_id,
                         obs_type="agent", op="reflect")
            span.set_attribute("langfuse.observation.input", _j({"strategy": strategy}))
            resp = self._t("/api/v1/ome/trigger", {"name": strategy, "force": True})
            detail = resp.get("_detail", {})

            with self._tracer.start_as_current_span("everos.reflect.consolidate") as g:
                self._common(g, session_id=session_id, user_id=user_id,
                             obs_type="generation", op="consolidate")
                g.set_attribute("gen_ai.request.model", detail.get("model", "gpt-4.1-mini"))
                g.set_attribute("langfuse.observation.input",
                                _j(detail.get("episodes_in", [])))
                g.set_attribute("langfuse.observation.output",
                                _j(detail.get("consolidated", {})))
                usage = detail.get("usage", {})
                g.set_attribute("gen_ai.usage.input_tokens", usage.get("input", 0))
                g.set_attribute("gen_ai.usage.output_tokens", usage.get("output", 0))
                time.sleep(detail.get("reflect_s", 0.08))

            span.set_attribute("langfuse.observation.output", _j(resp["data"]))
            return resp


class HTTPTransport:
    """Real transport for a running EverOS server (pip install everos)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def __call__(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
