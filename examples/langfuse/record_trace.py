"""Record a real EverOS trace into ``recorded_trace.json`` (maintainer tool).

This stands in for Langfuse's two ingestion endpoints on localhost, so a real
EverOS server exports its spans *and* its recall scores here instead of to
Langfuse. What lands in the fixture is exactly what EverOS emitted: no span is
synthesized, no attribute is invented. ``replay.py`` then pushes that recording
into any reader's own Langfuse project.

Both signals are captured by one sink because EverOS derives both endpoints
from ``langfuse_host``: spans go to ``<host>/api/public/otel/v1/traces`` and
scores to ``<host>/api/public/scores``.

Usage:
    1. Point EverOS at this sink in ``everos.toml``. Keep the LLM, embedding and
       rerank sections filled in — a recording with real generations is the
       point, since that is what gives Langfuse the token usage to cost out.

           [observability]
           enabled             = true
           langfuse_public_key = "pk-lf-local"     # any value; the sink ignores auth
           langfuse_secret_key = "sk-lf-local"
           langfuse_host       = "http://127.0.0.1:4318"
           capture_content     = true              # demo data is synthetic, so show it

    2. ``python record_trace.py``      # starts the sink on :4318
    3. ``everos server start``         # in another shell
    4. ``python demo.py``              # drives add -> flush -> search
    5. Ctrl-C the sink; it writes ``recorded_trace.json``

Requires the OTel protobuf definitions, which ship with the exporter EverOS
already needs::

    pip install opentelemetry-exporter-otlp-proto-http
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.trace.v1.trace_pb2 import Span as PbSpan
from opentelemetry.proto.trace.v1.trace_pb2 import Status as PbStatus

TRACES_PATH = "/api/public/otel/v1/traces"
SCORES_PATH = "/api/public/scores"

# Collected across requests; written out on shutdown.
_spans: list[dict[str, Any]] = []
_scores: list[dict[str, Any]] = []
_resource: dict[str, Any] = {}


def _any_value(value: Any) -> Any:
    """Decode an OTLP ``AnyValue`` into a plain Python value."""
    which = value.WhichOneof("value")
    if which == "array_value":
        return [_any_value(item) for item in value.array_value.values]
    if which == "kvlist_value":
        return {kv.key: _any_value(kv.value) for kv in value.kvlist_value.values}
    if which is None:
        return None
    return getattr(value, which)


def _attributes(pairs: Any) -> dict[str, Any]:
    return {kv.key: _any_value(kv.value) for kv in pairs}


def _ingest_traces(body: bytes) -> int:
    """Decode one OTLP export request, appending its spans to ``_spans``."""
    global _resource
    request = ExportTraceServiceRequest()
    request.ParseFromString(body)
    count = 0
    for resource_spans in request.resource_spans:
        if not _resource:
            _resource = _attributes(resource_spans.resource.attributes)
        for scope_spans in resource_spans.scope_spans:
            for span in scope_spans.spans:
                parent = span.parent_span_id.hex()
                _spans.append(
                    {
                        "trace_id": span.trace_id.hex(),
                        "span_id": span.span_id.hex(),
                        "parent_span_id": parent or None,
                        "name": span.name,
                        "kind": PbSpan.SpanKind.Name(span.kind),
                        "start_unix_nano": span.start_time_unix_nano,
                        "end_unix_nano": span.end_time_unix_nano,
                        "status": {
                            "code": PbStatus.StatusCode.Name(span.status.code),
                            "message": span.status.message,
                        },
                        "attributes": _attributes(span.attributes),
                    }
                )
                count += 1
    return count


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(length)
        if self.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        path = self.path.split("?", 1)[0]

        if path == TRACES_PATH:
            try:
                added = _ingest_traces(body)
            except Exception as exc:  # keep the sink alive; the export retries
                print(f"  ! failed to decode an export: {exc}", file=sys.stderr)
                self._respond(400, b"")
                return
            print(f"  spans   +{added} (total {len(_spans)})")
            self._respond(
                200,
                ExportTraceServiceResponse().SerializeToString(),
                content_type="application/x-protobuf",
            )
            return

        if path == SCORES_PATH:
            score = json.loads(body)
            _scores.append(score)
            print(
                f"  score   {score.get('name')}={score.get('value')} "
                f"({score.get('comment')})"
            )
            self._respond(201, b"{}", content_type="application/json")
            return

        self._respond(404, b"")

    def _respond(
        self, status: int, body: bytes, *, content_type: str | None = None
    ) -> None:
        self.send_response(status)
        if content_type:
            self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        """Silence the default per-request logging; we print our own summary."""


def _write_fixture(path: str, everos_version: str | None) -> None:
    if not _spans:
        print("\nNothing recorded — no fixture written.", file=sys.stderr)
        return
    _spans.sort(key=lambda span: span["start_unix_nano"])
    fixture = {
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "everos_version": everos_version or _resource.get("service.version"),
        "resource": _resource,
        "spans": _spans,
        "scores": _scores,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    traces = len({span["trace_id"] for span in _spans})
    print(
        f"\nWrote {path}: {len(_spans)} span(s) across {traces} trace(s), "
        f"{len(_scores)} score(s)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4318)
    parser.add_argument("--out", default="recorded_trace.json")
    parser.add_argument(
        "--everos-version",
        default=None,
        help="Stamped into the fixture; defaults to the exporter's "
        "service.version resource attribute.",
    )
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), _Handler)
    print(
        f"Recording on http://127.0.0.1:{args.port}\n"
        f"  spans  <- POST {TRACES_PATH}\n"
        f"  scores <- POST {SCORES_PATH}\n"
        "Point everos.toml's [observability].langfuse_host at it, start the "
        "server, run demo.py, then Ctrl-C here.\n"
    )

    # Let KeyboardInterrupt break out of serve_forever, then write the fixture on
    # the way out. Calling server.shutdown() from a signal handler instead would
    # deadlock: it waits for the serve_forever loop that the handler is blocking.
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        server.server_close()
        _write_fixture(args.out, args.everos_version)


if __name__ == "__main__":
    main()
