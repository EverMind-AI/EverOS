"""Replay a recorded EverOS trace into your own Langfuse project.

No EverOS install and no model API keys: this pushes a trace that a real
EverOS server actually produced (``recorded_trace.json``, captured with
``record_trace.py``) into your Langfuse project, so you can see what the
integration looks like in your own UI before deciding to deploy anything.

It is a recording, not a live server. Span names, attributes, token usage,
parent/child structure and durations are EverOS's own output, replayed
verbatim. Three things are necessarily rewritten: trace/span ids are minted
fresh (so repeated runs do not collide), timestamps are shifted so the trace
lands at the current time, and the root spans get a ``replay`` tag so nobody
mistakes it for live traffic.

Usage::

    pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
    export LANGFUSE_PUBLIC_KEY="pk-lf-..."
    export LANGFUSE_SECRET_KEY="sk-lf-..."
    export LANGFUSE_HOST="https://cloud.langfuse.com"   # US: https://us.cloud.langfuse.com
    python replay.py

To trace your own EverOS server instead, see ``README.md`` — that needs no
replay at all, just ``[observability]`` in ``everos.toml``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import (
    Span,
    SpanKind,
    Status,
    StatusCode,
    set_span_in_context,
)

DEFAULT_HOST = "https://cloud.langfuse.com"
REPLAY_TAG = "replay"
SCORE_MAX_ATTEMPTS = 5
# Small gap between scores; cheaper than discovering the limiter one 429 at a time.
SCORE_PACE_SECONDS = 0.15


def _credentials() -> tuple[str, str]:
    """Langfuse OTLP endpoint + Basic auth header, from the standard env vars."""
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", DEFAULT_HOST).rstrip("/")
    if not (public_key and secret_key):
        sys.exit(
            "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be set "
            "(project settings in Langfuse)."
        )
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    return host, f"Basic {token}"


def _check_credentials(host: str, auth: str) -> None:
    """Fail fast, and say why, before pushing a few hundred spans.

    Langfuse keys are region-scoped, and the OTLP exporter only reports a
    rejected export through the SDK's own logging, so a wrong host otherwise
    looks like a successful run into an empty project.
    """
    request = urllib.request.Request(
        f"{host}/api/public/projects", headers={"Authorization": auth}
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        if exc.code not in {401, 403}:
            # Only auth is under test; any other response is the replay's problem.
            return
        other = (
            "https://cloud.langfuse.com"
            if "us." in host
            else "https://us.cloud.langfuse.com"
        )
        sys.exit(
            f"{host} rejected these keys ({exc.code}). Langfuse projects are "
            f"region-scoped, so if the project lives in the other region set "
            f"LANGFUSE_HOST={other} and try again."
        )
    except OSError:
        return  # unreachable host surfaces on the real export a moment later


def _span_kind(name: str) -> SpanKind:
    bare = name.removeprefix("SPAN_KIND_")
    if bare in {"", "UNSPECIFIED"}:
        return SpanKind.INTERNAL
    return SpanKind[bare]


def _status(record: dict[str, Any]) -> Status | None:
    code = record.get("code", "STATUS_CODE_UNSET").removeprefix("STATUS_CODE_")
    if code in {"", "UNSET"}:
        return None
    return Status(StatusCode[code], record.get("message") or None)


def _post_score(host: str, auth: str, payload: dict[str, Any]) -> None:
    """POST one score, backing off when Langfuse rate-limits the endpoint.

    Scores go one per request, so replaying a whole recording sends dozens in a
    row and reliably trips the limiter without this.
    """
    request = urllib.request.Request(
        f"{host}/api/public/scores",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "Authorization": auth},
        method="POST",
    )
    for attempt in range(SCORE_MAX_ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                response.read()
            return
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == SCORE_MAX_ATTEMPTS - 1:
                raise
            after = exc.headers.get("retry-after") if exc.headers else None
            delay = float(after) if after and after.isdigit() else 2.0**attempt
            time.sleep(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default="recorded_trace.json")
    args = parser.parse_args()

    host, auth = _credentials()
    _check_credentials(host, auth)
    try:
        with open(args.fixture, encoding="utf-8") as handle:
            fixture = json.load(handle)
    except FileNotFoundError:
        sys.exit(
            f"{args.fixture} not found. Fetch it next to this script from "
            "https://github.com/EverMind-AI/EverOS/tree/main/examples/langfuse"
        )

    spans: list[dict[str, Any]] = fixture["spans"]
    if not spans:
        sys.exit(f"{args.fixture} contains no spans.")

    provider = TracerProvider(resource=Resource.create(fixture.get("resource") or {}))
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=f"{host}/api/public/otel/v1/traces",
                headers={"Authorization": auth},
            )
        )
    )
    tracer = provider.get_tracer("everos.replay")

    # Land the recording at "now", preserving every relative duration.
    offset = time.time_ns() - min(span["start_unix_nano"] for span in spans)

    by_id = {span["span_id"]: span for span in spans}
    children: dict[str, list[dict[str, Any]]] = defaultdict(list)
    roots: list[dict[str, Any]] = []
    for span in spans:
        parent = span["parent_span_id"]
        if parent and parent in by_id:
            children[parent].append(span)
        else:
            roots.append(span)

    # old span id -> (new trace id hex, new span id hex), for remapping scores.
    remapped: dict[str, tuple[str, str]] = {}
    trace_remap: dict[str, str] = {}

    def emit(record: dict[str, Any], parent: Span | None) -> None:
        attributes = dict(record["attributes"])
        if parent is None:
            tags = attributes.get("langfuse.trace.tags")
            tags = list(tags) if isinstance(tags, list) else []
            if REPLAY_TAG not in tags:
                tags.append(REPLAY_TAG)
            attributes["langfuse.trace.tags"] = tags
            recorded_at = fixture.get("recorded_at")
            if recorded_at:
                attributes["langfuse.trace.metadata.replay_of"] = recorded_at

        span = tracer.start_span(
            record["name"],
            context=set_span_in_context(parent) if parent is not None else None,
            kind=_span_kind(record["kind"]),
            start_time=record["start_unix_nano"] + offset,
            attributes=attributes,
        )
        context = span.get_span_context()
        remapped[record["span_id"]] = (
            format(context.trace_id, "032x"),
            format(context.span_id, "016x"),
        )
        trace_remap.setdefault(record["trace_id"], format(context.trace_id, "032x"))

        for child in children[record["span_id"]]:
            emit(child, span)

        status = _status(record["status"])
        if status is not None:
            span.set_status(status)
        span.end(end_time=record["end_unix_nano"] + offset)

    for root in roots:
        emit(root, None)

    provider.force_flush()
    provider.shutdown()
    print(f"Replayed {len(spans)} span(s) in {len(roots)} trace(s) to {host}")

    sent = 0
    skipped = 0
    for score in fixture.get("scores", []):
        payload = dict(score)
        observation = score.get("observationId")
        if observation and observation in remapped:
            trace_id, span_id = remapped[observation]
            payload["traceId"] = trace_id
            payload["observationId"] = span_id
        elif score.get("traceId") in trace_remap:
            payload["traceId"] = trace_remap[score["traceId"]]
            payload.pop("observationId", None)
        else:
            skipped += 1
            continue
        try:
            _post_score(host, auth, payload)
            sent += 1
        except urllib.error.HTTPError as exc:
            print(f"  ! score {score.get('name')} rejected: {exc}", file=sys.stderr)
        time.sleep(SCORE_PACE_SECONDS)
    if sent or skipped:
        note = f", {skipped} unmapped" if skipped else ""
        print(f"Pushed {sent} recall score(s){note}")

    print(
        "\nOpen Langfuse -> Tracing and filter on the 'replay' tag. "
        "This is a recorded EverOS run, not a live server: "
        "see README.md to trace your own."
    )


if __name__ == "__main__":
    main()
