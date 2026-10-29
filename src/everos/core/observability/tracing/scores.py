"""Non-blocking recall-score push to Langfuse.

Recall-quality scores are a Langfuse-specific REST object (POST
``/api/public/scores``), independent of the OTLP span stream. To keep the
search request path free of any network time, scores go through a bounded
queue + a single background worker:

- ``enqueue`` is O(1) and never blocks or raises — when the queue is full it
  drops + counts (back-pressure never reaches the caller).
- the worker drains the queue and POSTs one score at a time (matching the
  Langfuse scores API + the reference prototype); every send is wrapped so a
  network failure only logs and the worker keeps going.

Attaches to the originating span via ``traceId`` (OTel trace_id, 032x hex) +
``observationId`` (OTel span_id, 016x hex) — exactly the mapping Langfuse's
OTLP ingestion uses.

Calibrated and uncalibrated top scores go out under *different names* — see
:data:`SCORE_TOP_CALIBRATED` / :data:`SCORE_TOP_RAW`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from everos.core.observability.logging import get_logger

if TYPE_CHECKING:
    from everos.config.settings import ObservabilitySettings

logger = get_logger(__name__)

Sender = Callable[[dict], Awaitable[None]]

# A Langfuse chart aggregates scores by *name*, so a single name may only ever
# carry values on one scale. HYBRID / AGENTIC top scores are calibrated to a
# comparable [0, 1]; KEYWORD (unbounded BM25) and single-route VECTOR are not,
# and averaging the two together would be meaningless. Hence two names: a
# dashboard built on ``recall_top_score`` stays comparable across methods and
# over time, and the raw scores remain available under their own name.
SCORE_TOP_CALIBRATED = "recall_top_score"
SCORE_TOP_RAW = "recall_top_score_raw"
SCORE_HIT = "recall_hit"


@dataclass(frozen=True)
class ScoreRecord:
    """One recall score bound for the Langfuse scores API."""

    trace_id: str
    observation_id: str | None
    name: str
    value: float
    comment: str | None
    metadata: dict[str, object] | None = None


def _to_payload(record: ScoreRecord) -> dict:
    payload: dict = {
        "traceId": record.trace_id,
        "name": record.name,
        "value": record.value,
        "dataType": "NUMERIC",
    }
    if record.observation_id:
        payload["observationId"] = record.observation_id
    if record.comment:
        payload["comment"] = record.comment
    if record.metadata:
        payload["metadata"] = record.metadata
    return payload


class RecallScoreSink:
    """Bounded queue + background worker that POSTs scores out-of-band."""

    def __init__(
        self,
        *,
        sender: Sender,
        closer: Callable[[], Awaitable[None]] | None = None,
        max_queue: int = 1000,
    ) -> None:
        self._sender = sender
        self._closer = closer
        self._queue: asyncio.Queue[ScoreRecord] = asyncio.Queue(maxsize=max_queue)
        self._task: asyncio.Task[None] | None = None
        self.dropped = 0

    def enqueue(self, record: ScoreRecord) -> None:
        """Hand a score to the worker; never blocks or raises."""
        try:
            self._queue.put_nowait(record)
        except asyncio.QueueFull:
            self.dropped += 1
            logger.warning("recall_score_dropped_queue_full", dropped=self.dropped)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            record = await self._queue.get()
            try:
                await self._sender(_to_payload(record))
            except Exception:  # telemetry must never break; log + continue
                logger.warning("recall_score_send_failed", exc_info=True)
            finally:
                self._queue.task_done()

    async def stop(self, *, drain_timeout: float = 5.0) -> None:
        """Drain pending scores (bounded by ``drain_timeout``), then tear down."""
        try:
            await asyncio.wait_for(self._queue.join(), timeout=drain_timeout)
        except TimeoutError:
            logger.warning("recall_score_drain_timeout")
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._closer is not None:
            with contextlib.suppress(Exception):
                await self._closer()


class _DisabledSink:
    """The sink while scores are off: every score is discarded."""

    def enqueue(self, record: ScoreRecord) -> None:
        """Discard ``record``."""

    async def stop(self) -> None:
        """Nothing to drain."""


_DISABLED = _DisabledSink()

# Process-wide, like the tracer provider: ``init_score_sink`` installs the
# live sink and ``shutdown_score_sink`` puts the disabled one back.
_sink: RecallScoreSink | _DisabledSink = _DISABLED


async def init_score_sink(
    settings: ObservabilitySettings, *, sender: Sender | None = None
) -> bool:
    """Build + start the sink when Langfuse creds + emit_recall_scores are set.

    Returns True if a sink was installed, False otherwise (disabled, scores
    off, or missing creds) — in which case ``emit_recall_scores`` is a no-op.

    Idempotent: a re-init without an intervening shutdown tears down the
    previous sink first, so its worker task + httpx client are not orphaned.

    Args:
        settings: The ``[observability]`` section.
        sender: Where each score payload goes; ``None`` posts it to the
            Langfuse scores API ``settings`` points at.
    """
    global _sink
    if not settings.enabled or not settings.emit_recall_scores:
        return False
    pk = settings.langfuse_public_key
    sk = settings.langfuse_secret_key
    host = settings.langfuse_host
    if not (pk and sk and host):
        return False

    await shutdown_score_sink()
    if sender is None:
        _sink = _langfuse_sink(host, pk, sk.get_secret_value())
    else:
        _sink = RecallScoreSink(sender=sender)
    _sink.start()
    return True


def _langfuse_sink(host: str, public_key: str, secret_key: str) -> RecallScoreSink:
    """A sink that POSTs each score to the Langfuse scores API at ``host``."""
    import httpx  # Deferred: only a deployment that pushes scores needs it.

    endpoint = host.rstrip("/") + "/api/public/scores"
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    auth = f"Basic {token}"
    client = httpx.AsyncClient(timeout=5.0)

    async def sender(payload: dict) -> None:
        resp = await client.post(
            endpoint, json=payload, headers={"Authorization": auth}
        )
        resp.raise_for_status()

    logger.info("recall_score_sink_started", endpoint=endpoint)
    return RecallScoreSink(sender=sender, closer=client.aclose)


def emit_recall_scores(
    *,
    trace_id: str,
    observation_id: str | None,
    top_score: float,
    hit: bool | None,
    method: str,
) -> None:
    """Enqueue a top score (always) + recall_hit (only when ``hit`` is a
    verdict); no-op when the sink is off.

    ``hit=None`` means the method's score is uncalibrated (KEYWORD /
    single-route VECTOR): no hit verdict is pushed, and the top score is
    reported as :data:`SCORE_TOP_RAW` instead of :data:`SCORE_TOP_CALIBRATED`
    so the two scales never land under one score name.

    ``method`` rides along as score metadata (a structured field Langfuse
    persists) as well as in the human-readable comment.
    """
    calibrated = hit is not None
    comment = f"method={method}"
    metadata: dict[str, object] = {"method": method, "calibrated": calibrated}
    _sink.enqueue(
        ScoreRecord(
            trace_id,
            observation_id,
            SCORE_TOP_CALIBRATED if calibrated else SCORE_TOP_RAW,
            float(top_score),
            comment,
            metadata,
        )
    )
    if hit is not None:
        _sink.enqueue(
            ScoreRecord(
                trace_id,
                observation_id,
                SCORE_HIT,
                1.0 if hit else 0.0,
                comment,
                metadata,
            )
        )


async def shutdown_score_sink() -> None:
    """Drain + tear down the sink; safe when uninitialized."""
    global _sink
    await _sink.stop()
    _sink = _DISABLED
