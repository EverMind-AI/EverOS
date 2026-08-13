"""``RecallScoreSink`` — non-blocking recall-score push.

The sink is the piece that guarantees the Langfuse scores REST call never
touches the search request path: ``enqueue`` is O(1) and never blocks / raises
(drops + counts when full); a background worker drains and sends. The network
``sender`` is injected so these tests assert the queue/worker contract offline.
"""

from __future__ import annotations

import asyncio

from pydantic import SecretStr

from everos.config.settings import ObservabilitySettings
from everos.core.observability.tracing.scores import (
    SCORE_HIT,
    SCORE_TOP_CALIBRATED,
    SCORE_TOP_RAW,
    RecallScoreSink,
    ScoreRecord,
    emit_recall_scores,
    init_score_sink,
    shutdown_score_sink,
)


async def test_init_score_sink_tears_down_previous() -> None:
    """Re-init without an intervening shutdown must not leak the previous
    sink's worker task + httpx client: the old sink is stopped first."""
    settings = ObservabilitySettings(
        enabled=True,
        emit_recall_scores=True,
        langfuse_public_key="pk",
        langfuse_secret_key=SecretStr("sk"),
        langfuse_host="https://us.cloud.langfuse.com",
    )
    from everos.core.observability.tracing import scores as scores_mod

    try:
        assert await init_score_sink(settings) is True
        first = scores_mod._sink
        assert first is not None and first._task is not None

        assert await init_score_sink(settings) is True
        # The previous sink was torn down (stop() nulls its task) and a new
        # sink installed in its place.
        assert first._task is None
        assert scores_mod._sink is not first
    finally:
        await shutdown_score_sink()


async def test_worker_sends_payload_in_langfuse_shape() -> None:
    sent: list[dict] = []
    done = asyncio.Event()

    async def sender(payload: dict) -> None:
        sent.append(payload)
        done.set()

    sink = RecallScoreSink(sender=sender, max_queue=10)
    sink.start()
    sink.enqueue(
        ScoreRecord(
            trace_id="tid",
            observation_id="oid",
            name="recall_top_score",
            value=0.8,
            comment="method=hybrid",
        )
    )
    await asyncio.wait_for(done.wait(), timeout=1.0)
    await sink.stop()

    assert sent[0] == {
        "traceId": "tid",
        "observationId": "oid",
        "name": "recall_top_score",
        "value": 0.8,
        "dataType": "NUMERIC",
        "comment": "method=hybrid",
    }


async def test_worker_forwards_score_metadata() -> None:
    """``metadata`` is a structured field Langfuse persists, so a dashboard can
    split by method without parsing the free-text comment."""
    sent: list[dict] = []
    done = asyncio.Event()

    async def sender(payload: dict) -> None:
        sent.append(payload)
        done.set()

    sink = RecallScoreSink(sender=sender, max_queue=10)
    sink.start()
    sink.enqueue(
        ScoreRecord("tid", "oid", "n", 0.5, "method=keyword", {"method": "keyword"})
    )
    await asyncio.wait_for(done.wait(), timeout=1.0)
    await sink.stop()

    assert sent[0]["metadata"] == {"method": "keyword"}


async def _emitted(**kwargs: object) -> list[dict]:
    """Install a capturing sink, run ``emit_recall_scores``, return the payloads."""
    from everos.core.observability.tracing import scores as scores_mod

    sent: list[dict] = []

    async def sender(payload: dict) -> None:
        sent.append(payload)

    sink = RecallScoreSink(sender=sender, max_queue=10)
    sink.start()
    previous, scores_mod._sink = scores_mod._sink, sink
    try:
        emit_recall_scores(**kwargs)  # type: ignore[arg-type]
        await sink.stop()  # drains before returning
    finally:
        scores_mod._sink = previous
    return sent


async def test_calibrated_method_reports_top_score_and_hit() -> None:
    sent = await _emitted(
        trace_id="tid",
        observation_id="oid",
        top_score=0.72,
        hit=True,
        method="hybrid",
    )

    assert [s["name"] for s in sent] == [SCORE_TOP_CALIBRATED, SCORE_HIT]
    assert [s["value"] for s in sent] == [0.72, 1.0]
    assert all(s["metadata"] == {"method": "hybrid", "calibrated": True} for s in sent)


async def test_uncalibrated_method_reports_raw_name_and_no_hit() -> None:
    """An unbounded BM25 top score must not land under the same score name as a
    calibrated probability — a chart on that name would average both scales."""
    sent = await _emitted(
        trace_id="tid",
        observation_id="oid",
        top_score=8.4,
        hit=None,
        method="keyword",
    )

    assert [s["name"] for s in sent] == [SCORE_TOP_RAW]
    assert sent[0]["value"] == 8.4
    assert sent[0]["metadata"] == {"method": "keyword", "calibrated": False}


async def test_emit_is_a_noop_without_a_sink() -> None:
    from everos.core.observability.tracing import scores as scores_mod

    previous, scores_mod._sink = scores_mod._sink, None
    try:
        emit_recall_scores(
            trace_id="tid",
            observation_id="oid",
            top_score=1.0,
            hit=True,
            method="hybrid",
        )
    finally:
        scores_mod._sink = previous


async def test_enqueue_never_blocks_or_raises_when_full() -> None:
    async def slow(_: dict) -> None:
        await asyncio.sleep(10)

    sink = RecallScoreSink(sender=slow, max_queue=1)
    # No worker started → queue fills. enqueue must stay non-blocking.
    sink.enqueue(ScoreRecord("t", "o", "n", 1.0, None))  # fills the single slot
    sink.enqueue(ScoreRecord("t", "o", "n", 1.0, None))  # dropped, no raise
    assert sink.dropped == 1


async def test_sender_failure_does_not_crash_worker() -> None:
    calls: list[dict] = []
    second = asyncio.Event()

    async def flaky(payload: dict) -> None:
        calls.append(payload)
        if len(calls) == 1:
            raise RuntimeError("boom")
        second.set()

    sink = RecallScoreSink(sender=flaky, max_queue=10)
    sink.start()
    sink.enqueue(ScoreRecord("t", "o", "n1", 1.0, None))  # sender raises
    sink.enqueue(ScoreRecord("t", "o", "n2", 2.0, None))  # worker must survive
    await asyncio.wait_for(second.wait(), timeout=1.0)
    await sink.stop()
    assert len(calls) == 2  # first failed but worker kept going


async def test_stop_drains_pending() -> None:
    sent: list[dict] = []

    async def sender(payload: dict) -> None:
        sent.append(payload)

    sink = RecallScoreSink(sender=sender, max_queue=10)
    sink.start()
    for i in range(3):
        sink.enqueue(ScoreRecord("t", "o", f"n{i}", float(i), None))
    await sink.stop()  # should drain the queue before returning
    assert len(sent) == 3
