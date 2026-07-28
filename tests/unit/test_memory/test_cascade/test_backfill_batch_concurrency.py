"""Phase 1 embed-batch concurrency (finding #10).

Pins the invariant that ``_backfill_table`` fires the primary and
subject embed batches under ``asyncio.gather`` when a batch carries
subject rows. Prior to the fix the two calls ran serially — a
50%+ wall-clock regression on any table with subjects (episode is the
only such table today, but Phase 1's whole point is to speed up the
bulk migration).

Also pins the paired negative: on batches without subject rows, the
subject batch is skipped (no wasted zero-length ``embed_batch({})``).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from everos.memory.cascade._backfill import (
    NullBackfillPresenter,
    _backfill_table,
    _NullVectorRow,
    _TableBacklog,
    _TableSpec,
)


class _RecordingProvider:
    """Fake ``EmbeddingProvider`` that logs each ``embed_batch`` call's
    wall-clock window.

    Records ``(kind, start, end)`` where ``kind`` is inferred from the
    call order per batch: the first call in a batch is treated as
    "primary" and the second (if any) as "subject". A configurable
    ``latency`` per call makes concurrency observable.
    """

    def __init__(self, latency: float = 0.05) -> None:
        self.latency = latency
        self.calls: list[tuple[str, float, float]] = []
        self._batch_call_idx = 0
        self._lock = asyncio.Lock()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Serialize the call-index bump only; the actual sleep runs
        # concurrently under asyncio.gather, which is the whole point.
        async with self._lock:
            kind = "primary" if self._batch_call_idx % 2 == 0 else "subject"
            self._batch_call_idx += 1
        start = time.perf_counter()
        await asyncio.sleep(self.latency)
        end = time.perf_counter()
        self.calls.append((kind, start, end))
        return [[0.0, 0.0, 0.0] for _ in texts]


class _NoopRepo:
    """Repo double: silently accept every ``update`` so the test can
    focus on the embed-batch call shape."""

    async def update(self, values: dict[str, Any], *, where: str) -> None:
        return None


class _FakeSchema:
    """Minimal stand-in for :class:`BaseLanceTable` — ``_backfill_table``
    only reads ``schema.TABLE_NAME``."""

    TABLE_NAME = "fake_table"


def _build_spec(*, with_subject: bool) -> _TableSpec:
    """Construct a frozen :class:`_TableSpec` wired to a NoopRepo.

    ``with_subject`` toggles ``subject_of`` so a single helper covers
    both the concurrent (episode-like) and single-batch (fact-like)
    call shapes.
    """
    return _TableSpec(
        schema=_FakeSchema,  # type: ignore[arg-type]
        repo=_NoopRepo(),  # type: ignore[arg-type]
        text_of=lambda r: r["text"],
        subject_of=(lambda r: r.get("subject")) if with_subject else None,
    )


async def test_backfill_table_gathers_primary_and_subject_when_both_needed() -> None:
    """Episode-like batch with subject rows: primary + subject run
    concurrently. Concurrency is verified by asserting the two calls'
    ``[start, end]`` intervals temporally overlap — a scheduler-latency-
    free formulation of "in flight simultaneously".
    """
    spec = _build_spec(with_subject=True)
    rows = [
        _NullVectorRow(id=f"r{i}", text=f"text {i}", subject_text=f"sub {i}", tokens=5)
        for i in range(3)
    ]
    backlog = _TableBacklog(spec=spec, rows=rows)

    # ``latency=0.05`` is small enough to keep the test fast, but large
    # enough that ``asyncio.gather`` can actually interleave the two
    # coroutines' sleeps — a zero-latency provider would race the
    # ``perf_counter`` grain and give a spurious non-overlap.
    provider = _RecordingProvider(latency=0.05)
    result = await _backfill_table(backlog, provider, presenter=NullBackfillPresenter())  # type: ignore[arg-type]

    assert result.rows_processed == len(rows)
    assert result.rows_failed == 0

    assert len(provider.calls) == 2, provider.calls
    (_, p_start, p_end), (_, s_start, s_end) = provider.calls
    # Assert temporal overlap rather than wall-clock ratio. Concurrent
    # execution means the two calls' [start, end] windows intersect —
    # no scheduler-latency assumption needed. A CI runner under load
    # can widen individual intervals without disproving concurrency.
    assert p_start < s_end and s_start < p_end, (
        f"expected primary and subject batches to overlap; got "
        f"primary=[{p_start:.3f}, {p_end:.3f}] "
        f"subject=[{s_start:.3f}, {s_end:.3f}] (calls={provider.calls!r})"
    )


async def test_backfill_table_skips_subject_gather_when_no_subjects() -> None:
    """Non-subject spec: only the primary call fires; no wasted
    zero-length subject batch."""
    spec = _build_spec(with_subject=False)
    rows = [
        _NullVectorRow(id=f"r{i}", text=f"text {i}", subject_text=None, tokens=5)
        for i in range(3)
    ]
    backlog = _TableBacklog(spec=spec, rows=rows)

    provider = _RecordingProvider(latency=0.05)
    result = await _backfill_table(backlog, provider, presenter=NullBackfillPresenter())  # type: ignore[arg-type]

    assert result.rows_processed == len(rows)
    assert result.rows_failed == 0
    assert len(provider.calls) == 1, provider.calls
    assert provider.calls[0][0] == "primary"


async def test_backfill_table_gathers_only_on_batches_with_subjects() -> None:
    """Spec with ``subject_of`` set but a batch whose rows all carry
    ``subject_text=None`` still skips the subject gather — the
    decision is per-batch (`any(...)` check), not per-table."""
    spec = _build_spec(with_subject=True)
    rows = [
        _NullVectorRow(id=f"r{i}", text=f"text {i}", subject_text=None, tokens=5)
        for i in range(2)
    ]
    backlog = _TableBacklog(spec=spec, rows=rows)

    provider = _RecordingProvider(latency=0.02)
    result = await _backfill_table(backlog, provider, presenter=NullBackfillPresenter())  # type: ignore[arg-type]

    assert result.rows_processed == len(rows)
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == "primary"
