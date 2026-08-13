"""Round-2 finding M8: per-row fallback for embed batch failures.

Before M8, a single poison row (text over provider context limit, etc.)
would 400 the whole ``embed_batch`` call. ``_embed_primary_batch``
caught the exception and returned ``None``, and ``_backfill_table``
tallied every row in the batch as failed — 31 healthy rows would
re-enter the next scan and hit the same batch layout, failing the same
way. Infinite retry loop until an operator manually identified the
poison row.

Round-2 rewires ``_embed_primary_batch`` and ``_embed_subject_batch``
to fall back to per-row ``provider.embed(...)`` when the batch call
raises. Only rows whose per-row retry also fails count as failed —
31 rows recover, 1 poison row logs its ``row_id`` + truncated
``text_prefix`` so an operator can quarantine it.

This file pins the fallback contract:

- Batch-happy path returns every vector without touching the fallback.
- Batch failure logs the fallback-triggered event and iterates per row.
- A poison row within the per-row pass leaves a ``None`` in that slot
  while its neighbors recover, and the per-row failure log carries the
  row's id plus a truncated (100-char) text prefix.
"""

from __future__ import annotations

import pytest

from everos.core.observability.logging import configure_logging
from everos.memory.cascade._backfill import (
    _embed_primary_batch,
    _embed_subject_batch,
    _NullVectorRow,
    _truncated_text_prefix,
)


@pytest.fixture(autouse=True)
def _wire_structlog_to_stdlib() -> None:
    """Unit tests don't run through the CLI entry, so the structlog →
    stdlib bridge is uninitialised. Wire it up so pytest's ``caplog``
    fixture sees the WARNING events emitted by the backfill code."""
    configure_logging(level="WARNING")


def _row(row_id: str, text: str, subject_text: str | None = None) -> _NullVectorRow:
    return _NullVectorRow(
        id=row_id,
        text=text,
        subject_text=subject_text,
        tokens=len(text.split()),
        needs_primary=True,
        needs_subject=subject_text is not None,
    )


class _HappyProvider:
    """Batch succeeds every time — the fallback must NOT trigger."""

    def __init__(self) -> None:
        self.batch_calls = 0
        self.per_row_calls = 0

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        return [[float(i)] * 3 for i, _ in enumerate(texts)]

    async def embed(self, text: str) -> list[float]:
        self.per_row_calls += 1
        return [42.0] * 3


class _BatchFailsPerRowSucceedsProvider:
    """``embed_batch`` always raises; ``embed`` always succeeds. Every
    row must come back through the per-row fallback with a vector."""

    def __init__(self) -> None:
        self.batch_calls = 0
        self.per_row_calls = 0

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        raise RuntimeError("simulated batch failure (e.g. provider 500)")

    async def embed(self, text: str) -> list[float]:
        self.per_row_calls += 1
        return [1.0, 2.0, 3.0]


class _PoisonRowProvider:
    """``embed_batch`` raises; ``embed`` raises for one specific text
    ("poison") and succeeds for everything else. This is the M8
    scenario — 31 healthy rows recover, 1 poison row stays NULL."""

    def __init__(self, poison_marker: str = "POISON") -> None:
        self.poison_marker = poison_marker
        self.batch_calls = 0
        self.per_row_calls = 0

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        raise RuntimeError("simulated batch failure — one poison row")

    async def embed(self, text: str) -> list[float]:
        self.per_row_calls += 1
        if self.poison_marker in text:
            raise RuntimeError("simulated per-row failure — this row is the poison")
        return [7.0, 7.0, 7.0]


async def test_embed_primary_batch_returns_all_vectors_when_batch_succeeds() -> None:
    """Sanity: happy path stays on the fast lane, no per-row calls."""
    provider = _HappyProvider()
    batch = [_row(f"r{i}", f"text {i}") for i in range(5)]

    result = await _embed_primary_batch(provider, batch, "episode")  # type: ignore[arg-type]

    assert len(result) == len(batch)
    assert all(vec is not None for vec in result)
    assert provider.batch_calls == 1
    assert provider.per_row_calls == 0


async def test_embed_primary_batch_falls_back_per_row_when_batch_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Batch fails → per-row fallback fires, every row gets a vector,
    and the fallback log event names both the trigger and the row-level
    outcome so operators can distinguish the two."""
    provider = _BatchFailsPerRowSucceedsProvider()
    batch = [_row(f"r{i}", f"text {i}") for i in range(4)]

    with caplog.at_level("WARNING"):
        result = await _embed_primary_batch(provider, batch, "episode")  # type: ignore[arg-type]

    assert len(result) == len(batch)
    assert all(vec == [1.0, 2.0, 3.0] for vec in result)
    assert provider.batch_calls == 1
    assert provider.per_row_calls == len(batch)
    # The fallback-trigger log must fire exactly once for the batch.
    assert "cascade_backfill_batch_embed_failed_falling_back_per_row" in caplog.text


async def test_embed_primary_batch_isolates_poison_row_in_per_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Poison row scenario: 5 rows, row index 2 fails per-row. The
    result list carries ``None`` in slot 2 and vectors in slots 0/1/3/4.
    The per-row failure log carries the poison row's ``row_id`` and a
    truncated ``text_prefix`` — an operator reading the logs can pick
    the poison row out and quarantine it."""
    provider = _PoisonRowProvider()
    batch = [
        _row("r0", "healthy 0"),
        _row("r1", "healthy 1"),
        _row("r2", "POISON body — this one triggers a 400 at the provider"),
        _row("r3", "healthy 3"),
        _row("r4", "healthy 4"),
    ]

    with caplog.at_level("WARNING"):
        result = await _embed_primary_batch(provider, batch, "episode")  # type: ignore[arg-type]

    assert result[0] == [7.0, 7.0, 7.0]
    assert result[1] == [7.0, 7.0, 7.0]
    assert result[2] is None
    assert result[3] == [7.0, 7.0, 7.0]
    assert result[4] == [7.0, 7.0, 7.0]
    assert provider.batch_calls == 1
    assert provider.per_row_calls == len(batch)

    # The per-row failure log must carry the row_id + text_prefix so an
    # operator can identify and quarantine the poison row. structlog's
    # ConsoleRenderer emits kwargs as ``key=value`` in the message text.
    assert "cascade_backfill_row_embed_failed" in caplog.text
    assert "row_id" in caplog.text
    assert "r2" in caplog.text
    assert "POISON body" in caplog.text
    # And only ONE per-row event fires — the four healthy rows don't log.
    assert caplog.text.count("cascade_backfill_row_embed_failed") == 1


async def test_embed_subject_batch_isolates_poison_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Mirror of the poison-row assertion for ``_embed_subject_batch``.
    Rows without a subject skip the call entirely; rows with a subject
    whose per-row retry fails are absent from the returned map (so
    their ``subject_vector`` stays NULL and the widened NULL-scan filter
    picks them up next run)."""
    provider = _PoisonRowProvider()
    batch = [
        _row("r0", "primary 0", subject_text="healthy subject 0"),
        _row("r1", "primary 1", subject_text=None),  # no subject work
        _row("r2", "primary 2", subject_text="POISON subject text"),
        _row("r3", "primary 3", subject_text="healthy subject 3"),
    ]

    with caplog.at_level("WARNING"):
        result = await _embed_subject_batch(provider, batch, "episode")  # type: ignore[arg-type]

    # Only rows with a subject and a successful per-row retry appear.
    assert set(result) == {"r0", "r3"}
    assert result["r0"] == [7.0, 7.0, 7.0]
    assert result["r3"] == [7.0, 7.0, 7.0]
    # r1 has no subject, r2 is the poison — neither in the map.
    assert "r1" not in result
    assert "r2" not in result

    # The subject-specific per-row event carries r2's id + prefix.
    assert "cascade_backfill_row_subject_embed_failed" in caplog.text
    assert "r2" in caplog.text
    assert "POISON subject" in caplog.text


async def test_embed_subject_batch_happy_path_skips_fallback() -> None:
    """Sanity for the subject side: batch succeeds → no per-row calls."""
    provider = _HappyProvider()
    batch = [_row("r0", "primary 0", subject_text="subject 0")]

    result = await _embed_subject_batch(provider, batch, "episode")  # type: ignore[arg-type]

    assert set(result) == {"r0"}
    assert provider.batch_calls == 1
    assert provider.per_row_calls == 0


def test_truncated_text_prefix_ellipsizes_at_100_chars() -> None:
    """The per-row failure log must never leak full user text (PII).
    Short text passes through verbatim; anything over 100 chars gets
    the ellipsis suffix appended after exactly 100 kept chars."""
    assert _truncated_text_prefix("short") == "short"
    long_text = "x" * 200
    out = _truncated_text_prefix(long_text)
    assert out == "x" * 100 + "…"
    # Exactly at the limit — no ellipsis.
    exact = "y" * 100
    assert _truncated_text_prefix(exact) == exact
