"""Integration test for ``everos cascade backfill --phase vectors``.

Exercises the real Phase 1 re-embed path against actual LanceDB tables
under a tmp memory root: seeds rows with ``vector IS NULL``, drives
``run_backfill(phase="vectors", ...)``, and asserts the rows now carry
a vector. A stub embedder replaces the process-wide embedding
capability singleton so no real network call ever happens (mirrors the
pattern in ``tests/integration/test_cascade_all_kinds_consistency.py``).

Covers: empty DB (nothing to backfill), declined confirmation (nothing
written, exit 1), a small DB with rows that need backfilling (rows
updated, pre-existing vectors left untouched, token estimate rendered
in K/M notation), a batch-embed failure that is tallied without
aborting the phase, an episode's ``subject_vector`` secondary embed,
and a missing embedding capability returning exit 2.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingCapability, EmbeddingProvider
from everos.component.utils.datetime import get_utc_now
from everos.config import load_settings
from everos.entrypoints.cli.commands._backfill_cmd import run_backfill
from everos.infra.persistence.lancedb import (
    AtomicFact,
    Episode,
    atomic_fact_repo,
    dispose_connection,
    episode_repo,
)

_DIM = 1024


class _StubEmbedder(EmbeddingProvider):
    dim = _DIM

    async def embed(self, text: str) -> list[float]:
        return [float(len(text) % 7)] * self.dim

    async def embed_batch(self, texts):  # type: ignore[no-untyped-def]
        return [[float(len(t) % 7)] * self.dim for t in texts]


@pytest.fixture
async def backfill_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Path]:
    """Tmp memory root + stub embedder; disposes the lancedb singleton
    around the test so no stale connection from a neighbouring test
    leaks in (mirrors ``cascade_runtime`` in the cascade test suite)."""
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    await dispose_connection()

    import everos.component.embedding.accessor as acc

    monkeypatch.setattr(
        acc, "_capability", EmbeddingCapability(provider=_StubEmbedder())
    )

    yield tmp_path
    await dispose_connection()


def _episode(
    entry_id: str,
    *,
    text: str = "",
    vector: list[float] | None = None,
    subject: str | None = None,
    subject_vector: list[float] | None = None,
) -> Episode:
    body = text or f"episode body {entry_id}"
    return Episode(
        id=f"u1_{entry_id}",
        entry_id=entry_id,
        owner_id="u1",
        owner_type="user",
        timestamp=get_utc_now(),
        parent_id="mc1",
        sender_ids=["u1"],
        subject=subject,
        episode=body,
        episode_tokens=body,
        md_path="users/u1/episodes/episode-2026-01-01.md",
        content_sha256=hashlib.sha256(entry_id.encode()).hexdigest(),
        vector=vector,
        subject_vector=subject_vector,
    )


def _fact(entry_id: str, *, vector: list[float] | None = None) -> AtomicFact:
    return AtomicFact(
        id=f"u1_{entry_id}",
        entry_id=entry_id,
        owner_id="u1",
        owner_type="user",
        timestamp=get_utc_now(),
        parent_id="mc1",
        sender_ids=["u1"],
        fact=f"fact body {entry_id}",
        fact_tokens=f"fact body {entry_id}",
        md_path="users/u1/.atomic_facts/atomic_fact-2026-01-01.md",
        content_sha256=hashlib.sha256(entry_id.encode()).hexdigest(),
        vector=vector,
    )


async def test_empty_db_reports_nothing_to_backfill(
    backfill_runtime: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = await run_backfill(phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "Nothing to backfill" in out


async def test_declined_confirmation_writes_nothing_and_exits_one(
    backfill_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await episode_repo.add([_episode("ep1")])
    monkeypatch.setattr("typer.confirm", lambda *a, **k: False)

    code = await run_backfill(phase="vectors", auto_yes=False)

    assert code == 1
    row = await episode_repo.get_by_id("u1_ep1")
    assert row is not None
    assert row.vector is None


async def test_small_db_backfills_null_vector_rows_only(
    backfill_runtime: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pre_embedded = [0.5] * _DIM
    await episode_repo.add([_episode("ep1"), _episode("ep2", vector=pre_embedded)])
    await atomic_fact_repo.add([_fact("af1")])

    code = await run_backfill(phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    ep1 = await episode_repo.get_by_id("u1_ep1")
    ep2 = await episode_repo.get_by_id("u1_ep2")
    af1 = await atomic_fact_repo.get_by_id("u1_af1")
    assert ep1 is not None and ep1.vector is not None
    assert ep2 is not None and ep2.vector == pytest.approx(pre_embedded)
    assert af1 is not None and af1.vector is not None
    assert "memories to process:  2" in out
    assert "phase 1 complete" in out


async def test_token_estimate_uses_kilo_notation(
    backfill_runtime: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    long_text = "word " * 3000
    await episode_repo.add([_episode("epbig", text=long_text)])

    code = await run_backfill(phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    assert "input tokens:" in out
    assert "K" in out


async def test_failed_row_is_tallied_and_does_not_abort_the_phase(
    backfill_runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Round-2 finding M8: a batch failure now falls back to per-row
    ``embed(...)`` calls (see ``_embed_primary_batch``). A poison row —
    one whose per-row retry also fails — must (a) not abort the phase,
    (b) not touch the healthy rows sharing its batch, and (c) drive the
    final exit code to ``4`` (COMPLETED_WITH_FAILURES) rather than ``0``
    so automation can distinguish clean success from partial success.

    ``_FlakyEmbedder`` fails BOTH ``embed_batch`` and ``embed`` on the
    poison text ("boom"), so the row remains unwritable after fallback.
    Healthy rows still succeed via the per-row retry path.
    """
    import everos.component.embedding.accessor as acc
    from everos.component.embedding import EmbeddingCapability

    class _FlakyEmbedder(_StubEmbedder):
        async def embed_batch(self, texts: list[str]) -> list[list[float]]:
            if any("boom" in t for t in texts):
                raise RuntimeError("simulated batch embed failure")
            return await super().embed_batch(texts)

        async def embed(self, text: str) -> list[float]:
            if "boom" in text:
                raise RuntimeError("simulated per-row embed failure")
            return await super().embed(text)

    monkeypatch.setattr(
        acc, "_capability", EmbeddingCapability(provider=_FlakyEmbedder())
    )
    await episode_repo.add(
        [_episode("epok", text="fine content"), _episode("epbad", text="boom content")]
    )

    code = await run_backfill(phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 4
    ok_row = await episode_repo.get_by_id("u1_epok")
    bad_row = await episode_repo.get_by_id("u1_epbad")
    assert ok_row is not None and ok_row.vector is not None
    assert bad_row is not None and bad_row.vector is None
    assert "phase 1 complete" in out
    assert "COMPLETED_WITH_FAILURES" in out
    assert "1 rows failed embedding" in out


async def test_episode_subject_vector_also_backfilled(
    backfill_runtime: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Episode is the only table with a second, independent embed
    (``subject_vector`` — see ``EpisodeHandler._build_row``). A row that
    has a ``subject`` must come out of Phase 1 with *both* vectors set,
    exercising ``_embed_subject_batch``'s secondary ``embed_batch`` call.
    """
    await episode_repo.add(
        [_episode("epsub", text="episode body", subject="What is the meaning of X?")]
    )

    code = await run_backfill(phase="vectors", auto_yes=True)
    out = capsys.readouterr().out

    assert code == 0
    row = await episode_repo.get_by_id("u1_epsub")
    assert row is not None
    assert row.vector is not None and len(row.vector) == _DIM
    assert row.subject_vector is not None and len(row.subject_vector) == _DIM
    assert "phase 1 complete" in out


async def test_missing_embedding_capability_returns_exit_two(
    backfill_runtime: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the embedding capability is unavailable (no provider
    configured), Phase 1 must fail cleanly with exit 2 rather than
    silently skipping rows or crashing uncaught."""
    import everos.component.embedding.accessor as acc

    await episode_repo.add([_episode("ep1")])
    monkeypatch.setattr(acc, "_capability", EmbeddingCapability(provider=None))

    code = await run_backfill(phase="vectors", auto_yes=True)

    assert code == 2
    row = await episode_repo.get_by_id("u1_ep1")
    assert row is not None and row.vector is None
