"""Real-LanceDB regression: OR-mode BooleanQuery sparse recall.

Locks the fix for the tantivy implicit-AND poison: when a query
contains an IDF≈0 token (typically the partition owner's own name on
an owner-scoped corpus), the entire query used to return 0 hits. The
fixed path wraps each token in a ``BooleanQuery`` with ``SHOULD``
clauses (mirrors enterprise ES ``bool.should + minimum_should_match=1``)
so other tokens can carry the query.

These tests build a tiny in-memory corpus where one term is 100% DF
(the "poison" term) and verify that mixing it with informative
content tokens still surfaces results.

White-box surfaces:
- LanceDB ``episode`` table (real, per-test tmp root)
- ``EpisodeRecaller.sparse_recall``
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.component.tokenizer import Tokenizer
from everos.infra.persistence.lancedb import (
    Episode,
    ParentType,
    episode_repo,
    lancedb_manager,
)
from everos.memory.search.recall.base import RecallerDeps, build_or_query
from everos.memory.search.recall.episode import EpisodeRecaller


class _WhitespaceTokenizer(Tokenizer):
    """Split-on-whitespace tokenizer, lowercased.

    The OR-semantics fix is independent of jieba's behaviour, so a
    trivial tokenizer keeps the test focused.
    """

    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.lower().split() if tok]


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _episode_row(
    *,
    eid: str,
    owner_id: str,
    body_tokens: str,
) -> Episode:
    """Build an Episode row with ``body_tokens`` indexed as ``episode_tokens``."""
    return Episode(
        id=f"{owner_id}_{eid}",
        entry_id=eid,
        owner_id=owner_id,
        owner_type="user",
        session_id="sess_1",
        timestamp=_ts(),
        parent_type=ParentType.MEMCELL.value,
        parent_id="mc_test",
        sender_ids=[owner_id],
        episode=body_tokens,
        episode_tokens=body_tokens,
        md_path=f"users/{owner_id}/episodes/episode-2026-01-01.md",
        content_sha256="x" * 64,
        vector=[0.0] * 1024,
    )


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


def _recaller() -> EpisodeRecaller:
    return EpisodeRecaller(RecallerDeps(tokenizer=_WhitespaceTokenizer()))


# ── build_or_query helper unit-level checks ────────────────────────────


def test_build_or_query_empty_returns_none() -> None:
    """Empty / whitespace-only query → ``None`` (caller must short-circuit)."""
    tk = _WhitespaceTokenizer()
    assert build_or_query(tk, "", column="episode_tokens") is None
    assert build_or_query(tk, "   ", column="episode_tokens") is None


def test_build_or_query_single_token_returns_match_query() -> None:
    """One token → bare MatchQuery (no boolean-wrapper overhead)."""
    from lancedb.query import MatchQuery

    q = build_or_query(_WhitespaceTokenizer(), "hello", column="episode_tokens")
    assert isinstance(q, MatchQuery)


def test_build_or_query_multi_token_returns_boolean_query() -> None:
    """≥2 tokens → BooleanQuery with one SHOULD clause per token."""
    from lancedb.query import BooleanQuery

    q = build_or_query(
        _WhitespaceTokenizer(), "alice support group", column="episode_tokens"
    )
    assert isinstance(q, BooleanQuery)


# ── Live recall: poison token + informative token must surface results ──


async def test_or_semantics_poison_token_does_not_kill_query() -> None:
    """Two episodes, owner name in every doc (DF=100%), plus distinct content.

    Pre-fix, querying ``"alice support group"`` against owner=alice would
    return 0 hits — the ``alice`` token (DF=100% → IDF≈0) poisoned the
    implicit-AND query parser and dragged the score-conjunction to zero.
    Post-fix, ``BooleanQuery + SHOULD`` lets ``support`` / ``group`` carry
    the query on their own.
    """
    await episode_repo.upsert(
        [
            _episode_row(
                eid="ep_1",
                owner_id="alice",
                body_tokens="alice attended lgbtq support group last tuesday",
            ),
            _episode_row(
                eid="ep_2",
                owner_id="alice",
                body_tokens="alice tried watercolor painting on saturday morning",
            ),
        ]
    )
    # LanceDB FTS only sees data merged into the index after optimize().
    # Tests treat that as part of "the corpus is ready to query".
    from everos.infra.persistence.lancedb import get_table

    tbl = await get_table(Episode.TABLE_NAME, Episode)
    await tbl.optimize()

    where = "owner_id = 'alice' AND owner_type = 'user'"
    cands = await _recaller().sparse_recall("alice support group", where, limit=10)
    assert cands, "alice + support + group should recall ep_1 via SHOULD"
    # ep_1 is the support-group episode; should rank above ep_2 (no support).
    assert cands[0].id == "alice_ep_1"
    assert cands[0].score > 0.0


async def test_or_semantics_single_informative_token() -> None:
    """Single non-poison token still recalls (regression for ``painting``)."""
    await episode_repo.upsert(
        [
            _episode_row(
                eid="ep_1",
                owner_id="alice",
                body_tokens="alice attended lgbtq support group",
            ),
            _episode_row(
                eid="ep_2",
                owner_id="alice",
                body_tokens="alice tried watercolor painting on saturday",
            ),
        ]
    )
    from everos.infra.persistence.lancedb import get_table

    tbl = await get_table(Episode.TABLE_NAME, Episode)
    await tbl.optimize()

    where = "owner_id = 'alice' AND owner_type = 'user'"
    cands = await _recaller().sparse_recall("painting", where, limit=10)
    assert cands, "single informative token must recall the matching episode"
    assert cands[0].id == "alice_ep_2"


async def test_or_semantics_empty_query_returns_empty() -> None:
    """Tokenisation yields nothing → recall returns ``[]`` without hitting LanceDB."""
    cands = await _recaller().sparse_recall("   ", "owner_id = 'alice'", limit=10)
    assert cands == []
