"""Real-LanceDB tests for ``AtomicFactRecaller.facts_for_episodes``.

The MRAG bridge is the only path that links facts back to episodes, and
the previous ``parent_type='episode' AND parent_id IN (episode_ids)``
query never matched: cascade writes facts with
``parent_type='memcell'``, ``parent_id=memcell_id``. The fixed version
takes an ``episode → memcell`` map from the caller, queries by the
deduped memcell set, and re-buckets results under every episode that
shares each memcell.

These tests exercise the real LanceDB query path (no recaller stubs):
- shared memcell → fact appears under both episodes,
- distinct memcells → facts bucket exclusively to their owning episode,
- empty / unknown memcells → empty result, no LanceDB call surprise.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.component.tokenizer import Tokenizer
from everos.infra.persistence.lancedb import (
    AtomicFact,
    ParentType,
    atomic_fact_repo,
    lancedb_manager,
)
from everos.memory.search.recall.atomic_fact import AtomicFactRecaller
from everos.memory.search.recall.base import RecallerDeps


class _WhitespaceTokenizer(Tokenizer):
    """Trivial tokenizer — the bridge doesn't touch text tokenisation."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


def _ts() -> _dt.datetime:
    return _dt.datetime(2026, 1, 1, tzinfo=_dt.UTC)


def _fact_row(
    *,
    fid: str,
    memcell_id: str,
    fact: str,
    owner_id: str = "alice",
) -> AtomicFact:
    return AtomicFact(
        id=fid,
        entry_id=fid.split("_", 1)[1] if "_" in fid else fid,
        owner_id=owner_id,
        owner_type="user",
        session_id="sess_1",
        timestamp=_ts(),
        parent_type=ParentType.MEMCELL.value,
        parent_id=memcell_id,
        sender_ids=[owner_id],
        fact=fact,
        fact_tokens=fact,
        md_path=f"users/{owner_id}/.atomic_facts/atomic_fact-2026-01-01.md",
        content_sha256="x" * 64,
        vector=[0.0] * 1024,
    )


@pytest.fixture(autouse=True)
async def _reset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate LanceDB to a tmp memory root per test."""
    monkeypatch.setenv("EVEROS_MEMORY__ROOT", str(tmp_path))
    lancedb_manager._conn = None
    lancedb_manager._tables.clear()
    yield
    await lancedb_manager.dispose_connection()


def _recaller() -> AtomicFactRecaller:
    return AtomicFactRecaller(RecallerDeps(tokenizer=_WhitespaceTokenizer()))


async def test_facts_for_episodes_buckets_by_shared_memcell() -> None:
    """Two episodes sharing one memcell both see the same fact pool.

    Episode-level fan-out (Episode pipeline runs once per cell but emits
    one Episode per user sender) gives multiple LanceDB episode rows
    pointing at the same memcell. The bridge must surface every fact
    that hangs off that memcell under both episode ids.
    """
    await atomic_fact_repo.upsert(
        [
            _fact_row(fid="alice_af_1", memcell_id="mc_shared", fact="likes hiking"),
            _fact_row(fid="alice_af_2", memcell_id="mc_shared", fact="lives in tokyo"),
            _fact_row(fid="alice_af_3", memcell_id="mc_other", fact="prefers oat milk"),
        ]
    )

    ep_to_memcell = {
        "alice_ep_a": "mc_shared",
        "alice_ep_b": "mc_shared",
        "alice_ep_c": "mc_other",
    }
    where = "owner_id = 'alice' AND owner_type = 'user'"
    out = await _recaller().facts_for_episodes(ep_to_memcell, where, per_episode=10)

    assert sorted(out.keys()) == ["alice_ep_a", "alice_ep_b", "alice_ep_c"]
    assert sorted(f.id for f in out["alice_ep_a"]) == ["alice_af_1", "alice_af_2"]
    assert sorted(f.id for f in out["alice_ep_b"]) == ["alice_af_1", "alice_af_2"]
    assert [f.id for f in out["alice_ep_c"]] == ["alice_af_3"]
    # parent_episode_id is the *bucket* episode, not the underlying memcell:
    # the same fact_1 surfaces twice with different parent_episode_id values.
    fact1_in_a = next(f for f in out["alice_ep_a"] if f.id == "alice_af_1")
    fact1_in_b = next(f for f in out["alice_ep_b"] if f.id == "alice_af_1")
    assert fact1_in_a.parent_episode_id == "alice_ep_a"
    assert fact1_in_b.parent_episode_id == "alice_ep_b"


async def test_facts_for_episodes_returns_empty_for_no_episodes() -> None:
    out = await _recaller().facts_for_episodes({}, "owner_id = 'alice'", per_episode=10)
    assert out == {}


async def test_facts_for_episodes_skips_unknown_memcells() -> None:
    """Episodes whose memcell has no facts simply don't appear in the result."""
    await atomic_fact_repo.upsert(
        [_fact_row(fid="alice_af_1", memcell_id="mc_a", fact="hello")]
    )

    out = await _recaller().facts_for_episodes(
        {"alice_ep_a": "mc_a", "alice_ep_b": "mc_missing"},
        "owner_id = 'alice' AND owner_type = 'user'",
        per_episode=10,
    )
    assert "alice_ep_a" in out
    assert "alice_ep_b" not in out
    assert [f.id for f in out["alice_ep_a"]] == ["alice_af_1"]


async def test_facts_for_episodes_filters_by_where_clause() -> None:
    """The caller's where clause is preserved (e.g. owner pinning)."""
    await atomic_fact_repo.upsert(
        [
            _fact_row(
                fid="alice_af_1",
                memcell_id="mc_a",
                fact="alice fact",
                owner_id="alice",
            ),
            _fact_row(
                fid="bob_af_1",
                memcell_id="mc_a",
                fact="bob fact",
                owner_id="bob",
            ),
        ]
    )

    out = await _recaller().facts_for_episodes(
        {"alice_ep_a": "mc_a"},
        "owner_id = 'alice' AND owner_type = 'user'",
        per_episode=10,
    )
    assert [f.id for f in out["alice_ep_a"]] == ["alice_af_1"]


async def test_facts_for_episodes_drops_empty_memcell_ids() -> None:
    """Episodes whose parent_id is missing (empty string) are dropped silently.

    Real-world cause: a candidate row that lost its ``parent_id`` (data
    corruption, manual edit). The bridge must not crash and must not
    emit ``parent_id IN ('')`` — which would match every empty-string
    row in the table.
    """
    await atomic_fact_repo.upsert(
        [_fact_row(fid="alice_af_1", memcell_id="", fact="orphan fact")]
    )

    out = await _recaller().facts_for_episodes(
        {"alice_ep_a": ""},
        "owner_id = 'alice' AND owner_type = 'user'",
        per_episode=10,
    )
    assert out == {}


# ── MRAG fact-level scoring (regression for query_vector handling) ─────


def _unit_vector(direction: int, dim: int = 1024) -> list[float]:
    """Return a unit vector with 1.0 at ``direction`` axis, 0 elsewhere.

    Used to build deterministic cosine relationships in the tests below:
    same direction → distance 0 (score 1.0); orthogonal → distance 1
    (score 0.0). The ``vector`` field on AtomicFact requires 1024-dim,
    so any test that goes through ``.nearest_to`` needs full-width.
    """
    out = [0.0] * dim
    out[direction] = 1.0
    return out


async def test_facts_for_episodes_assigns_real_cosine_score_with_query_vector() -> None:
    """Regression: ``query_vector`` triggers cosine ANN, not flat scan.

    Pre-fix, ``facts_for_episodes`` only ran ``where parent_id IN (...)``
    and emitted every fact with ``score=0.0`` — the MRAG fact-level
    ranking collapsed to insertion order. Post-fix, ``query_vector``
    flows into ``.nearest_to(...).distance_type('cosine')`` and each
    fact lands with its real query↔fact relevance score.

    Setup:
    - fact A's vector = unit on axis 0 (same direction as the query) →
      cosine distance 0 → score ≈ 1.0.
    - fact B's vector = unit on axis 1 (orthogonal to the query) →
      cosine distance 1 → score ≈ 0.0.

    Assertion: A ranks first AND its score > B's score AND both are
    non-zero-distinguishable (catches the old hardcoded ``0.0`` bug).
    """
    row_a = _fact_row(fid="alice_af_1", memcell_id="mc_shared", fact="close fact")
    row_a.vector = _unit_vector(0)
    row_b = _fact_row(fid="alice_af_2", memcell_id="mc_shared", fact="far fact")
    row_b.vector = _unit_vector(1)
    await atomic_fact_repo.upsert([row_a, row_b])

    out = await _recaller().facts_for_episodes(
        {"alice_ep_a": "mc_shared"},
        "owner_id = 'alice' AND owner_type = 'user'",
        per_episode=10,
        query_vector=_unit_vector(0),
    )

    facts = out["alice_ep_a"]
    assert [f.id for f in facts] == ["alice_af_1", "alice_af_2"], (
        "facts must be ordered by cosine distance ascending (closest first)"
    )
    assert facts[0].score > facts[1].score, "real cosine scoring must differentiate"
    assert facts[0].score > 0.5, "near-identical vectors should score close to 1"
    assert facts[1].score < 0.5, "orthogonal vectors should score close to 0"


async def test_facts_for_episodes_score_zero_without_query_vector() -> None:
    """Backward-compat: omitting ``query_vector`` falls back to flat scan.

    Callers that don't need fact-level relevance (e.g. KV-style fetch
    where the parent ranking already encodes the signal) keep the old
    ``score=0.0`` semantics. Documents the explicit contract so the
    fallback path is intentional, not an oversight.
    """
    row = _fact_row(fid="alice_af_1", memcell_id="mc_a", fact="anything")
    row.vector = _unit_vector(0)
    await atomic_fact_repo.upsert([row])

    out = await _recaller().facts_for_episodes(
        {"alice_ep_a": "mc_a"},
        "owner_id = 'alice' AND owner_type = 'user'",
        per_episode=10,
        # no query_vector
    )

    assert out["alice_ep_a"][0].score == 0.0
