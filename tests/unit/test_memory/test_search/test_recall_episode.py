"""Unit tests for :class:`EpisodeRecaller`: index hits -> candidates + records."""

from __future__ import annotations

from unittest.mock import MagicMock

from everos.component.tokenizer import Tokenizer
from everos.memory.ports import eq
from everos.memory.search.recall.base import RecallerDeps
from everos.memory.search.recall.episode import EpisodeRecaller

from ._records import ScriptedIndex, hit, stored_episode

_ALICE = eq("owner_id", "alice")


def _recaller(
    index: ScriptedIndex, *, terms: list[str] | None = None
) -> EpisodeRecaller:
    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = ["hi"] if terms is None else terms
    return EpisodeRecaller(RecallerDeps(tokenizer=tok), index)  # type: ignore[arg-type]


async def test_sparse_recall_maps_hits_to_candidates_and_records() -> None:
    ep = stored_episode("ep_1")
    recaller = _recaller(ScriptedIndex(sparse=[hit(ep, 3.5)]))

    result = await recaller.sparse_recall("hi", _ALICE, limit=5)

    (cand,) = result.candidates
    assert cand.id == ep.ref.id
    assert cand.score == 3.5 and cand.source == "keyword"
    assert result.stored_by_id == {ep.ref.id: ep}


async def test_candidate_metadata_holds_algorithm_fields_only() -> None:
    """The record rides beside the candidate, never inside its metadata."""
    ep = stored_episode("ep_1")
    recaller = _recaller(ScriptedIndex(dense=[hit(ep, 0.9)]))

    (cand,) = (await recaller.dense_recall([0.1], _ALICE, limit=5)).candidates

    assert cand.metadata["entry_id"] == "ep_1"
    assert cand.metadata["episode"] == "body ep_1"
    assert cand.metadata["timestamp"] == ep.item.timestamp
    assert all(not hasattr(v, "ref") for v in cand.metadata.values())


async def test_a_missing_score_maps_to_zero() -> None:
    recaller = _recaller(ScriptedIndex(sparse=[hit(stored_episode("ep_1"), None)]))
    (cand,) = (await recaller.sparse_recall("hi", _ALICE, limit=5)).candidates
    assert cand.score == 0.0


async def test_empty_query_and_empty_vector_skip_the_index() -> None:
    index = ScriptedIndex(sparse=[hit(stored_episode("ep_1"), 1.0)])
    recaller = _recaller(index, terms=[])

    assert (await recaller.sparse_recall("", _ALICE, limit=5)).candidates == []
    assert (await recaller.dense_recall([], _ALICE, limit=5)).candidates == []
    assert index.calls == []


async def test_has_any_reads_one_row_at_most() -> None:
    index = ScriptedIndex(
        found=[hit(stored_episode("ep_1")), hit(stored_episode("e2"))]
    )
    assert await _recaller(index).has_any(_ALICE) is True
    assert index.calls[0][3] == 1
    assert await _recaller(ScriptedIndex()).has_any(_ALICE) is False
