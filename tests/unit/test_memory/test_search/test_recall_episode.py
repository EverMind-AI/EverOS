"""Unit tests for :class:`EpisodeRecaller`: index hits -> candidates + records."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

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
    assert (await recaller.dense_recall_subject([], _ALICE, limit=5)).candidates == []
    assert index.calls == []


@pytest.mark.parametrize(
    "method",
    [
        "sparse_recall_as_child",
        "dense_recall_as_child",
        "dense_recall_subject_as_child",
    ],
)
async def test_child_recalls_tag_parent_id_with_the_entry_id(method: str) -> None:
    ep = stored_episode("ep_7", parent_id="mc_9")
    index = ScriptedIndex(
        sparse=[hit(ep, 1.0)], dense=[hit(ep, 0.5)], subject=[hit(ep, 0.4)]
    )
    recaller = _recaller(index)
    query: object = "hi" if method.startswith("sparse") else [0.1]

    result = await getattr(recaller, method)(query, _ALICE, limit=5)

    (cand,) = result.candidates
    assert cand.metadata["parent_id"] == "ep_7"
    assert result.stored_by_id[cand.id] is ep


async def test_dense_recall_subject_uses_the_subject_search() -> None:
    ep = stored_episode("ep_1")
    index = ScriptedIndex(subject=[hit(ep, 0.8)])

    result = await _recaller(index).dense_recall_subject([0.1], _ALICE, limit=3)

    assert [c.score for c in result.candidates] == [pytest.approx(0.8)]
    assert result.candidates[0].source == "vector"
    assert [c[0] for c in index.calls] == ["subject"]


async def test_fetch_all_for_owner_keys_by_entry_id() -> None:
    """Cluster members are entry ids, so the corpus is keyed by them."""
    regular = stored_episode("ep_1")
    merged = stored_episode("ep_9", parent_id="cl_1", parent_type="cluster")
    index = ScriptedIndex(found=[hit(regular), hit(merged)])

    result = await _recaller(index).fetch_all_for_owner(_ALICE)

    assert [c.id for c in result.candidates] == ["ep_1", "ep_9"]
    assert result.stored_by_id["ep_9"] is merged
    assert index.calls[0][3] is None, "the whole partition is needed"


async def test_fetch_parents_keys_by_entry_id_and_filters_on_it() -> None:
    ep = stored_episode("ep_3")
    index = ScriptedIndex(found=[hit(ep)])

    result = await _recaller(index).fetch_parents(["ep_3"], _ALICE)

    assert [c.id for c in result.candidates] == ["ep_3"]
    assert result.stored_by_id["ep_3"] is ep
    assert index.calls[0][3] == 1


async def test_fetch_by_entry_ids_keys_by_global_id() -> None:
    ep = stored_episode("ep_3")
    result = await _recaller(ScriptedIndex(found=[hit(ep)])).fetch_by_entry_ids(
        ["ep_3"], _ALICE
    )
    assert [c.id for c in result.candidates] == [ep.ref.id]


async def test_fetches_with_no_ids_skip_the_index() -> None:
    index = ScriptedIndex(found=[hit(stored_episode("ep_1"))])
    recaller = _recaller(index)

    assert (await recaller.fetch_parents([], _ALICE)).candidates == []
    assert (await recaller.fetch_by_entry_ids([], _ALICE)).candidates == []
    assert index.calls == []
