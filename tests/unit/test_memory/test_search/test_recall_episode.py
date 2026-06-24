"""Unit tests for ``EpisodeRecaller.fetch_all_for_owner`` and ``fetch_by_entry_ids``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from everos.component.tokenizer import Tokenizer
from everos.memory.search.recall.base import RecallerDeps
from everos.memory.search.recall.episode import EpisodeRecaller


def _make_row(
    ep_id: str,
    mc_id: str,
    *,
    parent_type: str = "memcell",
    entry_id: str = "",
) -> dict[str, Any]:
    """Build a minimal episode LanceDB row dict for test fixtures."""
    return {
        "id": ep_id,
        "owner_id": "alice",
        "owner_type": "user",
        "session_id": "sess_1",
        "timestamp": 1000000,
        "sender_ids": ["alice"],
        "subject": f"subj {ep_id}",
        "summary": f"summary {ep_id}",
        "episode": f"body {ep_id}",
        "parent_id": mc_id,
        "parent_type": parent_type,
        "entry_id": entry_id or ep_id,
    }


def _mock_table(rows: list[dict[str, Any]]) -> MagicMock:
    tbl = MagicMock()
    tbl.query.return_value.where.return_value.to_list = AsyncMock(return_value=rows)
    return tbl


@pytest.fixture()
def recaller() -> EpisodeRecaller:
    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = ["hi"]
    return EpisodeRecaller(RecallerDeps(tokenizer=tok))


async def test_fetch_all_for_owner_returns_entry_id_keyed_candidates(
    recaller: EpisodeRecaller,
) -> None:
    """id must equal entry_id so acluster_retrieve membership works."""
    rows = [
        _make_row("ep_1", "mc_1"),
        _make_row("ep_2", "mc_2"),
    ]
    with patch(
        "everos.memory.search.recall.episode.get_table",
        new_callable=AsyncMock,
        return_value=_mock_table(rows),
    ):
        result = await recaller.fetch_all_for_owner("owner_id = 'alice'")

    assert len(result) == 2
    ids = {c.id for c in result}
    assert ids == {"ep_1", "ep_2"}, "id must be entry_id"


async def test_fetch_all_for_owner_stores_episode_id_in_metadata(
    recaller: EpisodeRecaller,
) -> None:
    """metadata['episode_id'] carries the real LanceDB episode id for final shaping."""
    rows = [_make_row("ep_abc", "mc_xyz")]
    with patch(
        "everos.memory.search.recall.episode.get_table",
        new_callable=AsyncMock,
        return_value=_mock_table(rows),
    ):
        result = await recaller.fetch_all_for_owner("owner_id = 'alice'")

    assert result[0].metadata["episode_id"] == "ep_abc"
    assert result[0].metadata["parent_id"] == "mc_xyz"


async def test_fetch_all_for_owner_skips_rows_without_entry_id(
    recaller: EpisodeRecaller,
) -> None:
    """Rows without entry_id are silently skipped."""
    rows = [
        {
            "id": "ep_bad",
            "owner_id": "alice",
            "owner_type": "user",
            "session_id": "s",
            "timestamp": 1,
            "sender_ids": [],
            "subject": "",
            "summary": "",
            "episode": "",
            "parent_id": "mc_x",
        },
    ]
    with patch(
        "everos.memory.search.recall.episode.get_table",
        new_callable=AsyncMock,
        return_value=_mock_table(rows),
    ):
        result = await recaller.fetch_all_for_owner("owner_id = 'alice'")

    assert result == []


async def test_fetch_all_for_owner_merged_episode_uses_entry_id(
    recaller: EpisodeRecaller,
) -> None:
    """Merged episodes (parent_type=cluster) must use entry_id as Candidate.id.

    This ensures acluster_retrieve membership matching works for
    member_type=episode cluster members whose member_id is the episode's
    entry_id, not the cluster_id stored in parent_id.
    """
    rows = [
        _make_row(
            "ep_merged",
            "cluster_abc",
            parent_type="cluster",
            entry_id="entry_xyz",
        ),
    ]
    with patch(
        "everos.memory.search.recall.episode.get_table",
        new_callable=AsyncMock,
        return_value=_mock_table(rows),
    ):
        result = await recaller.fetch_all_for_owner("owner_id = 'alice'")

    assert len(result) == 1
    assert result[0].id == "entry_xyz", "merged episode id must be entry_id"
    assert result[0].metadata["episode_id"] == "ep_merged"


async def test_fetch_all_for_owner_mixed_regular_and_merged(
    recaller: EpisodeRecaller,
) -> None:
    """Mixed rows: both regular and merged episodes key by entry_id."""
    rows = [
        _make_row("ep_regular", "mc_1"),
        _make_row(
            "ep_merged",
            "cluster_99",
            parent_type="cluster",
            entry_id="entry_42",
        ),
    ]
    with patch(
        "everos.memory.search.recall.episode.get_table",
        new_callable=AsyncMock,
        return_value=_mock_table(rows),
    ):
        result = await recaller.fetch_all_for_owner("owner_id = 'alice'")

    assert len(result) == 2
    ids = {c.id for c in result}
    assert ids == {"ep_regular", "entry_42"}


async def test_fetch_by_entry_ids_returns_candidates(
    recaller: EpisodeRecaller,
) -> None:
    """fetch_by_entry_ids queries by entry_id and returns valid candidates."""
    rows = [
        _make_row(
            "ep_merged",
            "cluster_abc",
            parent_type="cluster",
            entry_id="entry_xyz",
        ),
    ]
    mock_tbl = MagicMock()
    mock_tbl.query.return_value.where.return_value.limit.return_value.to_list = (
        AsyncMock(return_value=rows)
    )
    with patch(
        "everos.memory.search.recall.episode.get_table",
        new_callable=AsyncMock,
        return_value=mock_tbl,
    ):
        result = await recaller.fetch_by_entry_ids(["entry_xyz"], "owner_id = 'alice'")

    assert len(result) == 1
    assert result[0].id == "ep_merged"


async def test_fetch_by_entry_ids_empty_input_returns_empty(
    recaller: EpisodeRecaller,
) -> None:
    """Empty entry_ids list short-circuits without querying."""
    result = await recaller.fetch_by_entry_ids([], "owner_id = 'alice'")
    assert result == []
