"""Unit tests for ``EpisodeRecaller.fetch_all_for_owner``."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from everos.component.tokenizer import Tokenizer
from everos.memory.search.recall.base import RecallerDeps
from everos.memory.search.recall.episode import EpisodeRecaller


def _make_row(ep_id: str, mc_id: str) -> dict[str, Any]:
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


async def test_fetch_all_for_owner_returns_memcell_keyed_candidates(
    recaller: EpisodeRecaller,
) -> None:
    """id must equal parent_id (memcell_id) so acluster_retrieve membership works."""
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
    assert ids == {"mc_1", "mc_2"}, "id must be memcell_id, not episode_id"


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


async def test_fetch_all_for_owner_skips_rows_without_parent_id(
    recaller: EpisodeRecaller,
) -> None:
    """Rows without parent_id are silently skipped.

    They are incomplete episode records.
    """
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
            # no parent_id key
        },
    ]
    with patch(
        "everos.memory.search.recall.episode.get_table",
        new_callable=AsyncMock,
        return_value=_mock_table(rows),
    ):
        result = await recaller.fetch_all_for_owner("owner_id = 'alice'")

    assert result == []
