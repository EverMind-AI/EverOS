"""Milvus adapters of the memory index ports.

The query logic is shared with LanceDB (see ``test_lancedb/test_indexes.py``
for the real round trip); here the point is that the configured Milvus
backend binds each index to this engine's repository and turns its rows into
domain objects.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pymilvus")

from everos.config import load_settings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_derived_index, build_indexes
from everos.infra.persistence.index import DerivedIndex
from everos.infra.persistence.milvus import MilvusRepoBase
from everos.infra.persistence.records import Episode as EpisodeRow
from everos.memory.models import Episode
from everos.memory.ports import EpisodeIndex, Scope

_USER = Scope(app_id="default", project_id="default", owner_id="u1", owner_type="user")


def _bound(adapter: object) -> object:
    return adapter._repo  # type: ignore[attr-defined]  # adapters keep their repo private


@pytest.fixture
def milvus_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DerivedIndex:
    """The derived index the factory builds when Milvus is configured."""
    monkeypatch.setenv("EVEROS_INDEX__BACKEND", "milvus")
    load_settings.cache_clear()
    try:
        return build_derived_index(load_settings(), MemoryRoot(tmp_path))
    finally:
        load_settings.cache_clear()


def test_factory_selects_milvus_when_configured(milvus_index: DerivedIndex) -> None:
    assert milvus_index.name == "milvus"
    assert all(
        isinstance(repo, MilvusRepoBase) for repo in milvus_index.repositories.all()
    )


def test_each_index_is_bound_to_the_milvus_repository(
    milvus_index: DerivedIndex,
) -> None:
    repos = milvus_index.repositories
    indexes = build_indexes(milvus_index)
    assert _bound(indexes.episode) is repos.episode
    assert _bound(indexes.atomic_fact) is repos.atomic_fact
    assert _bound(indexes.agent_case) is repos.agent_case
    assert _bound(indexes.agent_skill) is repos.agent_skill
    assert _bound(indexes.profile) is repos.user_profile
    assert _bound(indexes.knowledge_topic) is repos.knowledge_topic
    assert isinstance(indexes.episode, EpisodeIndex)


async def test_rows_come_back_as_domain_episodes(
    milvus_index: DerivedIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = EpisodeRow(
        id="u1_ep_1",
        entry_id="ep_1",
        owner_id="u1",
        owner_type="user",
        timestamp=dt.datetime(2025, 5, 14, 10, tzinfo=dt.UTC),
        parent_id="mc_1",
        sender_ids=["u1"],
        episode="text",
        episode_tokens="text",
        md_path="users/u1/episodes/e.md",
        content_sha256="x" * 64,
    )

    async def find_by_owner_entries(*_: Any, **__: Any) -> list[EpisodeRow]:
        return [row]

    monkeypatch.setattr(
        milvus_index.repositories.episode,
        "find_by_owner_entries",
        find_by_owner_entries,
    )

    (stored,) = await build_indexes(milvus_index).episode.find_by_entry_ids(
        _USER, ["ep_1"]
    )

    assert stored.item == Episode(
        owner_id="u1",
        episode="text",
        timestamp=1_747_216_800_000,
        sender_ids=["u1"],
        parent_id="mc_1",
    )
    assert stored.ref.md_path == "users/u1/episodes/e.md"
