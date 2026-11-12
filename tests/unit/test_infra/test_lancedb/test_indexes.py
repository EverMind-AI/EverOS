"""LanceDB adapters of the memory index ports, against a real LanceDB.

The episode case goes all the way round: a domain episode is written by the
markdown store, projected by the cascade handler, and read back as a domain
episode through the index port.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from everos.component.embedding import EmbeddingProvider
from everos.component.tokenizer import Tokenizer
from everos.config import load_settings
from everos.infra.cascade._projections.episode import EpisodeHandler
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.index.adapters import (
    EpisodeIndexAdapter,
    GroupProfileIndexAdapter,
    ImageMemoryIndexAdapter,
    ProfileIndexAdapter,
)
from everos.infra.persistence.lancedb import (
    Episode as EpisodeRow,
)
from everos.infra.persistence.lancedb import (
    UserProfile as UserProfileRow,
)
from everos.infra.persistence.markdown import MarkdownEpisodeStore
from everos.memory.models import Episode, Profile
from everos.memory.ports import (
    NO_SOURCE_CHECK,
    EpisodeIndex,
    GroupProfileIndex,
    ImageMemoryIndex,
    Page,
    ProfileIndex,
    Scope,
    all_of,
    eq,
)
from tests.fakes import handler_deps
from tests.fakes.storage import Storage

_DIM = 1024
_TS = 1_747_216_800_000  # 2025-05-14T10:00:00Z
_USER = Scope(app_id="default", project_id="default", owner_id="u1", owner_type="user")


class _WhitespaceTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def tokenize_batch(self, texts: Sequence[str]) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]


class _ConstantEmbedder(EmbeddingProvider):
    dim = _DIM

    async def embed(self, text: str) -> list[float]:
        return [0.1] * _DIM

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


@pytest.fixture(autouse=True)
async def _real_lancedb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> AsyncIterator[None]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    load_settings.cache_clear()
    yield
    await storage.lance.close()
    load_settings.cache_clear()


def test_adapters_satisfy_their_ports(storage: Storage) -> None:
    assert isinstance(EpisodeIndexAdapter(storage.repos.episode), EpisodeIndex)
    assert isinstance(
        ImageMemoryIndexAdapter(storage.repos.image_memory), ImageMemoryIndex
    )
    assert isinstance(ProfileIndexAdapter(storage.repos.user_profile), ProfileIndex)
    assert isinstance(
        GroupProfileIndexAdapter(storage.repos.group_profile_item), GroupProfileIndex
    )


async def test_episode_round_trips_through_markdown_and_the_cascade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: Storage
) -> None:
    root = MemoryRoot(tmp_path)
    root.ensure()
    episode = Episode(
        owner_id="u1",
        episode="we planned a trip",
        timestamp=_TS,
        subject="Trip",
        summary="planning",
        session_id="s1",
        sender_ids=["u1", "u2"],
        parent_id="mc_1",
    )
    ref = await MarkdownEpisodeStore(root).append(
        _USER, episode, only_if=NO_SOURCE_CHECK
    )
    handler = EpisodeHandler(
        handler_deps(
            memory_root=root,
            tokenizer=_WhitespaceTokenizer(),
            index=storage.repos,
            embedding=_ConstantEmbedder(),
        )
    )
    await handler.handle_added_or_modified(ref.md_path)
    index = EpisodeIndexAdapter(storage.repos.episode)

    (found,) = await index.find(all_of(_USER.to_filter(), eq("entry_id", ref.entry_id)))
    stored = found.stored
    page = await index.paginate(_USER.to_filter(), Page(sort_by="timestamp"))
    hits = await index.sparse_search(["trip"], _USER.to_filter(), limit=5)
    dense = await index.dense_search([0.1] * _DIM, _USER.to_filter(), limit=5)
    subject = await index.dense_search_subject([0.1] * _DIM, _USER.to_filter(), limit=5)
    every = await index.find(_USER.to_filter())

    assert stored.item == episode
    assert stored.ref == ref
    assert stored.scope == _USER
    assert page.total == 1 and page.items[0].item == episode
    assert [h.stored.ref for h in hits] == [ref] and hits[0].score is not None
    assert [h.stored.item for h in hits] == [episode]
    assert [h.stored.ref.id for h in dense] == [ref.id]
    assert dense[0].score is not None and 0.0 <= dense[0].score <= 1.0
    assert [h.stored.ref.id for h in subject] == [ref.id]
    assert [(h.stored.ref.entry_id, h.score) for h in every] == [(ref.entry_id, None)]


async def test_episode_find_with_and_without_limit(storage: Storage) -> None:
    await storage.repos.episode.upsert(
        [
            EpisodeRow(
                id=f"u1_ep_{i}",
                entry_id=f"ep_{i}",
                owner_id="u1",
                owner_type="user",
                timestamp=dt.datetime(2025, 5, 14, tzinfo=dt.UTC),
                parent_id="mc_1",
                sender_ids=["u1"],
                episode=f"episode {i}",
                episode_tokens=f"episode {i}",
                md_path="users/u1/episodes/e.md",
                content_sha256="x" * 64,
            )
            for i in range(3)
        ]
    )
    index = EpisodeIndexAdapter(storage.repos.episode)

    unbounded = await index.find(eq("parent_id", "mc_1"))
    bounded = await index.find(eq("parent_id", "mc_1"), limit=2)

    assert sorted(h.stored.ref.id for h in unbounded) == [
        "u1_ep_0",
        "u1_ep_1",
        "u1_ep_2",
    ]
    assert len(bounded) == 2
    assert all(h.stored.ref.md_path for h in bounded)
    assert {h.stored.item.episode for h in unbounded} == {
        "episode 0",
        "episode 1",
        "episode 2",
    }


async def test_profile_get_decodes_the_json_buckets(storage: Storage) -> None:
    await storage.repos.user_profile.upsert(
        [
            UserProfileRow(
                id="u1",
                owner_id="u1",
                owner_type="user",
                summary="likes tea",
                explicit_info_json=json.dumps([{"description": "tea"}]),
                implicit_traits_json="{not json",
                profile_timestamp_ms=_TS,
                md_path="users/u1/user.md",
                content_sha256="x" * 64,
            )
        ]
    )

    stored = await ProfileIndexAdapter(storage.repos.user_profile).get(_USER)

    assert stored is not None
    assert stored.item == Profile(
        owner_id="u1",
        summary="likes tea",
        timestamp=_TS,
        explicit_info=[{"description": "tea"}],
        implicit_traits=[],
    )
    assert stored.ref.id == "u1"
    other = Scope(
        app_id="default", project_id="default", owner_id="u2", owner_type="user"
    )
    assert await ProfileIndexAdapter(storage.repos.user_profile).get(other) is None
