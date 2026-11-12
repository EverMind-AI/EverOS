"""Every picture a search returns names the answering model's copy.

What these pin:

- A picture reaches the answering model as the ``[images] model_edge`` copy,
  not the original: the original can be many times its size and visual tokens,
  and it is the size extraction read the picture at. Both a ranked picture and
  one attached to the episode that refers to it carry ``model_path``.
- The copy is asked of the picture store with the configured edge and quality,
  so the file is on disk when the response is read.
- A picture the store cannot copy keeps an empty ``model_path``; the search
  still answers, and the picture still carries its description.
- Without ``[images]`` limits nothing is asked and the field stays empty.
- ``thumbnail_path`` carries the extension the decider's copy really has: PNG
  for a greyscale picture, as its model copy says, never a ``.jpeg`` guess.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from everalgo.testing import FakeLLMClient
from PIL import Image

from everos.component.images import ImageLimits, derived_relpath
from everos.component.tokenizer import Tokenizer
from everos.config import SqliteSettings
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.images import FileImageStore
from everos.infra.persistence.sqlite import SqliteDatabase, SqliteRepos
from everos.memory.access import AccessPolicy
from everos.memory.models import CanonicalMessage, ImageMemory
from everos.memory.ports import MemoryRef, Scope, SearchHit, SessionKey, Stored
from everos.memory.search.dto import SearchImageItem, SearchRequest
from everos.memory.search.manager import SearchManager
from everos.memory.search.recall import EpisodeRecaller, ImageMemoryRecaller
from everos.memory.search.recall.base import RecallerDeps
from everos.memory.search.shaper import name_copies
from tests.fakes import InMemoryGroupStore, static_trace_path, static_tuning

from ._records import ScriptedIndex, hit, stored_episode

_SIGN = "a" * 64
_LATTE = "b" * 64
_LIMITS = ImageLimits(
    max_bytes=1,
    max_per_message=1,
    max_request_bytes=1,
    max_per_request=1,
    formats=("png",),
    model_edge=1024,
    cache_quality=85,
)


def _tokenizer() -> Tokenizer:
    tok = MagicMock(spec=Tokenizer)
    tok.tokenize.return_value = ["hi"]
    return tok


class _NoProfiles:
    async def fetch(self, owner_id: str, **_: Any) -> list[Any]:
        return []


class _NoBuffer:
    async def list(self, session: SessionKey) -> list[CanonicalMessage]:
        return []


class _Embedding:
    dim = 4

    async def embed_query(self, text: str) -> list[float]:
        return [0.1] * self.dim


def _keep_all(messages: Any, **_kwargs: Any) -> str:
    return json.dumps({"core": list(range(10)), "next_queries": []})


def _picture(entry_id: str, image_id: str) -> Stored[ImageMemory]:
    return Stored(
        ref=MemoryRef(id=f"alice_{entry_id}", entry_id=entry_id, md_path="i.md"),
        scope=Scope(
            app_id="default", project_id="default", owner_id="alice", owner_type="user"
        ),
        item=ImageMemory(
            owner_id="alice",
            description="a chalkboard sign reading EST. 2019",
            timestamp=1_747_216_800_000,
            ocr="EST. 2019",
            image_id=image_id,
            original_relpath=f"figures/{image_id[:2]}/{image_id}.png",
            media_type="png",
            parent_id="mc_1",
            entry_id=entry_id,
        ),
    )


class _PictureIndex:
    """Ranks ``ranked`` in both recalls; resolves every row by its file."""

    def __init__(
        self, rows: list[Stored[ImageMemory]], ranked: list[Stored[ImageMemory]]
    ) -> None:
        self.rows = rows
        self.ranked = ranked

    async def sparse_search(self, terms, where, *, limit: int):
        return [SearchHit(stored=row, score=0.9) for row in self.ranked][:limit]

    async def dense_search(self, vector, where, *, limit: int):
        return [SearchHit(stored=row, score=0.9) for row in self.ranked][:limit]

    async def find(self, where, *, limit: int | None = None):
        return [SearchHit(stored=row, score=None) for row in self.rows][:limit]

    async def find_by_image_ids(self, scope, image_ids):
        return [row for row in self.rows if row.item.image_id in set(image_ids)]


class _CopyStore:
    """Answers ``ensure_derived`` with the cache path; ``missing`` cannot be read,
    ``grey`` pictures have PNG copies."""

    def __init__(
        self, *, missing: set[str] = frozenset(), grey: set[str] = frozenset()
    ) -> None:
        self.missing = set(missing)
        self.grey = set(grey)
        self.asked: list[tuple[str, int, int]] = []

    async def ensure_derived(self, image_id: str, *, edge: int, quality: int) -> str:
        self.asked.append((image_id, edge, quality))
        if image_id in self.missing:
            raise FileNotFoundError(image_id)
        extension = "png" if image_id in self.grey else "jpeg"
        return f"figures/.cache/{image_id}-{edge}.{extension}"

    async def read_derived(self, image_id: str, *, edge: int, quality: int) -> bytes:
        raise AssertionError("the decider is shown no picture here")


def _manager(
    *, store: _CopyStore, limits: ImageLimits | None = _LIMITS, ranked: bool = True
) -> SearchManager:
    episode = stored_episode("ep_1", parent_id="mc_ep_1")
    episode = dataclasses.replace(
        episode, ref=dataclasses.replace(episode.ref, id="ep_1")
    )
    episode.item.image_ids = [_LATTE]
    episode.item.episode = f"she sent a photo of her latte (img-1: {_LATTE})"
    rows = [hit(episode, 0.8)]
    sign, latte = _picture("im_1", _SIGN), _picture("im_2", _LATTE)
    return SearchManager(
        access=AccessPolicy(InMemoryGroupStore()),
        episode_recaller=EpisodeRecaller(
            RecallerDeps(tokenizer=_tokenizer()),
            ScriptedIndex(sparse=rows, dense=rows, found=rows),  # type: ignore[arg-type]
        ),
        profile_recaller=_NoProfiles(),  # type: ignore[arg-type]
        session_buffer=_NoBuffer(),  # type: ignore[arg-type]
        embedding=_Embedding(),  # type: ignore[arg-type]
        llm_client=FakeLLMClient(handler=_keep_all),
        decider_client=FakeLLMClient(handler=_keep_all),
        tuning_source=static_tuning(),
        trace_path_source=static_trace_path(),
        image_recaller=ImageMemoryRecaller(
            RecallerDeps(tokenizer=_tokenizer()),  # type: ignore[arg-type]
            _PictureIndex([sign, latte], [sign] if ranked else []),  # type: ignore[arg-type]
        ),
        image_store=store,  # type: ignore[arg-type]
        image_limits=limits,
    )


def _request() -> SearchRequest:
    return SearchRequest(user_id="alice", query="the sign")


async def test_ranked_and_attached_pictures_name_the_answer_copy() -> None:
    store = _CopyStore()

    response = await _manager(store=store).search(_request())

    [ranked] = response.data.images
    assert ranked.image_id == _SIGN
    assert ranked.model_path == f"figures/.cache/{_SIGN}-1024.jpeg"
    assert ranked.original_path == f"figures/aa/{_SIGN}.png"  # kept for who wants it
    [attached] = response.data.episodes[0].images
    assert attached.image_id == _LATTE
    assert attached.model_path == f"figures/.cache/{_LATTE}-1024.jpeg"
    # Asked once per picture, at the configured size.
    assert sorted(store.asked) == [(_SIGN, 1024, 85), (_LATTE, 1024, 85)]


async def test_a_picture_without_a_copy_keeps_its_description() -> None:
    store = _CopyStore(missing={_LATTE})

    response = await _manager(store=store).search(_request())

    [attached] = response.data.episodes[0].images
    assert attached.model_path == ""
    assert "EST. 2019" in attached.description
    assert response.data.images[0].model_path.endswith("-1024.jpeg")


async def test_without_limits_no_copy_is_asked_for() -> None:
    store = _CopyStore()

    response = await _manager(store=store, limits=None, ranked=False).search(_request())

    assert response.data.episodes[0].images[0].model_path == ""
    assert store.asked == []


async def test_a_greyscale_picture_s_thumbnail_is_named_png() -> None:
    """The decider's copy of a greyscale picture is PNG, as its model copy is."""
    store = _CopyStore(grey={_LATTE})

    response = await _manager(store=store).search(_request())

    [attached] = response.data.episodes[0].images
    assert attached.model_path == f"figures/.cache/{_LATTE}-1024.png"
    assert attached.thumbnail_path == f"figures/.cache/{_LATTE}-448.png"
    [ranked] = response.data.images
    assert ranked.thumbnail_path == f"figures/.cache/{_SIGN}-448.jpeg"


async def test_a_picture_without_a_copy_keeps_its_first_thumbnail_name() -> None:
    store = _CopyStore(missing={_LATTE}, grey={_LATTE})

    response = await _manager(store=store).search(_request())

    [attached] = response.data.episodes[0].images
    assert attached.thumbnail_path == f"figures/.cache/{_LATTE}-448.jpeg"


def _encoded(mode: str) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (1600, 900), 120 if mode == "L" else (120, 30, 200)).save(
        buffer, format="PNG"
    )
    return buffer.getvalue()


async def test_the_names_are_the_files_a_real_store_writes(tmp_path: Path) -> None:
    """Through the filesystem store: each name is a file once its copy is made."""
    root = MemoryRoot(tmp_path)
    root.ensure()
    database = SqliteDatabase(root, SqliteSettings())
    await database.create_schema()
    store = FileImageStore(root, SqliteRepos.over(database).image_original)
    limits = dataclasses.replace(_LIMITS, max_bytes=10_000_000, formats=("png",))
    items = []
    for mode in ("L", "RGB"):
        stored = await store.store(_encoded(mode), limits=limits)
        items.append(
            SearchImageItem(
                id=f"alice_{mode}",
                entry_id=f"im_{mode}",
                image_id=stored.image_id,
                user_id="alice",
                timestamp=_dt.datetime(2026, 11, 12, tzinfo=_dt.UTC),
                description=mode,
                original_path=await store.relpath_of(stored.image_id),
                thumbnail_path=derived_relpath(stored.image_id, 448),
                score=1.0,
            )
        )

    grey, colour = await name_copies(
        [], items, store=store, limits=limits, thumbnail_edge=448
    )

    assert grey.model_path.endswith("-1024.png")
    assert grey.thumbnail_path == derived_relpath(grey.image_id, 448, "png")
    assert colour.model_path.endswith("-1024.jpeg")
    assert colour.thumbnail_path == derived_relpath(colour.image_id, 448, "jpeg")
    for item in (grey, colour):
        assert (tmp_path / item.model_path).is_file()
        await store.read_derived(item.image_id, edge=448, quality=85)
        assert (tmp_path / item.thumbnail_path).is_file()
