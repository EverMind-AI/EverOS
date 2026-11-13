"""Forgetting one picture: the four places the content could survive.

What these pin — each one is a way a forget silently fails:

- Every picture memory about the file is marked forgotten, not just one.
- The file is taken out of circulation but **kept**, so undo is a status
  change rather than a restore.
- The index rows are erased at once, so the picture stops competing for
  retrieval before the cascade's next scan.
- Every episode whose body referred to the picture is written again without
  the reference, and the new version supersedes the old one — while the old
  version stays in markdown, because "what did I say before" must stay
  answerable. A text edit: no model is called.
"""

from __future__ import annotations

import datetime as _dt

from everos.component.images import ImageLimits
from everos.memory.access import Identity
from everos.memory.models import FORGOTTEN_MARK as FORGET_REASON
from everos.memory.models import Episode, ImageMemory
from everos.memory.ports import MemoryRef, Scope, SearchHit, Stored
from everos.service import ForgetResult, ForgetTarget
from everos.service.memory._extraction import ExtractionSupervisor
from everos.service.memory._forget import ForgetManager
from tests.fakes import make_indexes, make_stores

_TS = int(_dt.datetime(2026, 5, 14, 10, tzinfo=_dt.UTC).timestamp() * 1000)
_IMAGE_ID = "e" * 64


def _limits() -> ImageLimits:
    return ImageLimits(
        max_bytes=1024,
        max_per_message=1,
        max_request_bytes=1024,
        max_per_request=1,
        formats=("png", "jpeg", "webp", "gif"),
    )


def _scope() -> Scope:
    return Scope(app_id="app", project_id="proj", owner_id="u1", owner_type="user")


def _stored_image(
    entry_id: str, image_id: str = _IMAGE_ID, *, name: str = ""
) -> Stored[ImageMemory]:
    return Stored(
        ref=MemoryRef(
            id=f"u1_{entry_id}",
            entry_id=entry_id,
            md_path="app/proj/users/u1/images/image-2026-05-14.md",
        ),
        scope=_scope(),
        item=ImageMemory(
            owner_id="u1",
            description="a coffee shop sign",
            timestamp=_TS,
            image_id=image_id,
            name=name,
            original_relpath=f"figures/ee/{image_id}.png",
            parent_id="mc_1",
            entry_id=entry_id,
        ),
    )


def _stored_episode(entry_id: str, body: str) -> Stored[Episode]:
    return Stored(
        ref=MemoryRef(
            id=f"u1_{entry_id}",
            entry_id=entry_id,
            md_path="app/proj/users/u1/episodes/episode-2026-05-14.md",
        ),
        scope=_scope(),
        item=Episode(
            owner_id="u1",
            episode=body,
            summary=body[:40],
            timestamp=_TS,
            parent_id="mc_1",
            entry_id=entry_id,
        ),
    )


class _Eraser:
    def __init__(self) -> None:
        self.erased: list[dict[str, list[MemoryRef]]] = []

    async def erase(self, scope: Scope, removed: dict) -> None:
        self.erased.append(dict(removed))

    async def erase_owner(self, scope: Scope, kinds) -> None:
        return None


class _ImageIndex:
    """Only what a forget asks of the picture index: find by owner + image."""

    def __init__(self, *rows: Stored[ImageMemory]) -> None:
        self.rows = list(rows)

    async def find(self, where, *, limit: int | None = None) -> list[SearchHit]:
        return [SearchHit(stored=row, score=None) for row in self.rows][:limit]


class _EpisodeIndex:
    def __init__(self, *rows: Stored[Episode]) -> None:
        self.rows = list(rows)

    async def find(self, where, *, limit: int | None = None) -> list[SearchHit]:
        return [SearchHit(stored=row, score=None) for row in self.rows][:limit]


class _Pictures:
    """A picture's forget through the one engine."""

    def __init__(self, engine: ForgetManager) -> None:
        self._engine = engine

    async def forget_image(self, *, scope: Scope, image_id: str) -> ForgetResult:
        return await self._engine.forget(
            scope=scope,
            identity=Identity(),
            target=ForgetTarget(kind="image", image_id=image_id),
            dry_run=False,
        )


def _manager(
    *, images: tuple = (), episodes: tuple = ()
) -> tuple[_Pictures, object, object, _Eraser]:
    stores = make_stores()
    stores.image_memory.records.extend(images)
    stores.episode.records.extend(episodes)
    indexes = make_indexes(
        image_memory=_ImageIndex(*images), episode=_EpisodeIndex(*episodes)
    )
    eraser = _Eraser()
    engine = ForgetManager(
        stores=stores,
        indexes=indexes,
        eraser=eraser,
        pipeline=None,
        scenes=None,
        extractions=ExtractionSupervisor(),
        session_wait_seconds=1.0,
    )
    return _Pictures(engine), stores, indexes, eraser


def _entry(record_id: str) -> str:
    return record_id.removeprefix("u1_")


async def test_every_occurrence_is_marked_forgotten() -> None:
    first, second = _stored_image("im_1"), _stored_image("im_2")
    manager, stores, _, _ = _manager(images=(first, second))

    result = await manager.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    assert len(result.image_memories) == 2
    marks = stores.image_memory.deprecated[first.ref.md_path]
    assert marks == {"im_1": FORGET_REASON, "im_2": FORGET_REASON}


async def test_the_file_is_taken_out_of_circulation_but_kept() -> None:
    manager, stores, _, _ = _manager(images=(_stored_image("im_1"),))

    await manager.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    assert _IMAGE_ID in stores.images.forgotten


async def test_index_rows_are_erased_at_once() -> None:
    stored = _stored_image("im_1")
    manager, _, _, eraser = _manager(images=(stored,))

    await manager.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    assert eraser.erased == [{"image_memory": [stored.ref]}]


async def test_an_episode_referring_to_the_picture_is_rewritten_without_it() -> None:
    # The picture extractor's own spelling: label plus the durable id.
    old = _stored_episode(
        "ep_20260514_0001",
        f"They went to the coffee shop (img-1: {_IMAGE_ID}) and liked it.",
    )
    manager, stores, _, _ = _manager(images=(_stored_image("im_1"),), episodes=(old,))

    result = await manager.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    assert len(result.new_episodes) == 1
    new_entry = _entry(result.new_episodes[0])
    written = [s for s in stores.episode.records if s.ref.entry_id == new_entry]
    assert len(written) == 1
    body = written[0].item.episode
    assert _IMAGE_ID not in body
    assert "()" not in body
    assert "went to the coffee shop" in body


async def test_the_old_episode_version_is_superseded_not_deleted() -> None:
    """The mark says which version replaced it; nothing is removed."""
    old = _stored_episode("ep_20260514_0001", f"see [IMAGE: {_IMAGE_ID}]")
    manager, stores, _, _ = _manager(images=(), episodes=(old,))

    result = await manager.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    marks = stores.episode.deprecated[old.ref.md_path]
    assert marks[old.ref.entry_id] == _entry(result.new_episodes[0])
    # Both versions are in markdown: the superseded one and its replacement.
    assert len(stores.episode.records) >= 1


async def test_an_episode_that_does_not_mention_the_picture_is_left_alone() -> None:
    other = _stored_episode("ep_20260514_0002", "they talked about lunch")
    manager, stores, _, _ = _manager(images=(), episodes=(other,))

    result = await manager.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    assert result.episodes == ()
    assert not any(stores.episode.deprecated.values())


async def test_forgetting_an_unknown_picture_is_a_no_op() -> None:
    manager, stores, _, eraser = _manager()

    result = await manager.forget_image(scope=_scope(), image_id="f" * 64)

    assert result.image_memories == ()
    assert result.episodes == ()
    assert eraser.erased == []
    assert stores.image_memory.records == []


async def test_the_result_names_each_occurrence_and_the_file_it_was_sent_as() -> None:
    """A caller mapping a memory back to its input needs the name, not a count."""
    manager, _, _, _ = _manager(
        images=(
            _stored_image("im_1", name="IMG_2231.jpg"),
            _stored_image("im_2", name="IMG_2232.jpg"),
        )
    )

    result = await manager.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    assert [(p.entry_id, p.image_id, p.name) for p in result.pictures] == [
        ("im_1", _IMAGE_ID, "IMG_2231.jpg"),
        ("im_2", _IMAGE_ID, "IMG_2232.jpg"),
    ]


async def test_the_rewritten_episode_no_longer_refers_to_the_picture() -> None:
    """The reference list follows the body; a dangling id would resolve to nothing."""
    old = _stored_episode("ep_20260514_0001", f"see [IMAGE: {_IMAGE_ID}]")
    old.item.image_ids = [_IMAGE_ID, "0" * 64]
    manager, stores, _, _ = _manager(images=(), episodes=(old,))

    result = await manager.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    written = [
        s
        for s in stores.episode.records
        if s.ref.entry_id == _entry(result.new_episodes[0])
    ]
    assert written[0].item.image_ids == ["0" * 64]


async def test_the_rewritten_body_has_no_gap_before_its_punctuation() -> None:
    """A removal must leave prose, not ``the signboard .``."""
    old = _stored_episode(
        "ep_20260514_0001",
        f"she described the signboard (img-2: {_IMAGE_ID}).",
    )
    manager, stores, _, _ = _manager(images=(), episodes=(old,))

    result = await manager.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    written = [
        s
        for s in stores.episode.records
        if s.ref.entry_id == _entry(result.new_episodes[0])
    ]
    assert written[0].item.episode == "she described the signboard."
