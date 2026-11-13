"""Undoing a forget: what comes back, and what stays gone.

What these pin — each is a way an undo silently fails:

- The picture file, every picture memory about it, every episode version the
  forget rewrote, and every profile item the forget removed come back.
- The pre-forget episode *text* is current again: the restored version carries
  the picture reference, and the forget's rewrite is superseded by it.
- A second undo of the same forget restores nothing, and an undo of a picture
  that was never forgotten is a no-op that still reports the current state.
- A picture forgotten *again* after an undo is not brought back by undoing the
  older forget: the newest live forget is the one an undo reverses.
- A group whose id is the user's never has its forget reversed by the user's
  undo: the log is read by owner id and type.
"""

from __future__ import annotations

import datetime as _dt

from everos.component.images import ImageLimits
from everos.memory.access import Identity
from everos.memory.models import FORGOTTEN_MARK as FORGET_REASON
from everos.memory.models import Episode, ImageMemory, Profile
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.service import ForgetResult, ForgetTarget, UndoResult
from everos.service.memory._extraction import ExtractionSupervisor
from everos.service.memory._forget import ForgetManager
from tests.fakes import make_indexes, make_stores

_TS = int(_dt.datetime(2026, 5, 14, 10, tzinfo=_dt.UTC).timestamp() * 1000)
_IMAGE_ID = "d" * 64


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


def _picture(entry_id: str, *, name: str = "IMG_2231.jpg") -> Stored[ImageMemory]:
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
            image_id=_IMAGE_ID,
            name=name,
            original_relpath=f"figures/dd/{_IMAGE_ID}.png",
            parent_id="mc_1",
            entry_id=entry_id,
        ),
    )


def _episode(entry_id: str, body: str) -> Stored[Episode]:
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


class _ImageIndex:
    def __init__(self, *rows: Stored[ImageMemory]) -> None:
        self.rows = list(rows)

    async def find(self, where, *, limit: int | None = None):
        from everos.memory.ports import SearchHit

        return [SearchHit(stored=row, score=None) for row in self.rows][:limit]


class _EpisodeIndex:
    def __init__(self, *rows: Stored[Episode]) -> None:
        self.rows = list(rows)

    async def find(self, where, *, limit: int | None = None):
        from everos.memory.ports import SearchHit

        return [SearchHit(stored=row, score=None) for row in self.rows][:limit]


class _Eraser:
    def __init__(self) -> None:
        self.requeued: list[tuple[str, set[str]]] = []

    async def erase(self, scope: Scope, removed: dict) -> None:
        return None

    async def erase_owner(self, scope: Scope, kinds) -> None:
        return None

    async def requeue(self, kind: str, md_paths) -> None:
        self.requeued.append((kind, set(md_paths)))


def _manager(images=(), episodes=(), *, profile: Profile | None = None):
    stores = make_stores()
    for row in images:
        stores.image_memory.records.append(row)
    for row in episodes:
        stores.episode.records.append(row)
    if profile is not None:
        stores.profile.profiles[(_scope(), "u1")] = profile
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
    return _Pictures(engine), _Pictures(engine), stores, eraser


class _Pictures:
    """A picture's forget and its undo by target, through the one engine."""

    def __init__(self, engine: ForgetManager) -> None:
        self._engine = engine

    async def forget_image(self, *, scope: Scope, image_id: str) -> ForgetResult:
        return await self._engine.forget(
            scope=scope,
            identity=Identity(),
            target=ForgetTarget(kind="image", image_id=image_id),
            dry_run=False,
        )

    async def undo_image(self, *, scope: Scope, image_id: str) -> UndoResult:
        return await self._engine.undo_picture(scope=scope, image_id=image_id)


async def test_the_picture_and_its_memories_come_back() -> None:
    forget, undo, stores, _ = _manager(images=(_picture("im_1"),))
    stores.images.images[_IMAGE_ID] = (b"bytes", "png")

    await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)
    assert _IMAGE_ID in stores.images.forgotten

    result = await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    assert result.applied
    assert _IMAGE_ID not in stores.images.forgotten
    assert len(result.image_memories) == 1
    assert [(p.entry_id, p.name) for p in result.pictures] == [("im_1", "IMG_2231.jpg")]
    marks = stores.image_memory.deprecated[_picture("im_1").ref.md_path]
    assert marks == {}


async def test_the_pre_forget_episode_text_is_current_again() -> None:
    old = _episode("ep_1", f"they went for coffee (img-1: {_IMAGE_ID})")
    forget, undo, stores, _ = _manager(images=(_picture("im_1"),), episodes=(old,))

    forgotten = await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)
    rewritten_id = forgotten.new_episodes[0].removeprefix("u1_")
    await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    # The forget's rewrite is superseded by the version the undo appended.
    restored = stores.episode.records[-1]
    assert restored.ref.entry_id not in {old.ref.entry_id, rewritten_id}
    assert _IMAGE_ID in restored.item.episode
    superseded_by = {
        key: value
        for marks in stores.episode.deprecated.values()
        for key, value in marks.items()
    }
    assert superseded_by.get(rewritten_id) == restored.ref.entry_id


async def test_the_picture_and_episode_files_are_projected_again() -> None:
    """Erased rows and cleared marks only reach the index through the cascade."""
    forget, undo, _, eraser = _manager(
        images=(_picture("im_1"),),
        episodes=(_episode("ep_1", f"[IMAGE: {_IMAGE_ID}]"),),
    )
    await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)
    eraser.requeued.clear()

    await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    kinds = {kind for kind, _ in eraser.requeued}
    assert kinds == {"image_memory", "episode"}


async def test_a_profile_item_the_picture_established_comes_back() -> None:
    profile = Profile(
        owner_id="u1",
        summary="",
        timestamp=_TS,
        explicit_info=[
            {
                "description": "Has a cat named Biscuit",
                "category": "pets",
                "evidence": "2026-05-14: the collar tag reads Biscuit",
                "observed_at": _TS,
                "source_image_ids": [_IMAGE_ID],
            }
        ],
    )
    forget, undo, stores, _ = _manager(images=(_picture("im_1"),), profile=profile)

    forgotten = await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)
    assert forgotten.profile_item_count == 1
    assert stores.profile.profiles[(_scope(), "u1")].explicit_info == []

    result = await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    assert result.profile_item_count == 1
    kept = stores.profile.profiles[(_scope(), "u1")].explicit_info
    assert [item.description for item in kept] == ["Has a cat named Biscuit"]
    assert kept[0].observed_at == _TS


async def test_a_replaced_value_the_picture_established_goes_and_comes_back() -> None:
    """``user.md``'s history holds old values; the picture surfaces from neither."""
    profile = Profile(
        owner_id="u1",
        summary="",
        timestamp=_TS,
        explicit_info=[{"description": "Has a dog named Rex", "observed_at": _TS}],
        history=[
            {
                "description": "Has a cat named Biscuit",
                "replaced_at": _TS,
                "source_image_ids": [_IMAGE_ID],
            }
        ],
    )
    forget, undo, stores, _ = _manager(images=(_picture("im_1"),), profile=profile)

    forgotten = await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)
    assert forgotten.profile_item_count == 0  # no item went; a replaced value did
    after = stores.profile.profiles[(_scope(), "u1")]
    assert after.history == []
    assert [item.description for item in after.explicit_info] == ["Has a dog named Rex"]

    result = await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    assert result.profile_item_count == 0
    back = stores.profile.profiles[(_scope(), "u1")]
    assert [entry["description"] for entry in back.history] == [
        "Has a cat named Biscuit"
    ]


async def test_a_second_undo_restores_nothing() -> None:
    forget, undo, _, _ = _manager(images=(_picture("im_1"),))
    await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)
    await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    again = await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    assert again.applied is False
    assert again.forget_id == ""
    assert len(again.pictures) == 1  # still current, and reported as such


async def test_undoing_a_picture_never_forgotten_is_a_no_op() -> None:
    _, undo, _, _ = _manager(images=(_picture("im_1"),))

    result = await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    assert result.applied is False
    assert result.episodes == ()
    assert [p.entry_id for p in result.pictures] == ["im_1"]


async def test_a_later_forget_is_not_undone_by_the_older_ones_undo() -> None:
    """Two forgets of one picture: undo reverses the newest, and only it."""
    forget, undo, stores, _ = _manager(images=(_picture("im_1"),))
    await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)
    await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)
    await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)

    result = await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    assert result.applied
    assert result.forget_id == stores.forget_log.entries[-1].forget_id
    assert stores.forget_log.undone == {
        stores.forget_log.entries[0].forget_id,
        stores.forget_log.entries[-1].forget_id,
    }


async def test_a_users_undo_never_reverses_a_same_named_groups_forget() -> None:
    """A group whose id is the user's forgot the picture last: the user's
    undo still reverses the user's own forget, and the group's stays live."""
    forget, undo, stores, _ = _manager(images=(_picture("im_1"),))
    await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)
    group = await stores.forget_log.record(
        target="image",
        target_id=_IMAGE_ID,
        app_id="app",
        project_id="proj",
        owner_id="u1",
        owner_type="group",
        undo={"withdrawn": []},
    )

    result = await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    assert result.applied
    assert result.forget_id == stores.forget_log.entries[0].forget_id
    assert group.forget_id not in stores.forget_log.undone


async def test_only_the_forgets_own_occurrences_come_back() -> None:
    """An occurrence the forget did not mark is not touched by its undo."""
    marked, other = _picture("im_1"), _picture("im_2")
    forget, undo, stores, _ = _manager(images=(marked,))
    await forget.forget_image(scope=_scope(), image_id=_IMAGE_ID)
    # A new arrival after the forget: current, and not part of the log entry.
    stores.image_memory.records.append(other)

    await undo.undo_image(scope=_scope(), image_id=_IMAGE_ID)

    marks = stores.image_memory.deprecated[marked.ref.md_path]
    assert marked.ref.entry_id not in marks
    assert FORGET_REASON not in marks.values()
