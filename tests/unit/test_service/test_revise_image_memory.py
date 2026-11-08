"""Revising a picture memory: a new version, the old one superseded.

What these pin:

- A revision writes a **new** entry and marks the old one superseded — the old
  text stays on disk, so "what did this memory say before" stays answerable.
- Only the text changes. The picture's identity, path and time are the
  arrival's, not the memory's, and a revision that moved them would be a
  different picture.
- The current occurrence is the default target; a named occurrence can be
  revised instead, and an unknown one is a 404 rather than a silent new memory.
- A picture never written is not revised into existence.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from everos.core.errors import NotFoundError
from everos.memory.models import ImageMemory
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.service.memory._revise import RevisionManager
from tests.fakes import make_stores

_TS = int(_dt.datetime(2026, 5, 14, 10, tzinfo=_dt.UTC).timestamp() * 1000)
_IMAGE_ID = "b" * 64


def _scope() -> Scope:
    return Scope(app_id="app", project_id="proj", owner_id="u1", owner_type="user")


def _picture(
    entry_id: str,
    *,
    description: str = "a sign reading DOU SHE EST 2021",
    ocr: str = "EST 2021",
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
            description=description,
            timestamp=_TS,
            ocr=ocr,
            image_id=_IMAGE_ID,
            name="IMG_2232.jpg",
            original_relpath=f"figures/bb/{_IMAGE_ID}.png",
            media_type="png",
            width=640,
            height="400",  # type: ignore[arg-type]
            parent_id="mc_1",
            entry_id=entry_id,
        ),
    )


def _manager(*rows: Stored[ImageMemory]) -> tuple[RevisionManager, object]:
    stores = make_stores()
    stores.image_memory.records.extend(rows)
    return RevisionManager(stores=stores), stores


async def test_a_revision_writes_a_new_version_and_supersedes_the_old() -> None:
    manager, stores = _manager(_picture("im_1"))

    result = await manager.revise_image(
        scope=_scope(),
        image_id=_IMAGE_ID,
        description="a sign reading DOU SHE EST 2019",
    )

    assert result.entry_id != "im_1"
    assert result.supersedes == "im_1"
    assert result.description == "a sign reading DOU SHE EST 2019"
    records = stores.image_memory.items_for("u1")
    assert [item.description for item in records] == [
        "a sign reading DOU SHE EST 2021",
        "a sign reading DOU SHE EST 2019",
    ]


async def test_the_old_version_stays_in_place_and_is_marked() -> None:
    manager, stores = _manager(_picture("im_1"))

    result = await manager.revise_image(
        scope=_scope(), image_id=_IMAGE_ID, ocr="EST 2019"
    )

    marks = stores.image_memory.deprecated[_picture("im_1").ref.md_path]
    assert marks == {"im_1": result.entry_id}
    # Nothing was deleted: both versions are in the same file.
    assert len(stores.image_memory.records) == 2


async def test_only_the_text_changes() -> None:
    manager, stores = _manager(_picture("im_1"))

    await manager.revise_image(
        scope=_scope(), image_id=_IMAGE_ID, description="new text"
    )

    revised = stores.image_memory.items_for("u1")[-1]
    assert revised.image_id == _IMAGE_ID
    assert revised.name == "IMG_2232.jpg"
    assert revised.original_relpath == f"figures/bb/{_IMAGE_ID}.png"
    assert revised.timestamp == _TS
    assert revised.parent_id == "mc_1"
    assert revised.ocr == "EST 2021"  # untouched


async def test_the_current_occurrence_is_the_default_target() -> None:
    old, current = _picture("im_1"), _picture("im_2")
    manager, stores = _manager(old, current)
    stores.image_memory.deprecated[_picture("im_1").ref.md_path]["im_1"] = "im_2"

    result = await manager.revise_image(
        scope=_scope(), image_id=_IMAGE_ID, description="revised current"
    )

    assert result.supersedes == "im_2"


async def test_a_named_occurrence_can_be_revised() -> None:
    old, current = _picture("im_1"), _picture("im_2")
    manager, stores = _manager(old, current)
    stores.image_memory.deprecated[_picture("im_1").ref.md_path]["im_1"] = "im_2"

    result = await manager.revise_image(
        scope=_scope(), image_id=_IMAGE_ID, description="revised old", entry_id="im_1"
    )

    assert result.supersedes == "im_1"


async def test_an_unknown_occurrence_is_not_created() -> None:
    manager, stores = _manager(_picture("im_1"))

    with pytest.raises(NotFoundError):
        await manager.revise_image(
            scope=_scope(), image_id=_IMAGE_ID, description="x", entry_id="im_9"
        )

    assert len(stores.image_memory.records) == 1


async def test_a_picture_never_written_is_not_revised_into_existence() -> None:
    manager, stores = _manager()

    with pytest.raises(NotFoundError):
        await manager.revise_image(scope=_scope(), image_id=_IMAGE_ID, description="x")

    assert stores.image_memory.records == []


async def test_a_revision_with_nothing_to_change_is_refused() -> None:
    manager, _ = _manager(_picture("im_1"))

    with pytest.raises(ValueError):
        await manager.revise_image(scope=_scope(), image_id=_IMAGE_ID)
