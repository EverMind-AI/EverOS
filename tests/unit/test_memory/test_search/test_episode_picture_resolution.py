"""An episode's picture references resolve for the reader.

What these pin — the gap this closes:

- An episode body names its pictures by id and never restates what they show
  (the picture has its own memory). So an episode selected **without** its
  picture memory must still deliver the picture's description and in-image
  text, or the answering model gets a dangling id and no content — worse than
  the 1.0 behaviour of inlining the parsed picture text.
- A forgotten or superseded picture is **not** revived by an episode that still
  refers to it: the reference stays, its content does not. That is what a
  forget means.
- A picture selected as core memory and also referred to by an episode appears
  once in the result, not twice.
"""

from __future__ import annotations

import datetime as _dt

from everos.memory.models import ImageMemory
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.memory.search.dto import SearchEpisodeItem
from everos.memory.search.shaper import attach_referenced_pictures

_TS = int(_dt.datetime(2026, 5, 14, 10, tzinfo=_dt.UTC).timestamp() * 1000)
_SIGN = "a" * 64
_LATTE = "b" * 64


def _scope() -> Scope:
    return Scope(app_id="app", project_id="proj", owner_id="u1", owner_type="user")


def _picture(
    entry_id: str,
    image_id: str,
    *,
    description: str = "a chalkboard sign reading DOU SHE EST 2019",
    ocr: str = "EST. 2019",
    superseded_by: str | None = None,
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
            image_id=image_id,
            name="IMG_2232.png",
            original_relpath=f"figures/aa/{image_id}.png",
            media_type="png",
            parent_id="mc_1",
            entry_id=entry_id,
            status="superseded" if superseded_by else "current",
            superseded_by=superseded_by,
        ),
    )


def _episode(body: str, image_ids: list[str]) -> SearchEpisodeItem:
    return SearchEpisodeItem(
        id="u1_ep_1",
        user_id="u1",
        timestamp=_dt.datetime(2026, 5, 14, 10, tzinfo=_dt.UTC),
        summary="",
        subject="",
        episode=body,
        type="Conversation",
        score=1.0,
        image_ids=image_ids,
    )


def _resolve(
    episode: SearchEpisodeItem, stored: list[Stored[ImageMemory]], *, selected=()
):
    by_id = {row.item.image_id: row for row in stored}
    return attach_referenced_pictures(
        [episode],
        image_ids=[row.item.image_id for row in stored],
        stored_images=by_id,
        selected_ids=set(selected),
        thumbnail_edge=448,
    )


def test_an_episode_kept_without_its_picture_still_carries_the_content() -> None:
    episode = _episode(f"she sent a photo of the sign (img-2: {_SIGN})", [_SIGN])

    extra = _resolve(episode, [_picture("im_1", _SIGN)])

    assert len(episode.images) == 1
    attached = episode.images[0]
    assert attached.image_id == _SIGN
    assert "DOU SHE" in attached.description
    assert attached.ocr == "EST. 2019"
    assert attached.original_path.endswith(f"{_SIGN}.png")
    assert attached.thumbnail_path
    # Reported alongside the ranked pictures so a client sees what resolved.
    assert [item.image_id for item in extra] == [_SIGN]


def test_the_attachment_uses_the_label_the_body_used() -> None:
    """A reader needs to tie the description to the sentence that named it."""
    episode = _episode(
        f"a latte (img-1: {_LATTE}), then the sign (img-2: {_SIGN})", [_LATTE, _SIGN]
    )

    _resolve(episode, [_picture("im_1", _SIGN), _picture("im_2", _LATTE)])

    labels = {image.image_id: image.label for image in episode.images}
    assert labels == {_LATTE: "img-1", _SIGN: "img-2"}


def test_a_forgotten_picture_is_not_revived_by_an_episode_reference() -> None:
    """The episode keeps its reference; the content is gone, which is the point."""
    episode = _episode(f"she sent a photo of the sign (img-2: {_SIGN})", [_SIGN])

    extra = _resolve(episode, [])  # the lookup returns only current pictures

    assert episode.images == []
    assert extra == []


def test_a_superseded_picture_is_not_attached() -> None:
    """Only a current occurrence resolves: an old one is history, not content."""
    episode = _episode(f"the sign (img-2: {_SIGN})", [_SIGN])
    old = _picture("im_0", _SIGN, superseded_by="im_1")

    # The index lookup itself filters superseded rows; the resolver honours it.
    extra = _resolve(episode, [])

    assert extra == []
    assert old.item.superseded_by == "im_1"


def test_a_selected_picture_is_not_duplicated_inside_the_episode() -> None:
    episode = _episode(f"the sign (img-2: {_SIGN})", [_SIGN])

    extra = _resolve(episode, [_picture("im_1", _SIGN)], selected={_SIGN})

    assert episode.images == []
    assert extra == []


def test_an_episode_with_no_pictures_gets_nothing() -> None:
    episode = _episode("they talked about lunch", [])

    assert _resolve(episode, [_picture("im_1", _SIGN)]) == []
    assert episode.images == []
