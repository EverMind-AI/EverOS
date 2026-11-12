"""Forgetting a conversation: marked, not deleted, and reversible.

What these pin:

- Every episode and picture memory that came from one session is marked
  forgotten, through the same forget log a picture forget writes.
- A picture *file* is taken out of circulation only when this session held its
  last live reference; a picture that also arrived in another conversation
  stays.
- Undo clears exactly the marks the forget wrote, and brings the files back.
- Another participant's memories of the same conversation are untouched.
- A profile item (or a replaced value) goes once every segment it rests on is
  forgotten, and comes back, once, when one of them is remembered again.
"""

from __future__ import annotations

import datetime as _dt

from everos.memory.models import (
    Episode,
    ExplicitInfoItem,
    ImageMemory,
    MemCell,
    Profile,
)
from everos.memory.ports import MemoryRef, Scope, SessionKey, Stored
from everos.service.memory._forget import FORGET_REASON
from everos.service.memory._forget_session import SessionForgetManager
from everos.service.memory._undo import UndoManager
from tests.fakes import make_stores

_TS = int(_dt.datetime(2026, 5, 14, 10, tzinfo=_dt.UTC).timestamp() * 1000)
_IMAGE_A = "1" * 64
_IMAGE_B = "2" * 64


def _scope(owner: str = "u1") -> Scope:
    return Scope(app_id="app", project_id="proj", owner_id=owner, owner_type="user")


def _episode(
    entry_id: str, *, session: str = "s1", owner: str = "u1"
) -> Stored[Episode]:
    return Stored(
        ref=MemoryRef(
            id=f"{owner}_{entry_id}",
            entry_id=entry_id,
            md_path=f"app/proj/users/{owner}/episodes/episode-2026-05-14.md",
        ),
        scope=_scope(owner),
        item=Episode(
            owner_id=owner,
            episode="they talked about the cafe",
            summary="the cafe",
            timestamp=_TS,
            parent_id="mc_1",
            session_id=session,
            entry_id=entry_id,
        ),
    )


def _picture(
    entry_id: str,
    image_id: str,
    *,
    session: str = "s1",
    owner: str = "u1",
) -> Stored[ImageMemory]:
    return Stored(
        ref=MemoryRef(
            id=f"{owner}_{entry_id}",
            entry_id=entry_id,
            md_path=f"app/proj/users/{owner}/images/image-2026-05-14.md",
        ),
        scope=_scope(owner),
        item=ImageMemory(
            owner_id=owner,
            description="a coffee shop sign",
            timestamp=_TS,
            image_id=image_id,
            original_relpath=f"figures/{image_id[:2]}/{image_id}.png",
            session_id=session,
            parent_id="mc_1",
            entry_id=entry_id,
        ),
    )


class _Eraser:
    def __init__(self) -> None:
        self.erased: list[dict] = []
        self.requeued: list[tuple[str, set[str]]] = []

    async def erase(self, scope: Scope, removed: dict) -> None:
        self.erased.append(removed)

    async def erase_owner(self, scope: Scope, kinds) -> None:
        return None

    async def requeue(self, kind: str, md_paths) -> None:
        self.requeued.append((kind, set(md_paths)))


def _manager(*stored):
    stores = make_stores()
    for row in stored:
        target = (
            stores.episode if isinstance(row.item, Episode) else stores.image_memory
        )
        target.records.append(row)
        if isinstance(row.item, ImageMemory):
            stores.images.images[row.item.image_id] = (b"bytes", "png")
    eraser = _Eraser()
    forget = SessionForgetManager(stores=stores, images=stores.images, eraser=eraser)
    undo = UndoManager(stores=stores, images=stores.images, eraser=eraser)
    return forget, undo, stores, eraser


async def test_the_sessions_episodes_and_pictures_are_marked() -> None:
    forget, _, stores, _ = _manager(
        _episode("ep_1"),
        _picture("im_1", _IMAGE_A),
        _episode("ep_2", session="s2"),
        _picture("im_2", _IMAGE_B, session="s2"),
    )

    result = await forget.forget_session(scope=_scope(), session_id="s1")

    assert result.episodes == ["ep_1"]
    assert result.image_memories == 1
    assert stores.episode.deprecated[_episode("ep_1").ref.md_path] == {
        "ep_1": FORGET_REASON
    }
    assert stores.image_memory.deprecated[_picture("im_1", _IMAGE_A).ref.md_path] == {
        "im_1": FORGET_REASON
    }
    # The other conversation is untouched.
    assert _episode("ep_2", session="s2").ref.entry_id not in str(
        stores.episode.deprecated
    )


async def test_a_file_goes_only_when_the_last_reference_does() -> None:
    """The same picture in two conversations survives forgetting one of them."""
    forget, _, stores, _ = _manager(
        _picture("im_1", _IMAGE_A),
        _picture("im_2", _IMAGE_A, session="s2"),
        _picture("im_3", _IMAGE_B),
    )

    result = await forget.forget_session(scope=_scope(), session_id="s1")

    assert result.images == [_IMAGE_B]
    assert _IMAGE_B in stores.images.forgotten
    assert _IMAGE_A not in stores.images.forgotten


async def test_only_the_owners_memories_are_forgotten() -> None:
    forget, _, stores, _ = _manager(
        _episode("ep_1", owner="u1"),
        _episode("ep_1", owner="u2"),
    )

    await forget.forget_session(scope=_scope("u1"), session_id="s1")

    assert stores.episode.deprecated[_episode("ep_1", owner="u2").ref.md_path] == {}


async def test_the_forget_is_recorded_in_the_log() -> None:
    forget, _, stores, _ = _manager(_episode("ep_1"))

    result = await forget.forget_session(scope=_scope(), session_id="s1")

    entry = stores.forget_log.entries[-1]
    assert entry.target == "session"
    assert entry.target_id == "s1"
    assert entry.forget_id == result.forget_id
    assert entry.episode_ids == (("ep_1", ""),)


async def test_index_rows_are_erased_with_their_owner_prefixed_key() -> None:
    forget, _, _, eraser = _manager(_episode("ep_1"), _picture("im_1", _IMAGE_A))

    await forget.forget_session(scope=_scope(), session_id="s1")

    erased = eraser.erased[0]
    assert [ref.id for ref in erased["episode"]] == ["u1_ep_1"]
    assert [ref.id for ref in erased["image_memory"]] == ["u1_im_1"]


async def test_undo_brings_the_conversation_back() -> None:
    forget, undo, stores, eraser = _manager(
        _episode("ep_1"), _picture("im_1", _IMAGE_A)
    )
    await forget.forget_session(scope=_scope(), session_id="s1")
    eraser.requeued.clear()

    result = await undo.undo_session(scope=_scope(), session_id="s1")

    assert result.applied
    assert result.episodes == ("ep_1",)
    assert result.image_memories == 1
    assert result.images == (_IMAGE_A,)
    assert _IMAGE_A not in stores.images.forgotten
    assert stores.episode.deprecated[_episode("ep_1").ref.md_path] == {}
    assert stores.image_memory.deprecated[_picture("im_1", _IMAGE_A).ref.md_path] == {}
    assert {kind for kind, _ in eraser.requeued} == {"episode", "image_memory"}


async def test_undo_hands_the_episode_files_back_to_the_cascade() -> None:
    """The cascade is handed the files the undo rewrote, not the entry ids.

    An entry id is no path: queued as one, the projection never re-read the
    file, and the restored episodes waited for the next full scan.
    """
    forget, undo, _, eraser = _manager(_episode("ep_1"), _episode("ep_2"))
    await forget.forget_session(scope=_scope(), session_id="s1")
    eraser.requeued.clear()

    await undo.undo_session(scope=_scope(), session_id="s1")

    episode_files = {_episode("ep_1").ref.md_path}
    assert ("episode", episode_files) in eraser.requeued


async def test_a_second_undo_of_a_conversation_restores_nothing() -> None:
    forget, undo, _, _ = _manager(_episode("ep_1"))
    await forget.forget_session(scope=_scope(), session_id="s1")
    await undo.undo_session(scope=_scope(), session_id="s1")

    again = await undo.undo_session(scope=_scope(), session_id="s1")

    assert again.applied is False
    assert again.episodes == ()


async def test_undoing_a_conversation_never_forgotten_is_a_no_op() -> None:
    _, undo, _, _ = _manager(_episode("ep_1"))

    result = await undo.undo_session(scope=_scope(), session_id="s1")

    assert result.applied is False
    assert result.forget_id == ""


def _segments(stores, session: str, *memcell_ids: str) -> None:
    key = SessionKey(app_id="app", project_id="proj", session_id=session)
    for memcell_id in memcell_ids:
        stores.memcell.seed(memcell_id, MemCell(items=[], timestamp=_TS), key)


def _fact(description: str, *sources: str) -> ExplicitInfoItem:
    extra = {"source_memcell_ids": list(sources)} if sources else {}
    return ExplicitInfoItem(description=description, **extra)


def _with_profile(stores, *items: ExplicitInfoItem, history=()) -> None:
    stores.profile.profiles[(_scope(), "u1")] = Profile(
        owner_id="u1",
        summary="",
        timestamp=_TS,
        explicit_info=list(items),
        history=list(history),
    )


def _facts(stores) -> list[str]:
    return [
        i.description for i in stores.profile.profiles[(_scope(), "u1")].explicit_info
    ]


async def test_a_forgotten_conversation_takes_the_items_resting_on_it_alone() -> None:
    forget, _, stores, _ = _manager(_episode("ep_1"))
    _segments(stores, "s1", "mc_1")
    _segments(stores, "s2", "mc_2")
    _with_profile(
        stores,
        _fact("Lives in Paris", "mc_1"),
        _fact("Works in Python", "mc_1", "mc_2"),
        _fact("Likes tea"),
    )

    result = await forget.forget_session(scope=_scope(), session_id="s1")

    assert result.profiles == 1
    assert _facts(stores) == ["Works in Python", "Likes tea"]
    ((bucket, item),) = stores.forget_log.entries[-1].removed_profile_items
    assert (bucket, item["description"]) == ("explicit_info", "Lives in Paris")


async def test_the_summary_goes_with_an_item_and_comes_back_with_the_undo() -> None:
    """Written over every item, it may repeat the forgotten one."""
    forget, undo, stores, _ = _manager(_episode("ep_1"))
    _segments(stores, "s1", "mc_1")
    _segments(stores, "s2", "mc_2")
    _with_profile(stores, _fact("Lives in Paris", "mc_1"), _fact("Likes tea", "mc_2"))
    stores.profile.profiles[(_scope(), "u1")].summary = "Lives in Paris"

    await forget.forget_session(scope=_scope(), session_id="s1")
    assert stores.profile.profiles[(_scope(), "u1")].summary == ""

    result = await undo.undo_session(scope=_scope(), session_id="s1")
    assert result.profiles == 1
    assert stores.profile.profiles[(_scope(), "u1")].summary == "Lives in Paris"


async def test_an_item_goes_with_the_last_conversation_it_rests_on() -> None:
    forget, _, stores, _ = _manager(_episode("ep_1"))
    _segments(stores, "s1", "mc_1")
    _segments(stores, "s2", "mc_2")
    _with_profile(stores, _fact("Works in Python", "mc_1", "mc_2"))

    first = await forget.forget_session(scope=_scope(), session_id="s1")
    second = await forget.forget_session(scope=_scope(), session_id="s2")

    assert (first.profiles, second.profiles) == (0, 1)
    assert _facts(stores) == []


async def test_undo_brings_back_the_items_and_what_a_later_forget_took() -> None:
    forget, undo, stores, _ = _manager(_episode("ep_1"))
    _segments(stores, "s1", "mc_1")
    _segments(stores, "s2", "mc_2")
    _with_profile(
        stores,
        _fact("Lives in Paris", "mc_1"),
        _fact("Works in Python", "mc_1", "mc_2"),
        history=[
            {
                "description": "Lived in Lyon",
                "replaced_at": _TS,
                "source_memcell_ids": ["mc_1"],
            }
        ],
    )
    await forget.forget_session(scope=_scope(), session_id="s1")
    await forget.forget_session(scope=_scope(), session_id="s2")
    assert _facts(stores) == []

    result = await undo.undo_session(scope=_scope(), session_id="s1")

    # "Works in Python" went with s2's forget, but rests on s1 too.
    assert result.profiles == 2
    assert sorted(_facts(stores)) == ["Lives in Paris", "Works in Python"]
    history = stores.profile.profiles[(_scope(), "u1")].history
    assert [entry["description"] for entry in history] == ["Lived in Lyon"]

    again = await undo.undo_session(scope=_scope(), session_id="s2")
    assert again.applied and again.profiles == 0  # nothing doubled
    assert sorted(_facts(stores)) == ["Lives in Paris", "Works in Python"]
