"""A forget or an undo whose process stops part-way is settled afterwards.

docs/api.md, "Forget durability". A forget writes its log entry before its
first change, in the ``forgetting`` stage, and saves into it ahead of each step
the change the step makes; no read counts the entry until the run completes
it. What these pin, each with the process "stopped" at one point -- the step
raises and the run's own reversal never runs, as when the process is killed:

- the next start (``finish_interrupted_forgets``) reverses the run from its
  entry: every mark, every version it wrote (also one it was writing when it
  stopped), every profile value -- put back once, whether the document had
  taken the change or not -- and the entry ends ``rolled_back``; the same
  forget sent again works and its undo brings everything back;
- with no restart, the next forget or undo of the owner settles it first;
- an undo cut short (``undoing``) is finished, a copy it already wrote taken
  as written, never written twice;
- a run that fails in-process is reversed by the same steps; when even that
  fails, it stays an incomplete forget its undo finishes.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from everos.memory.models import FORGOTTEN_MARK, Profile
from everos.memory.ports import SessionKey
from everos.memory.search import SearchRequest
from everos.service import ForgetTarget
from everos.service.memory._forget import ForgetManager
from everos.service.memory._forget_apply import ForgetWriter
from everos.service.memory._forget_rollback import Rollback, _took
from everos.service.memory._forget_undo import Undoer
from tests.helpers.forgets import (
    T0,
    ForgetModel,
    forget,
    items,
    live,
    message_ids,
    open_world,
    profile_update,
    send,
    undo,
)
from tests.helpers.groups import APP, PROJECT


class _Stop(BaseException):
    """The process stops here."""


def _stop(
    monkeypatch: pytest.MonkeyPatch,
    owner: Any,
    name: str,
    *,
    after: bool = False,
    when: Callable[..., bool] | None = None,
) -> None:
    """Make ``owner.name`` stop the process: before it runs, or ``after``."""
    original = getattr(owner, name)

    async def stopped(*args: Any, **kwargs: Any) -> Any:
        if when is not None and not when(*args, **kwargs):
            return await original(*args, **kwargs)
        if after:
            await original(*args, **kwargs)
        raise _Stop

    monkeypatch.setattr(owner, name, stopped)


def _no_reversal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stopped process runs nothing after the stop: no reversal either."""

    async def gone(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(ForgetManager, "_roll_back", gone)


_SESSION = ForgetTarget(kind="session", session_id="p1")


async def _seed(world: Any) -> None:
    await send(
        world,
        "p1",
        ("amy", "Keep ZQFONE10 and ZQA11.", 1_000),
        ("amy", "Also ZQC12.", 1_001),
        ("bot", "Ok.", 1_002),
    )
    await send(
        world, "p2", ("amy", "Then ZQFTWO20 and ZQB21.", 60_000), ("bot", "Ok.", 60_001)
    )
    await profile_update(world)


async def _entries(world: Any) -> list[tuple[str, str | None]]:
    """Every episode entry of amy's files with its mark."""
    return sorted(
        (
            (s.item.episode, mark)
            for s, mark in await world.stores.episode.read_all(world.personal("amy"))
        ),
        key=lambda row: (row[0], row[1] or ""),
    )


async def _searchable(world: Any, marker: str) -> bool:
    await world.project()
    found = await world.memory.search(
        SearchRequest(app_id=APP, project_id=PROJECT, user_id="amy", query=marker)
    )
    return marker in found.data.model_dump_json()


async def test_a_forget_stopped_after_its_marks_is_reversed_at_the_next_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        await _seed(world)
        before, held = await _entries(world), sorted(await items(world))
        assert held == ["Holds ZQFONE10", "Holds ZQFTWO20"]
        _no_reversal(monkeypatch)
        _stop(monkeypatch, ForgetWriter, "profiles")
        with pytest.raises(_Stop):
            await forget(world, _SESSION)
        monkeypatch.undo()

        # Half done, and not a forget: no read counts the entry.
        assert not [t for t in await live(world) if "ZQA11" in t]
        (entry,) = await world.stores.forget_log.staged()
        assert entry.stage == "forgetting" and entry.target_id == "p1"
        log = world.stores.forget_log
        space = {"app_id": APP, "project_id": PROJECT, "owner_id": "amy"}
        assert await log.live_entries(owner_type="user", **space) == []

        assert await world.memory.finish_interrupted_forgets() == 1
        assert await _entries(world) == before
        assert sorted(await items(world)) == held
        assert await _searchable(world, "ZQA11")
        cut = await log.get(entry.forget_id)
        assert cut is not None and cut.undone and cut.stage == "rolled_back"
        assert await world.memory.finish_interrupted_forgets() == 0

        # The same forget sent again forgets; its undo brings all of it back.
        done = await forget(world, _SESSION)
        assert not [t for t in await live(world) if "ZQA11" in t]
        assert await items(world) == ["Holds ZQFTWO20"]
        await undo(world, done.forget_id)
        assert [m for _, m in await _entries(world)] == [None] * len(before)
        assert sorted(t for t, _ in await _entries(world)) == sorted(
            t for t, _ in before
        )
        assert sorted(await items(world)) == held


@pytest.mark.parametrize("landed", [True, False], ids=["written", "not-written"])
async def test_a_profile_change_comes_back_once_whether_or_not_it_landed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, landed: bool
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        await _seed(world)
        held = await world.stores.profile.read(world.personal("amy"))
        _no_reversal(monkeypatch)
        _stop(monkeypatch, world.stores.profile, "write", after=landed)
        with pytest.raises(_Stop):
            await forget(world, _SESSION)
        monkeypatch.undo()
        assert (await items(world) == ["Holds ZQFTWO20"]) is landed

        assert await world.memory.finish_interrupted_forgets() == 1
        again = await world.stores.profile.read(world.personal("amy"))
        assert again is not None and held is not None
        assert again.explicit_info == held.explicit_info
        assert again.pending_memcell_ids == held.pending_memcell_ids


async def _message_target(world: Any) -> ForgetTarget:
    ids = await message_ids(world, "p1")
    return ForgetTarget(
        kind="message", session_id="p1", message_ids=(ids["Keep ZQFONE10 and ZQA11."],)
    )


async def test_a_version_being_written_when_the_forget_stopped_is_taken_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopped after the re-extracted episode reached the file and before the
    entry named it: the entry says what was being written, which finds it."""
    async with open_world(tmp_path, ForgetModel()) as world:
        await _seed(world)
        before = await _entries(world)
        target = await _message_target(world)
        _no_reversal(monkeypatch)
        _stop(
            monkeypatch,
            world.stores.forget_log,
            "journal",
            when=lambda _id, undo: bool((undo.get("pending") or {}).get("written")),
        )
        with pytest.raises(_Stop):
            await forget(world, target)
        monkeypatch.undo()
        assert len(await _entries(world)) == len(before) + 1  # the new version

        assert await world.memory.finish_interrupted_forgets() == 1
        assert await _entries(world) == before
        assert await _searchable(world, "ZQA11")


async def test_a_version_written_over_the_old_one_is_taken_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stopped once the new version replaced the old one: taking it out makes
    the old one current again."""
    async with open_world(tmp_path, ForgetModel()) as world:
        await _seed(world)
        before = await _entries(world)
        target = await _message_target(world)
        _no_reversal(monkeypatch)
        _stop(
            monkeypatch,
            world.stores.episode,
            "deprecate",
            after=True,
            when=lambda *_a, replaced_by, **_k: replaced_by != FORGOTTEN_MARK,
        )
        with pytest.raises(_Stop):
            await forget(world, target)
        monkeypatch.undo()
        assert not [t for t in await live(world) if "ZQA11" in t]

        assert await world.memory.finish_interrupted_forgets() == 1
        assert await _entries(world) == before


async def test_the_next_forget_of_the_owner_settles_a_stopped_one_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        await _seed(world)
        _no_reversal(monkeypatch)
        _stop(monkeypatch, ForgetWriter, "profiles")
        with pytest.raises(_Stop):
            await forget(world, _SESSION)
        monkeypatch.undo()
        (stopped,) = await world.stores.forget_log.staged()

        done = await forget(world, ForgetTarget(kind="session", session_id="p2"))
        texts = await live(world)
        assert [t for t in texts if "ZQA11" in t] and not [
            t for t in texts if "ZQB21" in t
        ]
        assert await items(world) == ["Holds ZQFONE10"]
        assert (await world.stores.forget_log.get(stopped.forget_id)).stage == (  # type: ignore[union-attr]
            "rolled_back"
        )
        assert await world.stores.forget_log.staged() == []
        await undo(world, done.forget_id)
        assert sorted(await items(world)) == ["Holds ZQFONE10", "Holds ZQFTWO20"]


async def _picture_forget(world: Any) -> tuple[str, Any]:
    scope = world.personal("amy")
    kite = await world.picture(scope, "a red kite", at=T0, shade=45, poster="amy")
    await world.personal_episode("amy", f"A kite (img-1: {kite}) flew.", at=T0)
    done = await forget(world, ForgetTarget(kind="image", image_id=kite))
    assert await live(world) == ["A kite flew."]
    return kite, done


@pytest.mark.parametrize("stop_at", ["after-copies", "between-copy-and-mark"])
async def test_an_undo_cut_short_is_finished_without_a_second_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop_at: str
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        kite, done = await _picture_forget(world)
        if stop_at == "after-copies":
            _stop(monkeypatch, Undoer, "_unmark")
        else:
            _stop(monkeypatch, world.stores.episode, "deprecate")
        with pytest.raises(_Stop):
            await undo(world, done.forget_id)
        monkeypatch.undo()
        (entry,) = await world.stores.forget_log.staged()
        assert (entry.forget_id, entry.stage) == (done.forget_id, "undoing")

        assert await world.memory.finish_interrupted_forgets() == 1
        assert await live(world) == [f"A kite (img-1: {kite}) flew."]
        texts = [t for t, _ in await _entries(world)]
        assert texts.count(f"A kite (img-1: {kite}) flew.") == 2  # original + copy
        undone = await world.stores.forget_log.get(done.forget_id)
        assert undone is not None and undone.undone and undone.stage is None
        assert (undone.undo or {}).get("restored", {}).get("episode")
        pictures = await world.stores.image_memory.read_by_image(
            world.personal("amy"), kite
        )
        assert [p.item.superseded_by for p in pictures] == [None]


async def test_an_undo_sent_again_finishes_the_one_cut_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        kite, done = await _picture_forget(world)
        _stop(monkeypatch, Undoer, "_unmark")
        with pytest.raises(_Stop):
            await undo(world, done.forget_id)
        monkeypatch.undo()

        again = await undo(world, done.forget_id)
        assert again.applied and again.forget_id == done.forget_id
        assert await live(world) == [f"A kite (img-1: {kite}) flew."]
        assert await world.stores.forget_log.staged() == []


async def test_a_group_forget_stopped_after_its_message_marks_releases_them(
    tmp_path: Path,
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        session = SessionKey(app_id=APP, project_id=PROJECT, session_id="s_trip")
        log = world.stores.forget_log
        await log.begin(
            forget_id="fg_stopped",
            target="message",
            target_id="s_trip",
            app_id=APP,
            project_id=PROJECT,
            owner_id="trip",
            owner_type="group",
            undo={"format": 2, "messages": [["s_trip", ["m-1"]]]},
            requested_by="u_chen",
        )
        await world.stores.message_forget.record(
            session, ["m-1"], forget_id="fg_stopped"
        )
        assert await world.stores.message_forget.forgotten(session) == {"m-1"}

        assert await world.memory.finish_interrupted_forgets() == 1
        assert await world.stores.message_forget.forgotten(session) == set()
        stopped = await log.get("fg_stopped")
        assert stopped is not None and stopped.stage == "rolled_back"


async def test_a_failed_forget_is_reversed_and_stamped_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        await _seed(world)
        before, held = await _entries(world), sorted(await items(world))

        async def full(*_args: Any, **_kwargs: Any) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(world.stores.profile, "write", full)
        with pytest.raises(OSError):
            await forget(world, _SESSION)
        monkeypatch.undo()
        assert await _entries(world) == before
        assert sorted(await items(world)) == held
        assert await world.stores.forget_log.staged() == []
        log = world.stores.forget_log
        space = {"app_id": APP, "project_id": PROJECT, "owner_id": "amy"}
        assert await log.live_entries(owner_type="user", **space) == []


async def test_a_reversal_that_fails_too_leaves_an_incomplete_forget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with open_world(tmp_path, ForgetModel()) as world:
        await _seed(world)
        before = await _entries(world)
        _stop(monkeypatch, ForgetWriter, "profiles")

        async def broken(*_args: Any, **_kwargs: Any) -> None:
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(Rollback, "run", broken)
        with pytest.raises(_Stop):
            await forget(world, _SESSION)
        monkeypatch.undo()
        log = world.stores.forget_log
        space = {"app_id": APP, "project_id": PROJECT, "owner_id": "amy"}
        (entry,) = await log.live_entries(owner_type="user", **space)
        assert entry.reasons == "incomplete forget" and entry.stage is None
        assert "pending" not in (entry.undo or {})
        assert await log.staged() == []

        await undo(world, entry.forget_id)
        assert [m for _, m in await _entries(world)] == [None] * len(before)


def test_a_profile_document_shows_whether_it_took_a_change() -> None:
    profile = Profile.model_validate(
        {
            "owner_id": "amy",
            "timestamp": T0,
            "summary": "Amy.",
            "explicit_info": [
                {"description": "Holds ZQF1", "item_id": "pi_1", "category": "facts"}
            ],
            "history": [{"description": "Was ZQF0", "replaced_at": 1}],
            "pending_memcell_ids": ["mc_1"],
        }
    )
    item = {"description": "Holds ZQF1", "item_id": "pi_1"}
    assert not _took({"items": [["explicit_info", 0, item]]}, profile)
    gone = {"description": "Holds ZQF9", "item_id": "pi_9"}
    assert _took({"items": [["explicit_info", 0, gone]]}, profile)
    entry = {"description": "Was ZQF0", "replaced_at": 1}
    assert not _took({"history": [[0, entry]]}, profile)
    assert _took({"history": [[0, {"description": "Was ZQF8"}]]}, profile)
    assert not _took({"summary": "Amy."}, profile)
    assert _took({"summary": "Someone else."}, profile)
    assert _took({"queued": ["mc_1"]}, profile)
    assert not _took({"queued": ["mc_2"]}, profile)
