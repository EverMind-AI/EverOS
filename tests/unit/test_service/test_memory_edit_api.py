"""A person's update, the history and the forget of a user's memory (docs/api.md).

Over the real stack (markdown, ``system.db``, LanceDB, the memory service),
through the facade the routes call. What these pin:

- **Update.** Every kind -- an episode, a picture memory, a scene version, a
  profile item -- gets a new version that names the old one and who wrote it;
  the old version is superseded, not edited; search returns only the new one;
  a stale version answers 409 ``VERSION_CONFLICT`` naming the current one, a
  forgotten memory 409 ``MEMORY_FORGOTTEN``; the new version is held unless
  the caller releases it.
- **History.** Every version of a memory, current first, whichever of its ids
  is asked: state, time, the link to the next version, who wrote it.
- **Forget.** Every target (records of each kind, a conversation, everything,
  a kind and a time range, a picture) has a dry run that changes nothing and
  a real run whose ``forget_id`` undoes exactly it; a forget follows the
  sources (scenes, picture files, profile items); an entry two forgets name
  stays forgotten until both are undone.
- **Delete is a forget.** ``/memory/delete`` and ``/delete_by_ids`` answer with
  forgets (2.0 has no hard delete); an agent's memory is refused.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.core.errors import (
    ForgottenMemoryError,
    InvalidInputError,
    NotFoundError,
    VersionConflictError,
)
from everos.memory.models import (
    FORGOTTEN_MARK,
    Episode,
    ExplicitInfoItem,
    Profile,
    Scene,
)
from everos.memory.ports import NoSourceCheck, Scope
from everos.service import MemoryChanges, SearchRequest, UserForgetTarget
from tests.helpers.groups import APP, PROJECT, GroupWorld

T0 = 1_772_323_200_000
DAY = 86_400_000


def _amy(world: GroupWorld) -> Scope:
    return world.personal("amy")


async def _episode(
    world: GroupWorld, text: str, *, at: int, session: str = "s1", **fields: object
) -> str:
    ref = await world.stores.episode.append(
        _amy(world),
        Episode(
            owner_id="amy",
            episode=text,
            summary=text[:40],
            subject=text[:20],
            timestamp=at,
            session_id=session,
            sender_ids=["amy"],
            parent_id=f"mc_{session}_{at}",
            **fields,  # type: ignore[arg-type]
        ),
        only_if=NoSourceCheck(),
    )
    assert ref is not None
    return ref.id


async def _profile(world: GroupWorld, *items: tuple[str, str, list[str]]) -> None:
    """``user.md`` with items ``(item_id, description, source segments)``."""
    await world.stores.profile.write(
        _amy(world),
        Profile(
            owner_id="amy",
            summary="Amy lives in Lyon and has a cat.",
            timestamp=T0,
            explicit_info=[
                ExplicitInfoItem(
                    description=text,
                    category="life",
                    observed_at=T0,
                    item_id=item_id,
                    source_memcell_ids=segments,
                    source_start=T0,
                    source_end=T0,
                )
                for item_id, text, segments in items
            ],
        ),
    )


async def _search_ids(world: GroupWorld, query: str) -> list[str]:
    await world.project()
    found = await world.memory.search(
        SearchRequest(
            app_id=APP, project_id=PROJECT, user_id="amy", query=query, top_k=20
        )
    )
    return [e.id for e in found.data.episodes] + [i.id for i in found.data.images]


async def _update(world: GroupWorld, record_id: str, **kwargs: object) -> object:
    changes = {
        k: kwargs.pop(k)
        for k in ("subject", "summary", "content", "description", "ocr")
        if k in kwargs
    }
    return await world.memory.update_memory(
        app_id=APP,
        project_id=PROJECT,
        user_id="amy",
        record_id=record_id,
        changes=MemoryChanges(**changes),  # type: ignore[arg-type]
        **kwargs,
    )


async def _forget(world: GroupWorld, dry_run: bool = False, **target: object) -> object:
    return await world.memory.forget(
        app_id=APP,
        project_id=PROJECT,
        user_id="amy",
        target=UserForgetTarget(**target),  # type: ignore[arg-type]
        dry_run=dry_run,
    )


async def _history(world: GroupWorld, record_id: str) -> object:
    return await world.memory.memory_history(
        app_id=APP, project_id=PROJECT, user_id="amy", record_id=record_id
    )


# ── update ───────────────────────────────────────────────────────────────


async def test_an_episode_update_is_a_new_version_and_search_returns_only_it(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        old = await _episode(world, "Amy adopted a cat named Biscuit.", at=T0)
        assert old in await _search_ids(world, "Biscuit adopted")

        result = await _update(
            world, old, content="Amy adopted a dog named Biscuit.", subject="Biscuit"
        )
        assert result.changed and result.supersedes == old and result.id != old
        assert result.held and result.edited_by == "amy"

        ids = await _search_ids(world, "Biscuit adopted")
        assert result.id in ids and old not in ids

        history = await _history(world, old)
        assert [v.id for v in history.versions] == [result.id, old]
        assert [v.state for v in history.versions] == ["current", "superseded"]
        current, replaced = history.versions
        assert current.supersedes == old and current.edited_by == "amy"
        assert current.held and current.content["content"].startswith(
            "Amy adopted a dog"
        )
        assert replaced.next == result.id and replaced.edited_by is None
        # The version keeps what it rests on: its time and its segment.
        assert current.timestamp == replaced.timestamp == T0
        assert current.sources["memcell_id"] == replaced.sources["memcell_id"]
        assert history.current == result.id
        assert (await _history(world, result.id)).versions == history.versions


async def test_a_stale_version_answers_409_naming_the_current_one(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        old = await _episode(world, "Amy's train leaves at nine.", at=T0)
        first = await _update(world, old, content="Amy's train leaves at ten.")

        with pytest.raises(VersionConflictError) as conflict:
            await _update(world, old, content="Amy's train leaves at eleven.")
        assert conflict.value.current == first.id
        assert first.id in str(conflict.value)

        # Naming the version it read lets a caller change it by any id.
        second = await _update(
            world,
            old,
            content="Amy's train leaves at eleven.",
            expected_version=first.id,
        )
        assert second.supersedes == first.id
        history = await _history(world, old)
        assert [v.state for v in history.versions] == [
            "current",
            "superseded",
            "superseded",
        ]


async def test_a_forgotten_memory_cannot_be_updated(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        old = await _episode(world, "Amy's locker code is 4512.", at=T0)
        await world.project()
        forgotten = await _forget(world, kind="record", record_ids=(old,))
        with pytest.raises(ForgottenMemoryError):
            await _update(world, old, content="Amy's locker code is 9999.")
        await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=forgotten.forget_id
        )
        assert (await _update(world, old, content="Code 9999.")).changed


async def test_an_update_checks_its_fields_and_its_owner(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        old = await _episode(world, "Amy plays the cello.", at=T0)
        with pytest.raises(InvalidInputError):
            await _update(world, old, description="a picture field")
        with pytest.raises(InvalidInputError):
            await _update(world, "xx_19990101_0001", content="no such kind")
        with pytest.raises(NotFoundError):
            await _update(world, "amy_ep_19990101_0001", content="nothing there")


async def test_release_and_hold(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        old = await _episode(world, "Amy's favourite tea is oolong.", at=T0)
        held = await _update(world, old, content="Amy's favourite tea is sencha.")
        assert held.held
        # Nothing to change: the same text and the same hold write nothing.
        again = await _update(world, held.id, content="Amy's favourite tea is sencha.")
        assert not again.changed and again.id == held.id
        released = await _update(world, held.id, hold=False)
        assert released.changed and not released.held
        assert released.supersedes == held.id
        history = await _history(world, old)
        assert [v.held for v in history.versions] == [False, True, False]


async def test_a_picture_memory_update_and_the_revise_alias(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        image_id = await world.picture(_amy(world), "a cat on a sofa", at=T0, shade=10)
        (row,) = await world.stores.image_memory.read_by_image(_amy(world), image_id)
        first = await _update(world, row.ref.id, description="a dog on a sofa")
        assert first.kind == "image_memory" and first.supersedes == row.ref.id

        revised = await world.memory.revise_image_memory(
            app_id=APP,
            project_id=PROJECT,
            user_id="amy",
            image_id=image_id,
            ocr="WELCOME",
        )
        assert revised.supersedes == first.id.removeprefix("amy_")
        assert revised.description == "a dog on a sofa" and revised.ocr == "WELCOME"

        history = await _history(world, row.ref.id)
        assert [v.state for v in history.versions] == [
            "current",
            "superseded",
            "superseded",
        ]
        assert history.versions[0].content == {
            "description": "a dog on a sofa",
            "ocr": "WELCOME",
        }
        ids = await _search_ids(world, "dog sofa")
        assert row.ref.id not in ids and first.id not in ids


async def test_a_scene_update_is_its_next_version(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        members = [
            (await _episode(world, f"Lisbon trip day {n}.", at=T0 + n)).removeprefix(
                "amy_"
            )
            for n in (1, 2)
        ]
        ref = await world.stores.scene.append(
            _amy(world),
            Scene(
                owner_id="amy",
                scene_id="cl_lisbon",
                version=1,
                member_ids=members,
                subject="Lisbon",
                summary="The Lisbon trip.",
                narrative="Amy planned the Lisbon trip.",
                timestamp=T0 + 2,
                source_start=T0 + 1,
            ),
            supersedes=None,
            only_if=NoSourceCheck(),
        )
        assert ref is not None
        result = await _update(
            world, ref.id, content="Amy cancelled the Lisbon trip.", hold=True
        )
        assert result.kind == "scene" and result.supersedes == ref.id
        newest = await world.stores.scene.latest(_amy(world), "cl_lisbon")
        assert newest is not None and newest.item.version == 2
        assert newest.item.held and newest.item.member_ids == members
        history = await _history(world, ref.id)
        assert [(v.state, v.sources["version"]) for v in history.versions] == [
            ("current", 2),
            ("superseded", 1),
        ]
        ids = await _search_ids(world, "Lisbon trip")
        assert result.id in ids and ref.id not in ids


async def test_a_profile_item_update_is_a_new_value_linked_to_the_old(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _profile(world, ("pi_000000000001", "Lives in Lyon.", ["mc_1"]))
        result = await _update(world, "pi_000000000001", description="Lives in Berlin.")
        assert result.kind == "profile_item" and result.supersedes == "pi_000000000001"
        stored = await world.stores.profile.read(_amy(world))
        assert stored is not None
        (item,) = stored.explicit_info
        extra = item.model_extra or {}
        assert item.description == "Lives in Berlin."
        assert extra["supersedes"] == "pi_000000000001" and extra["held"] is True
        assert extra["source_memcell_ids"] == ["mc_1"]  # provenance kept
        assert stored.history[0]["superseded_by"] == result.id

        history = await _history(world, "pi_000000000001")
        assert [(v.id, v.state) for v in history.versions] == [
            (result.id, "current"),
            ("pi_000000000001", "superseded"),
        ]
        with pytest.raises(VersionConflictError):
            await _update(world, "pi_000000000001", description="Lives in Rome.")
        released = await _update(world, result.id, hold=False)
        assert not released.held and released.supersedes is None
        assert released.id == result.id  # a release of an item is no new value


# ── forget ───────────────────────────────────────────────────────────────


async def test_a_record_forget_dry_run_then_real_then_undo(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        keep = await _episode(world, "Amy likes jazz.", at=T0, session="s1")
        gone = await _episode(world, "Amy's password hint is Biscuit.", at=T0 + 1)
        await _profile(
            world,
            ("pi_000000000001", "Likes jazz.", [f"mc_s1_{T0}"]),
            ("pi_000000000002", "Password hint Biscuit.", [f"mc_s1_{T0 + 1}"]),
        )
        assert gone in await _search_ids(world, "password hint Biscuit")

        plan = await _forget(world, dry_run=True, kind="record", record_ids=(gone,))
        assert plan.dry_run and plan.forget_id == ""
        assert plan.episodes == (gone,)
        assert plan.profile_items == ("pi_000000000002",)  # it rests on that alone
        assert gone in await _search_ids(world, "password hint Biscuit")

        done = await _forget(world, kind="record", record_ids=(gone,))
        assert done.forget_id.startswith("fg_") and done.episodes == (gone,)
        ids = await _search_ids(world, "password hint Biscuit")
        assert gone not in ids and keep in ids
        profile = await world.stores.profile.read(_amy(world))
        assert profile is not None
        assert [i.description for i in profile.explicit_info] == ["Likes jazz."]
        assert profile.summary == ""  # written over every item, the next update's
        history = await _history(world, gone)
        assert [v.state for v in history.versions] == ["forgotten"]

        undone = await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=done.forget_id
        )
        assert undone.applied and undone.episodes == (gone,)
        assert undone.profile_items == 1
        profile = await world.stores.profile.read(_amy(world))
        assert profile is not None
        assert profile.summary == "Amy lives in Lyon and has a cat."
        assert gone in await _search_ids(world, "password hint Biscuit")
        again = await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=done.forget_id
        )
        assert not again.applied


async def test_a_record_forget_of_a_scene_and_of_a_profile_item(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        members = [
            (await _episode(world, f"Lisbon day {n}.", at=T0 + n)).removeprefix("amy_")
            for n in (1, 2)
        ]
        scene = await world.stores.scene.append(
            _amy(world),
            Scene(
                owner_id="amy",
                scene_id="cl_lisbon",
                version=1,
                member_ids=members,
                subject="Lisbon",
                summary="s",
                narrative="Amy planned the Lisbon trip.",
                timestamp=T0 + 2,
                source_start=T0 + 1,
            ),
            supersedes=None,
            only_if=NoSourceCheck(),
        )
        assert scene is not None
        await _profile(world, ("pi_000000000001", "Lives in Lyon.", ["mc_x"]))

        done = await _forget(
            world, kind="record", record_ids=(scene.id, "pi_000000000001")
        )
        assert done.scenes == (scene.id,) and done.episodes == ()
        assert done.profile_items == ("pi_000000000001",)
        marks = await world.stores.scene.deprecations(_amy(world), scene.md_path)
        assert marks == {scene.entry_id: FORGOTTEN_MARK}
        profile = await world.stores.profile.read(_amy(world))
        assert profile is not None and profile.explicit_info == []
        with pytest.raises(ForgottenMemoryError):
            await _update(world, "pi_000000000001", description="Lives in Paris.")

        await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=done.forget_id
        )
        assert await world.stores.scene.deprecations(_amy(world), scene.md_path) == {}
        profile = await world.stores.profile.read(_amy(world))
        assert profile is not None
        assert [i.description for i in profile.explicit_info] == ["Lives in Lyon."]


async def test_a_record_id_that_names_nothing_is_404(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        with pytest.raises(NotFoundError):
            await _forget(world, kind="record", record_ids=("amy_ep_20990101_0001",))


async def test_a_session_forget_dry_run_and_undo_by_id(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        a = await _episode(world, "Amy talked about the Alps.", at=T0, session="sA")
        b = await _episode(world, "Amy talked about the sea.", at=T0 + 1, session="sB")
        plan = await _forget(world, dry_run=True, kind="session", session_id="sA")
        assert plan.episodes == (a,) and plan.forget_id == ""
        done = await _forget(world, kind="session", session_id="sA")
        assert done.episodes == (a,)
        ids = await _search_ids(world, "Amy talked")
        assert a not in ids and b in ids
        undone = await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=done.forget_id
        )
        assert undone.applied and undone.episodes == (a,)
        assert a in await _search_ids(world, "Amy talked")


async def test_a_session_forget_keeps_the_chain_of_an_updated_episode(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        old = await _episode(world, "Amy met Bo at noon.", at=T0, session="sA")
        new = await _update(world, old, content="Amy met Bo at one.")
        done = await _forget(world, kind="session", session_id="sA")
        assert done.episodes == (new.id,)  # the current version only
        await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=done.forget_id
        )
        history = await _history(world, old)
        assert [(v.id, v.state) for v in history.versions] == [
            (new.id, "current"),
            (old, "superseded"),
        ]


async def test_an_owner_forget_empties_the_profile_and_undo_puts_it_back(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        first = await _episode(world, "Amy studies law.", at=T0, session="sA")
        second = await _episode(world, "Amy runs on Sundays.", at=T0 + 1, session="sB")
        image_id = await world.picture(_amy(world), "a running shoe", at=T0, shade=20)
        await _profile(world, ("pi_000000000001", "Studies law.", [f"mc_sA_{T0}"]))

        plan = await _forget(world, dry_run=True, kind="owner")
        assert set(plan.episodes) == {first, second} and len(plan.image_memories) == 1
        assert plan.images == (image_id,)

        done = await _forget(world, kind="owner")
        assert set(done.episodes) == {first, second}
        assert done.profile_items == ("pi_000000000001",)
        assert await _search_ids(world, "Amy law running shoe") == []
        profile = await world.stores.profile.read(_amy(world))
        assert profile is not None
        assert profile.explicit_info == [] and profile.summary == ""

        await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=done.forget_id
        )
        profile = await world.stores.profile.read(_amy(world))
        assert profile is not None
        assert [i.description for i in profile.explicit_info] == ["Studies law."]
        assert profile.summary == "Amy lives in Lyon and has a cat."
        assert set(await _search_ids(world, "Amy law runs Sundays")) >= {first, second}


async def test_a_condition_forget_takes_a_kind_in_a_time_range(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        early = await _episode(world, "Amy visited Rome.", at=T0, session="sA")
        late = await _episode(world, "Amy visited Oslo.", at=T0 + 3 * DAY, session="sB")
        plan = await _forget(
            world,
            dry_run=True,
            kind="condition",
            memory_types=("episode",),
            start=T0 + DAY,
            end=T0 + 5 * DAY,
        )
        assert plan.episodes == (late,)
        done = await _forget(
            world,
            kind="condition",
            memory_types=("episode",),
            start=T0 + DAY,
            end=T0 + 5 * DAY,
        )
        ids = await _search_ids(world, "Amy visited")
        assert early in ids and late not in ids
        await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=done.forget_id
        )
        assert late in await _search_ids(world, "Amy visited")


async def test_a_picture_forget_through_the_general_target(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        image_id = await world.picture(_amy(world), "a red bicycle", at=T0, shade=30)
        body = f"Amy showed her new bike (img-1: {image_id})."
        referring = await _episode(world, body, at=T0, image_ids=[image_id])
        await world.project()
        plan = await _forget(world, dry_run=True, kind="image", image_id=image_id)
        assert plan.episodes == (referring,) and plan.images == (image_id,)
        done = await _forget(world, kind="image", image_id=image_id)
        assert done.episodes == (referring,) and len(done.new_episodes) == 1
        (rewritten,) = done.new_episodes
        history = await _history(world, rewritten)
        assert image_id not in history.versions[0].content["content"]
        undone = await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=done.forget_id
        )
        assert undone.applied


async def test_an_entry_two_forgets_name_stays_forgotten_until_both_are_undone(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        one = await _episode(world, "Amy's first secret.", at=T0, session="sA")
        two = await _episode(world, "Amy's second secret.", at=T0 + 1, session="sA")
        by_id = await _forget(world, kind="record", record_ids=(one,))
        by_session = await _forget(world, kind="session", session_id="sA")
        assert by_session.episodes == (two,)  # one was forgotten already

        await world.memory.undo_forget(
            app_id=APP,
            project_id=PROJECT,
            user_id="amy",
            forget_id=by_session.forget_id,
        )
        ids = await _search_ids(world, "Amy secret")
        assert two in ids and one not in ids
        await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=by_id.forget_id
        )
        assert one in await _search_ids(world, "Amy secret")


async def test_a_forget_id_of_someone_else_is_404(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        old = await _episode(world, "Amy's note.", at=T0)
        done = await _forget(world, kind="record", record_ids=(old,))
        with pytest.raises(NotFoundError):
            await world.memory.undo_forget(
                app_id=APP, project_id=PROJECT, user_id="bo", forget_id=done.forget_id
            )


# ── delete is a forget ───────────────────────────────────────────────────


async def test_delete_forgets_and_answers_its_forget_ids(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        a = await _episode(world, "Amy's diary entry one.", at=T0, session="sA")
        await world.project()
        deleted = await world.memory.delete(
            app_id=APP, project_id=PROJECT, user_id=None, agent_id=None, session_id="sA"
        )
        assert deleted.filters == ["session_id"] and deleted.count == 1
        (forget_id,) = deleted.forget_ids
        assert a not in await _search_ids(world, "diary entry")
        # Nothing left markdown: the version is there, marked.
        history = await _history(world, a)
        assert [v.state for v in history.versions] == ["forgotten"]
        await world.memory.undo_forget(
            app_id=APP, project_id=PROJECT, user_id="amy", forget_id=forget_id
        )
        assert a in await _search_ids(world, "diary entry")

        with pytest.raises(InvalidInputError):
            await world.memory.delete(
                app_id=APP,
                project_id=PROJECT,
                user_id=None,
                agent_id="bot",
                session_id=None,
            )

        by_ids = await world.memory.delete_by_ids(
            app_id=APP,
            project_id=PROJECT,
            memory_type="episode",
            ids=[a, "amy_ep_20990101_0001"],
            reason_code="privacy",
        )
        assert len(by_ids) == 1
        assert a not in await _search_ids(world, "diary entry")


async def test_a_conversation_sent_again_after_its_delete_is_searchable(
    tmp_path: Path,
) -> None:
    """The harnesses' ``--replace-sessions``: delete, then send it again."""
    async with GroupWorld.open(tmp_path) as world:
        old = await _episode(world, "Amy's trip to Kyoto.", at=T0, session="sK")
        await world.memory.delete(
            app_id=APP, project_id=PROJECT, user_id=None, agent_id=None, session_id="sK"
        )
        again = await _episode(world, "Amy's trip to Kyoto.", at=T0, session="sK")
        ids = await _search_ids(world, "trip Kyoto")
        assert again in ids and old not in ids
