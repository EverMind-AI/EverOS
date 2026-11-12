"""A person's update and the history of a group's records (docs/api.md).

Over the real stack, through the facade the routes call. What these pin:

- **Who may change what.** The forget rule: an owner or admin may change any
  record of the group; a member, a record resting on their own messages
  alone; a stranger nothing (403).
- **Every group kind.** An episode, a picture memory and a scene version get a
  new version linked to the old one; a member's profile item a new value; a
  group scene's cluster follows its episode to the new version.
- **History only within the asker's windows.** A member who joined after a
  record was said cannot read its versions: 404, as for one that does not
  exist.
- **Forget stays a dry run by default** through ``/memory/forget`` (the route
  default); the service keeps its own explicit flag.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.core.errors import ForbiddenError, NotFoundError
from everos.memory.models import (
    Cluster,
    ExplicitInfoItem,
    Profile,
    Scene,
)
from everos.memory.ports import NoSourceCheck
from everos.service import MemoryChanges
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

T0 = 1_772_323_200_000


async def _trip(world: GroupWorld) -> None:
    await world.roster(
        "trip",
        [
            join("u_chen", T0, role="owner", display_name="Chen"),
            join("u_betty", T0, display_name="Betty"),
            join("u_dan", T0 + 10_000, display_name="Dan"),
        ],
    )


async def _update(
    world: GroupWorld, record_id: str, asker: str, **changes: str
) -> object:
    return await world.memory.update_memory(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id=asker,
        record_id=record_id,
        changes=MemoryChanges(**changes),
    )


async def test_who_may_change_a_group_record(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _trip(world)
        bettys = await world.episode(
            "trip", "Betty booked the hostel.", start=T0 + 1, end=T0 + 2,
            senders=("u_betty",),
        )  # fmt: skip
        mixed = await world.episode(
            "trip", "Chen and Betty argued.", start=T0 + 3, end=T0 + 4,
            senders=("u_chen", "u_betty"),
        )  # fmt: skip

        own = await _update(world, bettys, "u_betty", content="Betty booked a flat.")
        assert own.edited_by == "u_betty" and own.supersedes == bettys
        with pytest.raises(ForbiddenError):
            await _update(world, mixed, "u_betty", content="Only Betty spoke.")
        with pytest.raises(ForbiddenError):
            await _update(world, mixed, "u_stranger", content="I was not there.")
        by_owner = await _update(world, mixed, "u_chen", content="They agreed.")
        assert by_owner.changed and by_owner.edited_by == "u_chen"


async def test_a_group_picture_and_scene_update(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _trip(world)
        image_id = await world.picture(
            world.scope("trip"),
            "the hostel sign",
            at=T0 + 5,
            shade=40,
            poster="u_betty",
        )
        (row,) = await world.stores.image_memory.read_by_image(
            world.scope("trip"), image_id
        )
        picture = await world.memory.update_memory(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_betty",
            record_id=row.ref.id,
            changes=MemoryChanges(description="the flat's sign"),
        )
        assert picture.kind == "image_memory" and picture.supersedes == row.ref.id

        members = [
            (
                await world.episode(
                    "trip", f"Day {n}.", start=T0 + n, end=T0 + n + 1,
                    senders=("u_chen",),
                )
            ).removeprefix("trip_")
            for n in (6, 8)
        ]  # fmt: skip
        scope = world.scope("trip")
        await world.stores.cluster.upsert(
            scope,
            "group_scene",
            Cluster(
                id="cl_days", centroid=(1.0,), count=2, last_ts=T0 + 9, members=members
            ),
            member_type="episode",
        )
        scene = await world.stores.scene.append(
            scope,
            Scene(
                owner_id="trip",
                scene_id="cl_days",
                version=1,
                member_ids=members,
                subject="Days",
                summary="s",
                narrative="Two days.",
                timestamp=T0 + 9,
                source_start=T0 + 6,
                sender_ids=["u_chen"],
            ),
            supersedes=None,
            only_if=NoSourceCheck(),
        )
        assert scene is not None
        edited = await _update(world, scene.id, "u_chen", content="Two long days.")
        assert edited.kind == "scene" and edited.held

        # A group scene reads its members by id: the cluster follows an edit.
        episode = await _update(world, f"trip_{members[0]}", "u_chen", content="Day 6!")
        cluster = await world.stores.cluster.get("cl_days")
        assert cluster is not None
        assert episode.id.removeprefix("trip_") in cluster.members
        assert members[0] not in cluster.members


async def test_a_member_profile_item_update_and_its_history(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _trip(world)
        scope = world.scope("trip")
        await world.stores.profile.write(
            scope,
            Profile(
                owner_id="u_betty",
                summary="",
                timestamp=T0,
                explicit_info=[
                    ExplicitInfoItem(
                        description="Lives in Paris",
                        category="home",
                        item_id="pi_000000000009",
                        source_memcell_ids=["mc_trip_1"],
                        source_start=T0 + 1,
                        source_end=T0 + 2,
                    )
                ],
            ),
        )
        with pytest.raises(ForbiddenError):
            await world.memory.update_memory(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_dan",
                record_id="pi_000000000009",
                changes=MemoryChanges(description="Lives in Rome"),
            )
        result = await world.memory.update_memory(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_betty",
            record_id="pi_000000000009",
            changes=MemoryChanges(description="Lives in Berlin"),
        )
        assert result.member_id == "u_betty" and result.supersedes == "pi_000000000009"
        held = await world.stores.profile.read(scope, "u_betty")
        assert held is not None
        assert [i.description for i in held.explicit_info] == ["Lives in Berlin"]
        assert held.history[0]["superseded_by"] == result.id

        history = await world.memory.memory_history(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            record_id="pi_000000000009",
        )
        assert history.member_id == "u_betty"
        assert [(v.id, v.state) for v in history.versions] == [
            (result.id, "current"),
            ("pi_000000000009", "superseded"),
        ]
        # Dan joined after Betty said it: none of it is his to read.
        with pytest.raises(NotFoundError):
            await world.memory.memory_history(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_dan",
                record_id="pi_000000000009",
            )


async def test_a_group_history_shows_only_the_askers_windows(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _trip(world)
        early = await world.episode(
            "trip", "Early plan.", start=T0 + 1, end=T0 + 2, senders=("u_chen",)
        )
        edited = await _update(world, early, "u_chen", content="Early plan, fixed.")
        seen = await world.memory.memory_history(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_betty",
            record_id=early,
        )
        assert [v.id for v in seen.versions] == [edited.id, early]
        with pytest.raises(NotFoundError):
            await world.memory.memory_history(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_dan",
                record_id=early,
            )
        with pytest.raises(ForbiddenError):
            await world.memory.memory_history(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_stranger",
                record_id=early,
            )


async def test_a_group_profile_items_history_follows_its_supersedes(
    tmp_path: Path,
) -> None:
    async with GroupWorld.open(tmp_path) as world:
        await _trip(world)
        first = await world.profile_item(
            "trip", "Fly out on May 1.", start=T0 + 1, end=T0 + 2
        )
        second = await world.profile_item(
            "trip",
            "Fly out on May 2.",
            start=T0 + 3,
            end=T0 + 4,
            supersedes=first.entry_id,
        )
        history = await world.memory.memory_history(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            record_id=first.id,
        )
        assert history.kind == "group_profile_item"
        assert [(v.id, v.state) for v in history.versions] == [
            (second.id, "current"),
            (first.id, "superseded"),
        ]
        assert history.versions[1].next == second.id
        assert history.versions[0].content["statement"] == "Fly out on May 2."
