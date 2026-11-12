"""Project roll-up: a project's groups as one scope (task 17).

docs/group-memory.md, sections 4.4, 5.1, 5.3 and 8.7. Real markdown,
system.db and LanceDB. Chen is a current member of ``design`` and ``build``,
left ``archive``, was never in ``ops``; ``travel`` is another project's
group. What these pin:

- ``project_scope`` searches exactly the asker's current groups of the
  project, as one pool, each inside the asker's windows;
- ``GET /projects/{id}`` lists those groups and their members only;
  ``GET /projects/{id}/profile`` is the union of the visible current
  group-profile items of those groups, each tagged with its group, oldest
  first, duplicates across groups kept;
- an asker in none of the project's groups is refused (403).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from httpx import ASGITransport, AsyncClient

from everos.entrypoints.api.app import create_app
from everos.memory.models import Group
from everos.memory.ports import Scope
from everos.service import SearchRequest
from tests.fakes import make_application
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

T0 = 1_772_323_200_000


def _refuse(*_args: Any, **_kwargs: Any) -> str:
    raise AssertionError("no model call expected")


@pytest.fixture
async def world(tmp_path: Path) -> AsyncIterator[GroupWorld]:
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=_refuse)) as w:
        await w.roster("design", [join("u_chen", T0), join("u_betty", T0)])
        await w.roster("build", [join("u_chen", T0 + 5_000), join("u_dan", T0)])
        await w.roster("archive", [join("u_chen", T0), leave("u_chen", T0 + 100)])
        await w.roster("ops", [join("u_zoe", T0)])
        other = Scope(
            app_id=APP, project_id="elsewhere", owner_id="travel", owner_type="group"
        )
        await w.stores.groups.save(
            other,
            group=Group(group_id="travel", created_at=0),
            events=[join("u_chen", T0)],
        )
        for group, text, at in (
            ("design", "design marker plan", 1_000),
            ("build", "build early marker plan", 1_000),  # before Chen joined
            ("build", "build late marker plan", 6_000),
            ("archive", "archive marker plan", 50),
            ("ops", "ops marker plan", 1_000),
        ):
            await w.episode(group, text, start=T0 + at, end=T0 + at)
            await w.profile_item(group, f"{text} decision", start=T0 + at, end=T0 + at)
        # The same decision in two groups is two items.
        await w.profile_item(
            "design", "shared decision", start=T0 + 7_000, end=T0 + 7_000
        )
        await w.profile_item(
            "build", "shared decision", start=T0 + 7_000, end=T0 + 7_000
        )
        await w.project()
        yield w


async def test_project_scope_covers_exactly_the_askers_current_groups(
    world: GroupWorld,
) -> None:
    response = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            project_scope=True,
            asker_id="u_chen",
            query="marker plan",
            top_k=20,
        )
    )

    assert response.data.scope is not None
    assert response.data.scope.group_ids == ["build", "design"]
    texts = {e.episode for e in response.data.episodes}
    assert texts == {"design marker plan", "build late marker plan"}
    assert {e.group_id for e in response.data.episodes} == {"design", "build"}


async def test_the_project_profile_is_the_union_of_visible_items(
    world: GroupWorld,
) -> None:
    profile = await world.memory.project_profile(
        app_id=APP, project_id=PROJECT, asker_id="u_chen", asserted_asker=None
    )

    assert profile.group_ids == ["build", "design"]
    assert [(i.group_id, i.statement) for i in profile.items] == [
        ("design", "design marker plan decision"),
        ("build", "build late marker plan decision"),
        ("build", "shared decision"),
        ("design", "shared decision"),
    ]
    assert all(i.audience is not None for i in profile.items)


async def test_the_project_routes(world: GroupWorld) -> None:
    app = create_app(application=make_application(memory=world.memory))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        read = await c.get(
            f"/api/v2/projects/{PROJECT}",
            params={"app_id": APP, "asker_id": "u_chen"},
        )
        profile = await c.get(
            f"/api/v2/projects/{PROJECT}/profile",
            params={"app_id": APP, "asker_id": "u_chen"},
        )
        stranger = await c.get(
            f"/api/v2/projects/{PROJECT}", params={"app_id": APP, "asker_id": "u_amy"}
        )

    assert read.status_code == 200, read.text
    data = read.json()["data"]
    assert [g["group_id"] for g in data["groups"]] == ["build", "design"]
    members = {m["member_id"]: m["group_ids"] for m in data["members"]}
    assert members == {
        "u_betty": ["design"],
        "u_chen": ["build", "design"],
        "u_dan": ["build"],
    }
    assert "u_zoe" not in read.text  # ops is not Chen's
    assert profile.status_code == 200
    assert "ops marker" not in profile.text and "archive" not in profile.text
    assert stranger.status_code == 403
