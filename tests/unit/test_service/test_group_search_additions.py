"""What a group search returns besides its ranking (tasks 15 and 16).

docs/group-memory.md, sections 5.3, 8.5 and 8.6. Real markdown, system.db
and LanceDB; a scripted decider. What these pin:

- ``coverage`` is ``complete`` when the decider stops with a core,
  ``partial`` when the loop ends first (no new core for the patience, or the
  round budget), ``none`` without a core -- and the core is returned either
  way, never an empty result because coverage is incomplete;
- a group scope attaches, by default, only the current group-profile items
  linked to the core -- those the constraint block rests on, a chain's
  newest version, the open dissent about a linked decision, no role -- with
  or without the constraint block; ``include_group_profile: true`` attaches
  every visible current item, each kind capped; ``false`` leaves them out;
  a user scope gets neither a group profile nor a coverage;
- an item id is unique within its group only: in a pool of groups the
  linked items, their chains and the dissent about them are each read in
  their own group;
- every group record a search returns carries its audience: who was there
  when it was said, and which current members joined since; one-to-one
  records carry none.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.config import Settings
from everos.service import SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

T0 = 1_772_323_200_000


class _Decider:
    """Answers each round with the next scripted ``(core, next_queries)``."""

    def __init__(self, *rounds: tuple[list[int], list[str]]) -> None:
        self.rounds = list(rounds)

    def __call__(self, messages: Any, **_: Any) -> str:
        core, queries = self.rounds.pop(0) if self.rounds else ([], [])
        return json.dumps({"core": core, "next_queries": queries})


async def _world(
    tmp_path: Path, decider: _Decider, *, cap: int = 20
) -> AsyncIterator[GroupWorld]:
    base = Settings()
    settings = base.model_copy(
        update={"search": base.search.model_copy(update={"group_profile_cap": cap})}
    )
    async with GroupWorld.open(
        tmp_path, decider=FakeLLMClient(handler=decider), settings=settings
    ) as world:
        await world.roster(
            "trip",
            [
                join("u_chen", T0, display_name="Chen", role="owner"),
                join("u_betty", T0, display_name="Betty"),
                leave("u_betty", T0 + 6_000),
                join("u_dan", T0, display_name="Dan"),
                leave("u_dan", T0 + 2_500),
                join("u_dan", T0 + 7_500),
                join("u_evan", T0 + 5_000, display_name="Evan"),
            ],
        )
        await world.episode(
            "trip", "early plan marker Lisbon offsite", start=T0 + 1_000, end=T0 + 2_000
        )
        await world.episode(
            "trip", "late plan marker Lisbon offsite", start=T0 + 8_000, end=T0 + 9_000
        )
        await world.personal_episode("u_chen", "personal plan marker Lisbon", at=T0)
        for n in range(3):
            await world.profile_item(
                "trip",
                f"decision {n} marker Lisbon",
                start=T0 + 8_000 + n,
                end=T0 + 8_000 + n,
            )
        await world.profile_item(
            "trip", "Lisbon offsite", kind="topic", start=T0 + 8_100, end=T0 + 8_100
        )
        await world.project()
        yield world


def _request(asker: str = "u_chen", **fields: Any) -> SearchRequest:
    values: dict[str, Any] = {
        "app_id": APP,
        "project_id": PROJECT,
        "group_id": "trip",
        "asker_id": asker,
        "query": "plan marker Lisbon offsite",
        "method": "llm_multiround",
        "top_k": 10,
        **fields,
    }
    return SearchRequest(**values)


@pytest.mark.parametrize(
    ("rounds", "coverage"),
    [
        ([([0], [])], "complete"),
        ([([0], ["more about it"]), ([], ["and more"])], "partial"),
        ([([0], ["more"]), ([0, 1], ["more"]), ([0, 1, 2], ["more"])], "partial"),
        ([([], [])], "none"),
    ],
)
async def test_coverage_takes_each_of_its_values(
    tmp_path: Path, rounds: list[Any], coverage: str
) -> None:
    async for world in _world(tmp_path, _Decider(*rounds)):
        response = await world.memory.search(_request())

        assert response.data.coverage == coverage
        # Partial or not, the core and the ranking still come back.
        assert response.data.episodes


async def test_the_whole_group_profile_is_attached_when_asked_and_capped(
    tmp_path: Path,
) -> None:
    async for world in _world(tmp_path, _Decider(([0], [])), cap=2):
        attached = (
            await world.memory.search(_request(include_group_profile=True))
        ).data.group_profile
        left_out = (
            await world.memory.search(_request(include_group_profile=False))
        ).data.group_profile
        personal = await world.memory.search(
            _request(asker=None, group_id=None, user_id="u_chen")
        )

        assert attached is not None
        assert len(attached.decisions) == 2  # three current, capped at two
        assert [t.name for t in attached.topics] == ["Lisbon offsite"]
        assert left_out is None
        assert personal.data.group_profile is None and personal.data.coverage is None
        assert personal.data.constraints == []


async def test_group_records_carry_their_audience(tmp_path: Path) -> None:
    async for world in _world(tmp_path, _Decider(([0, 1], []))):
        response = await world.memory.search(
            _request(asker="u_chen", include_group_profile=True)
        )
        personal = await world.memory.search(
            _request(asker=None, group_id=None, user_id="u_chen")
        )

        by_text = {e.episode: e.audience for e in response.data.episodes}
        early = by_text["early plan marker Lisbon offsite"]
        late = by_text["late plan marker Lisbon offsite"]
        assert early is not None and late is not None
        assert early.present == ["u_betty", "u_chen", "u_dan"]
        assert early.joined_after == ["u_evan"]
        assert late.present == ["u_chen", "u_dan", "u_evan"]
        assert late.joined_after == []
        profile = response.data.group_profile
        assert profile is not None and profile.decisions
        assert all(d.audience is not None for d in profile.decisions)
        assert all(e.audience is None for e in personal.data.episodes)


async def test_a_pool_of_groups_attaches_each_groups_own_items(
    tmp_path: Path,
) -> None:
    """An item id is unique within its group only: the first item each group
    writes on a day has the same id, and the pool still shows both."""
    async with GroupWorld.open(
        tmp_path, decider=FakeLLMClient(handler=_Decider(([0], [])))
    ) as world:
        for group in ("trip", "ops"):
            await world.roster(group, [join("u_chen", T0, display_name="Chen")])
            await world.episode(
                group, f"{group} plan marker", start=T0 + 1_000, end=T0 + 2_000
            )
            await world.profile_item(
                group, f"{group} decision marker", start=T0 + 1_000, end=T0 + 2_000
            )
        await world.project()
        response = await world.memory.search(
            _request(
                group_id=None, group_ids=["trip", "ops"], include_group_profile=True
            )
        )

    profile = response.data.group_profile
    assert profile is not None
    assert sorted((d.group_id, d.statement) for d in profile.decisions) == [
        ("ops", "ops decision marker"),
        ("trip", "trip decision marker"),
    ]


def _keys(entries: Any) -> set[tuple[str | None, str | None]]:
    return {(e.group_id, e.item_id) for e in entries}


async def test_by_default_only_the_items_linked_to_the_core_are_attached(
    tmp_path: Path,
) -> None:
    """The default decider keeps every candidate: the one episode is the core.

    Linked: a decision and a topic quoting its segment, the newest version of
    a decision whose older version does, and the open dissent about a linked
    decision. Not linked: what rests on other segments, and a role, which is
    never a constraint.
    """
    async with GroupWorld.open(tmp_path) as world:
        await world.roster(
            "trip",
            [
                join("u_chen", T0, display_name="Chen", role="owner"),
                join("u_betty", T0, display_name="Betty"),
            ],
        )
        lines = {
            "mc_core": (T0 + 1_000, "u_chen", "venue; Lisbon offsite; May 1; I lead"),
            "mc_elsewhere": (T0 + 2_000, "u_chen", "catering; Budget"),
            "mc_later": (T0 + 3_000, "u_chen", "April 30"),
            "mc_dissent": (T0 + 4_000, "u_betty", "too far"),
        }
        for memcell, (at, who, text) in lines.items():
            await world.segment(memcell, (f"m-{at}", who, text, at))
        await world.episode(
            "trip",
            "venue plan marker Lisbon offsite",
            start=T0 + 1_000,
            end=T0 + 1_000,
            memcell_id="mc_core",
        )

        async def item(text: str, memcell: str, **fields: Any) -> Any:
            at, who, _ = lines[memcell]
            return await world.profile_item(
                "trip",
                text,
                start=at,
                end=at,
                memcell_id=memcell,
                speaker=who,
                **fields,
            )

        venue = await item("venue", "mc_core")
        await item("catering", "mc_elsewhere")
        await item("Lisbon offsite", "mc_core", kind="topic")
        await item("Budget", "mc_elsewhere", kind="topic")
        await item(
            "I lead",
            "mc_core",
            kind="role",
            member_id="u_chen",
            role="coordinator",
            confidence="strong",
        )
        old = await item("May 1", "mc_core")
        await item("April 30", "mc_later", supersedes=old.entry_id)
        await item(
            "too far",
            "mc_dissent",
            kind="dissent",
            held_by=["u_betty"],
            about=venue.entry_id,
        )
        await world.project()

        default = (await world.memory.search(_request())).data
        without_block = (await world.memory.search(_request(constraints=False))).data
        whole = (await world.memory.search(_request(include_group_profile=True))).data

    linked = default.group_profile
    assert linked is not None
    assert sorted(d.statement for d in linked.decisions) == ["April 30", "venue"]
    assert [t.name for t in linked.topics] == ["Lisbon offsite"]
    assert [d.statement for d in linked.dissent] == ["too far"]
    assert linked.roles == []
    assert all(e.audience is not None for e in [*linked.decisions, *linked.topics])
    # Exactly the items the constraint block rests on.
    attached = _keys([*linked.topics, *linked.decisions, *linked.dissent])
    assert attached == {
        (c.scope.group_id, c.record_id)
        for c in default.constraints
        if c.kind in ("decision", "dissent", "topic")
    }
    # The constraint block turned off changes nothing in the attachment.
    assert without_block.constraints == []
    assert without_block.group_profile == linked
    # Asked for, the whole current profile.
    assert whole.group_profile is not None
    assert sorted(d.statement for d in whole.group_profile.decisions) == [
        "April 30",
        "catering",
        "venue",
    ]
    assert sorted(t.name for t in whole.group_profile.topics) == [
        "Budget",
        "Lisbon offsite",
    ]
    assert [r.role for r in whole.group_profile.roles] == ["coordinator"]


async def _pool_world(tmp_path: Path) -> AsyncIterator[GroupWorld]:
    """Two groups whose items share ids: each group's first item of the day."""
    async with GroupWorld.open(tmp_path) as world:
        for group in ("trip", "ops"):
            await world.roster(group, [join("u_chen", T0, display_name="Chen")])
        yield world


async def test_a_pool_links_each_groups_own_items_and_dissent(tmp_path: Path) -> None:
    """Both groups' first decisions are linked; the dissent about ops's decision
    disputes it, never trip's decision of the same id."""
    async for world in _pool_world(tmp_path):
        for group in ("trip", "ops"):
            at = T0 + 1_000
            await world.segment(
                f"mc_{group}",
                (f"m-{at}", "u_chen", f"{group} venue", at),
                session_id=f"s_{group}",
            )
            await world.episode(
                group,
                f"{group} plan marker",
                start=at,
                end=at,
                memcell_id=f"mc_{group}",
                session_id=f"s_{group}",
            )
        trip = await world.profile_item(
            "trip", "trip venue", start=T0 + 1_000, end=T0 + 1_000, memcell_id="mc_trip"
        )
        ops = await world.profile_item(
            "ops", "ops venue", start=T0 + 1_000, end=T0 + 1_000, memcell_id="mc_ops"
        )
        assert trip.entry_id == ops.entry_id  # the fixture's premise
        await world.segment(
            "mc_ops_dissent",
            (f"m-{T0 + 2_000}", "u_chen", "ops objection", T0 + 2_000),
            session_id="s_ops",
        )
        await world.profile_item(
            "ops",
            "ops objection",
            kind="dissent",
            start=T0 + 2_000,
            end=T0 + 2_000,
            memcell_id="mc_ops_dissent",
            held_by=["u_chen"],
            about=ops.entry_id,
        )
        await world.project()

        data = (
            await world.memory.search(
                _request(group_id=None, group_ids=["trip", "ops"])
            )
        ).data

    status = {
        (c.scope.group_id, c.statement): c.status
        for c in data.constraints
        if c.kind != "episode"
    }
    assert status == {
        ("trip", "trip venue"): "current",
        ("ops", "ops venue"): "disputed",
        ("ops", "ops objection"): "open",
    }
    assert data.group_profile is not None
    assert sorted((d.group_id, d.statement) for d in data.group_profile.decisions) == [
        ("ops", "ops venue"),
        ("trip", "trip venue"),
    ]
    assert [(d.group_id, d.statement) for d in data.group_profile.dissent] == [
        ("ops", "ops objection")
    ]


async def test_a_chain_is_followed_in_its_own_group(tmp_path: Path) -> None:
    """Trip's first decision was replaced by trip's second; ops's second item,
    of the same id and written later, is never taken for trip's newer version."""
    async for world in _pool_world(tmp_path):
        lines = {
            "mc_trip": (T0 + 1_000, "trip old"),
            "mc_trip_later": (T0 + 2_000, "trip new"),
            "mc_ops": (T0 + 1_500, "ops first"),
            "mc_ops_later": (T0 + 2_500, "ops second"),
        }
        for memcell, (at, text) in lines.items():
            group = memcell.split("_")[1]
            await world.segment(
                memcell, (f"m-{at}", "u_chen", text, at), session_id=f"s_{group}"
            )
        await world.episode(
            "trip",
            "trip plan marker",
            start=T0 + 1_000,
            end=T0 + 1_000,
            memcell_id="mc_trip",
            session_id="s_trip",
        )
        old = await world.profile_item(
            "trip", "trip old", start=T0 + 1_000, end=T0 + 1_000, memcell_id="mc_trip"
        )
        await world.profile_item(
            "trip",
            "trip new",
            start=T0 + 2_000,
            end=T0 + 2_000,
            memcell_id="mc_trip_later",
            supersedes=old.entry_id,
        )
        await world.profile_item(
            "ops", "ops first", start=T0 + 1_500, end=T0 + 1_500, memcell_id="mc_ops"
        )
        await world.profile_item(
            "ops",
            "ops second",
            start=T0 + 2_500,
            end=T0 + 2_500,
            memcell_id="mc_ops_later",
            observed_at=T0 + 9_000,
        )
        await world.project()

        data = (
            await world.memory.search(
                _request(group_id=None, group_ids=["trip", "ops"])
            )
        ).data

    decisions = [
        (c.scope.group_id, c.statement, [s.statement for s in c.supersedes])
        for c in data.constraints
        if c.kind == "decision"
    ]
    assert decisions == [("trip", "trip new", ["trip old"])]
    assert data.group_profile is not None
    assert [(d.group_id, d.statement) for d in data.group_profile.decisions] == [
        ("trip", "trip new")
    ]
