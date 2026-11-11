"""What a group search returns besides its ranking (tasks 15 and 16).

docs/group-memory.md, sections 5.3, 8.5 and 8.6. Real markdown, system.db
and LanceDB; a scripted decider. What these pin:

- ``coverage`` is ``complete`` when the decider stops with a core,
  ``partial`` when the loop ends first (no new core for the patience, or the
  round budget), ``none`` without a core -- and the core is returned either
  way, never an empty result because coverage is incomplete;
- a group scope attaches the visible current group-profile items by default
  (``include_group_profile``), each kind capped; ``false`` leaves them out;
  a user scope gets neither a group profile nor a coverage;
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


async def test_the_group_profile_is_attached_by_default_and_capped(
    tmp_path: Path,
) -> None:
    async for world in _world(tmp_path, _Decider(([0], [])), cap=2):
        attached = (await world.memory.search(_request())).data.group_profile
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
        response = await world.memory.search(_request(asker="u_chen"))
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
        assert profile is not None
        assert all(d.audience is not None for d in profile.decisions)
        assert all(e.audience is None for e in personal.data.episodes)
