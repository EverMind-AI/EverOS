"""Agent cases learned in a group stay in that group's context (tasks 4, 7).

docs/group-memory.md, section 9.1: a case extracted from a group segment
carries ``source_group_id`` and is returned only to reads in that group's
context. Real markdown, ``system.db`` and LanceDB (``tests/helpers/groups.py``).
What these pin:

- the case's markdown entry and index row carry its group;
- an agent read without a group context sees only cases learned outside
  groups; a read in one group's context sees those and that group's, never
  another group's -- on search and on get;
- a group context needs a current member of the group as the asker (the
  agent itself by default): anyone else is refused (403);
- a group context is only for an agent scope (422 otherwise).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from everos.core.errors import ForbiddenError
from everos.memory.models import AgentCase
from everos.memory.ports import NoSourceCheck, Scope
from everos.service import GetRequest, SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join

_AGENT = Scope(app_id=APP, project_id=PROJECT, owner_id="bot_ava", owner_type="agent")
_CASES = {
    None: "Fix the flaky billing date test by parsing UTC",
    "trip": "Book the team flights for the Lisbon offsite",
    "ops": "Rotate the Lisbon ops pager schedule",
}


async def _world(tmp_path: Path) -> Any:
    cm = GroupWorld.open(tmp_path)
    world = await cm.__aenter__()
    await world.roster(
        "trip",
        [
            join("u_chen", 0),
            join("bot_ava", 0, kind="assistant", role="assistant"),
        ],
    )
    await world.roster("ops", [join("u_dan", 0)])
    for group, intent in _CASES.items():
        await world.stores.agent_case.append(
            _AGENT,
            AgentCase(
                owner_id="bot_ava",
                task_intent=intent,
                approach="Steps.",
                quality_score=0.9,
                timestamp=1_772_323_200_000,
                session_id=f"s_{group}",
                parent_id=f"mc_{group}",
                source_group_id=group,
            ),
            only_if=NoSourceCheck(),
        )
    await world.project()
    return world, cm


def _search(**fields: Any) -> SearchRequest:
    return SearchRequest(
        app_id=APP,
        project_id=PROJECT,
        agent_id="bot_ava",
        query="Lisbon flights billing pager",
        method="keyword",
        top_k=10,
        **fields,
    )


def _intents(response: Any) -> set[str]:
    return {c.task_intent for c in response.data.agent_cases}


async def test_group_sourced_cases_stay_in_their_groups_context(tmp_path: Path) -> None:
    world, cm = await _world(tmp_path)
    try:
        files = list((tmp_path / "default_app" / PROJECT / "agents").rglob("*.md"))
        assert any("**source_group_id**: trip" in f.read_text() for f in files)

        plain = await world.memory.search(_search())
        assert _intents(plain) == {_CASES[None]}
        in_trip = await world.memory.search(_search(context_group_id="trip"))
        assert _intents(in_trip) == {_CASES[None], _CASES["trip"]}

        listed = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                agent_id="bot_ava",
                memory_type="agent_case",
                context_group_id="trip",
            )
        )
        assert {c.task_intent for c in listed.data.agent_cases} == {
            _CASES[None],
            _CASES["trip"],
        }
        plain_get = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                agent_id="bot_ava",
                memory_type="agent_case",
            )
        )
        assert {c.task_intent for c in plain_get.data.agent_cases} == {_CASES[None]}

        # A human member may read the agent's cases in the group's context.
        chen = await world.memory.search(
            _search(context_group_id="trip", asker_id="u_chen")
        )
        assert _intents(chen) == {_CASES[None], _CASES["trip"]}
        # The agent is not in "ops", and Dan is not in "trip".
        with pytest.raises(ForbiddenError):
            await world.memory.search(_search(context_group_id="ops"))
        with pytest.raises(ForbiddenError):
            await world.memory.search(
                _search(context_group_id="trip", asker_id="u_dan")
            )
    finally:
        await cm.__aexit__(None, None, None)


def test_a_group_context_is_only_for_an_agent_scope() -> None:
    with pytest.raises(ValueError):
        SearchRequest(user_id="u_chen", query="q", context_group_id="trip")
    with pytest.raises(ValueError):
        GetRequest(user_id="u_chen", memory_type="episode", context_group_id="trip")
