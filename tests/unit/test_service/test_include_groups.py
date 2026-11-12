"""Personal search and get that also cover the user's groups (``include_groups``).

Real markdown, ``system.db`` and LanceDB (``tests/helpers/groups.py``); a
scripted decider that keeps the first ten candidates and records what it was
shown. What these pin:

- off by default: a request without the field, or with it ``false``, is
  answered exactly as before -- same response, same decider input;
- on, the personal memory and the user's current groups form one candidate
  pool: the decider sees both in one rendering, the question names the user,
  and every returned record names its owner (``user``, or ``group`` with its
  ``group_id``);
- each group is read as a group search by the user reads it: the same
  windows, the same audience blocks; a group the user left or never joined
  adds nothing, and ``include_group_ids`` narrows the groups;
- the only group attachments are the user's own profile in each group and
  the groups' current decisions (capped, ``include_group_profile=false``
  leaves them out); no constraint block, no coverage, no link expansion, and
  each round shows the decider no more candidates than a personal search;
- ``/memory/get`` lists episodes and scenes of both as one list, sorted,
  counted and paged together.

The privacy cases (a former member, a later joiner, a forgotten message,
another member's records) are in the leak suite.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.config import Settings, load_settings
from everos.memory.models import ExplicitInfoItem, Profile, Scene
from everos.memory.ports import NoSourceCheck
from everos.memory.search.llm_multiround import RoundDecision
from everos.service import GetRequest, SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

T0 = 1_772_323_200_000

EARLY = "early trip marker Lisbon offsite venue"  # trip [1000, 2000]
LATE = "late trip marker Lisbon offsite date"  # trip [8000, 9000]
OPS = "ops marker Lisbon offsite budget"  # ops [1000, 1500]
SECRET = "secret marker Lisbon offsite"  # a group Chen is not in
CHEN = "personal marker Lisbon offsite tiles of Chen"
EVAN = "personal marker Lisbon offsite of Evan"
GROUP_TEXTS = (EARLY, LATE, OPS, SECRET)


class _Decider:
    """Keeps the first ten candidates and stops; records every prompt."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, messages: Any, **_kwargs: Any) -> str:
        self.prompts.append("\n".join(str(getattr(m, "content", m)) for m in messages))
        return json.dumps({"core": list(range(10)), "next_queries": []})

    def taken(self) -> list[str]:
        prompts, self.prompts = self.prompts, []
        return prompts


@dataclass
class _World:
    world: GroupWorld
    decider: _Decider


def _member_profile(member: str, *items: tuple[str, int, int]) -> Profile:
    return Profile(
        owner_id=member,
        summary=f"member summary of {member}",
        timestamp=T0 + 9_000,
        explicit_info=[
            ExplicitInfoItem(
                description=text,
                category="plans",
                item_id=f"pi_{member}_{start}",
                source_memcell_ids=[f"mc_{start}"],
                source_start=T0 + start,
                source_end=T0 + end,
            )
            for text, start, end in items
        ],
    )


@pytest.fixture
async def seeded(tmp_path: Path) -> AsyncIterator[_World]:
    decider = _Decider()
    async with GroupWorld.open(
        tmp_path, decider=FakeLLMClient(handler=decider)
    ) as world:
        await world.roster(
            "trip",
            [
                join("u_chen", T0, role="owner", display_name="Chen"),
                join("u_evan", T0 + 5_000, display_name="Evan"),
                join("u_betty", T0, display_name="Betty"),
                leave("u_betty", T0 + 6_000),
            ],
        )
        await world.roster("ops", [join("u_chen", T0, display_name="Chen O.")])
        await world.roster("secret", [join("u_zoe", T0)])
        await world.episode("trip", EARLY, start=T0 + 1_000, end=T0 + 2_000)
        await world.episode("trip", LATE, start=T0 + 8_000, end=T0 + 9_000)
        await world.episode("ops", OPS, start=T0 + 1_000, end=T0 + 1_500)
        await world.episode("secret", SECRET, start=T0 + 1_000, end=T0 + 1_500)
        await world.personal_episode("u_chen", CHEN, at=T0 + 500)
        await world.personal_episode("u_evan", EVAN, at=T0 + 500)
        await world.profile("u_chen", "personal profile marker Chen lives in Porto")
        await world.stores.profile.write(
            world.scope("trip"),
            _member_profile(
                "u_chen",
                ("chen early profile marker", 1_000, 2_000),
                ("chen late profile marker", 8_000, 9_000),
            ),
        )
        await world.stores.profile.write(
            world.scope("trip"),
            _member_profile("u_evan", ("evan profile marker", 8_000, 9_000)),
        )
        await world.profile_item(
            "trip", "early decision marker", start=T0 + 1_000, end=T0 + 2_000
        )
        await world.profile_item(
            "trip", "late decision marker", start=T0 + 8_000, end=T0 + 9_000
        )
        await world.profile_item(
            "trip", "trip topic marker", kind="topic", start=T0 + 8_000, end=T0 + 8_000
        )
        await world.profile_item(
            "ops", "ops decision marker", start=T0 + 1_000, end=T0 + 1_500
        )
        await world.profile_item(
            "secret", "secret decision marker", start=T0 + 1_000, end=T0 + 1_500
        )
        await world.project()
        decider.taken()
        yield _World(world=world, decider=decider)


def _request(user: str = "u_chen", **fields: Any) -> SearchRequest:
    values: dict[str, Any] = {
        "app_id": APP,
        "project_id": PROJECT,
        "user_id": user,
        "query": "marker Lisbon offsite",
        "top_k": 20,
        **fields,
    }
    return SearchRequest(**values)


def _blob(response: Any) -> str:
    return json.dumps(response.data.model_dump(mode="json"))


async def test_off_by_default_a_request_is_answered_as_before(
    seeded: _World,
) -> None:
    world, decider = seeded.world, seeded.decider
    plain = await world.memory.search(_request(include_profile=True))
    plain_prompts = decider.taken()
    off = await world.memory.search(
        _request(include_profile=True, include_groups=False)
    )
    off_prompts = decider.taken()

    assert off.data.model_dump(mode="json") == plain.data.model_dump(mode="json")
    assert off_prompts == plain_prompts and plain_prompts
    blob = _blob(off)
    assert CHEN in blob and "personal profile marker" in blob
    assert not any(text in blob for text in GROUP_TEXTS)
    assert "decision marker" not in blob and "chen early profile" not in blob
    assert off.data.scope is None and off.data.group_profile is None
    assert all(e.owner_type is None for e in off.data.episodes)
    # The picture field this adds is absent, not null, on a response without it.
    assert all(
        "owner_type" not in image
        for image in off.data.model_dump(mode="json")["images"]
    )


async def test_one_pool_holds_both_and_every_record_names_its_owner(
    seeded: _World,
) -> None:
    world, decider = seeded.world, seeded.decider
    response = await world.memory.search(_request(include_groups=True))
    prompts = decider.taken()

    by_text = {e.episode: e for e in response.data.episodes}
    assert set(by_text) == {CHEN, EARLY, LATE, OPS}
    mine = by_text[CHEN]
    assert (mine.owner_type, mine.user_id, mine.group_id) == ("user", "u_chen", None)
    for text, group in ((EARLY, "trip"), (LATE, "trip"), (OPS, "ops")):
        record = by_text[text]
        assert (record.owner_type, record.group_id, record.user_id) == (
            "group",
            group,
            None,
        )
        assert record.audience is not None and record.source is not None
    # One round, one block: personal and group candidates side by side, in
    # the same rendering, under a question that names the user.
    (prompt,) = prompts
    shown = prompt[prompt.rindex("CANDIDATES THIS ROUND:") :]
    assert shown.count("[block:") == 1
    assert all(text in shown for text in (CHEN, EARLY, LATE, OPS))
    assert SECRET not in prompt
    assert "QUESTION:\nChen O.: marker Lisbon offsite" in prompt
    # The scope block: the user, and the groups searched with the windows.
    scope = response.data.scope
    assert scope is not None
    assert (scope.owner_type, scope.asker_id, scope.audience) == (
        "user",
        "u_chen",
        "asker",
    )
    assert scope.group_ids == ["ops", "trip"]
    assert {(w.group_id, w.from_, w.to) for w in scope.windows} == {
        ("ops", T0, None),
        ("trip", T0, None),
    }
    assert response.data.constraints == [] and response.data.coverage is None


async def test_each_group_is_read_as_the_member_reads_it(seeded: _World) -> None:
    world = seeded.world
    mixed = await world.memory.search(_request("u_evan", include_groups=True))
    group = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_evan",
            query="marker Lisbon offsite",
            top_k=20,
        )
    )

    texts = {e.episode for e in mixed.data.episodes}
    assert texts == {EVAN, LATE}  # before his join, and a group he is not in: out
    group_view = {e.id: e for e in group.data.episodes}
    for record in mixed.data.episodes:
        if record.owner_type == "group":
            # The same record, with the same audience, as his group search.
            assert record.id in group_view
            assert record.audience == group_view[record.id].audience
            assert record.source == group_view[record.id].source
    assert mixed.data.scope is not None and group.data.scope is not None
    assert mixed.data.scope.windows == group.data.scope.windows


@pytest.mark.parametrize(
    ("only", "expected"),
    [(["ops"], {CHEN, OPS}), (["secret", "nowhere"], {CHEN}), (["trip"], None)],
)
async def test_include_group_ids_narrows_the_groups(
    seeded: _World, only: list[str], expected: set[str] | None
) -> None:
    response = await seeded.world.memory.search(
        _request(include_groups=True, include_group_ids=only)
    )

    texts = {e.episode for e in response.data.episodes}
    assert texts == (expected or {CHEN, EARLY, LATE})
    assert response.data.scope is not None
    joined = [g for g in only if g in {"ops", "trip"}]
    assert response.data.scope.group_ids == joined
    if not joined:
        # A group the user is not in is ignored, and says nothing about it.
        assert response.data.scope.windows == []
        assert response.data.group_profile is None


async def test_the_attachments_are_the_users_own_profiles_and_decisions(
    seeded: _World,
) -> None:
    world = seeded.world
    chen = await world.memory.search(
        _request(
            include_groups=True,
            include_profile=True,
            query="What does Evan plan for the Lisbon offsite?",
        )
    )
    evan = await world.memory.search(
        _request("u_evan", include_groups=True, include_profile=True)
    )
    without = await world.memory.search(
        _request(include_groups=True, include_group_profile=False)
    )

    profiles = [(p.owner_type, p.group_id, p.member_id) for p in chen.data.profiles]
    # The personal profile, then his own in each group that has one; never
    # Evan's, although the question names him.
    assert profiles == [("user", None, None), ("group", "trip", "u_chen")]
    assert "evan profile marker" not in _blob(chen)
    decisions = chen.data.group_profile
    assert decisions is not None
    assert {d.statement for d in decisions.decisions} == {
        "early decision marker",
        "late decision marker",
        "ops decision marker",
    }
    assert decisions.topics == decisions.dissent == decisions.roles == []
    assert all(d.audience is not None for d in decisions.decisions)
    # Evan's own profile items and decisions are cut to his window.
    evan_profiles = [(p.owner_type, p.member_id) for p in evan.data.profiles]
    assert evan_profiles == [("group", "u_evan")]  # he has no personal profile
    evan_blob = _blob(evan)
    assert "evan profile marker" in evan_blob
    assert "chen late profile marker" not in evan_blob
    assert evan.data.group_profile is not None
    assert [d.statement for d in evan.data.group_profile.decisions] == [
        "late decision marker"
    ]
    assert without.data.group_profile is None


async def test_the_decision_cap_is_the_group_searchs(tmp_path: Path) -> None:
    base = Settings()
    settings = base.model_copy(
        update={"search": base.search.model_copy(update={"group_profile_cap": 2})}
    )
    async with GroupWorld.open(tmp_path, settings=settings) as world:
        await world.roster("trip", [join("u_chen", T0)])
        await world.personal_episode("u_chen", CHEN, at=T0 + 500)
        for n in range(4):
            await world.profile_item(
                "trip", f"decision {n} marker", start=T0 + n, end=T0 + n
            )
        await world.project()
        response = await world.memory.search(_request(include_groups=True))

    assert response.data.group_profile is not None
    statements = [d.statement for d in response.data.group_profile.decisions]
    assert statements == ["decision 3 marker", "decision 2 marker"]  # newest first


async def test_a_former_member_and_a_user_in_no_group(seeded: _World) -> None:
    world = seeded.world
    betty = await world.memory.search(
        _request("u_betty", include_groups=True, include_profile=True)
    )
    named = await world.memory.search(
        _request("u_betty", include_groups=True, include_group_ids=["trip"])
    )

    for response in (betty, named):
        assert response.data.episodes == [] and response.data.images == []
        assert response.data.profiles == []
        assert response.data.group_profile is None
        assert response.data.scope is not None
        assert response.data.scope.group_ids == []
        assert response.data.scope.asker_id == "u_betty"


async def test_the_rl_seam_searches_the_same_pool(seeded: _World) -> None:
    world = seeded.world
    manager = world.memory._search_manager

    class _Keep:
        async def __call__(self, *_args: Any, **_kwargs: Any) -> RoundDecision:
            return RoundDecision(stop=True, queries=[], core=list(range(10)))

    mixed = await manager.search_episodes_with_decider(
        _request("u_evan", include_groups=True), _Keep()
    )
    plain = await manager.search_episodes_with_decider(_request("u_evan"), _Keep())

    assert {(e.episode, e.owner_type) for e in mixed} == {
        (EVAN, "user"),
        (LATE, "group"),
    }
    assert [(e.episode, e.owner_type) for e in plain] == [(EVAN, None)]


async def test_each_round_shows_no_more_candidates_than_a_personal_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The groups widen what competes for a block, not the block: the
    decider's input per round is bounded by ``[decider] seed_topk`` either
    way."""
    monkeypatch.setenv("EVEROS_DECIDER__SEED_TOPK", "3")
    load_settings.cache_clear()
    decider = _Decider()
    async with GroupWorld.open(
        tmp_path, decider=FakeLLMClient(handler=decider)
    ) as world:
        await world.roster("trip", [join("u_chen", T0, display_name="Chen")])
        for n in range(4):
            await world.personal_episode("u_chen", f"{CHEN} {n}", at=T0 + n)
            await world.episode(
                "trip", f"{EARLY} {n}", start=T0 + 10 + n, end=T0 + 10 + n
            )
        await world.project()
        await world.memory.search(_request())
        (personal,) = decider.taken()
        await world.memory.search(_request(include_groups=True))
        (mixed,) = decider.taken()

    def shown(prompt: str) -> list[str]:
        block = prompt[prompt.rindex("CANDIDATES THIS ROUND:") :]
        return [line for line in block.splitlines() if line.startswith("  ")]

    assert len(shown(personal)) == len(shown(mixed)) == 3
    assert any(EARLY in line for line in shown(mixed))


# ── /memory/get ──────────────────────────────────────────────────────────


def _listing(user: str = "u_chen", **fields: Any) -> GetRequest:
    values: dict[str, Any] = {
        "app_id": APP,
        "project_id": PROJECT,
        "user_id": user,
        "memory_type": "episode",
        "page_size": 50,
        **fields,
    }
    return GetRequest(**values)


async def test_get_lists_both_as_one_paged_list(seeded: _World) -> None:
    world = seeded.world
    plain = await world.memory.get(_listing())
    mixed = await world.memory.get(_listing(include_groups=True))
    first = await world.memory.get(
        _listing(include_groups=True, page_size=2, sort_order="asc")
    )
    evan = await world.memory.get(_listing("u_evan", include_groups=True))
    narrowed = await world.memory.get(
        _listing(include_groups=True, include_group_ids=["ops"])
    )

    assert [(e.episode, e.owner_type) for e in plain.data.episodes] == [(CHEN, None)]
    listed = [(e.episode, e.owner_type, e.group_id) for e in mixed.data.episodes]
    assert listed == [  # newest first; a group episode is dated by its end
        (LATE, "group", "trip"),
        (EARLY, "group", "trip"),
        (OPS, "group", "ops"),
        (CHEN, "user", None),
    ]
    assert mixed.data.total_count == 4
    assert [e.episode for e in first.data.episodes] == [CHEN, OPS]
    assert (first.data.count, first.data.total_count) == (2, 4)
    assert {e.episode for e in evan.data.episodes} == {EVAN, LATE}
    assert {e.episode for e in narrowed.data.episodes} == {CHEN, OPS}


async def test_get_lists_scenes_of_both(seeded: _World) -> None:
    world = seeded.world
    for owner, text, start in (
        (world.scope("trip"), "trip scene marker", T0 + 1_000),
        (world.personal("u_chen"), "chen scene marker", T0 + 100),
    ):
        await world.stores.scene.append(
            owner,
            Scene(
                owner_id=owner.owner_id,
                scene_id=f"cl_{owner.owner_id}",
                version=1,
                member_ids=["a", "b"],
                subject=text,
                summary=text,
                narrative=text,
                timestamp=T0 + 9_000,
                source_start=start,
            ),
            supersedes=None,
            only_if=NoSourceCheck(),
        )
    await world.project()

    mixed = await world.memory.get(_listing(memory_type="scene", include_groups=True))
    evan = await world.memory.get(
        _listing("u_evan", memory_type="scene", include_groups=True)
    )

    assert {(s.episode, s.owner_type) for s in mixed.data.scenes} == {
        ("trip scene marker", "group"),
        ("chen scene marker", "user"),
    }
    assert mixed.data.episodes == []
    assert evan.data.scenes == []  # the group scene began before he joined


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"memory_type": "profile"}, "include_groups lists episodes and scenes"),
        ({"audience": "group"}, "audience 'group'"),
        ({"user_id": None, "group_id": "trip", "asker_id": "u_chen"}, "user_id"),
    ],
)
def test_get_include_groups_is_refused_where_it_does_not_apply(
    fields: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _listing(include_groups=True, **fields)
    with pytest.raises(ValueError, match="include_group_ids goes with"):
        _listing(include_group_ids=["trip"])
